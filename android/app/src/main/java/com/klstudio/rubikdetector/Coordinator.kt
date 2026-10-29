// Tracking every frame, detecting in the background. No Android imports: the host check drives it on a simulated
// two-thread timeline, the app (CameraSession) on two real threads.
//
//	tracker thread   every camera frame: push() it into the ring, trackStep() — apply the detections finished
//	                 since (re-seeding the tracker), track the frame (FrameTracker: lattice vertices by Lucas–Kanade,
//	                 a few ms), and return the Hud: the tracked pose's facelet quads in the last detection's colours.
//	detector thread  whenever free: detectStep() on the newest frame — the Pipeline (presence, focal, evidence,
//	                 state, follow) with the tracker's pose at that frame as the local search's prior — and post() the
//	                 outcome for the tracker thread.
//
// A detection is for an older frame k than the tracker's newest m. It is compared with the tracker's own pose at k:
// agreeing, the tracker is re-seeded at its own pose pulled by `blend` towards the detection moved on to m by the
// tracker's own motion k → m (fresh points, drift corrected without taking on each detection's own error); not
// agreeing while the tracker is healthy, it waits for a second one (a detection a cubie off is no reason to drop a
// good track); with no track yet (or a lost one), the tracker is seeded at k from the ring and caught up to m.
// Several detections in a row without a cube end the track.
package com.klstudio.rubikdetector

import java.nio.ByteBuffer
import java.nio.ByteOrder

class TrackSettings(
	var agreeCentre: Double = 0.35,     // cubie: a detection at frame k agrees with the tracker's pose at k
	var agreeDeg: Double = 8.0,         // rotation, up to a renaming of the view frame
	var healthyInliers: Int = 12,       // the tracker's pose is trusted above these…
	var healthyReprojection: Double = 2.5,   // …and below this median reprojection error (px)
	var noCubeDrop: Int = 2,            // detections in a row without a cube that end the track
	var ring: Int = 24,                 // camera frames kept for catching up (at 30 fps 0.8 s; 16 lost detections on v103)
	var historyLength: Int = 64,        // tracked poses kept by frame index
	var blend: Double = 0.3,            // share of an agreeing detection taken into the tracked pose (1: replace it)
	var catchUpSteps: Int = 3,          // frames tracked through when catching up from a detection's frame (fewer: the
	                                    // tracker thread back sooner; on a real video at 2× latency 3 beat 6, 12, 24)
)

enum class HudMode { NONE, TRACKED, LOST }

class Hud(
	val index: Long,
	val time: Double,
	val frame: FrameInfo,
	val mode: HudMode,
	val R: DoubleArray?,                // facing pose of the tracker at this frame
	val t: DoubleArray?,
	val quads: List<Pair<Int, FloatArray>>,   // per R U F facelet (view frame) of the pose, its sampled-part quad
	val colours: IntArray,              // per those facelets, ARGB of the last detection's sample (0: none)
	val track: Track?,
	val detectionAge: Double,           // seconds since the frame of the detection last applied (NaN: none)
	val snapshot: Snapshot?,            // the last detection's
	val msTrack: Double,                // this frame on the tracker thread: rotate + track + apply
	val msApply: Double,                // of it: applying detections (catching up included)
	val events: List<String>,
)

// One camera frame as delivered (sensor orientation): a copy owned by the ring.
class RawFrame(var buffer: ByteBuffer, var width: Int, var height: Int, var stride: Int, var rotation: Int, var time: Double, var index: Long)

class Coordinator(val settings: TrackSettings = TrackSettings(), val pipeline: Pipeline = Pipeline()) {
	private val tracker = Tracker()
	private val ring = arrayOfNulls<RawFrame>(settings.ring)
	// the frames handed to the detector, kept until their detection is applied (a detection can outlast the ring)
	private val pinned = LinkedHashMap<Long, RawFrame>()
	private val history = LinkedHashMap<Long, Pair<Track, FrameInfo>>()   // tracked frames by index (facing poses)
	private val outcomes = ArrayList<Pair<Long, Snapshot>>()            // detections finished, not applied yet
	private var seeded = false
	private var lastTracked = -1L
	private var lastFit: Pair<Long, Detection>? = null                  // the last detection applied with a fitting pose
	private var lastFitTime = Double.NaN
	private var lastSnapshot: Snapshot? = null
	private var noCube = 0
	private var disagreements = 0
	private var everTracked = false
	private val events = ArrayList<String>()
	// counts of what happened to detections and tracks
	val stats = java.util.concurrent.ConcurrentHashMap<String, Int>()
	private fun count(key: String) = stats.merge(key, 1, Int::plus)

	private fun event(text: String, time: Double) {
		synchronized(events) {
			events.add(String.format("%7.2f s  %s", time, text))
			if (events.size > 12)
				events.removeAt(0)
		}
	}

	// --- the ring of camera frames (both threads) ----------------------------------------------------------------

	fun push(buffer: ByteBuffer, width: Int, height: Int, stride: Int, rotation: Int, time: Double, index: Long) {
		synchronized(ring) {
			val slot = (index % ring.size).toInt()
			val bytes = stride * height
			var f = ring[slot]
			if (f == null || f.buffer.capacity() < bytes) {
				f = RawFrame(ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()), width, height, stride, rotation, time, index)
				ring[slot] = f
			}
			buffer.rewind()
			f.buffer.clear()
			val src = buffer.duplicate()
			src.limit(minOf(src.capacity(), bytes))
			f.buffer.put(src)
			f.buffer.flip()
			f.width = width
			f.height = height
			f.stride = stride
			f.rotation = rotation
			f.time = time
			f.index = index
		}
	}

	// the frame `index` from the ring (or pinned), set upright by `set` while held (null if gone)
	private fun <T> withFrame(index: Long, set: (RawFrame) -> T): T? = synchronized(ring) {
		val f = ring[(index % ring.size).toInt()]?.takeIf { it.index == index } ?: pinned[index]
		if (f == null) null else set(f)
	}

	private fun pin(index: Long) = synchronized(ring) {
		val f = ring[(index % ring.size).toInt()]?.takeIf { it.index == index } ?: return@synchronized
		val copy = RawFrame(ByteBuffer.allocateDirect(f.buffer.limit()).order(ByteOrder.nativeOrder()), f.width, f.height, f.stride, f.rotation, f.time, f.index)
		val src = f.buffer.duplicate()
		src.rewind()
		copy.buffer.put(src)
		copy.buffer.flip()
		pinned[index] = copy
		while (pinned.size > 2)
			pinned.remove(pinned.keys.first())
	}

	// --- detector thread -----------------------------------------------------------------------------------------

	// The newest frame for a detection, with the tracker's pose at it as the prior (if healthy).
	fun detectStep(detector: Detector, index: Long): Snapshot? {
		pin(index)
		var time = 0.0
		val frame = withFrame(index) { f ->
			time = f.time
			detector.setFrame(f.buffer, f.width, f.height, f.stride, f.rotation)
		} ?: return null
		// the tracker's pose at this frame, if healthy (a lost track's last pose was tried: local searches from it found
		// a cubie-shifted fit often enough to start wrong tracks — 90 vs 16 frames > 0.3 cubie on a real video)
		val prior = synchronized(history) { history[index]?.first?.takeIf { healthy(it) } }?.let { Pair(it.R, it.t) }
		return pipeline.process(detector, frame, time, prior)
	}

	fun post(index: Long, snapshot: Snapshot) {
		synchronized(outcomes) { outcomes.add(Pair(index, snapshot)) }
	}

	// --- tracker thread ------------------------------------------------------------------------------------------

	private fun healthy(t: Track) = t.ok && t.inliers >= settings.healthyInliers && t.reprojection <= settings.healthyReprojection

	fun trackStep(index: Long): Hud? {
		val started = System.nanoTime()
		val pending = synchronized(outcomes) { val list = outcomes.toList(); outcomes.clear(); list }
		var time = 0.0
		val t0 = System.nanoTime()
		for ((k, s) in pending.sortedBy { it.first })
			apply(k, s)
		val msApply = (System.nanoTime() - t0) / 1e6
		val frame = withFrame(index) { f ->
			time = f.time
			tracker.setFrame(f.buffer, f.width, f.height, f.stride, f.rotation)
		} ?: return null
		var track = tracker.track(frame)
		if (seeded && !track.ok) {
			// one retry from the previous frame with fresh points at its pose (a blurred frame or a passing finger can
			// take most points at once)
			val last = synchronized(history) { history[lastTracked]?.first }
			val previous = if (last != null) withFrame(lastTracked) { f -> tracker.setFrame(f.buffer, f.width, f.height, f.stride, f.rotation) } else null
			if (last != null && previous != null) {
				tracker.track(previous)   // not tracking: only takes it as the frame to track from
				if (tracker.reseed(last.R, last.t)) {
					withFrame(index) { f -> tracker.setFrame(f.buffer, f.width, f.height, f.stride, f.rotation) }
					track = tracker.track(frame)
					count(if (track.ok) "recovered by a retry" else "retry failed")
				}
			}
		}
		lastTracked = index
		if (seeded) {
			if (track.ok) {
				remember(index, track, frame)
				everTracked = true
			}
			else {
				seeded = false
				count("track lost")
				event("track lost (${track.tracked} points followed)", time)
			}
		}
		return hud(index, time, frame, if (seeded) track else null, (System.nanoTime() - started) / 1e6, msApply)
	}

	private fun remember(index: Long, track: Track, frame: FrameInfo) {
		synchronized(history) {
			history[index] = Pair(track, frame)
			while (history.size > settings.historyLength)
				history.remove(history.keys.first())
		}
	}

	// a pose detected at frame k moved on to frame m by the tracker's motion k → m (camera frame). The two tracked poses
	// can be in different frames of the cube (a re-seed between them takes the detection's): the motion is the small
	// rotation between them up to a renaming (frameMatch), Rm · (Rk · G)ᵀ
	private fun movedOn(R: DoubleArray, t: DoubleArray, k: Track, m: Track): Pair<DoubleArray, DoubleArray> {
		val (_, g) = Cube.frameMatch(k.R, m.R)
		val Rk = Cube.renamed(k.R, g)
		val Rm = m.R
		val dR = Cube.mul(Rm, Cube.transpose(Rk))
		val dt = DoubleArray(3) { r -> m.t[r] - (dR[r * 3] * k.t[0] + dR[r * 3 + 1] * k.t[1] + dR[r * 3 + 2] * k.t[2]) }
		val R2 = Cube.mul(dR, R)
		val t2 = DoubleArray(3) { r -> dR[r * 3] * t[0] + dR[r * 3 + 1] * t[1] + dR[r * 3 + 2] * t[2] + dt[r] }
		return Pair(R2, t2)
	}

	private fun apply(k: Long, s: Snapshot) {
		lastSnapshot = s
		val d = s.detection ?: return
		if (!s.present) {
			noCube++
			if (seeded && noCube >= settings.noCubeDrop) {
				seeded = false
				count("ended: no cube")
				tracker.reset()
				event("no cube in $noCube detections: track ended", s.time)
			}
			return
		}
		noCube = 0
		// a track is corrected only by evidence-grade fits; with none, any good fit with a cube starts one
		if (!s.fits && (seeded || !s.good))
			return
		lastFit = Pair(k, d)
		lastFitTime = s.time
		val atK = synchronized(history) { history[k] }?.first
		val atM = synchronized(history) { history[lastTracked] }?.first
		if (seeded && atK != null && atM != null) {
			val dc = Math.sqrt((0 until 3).sumOf { (d.t[it] - atK.t[it]) * (d.t[it] - atK.t[it]) })
			val (angle, _) = Cube.frameMatch(atK.R, d.R)
			val agree = dc <= settings.agreeCentre && angle <= settings.agreeDeg
			if (agree || !healthy(atK) || disagreements >= 1) {
				var (R, t) = movedOn(d.R, d.t, atK, atM)
				if (agree && settings.blend < 1) {
					// the tracked pose pulled towards the detection (rotation: along the geodesic, in the tracker's naming)
					val (_, g) = Cube.frameMatch(atM.R, R)
					val Rd = Cube.renamed(R, g)
					val step = Cube.log(Cube.mul(Cube.transpose(atM.R), Rd)).map { it * settings.blend }.toDoubleArray()
					R = Cube.mul(atM.R, Cube.exp(step))
					t = DoubleArray(3) { atM.t[it] + settings.blend * (t[it] - atM.t[it]) }
				}
				if (tracker.reseed(R, t)) {
					count(if (agree) "re-seeded (agreeing)" else "corrected")
					if (!agree)
						event(String.format("track corrected by the detection (%.2f cubie, %.1f°)", dc, angle), s.time)
					disagreements = 0
					return
				}
			}
			else {
				count("kept against a detection")
				disagreements++
				event(String.format("detection disagrees with a healthy track (%.2f cubie, %.1f°): kept the track", dc, angle), s.time)
				return
			}
		}
		catchUp(k, d)
	}

	// seed at the detection's frame from the ring and track forward to the newest one
	private fun catchUp(k: Long, d: Detection) {
		val frame = withFrame(k) { f -> tracker.setFrame(f.buffer, f.width, f.height, f.stride, f.rotation) }
		if (frame == null) {
			count("frame left the ring")
			event("detection's frame left the ring: waiting for the next", d.time)
			return
		}
		tracker.seed(frame, pipeline.fx, d.R, d.t)
		// forward through the frames the ring holds after k (frames the tracker thread skipped never entered it), at most
		// catchUpSteps of them, the newest always (pyramidal flow takes a few frames' motion in one step)
		val held = synchronized(ring) { ring.filterNotNull().map { it.index }.filter { it in (k + 1)..lastTracked }.sorted() }
		synchronized(ring) { pinned.remove(k) }
		val steps = if (held.size <= settings.catchUpSteps) held
			else (1..settings.catchUpSteps).map { held[(it * held.size) / settings.catchUpSteps - 1] }
		var ok = true
		for (i in steps) {
			val fi = withFrame(i) { f -> tracker.setFrame(f.buffer, f.width, f.height, f.stride, f.rotation) }
			if (fi == null) {
				ok = false
				break
			}
			val tr = tracker.track(fi)
			if (!tr.ok) {
				ok = false
				break
			}
			remember(i, tr, fi)
		}
		seeded = ok
		count(if (ok) "caught up" else "catch-up failed")
		if (ok) {
			event("tracking from the detection of ${lastTracked - k} frames ago", d.time)
			disagreements = 0
		}
		else
			event("catching up from the detection failed", d.time)
	}

	private fun hud(index: Long, time: Double, frame: FrameInfo, track: Track?, ms: Double, msApply: Double): Hud {
		val mode = when {
			track != null && track.ok -> HudMode.TRACKED
			everTracked || lastFit != null -> HudMode.LOST
			else -> HudMode.NONE
		}
		var quads: List<Pair<Int, FloatArray>> = emptyList()
		var colours = IntArray(0)
		if (mode == HudMode.TRACKED && track != null && pipeline.fx > 0) {
			quads = Engine.projectFacelets(frame, pipeline.fx, track.R, track.t)
			// the last detection's colours on the tracked pose: its facelet k is the tracker's facelet perms[g][k]
			val fit = lastFit
			colours = IntArray(quads.size)
			if (fit != null) {
				val (_, g) = Cube.frameMatch(track.R, fit.second.R)
				val inverse = IntArray(54)
				Cube.perms[g].forEachIndexed { a, b -> inverse[b] = a }
				val byFacelet = fit.second.samples.filter { it.valid }.associateBy { it.facelet }
				quads.forEachIndexed { q, (facelet, _) ->
					byFacelet[inverse[facelet]]?.let { colours[q] = Colors.labToArgb(it.lab) }
				}
			}
		}
		val evs = synchronized(events) { events.toList() }
		return Hud(index, time, frame, mode, track?.R, track?.t, quads, colours, track,
			if (lastFitTime.isNaN()) Double.NaN else time - lastFitTime, lastSnapshot, ms, msApply, evs)
	}

	// tracker thread: drop the track and what it was seeded from (a reset of the app)
	fun resetTrack() {
		seeded = false
		everTracked = false
		lastFit = null
		lastFitTime = Double.NaN
		lastSnapshot = null
		noCube = 0
		disagreements = 0
		tracker.reset()
		synchronized(history) { history.clear() }
		synchronized(outcomes) { outcomes.clear() }
	}

	fun close() = tracker.close()
}
