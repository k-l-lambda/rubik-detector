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
	var hints: Boolean = true,          // pass the tracker's turn signal to the follower
	var turnFrames: Int = 3,            // tracked frames in a row with the same turning layer that make a turn hint
	var hintSeconds: Double = 1.5,      // a hint is offered to the follower's detections for this long
	var twistSeconds: Double = 0.35,    // the mini cube's animation of a quarter turn (a half turn 1.6×)
	var catchUpSteps: Int = 3,          // frames tracked through when catching up from a detection's frame (fewer: the
	                                    // tracker thread back sooner; on a real video at 2× latency 3 beat 6, 12, 24)
)

enum class HudMode { NONE, TRACKED, LOST }

class Hud(
	val mini: MiniView,                 // the reconstructed cube, at the tracked orientation, a twist being played
	val stateText: String,              // the state's overall confidence
	val turning: Boolean,               // the tracker sees a layer turn (a run of turnFrames)
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
	// turn signal: the current run of frames with one turning layer, and the hints (run start → normal, pose, time)
	private var turnRun = 0
	private var turnRunStart = -1L
	private var turnRunNormal: DoubleArray? = null
	private var turnRunTime = 0.0
	private class Signal(val normal: DoubleArray, val track: Track, val time: Double, val start: Double)
	private val signals = LinkedHashMap<Long, Signal>()
	// the mini cube: the follower's moves animated, newest last (move, start time)
	private val animations = ArrayList<Pair<String, Double>>()
	private var movesSeen = 0
	private var lastPlacement = -1
	private var lastPlacementR: DoubleArray? = null
	// the tracked view frame → the state's frame (per view facelet the state facelet), and the facing pose it is for:
	// carried from frame to frame (the renaming between consecutive frames is unambiguous; between a detection and a
	// frame after a whole-cube turn it is not)
	private var viewMap: IntArray? = null
	private var viewMapR: DoubleArray? = null

	// b's facelet j as a's facelet: b's facelet k is a's facelet perms[g][k]
	private fun carried(map: IntArray, from: DoubleArray, to: DoubleArray): IntArray {
		val (_, g) = Cube.frameMatch(from, to)
		return IntArray(54) { map[Cube.perms[g][it]] }
	}

	// a placement for the detection of frame k (rotation Rk): the map from its view frame, carried along the frames
	// tracked since
	private fun placeAt(k: Long, placement: Int, Rk: DoubleArray) {
		var map = IntArray(54) { Cube.perms[placement][it] }
		var R = Rk
		val later = synchronized(history) { history.entries.filter { it.key >= k }.sortedBy { it.key }.map { it.value.first.R } }
		for (Ri in later) {
			map = carried(map, R, Ri)
			R = Ri
		}
		viewMap = map
		viewMapR = R
	}
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

	// a copy of the frame `index` (ring or pinned; null if gone): the frame a snapshot was detected on, for dumps
	fun copyFrame(index: Long): RawFrame? = synchronized(ring) {
		val f = ring[(index % ring.size).toInt()]?.takeIf { it.index == index } ?: pinned[index] ?: return@synchronized null
		val copy = RawFrame(ByteBuffer.allocateDirect(f.buffer.limit()).order(ByteOrder.nativeOrder()), f.width, f.height, f.stride, f.rotation, f.time, f.index)
		val src = f.buffer.duplicate()
		src.rewind()
		copy.buffer.put(src)
		copy.buffer.flip()
		copy
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
		return pipeline.process(detector, frame, time, prior, hint(index, time))
	}

	// the newest turn the tracker saw (within hintSeconds), its face's normal moved on to frame `index`
	private fun hint(index: Long, time: Double): TurnHint? {
		if (!settings.hints)
			return null
		val (id, sig) = synchronized(signals) { signals.entries.lastOrNull()?.toPair() } ?: return null
		if (time - sig.time > settings.hintSeconds)
			return null
		val now = synchronized(history) { history[index]?.first } ?: return TurnHint(id, sig.normal, sig.start, sig.time)
		val (_, g) = Cube.frameMatch(sig.track.R, now.R)
		val dR = Cube.mul(now.R, Cube.transpose(Cube.renamed(sig.track.R, g)))
		val n = sig.normal
		return TurnHint(id, DoubleArray(3) { r -> dR[r * 3] * n[0] + dR[r * 3 + 1] * n[1] + dR[r * 3 + 2] * n[2] }, sig.start, sig.time)
	}

	fun post(index: Long, snapshot: Snapshot) {
		synchronized(outcomes) { outcomes.add(Pair(index, snapshot)) }
	}

	// --- tracker thread ------------------------------------------------------------------------------------------

	private fun healthy(t: Track) = t.ok && t.inliers >= settings.healthyInliers && t.reprojection <= settings.healthyReprojection

	private var lastTrackTime = 0.0

	private fun duration(move: String) = settings.twistSeconds * (if (move.endsWith("2")) 1.6 else 1.0)

	private fun inverse(move: String) = when {
		move.endsWith("2") -> move
		move.endsWith("'") -> move.dropLast(1)
		else -> "$move'"
	}

	// the mini cube at frame time `time`: the state (follower's or the estimate's) with the follower's newest turns played
	private fun miniView(time: Double, track: Track?): Pair<MiniView, String> {
		val s = lastSnapshot
		// tracked: the tracked orientation with the carried map; not: the last placed detection's own frame
		val tracked = track?.takeIf { it.ok && viewMap != null && viewMapR === it.R }
		val R = tracked?.R ?: lastPlacementR ?: MiniCube.defaultR()
		val viewToState: IntArray? = when {
			tracked != null -> viewMap
			lastPlacementR != null && lastPlacement >= 0 -> IntArray(54) { Cube.perms[lastPlacement][it] }
			else -> null
		}
		val follow = s?.follow
		var colours = IntArray(54) { MiniCube.GREY }
		var letters: CharArray? = null
		var twist: MiniTwist? = null
		val text: String
		if (follow != null) {
			// play the turns still running: the state before them, the current one's layer part way
			animations.removeAll { (m, t) -> t + duration(m) <= time }
			var shown = follow.state
			val playing = animations.filter { (m, t) -> t + duration(m) > time }
			for ((m, _) in playing.reversed())
				shown = Engine.applyMoves(shown, inverse(m))
			if (playing.isNotEmpty() && viewToState != null) {
				val (m, t0) = playing.first()
				val progress = ((time - t0) / duration(m)).coerceIn(0.0, 1.0)
				if (progress > 0) {
					// the turned face of the state's frame in the view frame: the view face whose centre maps to it
					val viewFace = (0 until 6).firstOrNull { f -> Cube.FACES[viewToState[f * 9 + 4] / 9] == m[0] }
					if (viewFace != null)
						twist = MiniCube.twistFor(viewFace, m.substring(1), progress)
				}
			}
			if (viewToState != null) {
				colours = IntArray(54) { k -> follow.colours[shown[viewToState[k]]] ?: MiniCube.GREY }
				letters = CharArray(54) { k -> shown[viewToState[k]] }
			}
			text = String.format("state locked (confidence at lock %.2f), %d turns followed; %s", follow.lockConfidence, follow.moves.size, follow.decision)
		}
		else {
			val e = s?.estimate
			val net = s?.net
			if (net != null && viewToState != null && e?.state != null && e.legal)
				letters = CharArray(54) { k -> e.state[viewToState[k]] }
			if (net != null && viewToState != null)
				colours = IntArray(54) { k ->
					val j = viewToState[k]
					if (!net.known[j]) MiniCube.GREY else MiniCube.withConfidence(net.colour[j], net.confidence[j])
				}
			text = when {
				e == null || !e.ok -> "state: no evidence yet"
				e.legal -> String.format("state confidence %.2f (Π over the facelets; min %.2f), %d/54 observed, colours %.0f apart",
					e.confidence.fold(1.0) { a, c -> a * c }, e.minConfidence, e.coverage, s.clusterGap)
				else -> String.format("colours only: %d/54 observed (a legal state from 45)", e.coverage)
			}
		}
		return Pair(MiniView(R, colours, twist, letters), text)
	}

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
		lastTrackTime = time
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
				val map = viewMap
				val from = viewMapR
				if (map != null && from != null) {
					viewMap = carried(map, from, track.R)
					viewMapR = track.R
				}
				// a run of frames with one layer turning is a hint for the follower (one per run: its start is its id)
				val n0 = turnRunNormal
				if (track.turning && n0 != null && n0[0] * track.turnNormal[0] + n0[1] * track.turnNormal[1] + n0[2] * track.turnNormal[2] > 0.9)
					turnRun++
				else if (track.turning) {
					turnRun = 1
					turnRunStart = index
					turnRunTime = time
				}
				else
					turnRun = 0
				turnRunNormal = if (track.turning) track.turnNormal else null
				if (turnRun >= settings.turnFrames)
					synchronized(signals) {
						signals[turnRunStart] = Signal(track.turnNormal, track, time, turnRunTime)
						while (signals.size > 8)
							signals.remove(signals.keys.first())
					}
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
		// the placement of a view frame into the state's frame, with the rotation of that same detection: following,
		// the follower's last scored one (not necessarily this snapshot's detection); acquiring, this detection's
		val f0 = s.follow
		val d0 = s.detection
		if (f0 != null) {
			if (f0.placement >= 0 && f0.placementR != null) {
				lastPlacement = f0.placement
				lastPlacementR = f0.placementR
				if (d0 != null && f0.placementR === d0.R)   // the follower scored this frame's detection
					placeAt(k, f0.placement, d0.R)
			}
		}
		else if (s.placement >= 0 && d0 != null && s.good) {
			lastPlacement = s.placement
			lastPlacementR = d0.R
			placeAt(k, s.placement, d0.R)
		}
		s.follow?.let { f ->
			// turns the follower took since: animated on the mini cube, after the ones still playing
			if (f.moves.size > movesSeen) {
				val start = maxOf(lastTrackTime, animations.lastOrNull()?.let { (m, t) -> t + duration(m) } ?: 0.0)
				var t = start
				for (entry in f.moves.drop(movesSeen))
					for (m in entry.split(" ").filter { it.isNotEmpty() }) {
						animations.add(Pair(m, t))
						t += duration(m)
					}
			}
			movesSeen = f.moves.size
		}
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
		if (!s.good)
			return
		if (!s.fits) {
			// a good fit short of evidence grade: it starts a track when there is none, and votes against a track it
			// disagrees with (a track can slide onto a hand moving over the cube and stay healthy); it never blends in
			if (seeded) {
				val atK = synchronized(history) { history[k] }?.first
				val atM = synchronized(history) { history[lastTracked] }?.first
				if (atK != null && atM != null) {
					val dc = Math.sqrt((0 until 3).sumOf { (d.t[it] - atK.t[it]) * (d.t[it] - atK.t[it]) })
					val (angle, _) = Cube.frameMatch(atK.R, d.R)
					if (dc > settings.agreeCentre || angle > settings.agreeDeg) {
						disagreements++
						if (disagreements >= 2) {
							val (R, t) = movedOn(d.R, d.t, atK, atM)
							if (tracker.reseed(R, t)) {
								count("corrected by good fits")
								event(String.format("track corrected by two disagreeing detections (%.2f cubie, %.1f°)", dc, angle), s.time)
								disagreements = 0
							}
						}
						else
							count("disagreeing good fit")
					}
				}
				return
			}
		}
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
		val (mini, stateText) = miniView(time, if (mode == HudMode.TRACKED) track else null)
		return Hud(mini, stateText, turnRun >= settings.turnFrames, index, time, frame, mode, track?.R, track?.t, quads, colours, track,
			if (lastFitTime.isNaN()) Double.NaN else time - lastFitTime, lastSnapshot, ms, msApply, evs)
	}

	// tracker thread: drop the track and what it was seeded from (a reset of the app)
	fun resetTrack() {
		viewMap = null
		viewMapR = null
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
