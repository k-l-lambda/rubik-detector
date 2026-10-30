// The application layer: the state_tracker.py / follow_tracker.py prototype as one pipeline over camera frames.
// No Android imports (the host check runs it on rendered videos).
//
//	presence  every frame first: is there a cube at all (the engine's presence score of the detection, ≥
//	          presenceThreshold)? Nothing below it is evidence, followed, or drawn. Before the focal length is known
//	          the check runs with a provisional fx (provisionalFocal × long side; the score holds within ±25 %).
//	focal     the focal length from the first frame with a cube (Detector.estimateFocal), or given (camera intrinsics)
//	acquire   every good frame is evidence: frames linked to the previous one (small rotation up to a renaming,
//	          agreeing colours) form a chain placed through the renamings; a chain's first frame is placed by
//	          colour. Before each estimate: per facelet the samples far from its consensus colour (across chains)
//	          are left out (a finger over a sticker), and a layer the tracker saw turn loses its older evidence.
//	          Per facelet the Bayesian colour posterior (each chain — the frames of one hold stand or fall together
//	          — right with an unknown probability, uniform prior), from five faces the legal state. Lock when it is
//	          observed on all 54 facelets, every confidence ≥ lockConfidence, the same in lockStable estimates in a
//	          row.
//	follow    each well-fitted frame is matched against the states at most two face turns away (262, cached per
//	          state) over the 24 view renamings, under the locked colour model; a match has ≤ maxConflicts
//	          contradicting stickers. A new state replaces the followed one when it wins by ≥ switchMargin on
//	          `confirm` frames in a row (or ≥ strongMargin on one).
//	global    no match on globalAfter frames in a row: acquire again from fresh evidence.
package com.klstudio.rubikdetector

class Settings(
	var fallbackCost: Double = 0.15,
	var evidenceCost: Double = 0.09,
	var relativeCost: Double = 1.5,
	var linkDeg: Double = 30.0,       // a detection links to the last one turned at most this far (hand-held: 20 broke chains, misplaced by colour)
	var agreeColour: Double = 12.0,
	var legalFrom: Int = 45,
	var lockConfidence: Double = 0.99,
	var lockCoverage: Int = 51,       // facelets observed to lock (the rest from legality); hand-held turning leaves a few unseen
	var lockStable: Int = 3,
	var moveCost: Double = 10.0,
	var switchMargin: Double = 15.0,
	var strongMargin: Double = 80.0,
	var confirm: Int = 2,
	var hintBonus: Double = 8.0,        // follow: nll off the turn prior of candidates turning the face the tracker saw turn
	var consensusDelta: Double = 12.0,  // acquire: a sample further than this (Lab, L halved) from its facelet's consensus is left out
	// acquire: the unknown reliability per block of a chain (its frames within reliabilitySeconds stand or fall together:
	// consecutive frames of one hold share a finger, a reflection; 0 = per frame, which is overconfident: on a real video
	// confidence 0.998 for a state with two colours merged; a whole chain or 1 s blocks are too strict: no lock on v103;
	// 0.5 s locks it at 16 s, per frame at 12 s)
	var reliabilitySeconds: Double = 0.5,
	var dropCheckFacelets: Int = 4,     // acquire: a turn hint is decided once this many of the layer's facelets are seen before and after
	var dropChangedShare: Double = 0.4, // …and the layer's evidence dropped if at least this share of them changed colour (else kept)
	var carrySeconds: Double = 0.8,     // acquire: a chain begun in a hinted turn is placed by the geometry from a good frame this recent
	var minClusterGap: Double = 12.0,   // lock: the closest two letters' colours (median Lab of their facelets, L halved) at least this apart
	var conflictNll: Double = 4.0,
	var maxConflicts: Int = 2,
	var globalAfter: Int = 3,
	var maxEvidence: Int = 200,
	var workWidth: Int = 1024,
	// wide detection: cube-axes hypotheses (the engine's 8 suit photos; on a phone a frame that fails tries all 8 — 10 s on
	// a OnePlus 5T — and the next frame tries again anyway: real video p90 745 → 457 ms on the desktop, accepted 78 → 76/109)
	var axisHypotheses: Int = 2,
	var keepSegments: Boolean = true,
	var presenceThreshold: Double = 0.5,
	var provisionalFocal: Double = 1.0,
)

enum class Stage { FOCAL, ACQUIRE, FOLLOW }

class LinkInfo(val angle: Double, val colour: Double, val linked: Boolean)

// The tracker saw a layer turn (its points left the rigid pose for a few frames): the face's outward normal in the
// camera frame of the frame being detected. id: the tracked frame it was seen on (a hint is used for one switch).
class TurnHint(val id: Long, val normal: DoubleArray, val start: Double = 0.0, val end: Double = 0.0)   // start / end: the run's times (s)

class FollowInfo(
	val state: String,
	val moves: List<String>,
	val decision: String,
	val candidates: Int,
	val conflicts: Int,
	val stateConflicts: Int,
	val margin: Double?,
	val scoreMs: Double,
	val colours: Map<Char, Int>,
	val placementR: DoubleArray?,   // the detection the last step scored, and its view frame's placement into the
	val placement: Int,             // state's frame (rotation index; -1 none)
	val hinted: Boolean,            // the last switch was a turn of the face the tracker saw turn
	val lockConfidence: Double,     // Π of the facelet confidences of the estimate locked
)

// The cube net to draw: per facelet (common frame) whether anything is known, its colour (ARGB), the best component
// of its colour posterior (NaN: none to show), the confidence of the state's colour there, completed (not observed).
class Net(val known: BooleanArray, val colour: IntArray, val probability: DoubleArray, val confidence: DoubleArray, val inferred: BooleanArray) {
	companion object {
		val EMPTY = Net(BooleanArray(54), IntArray(54), DoubleArray(54) { Double.NaN }, DoubleArray(54), BooleanArray(54))
	}
}

// Where one frame's processing time went (ms): engine detections (first try / wide retry after a failed local fit),
// evidence placement (fusion by colour), state estimate, follow scoring, snapshot building (net, coverage).
class StepTimes(
	val first: Double = 0.0,
	val retry: Double = 0.0,
	val place: Double = 0.0,
	val estimate: Double = 0.0,
	val follow: Double = 0.0,
	val snapshot: Double = 0.0,
) {
	val total get() = first + retry + place + estimate + follow + snapshot
}

// What one processed frame leaves for the UI (immutable).
class Snapshot(
	val stage: Stage,
	val present: Boolean,           // a cube in the frame (presence ≥ threshold): only then are detection and HUD shown
	val presence: Double,
	val time: Double,
	val frame: FrameInfo,
	val fx: Double,
	val focal: Focal?,
	val detection: Detection?,
	val good: Boolean,
	val link: LinkInfo?,
	val evidenceFrames: Int,
	val chains: Int,
	val covered: Int,
	val estimate: Estimate?,
	val stable: Int,
	val follow: FollowInfo?,
	val net: Net,
	val processMs: Double,
	val events: List<String>,
	val steps: StepTimes,
	val fits: Boolean,              // good and evidence-grade (the relative cost gate): a pose to re-seed a tracker from
	val placement: Int,             // the detection's view frame placed into the estimate's frame (rotation index; -1 none)
	val clusterGap: Double,         // the estimate's closest two letters' colours (minClusterGap); NaN none
)

class Pipeline(val settings: Settings = Settings()) {
	var stage = Stage.FOCAL
		private set
	@Volatile var fx = 0.0          // focal length of the upright frame, pixels (read by the tracker thread)
	var focal: Focal? = null
	private var prev: Detection? = null
	private val costs = ArrayList<Double>()
	private val frames = ArrayList<Detection>()
	private val rotations = ArrayList<Int>()
	private val pending = ArrayList<Detection>()
	private var lastGood: Detection? = null   // the last good detection (prev is cleared on a bad one)
	// per chain begun with its link broken by colour only: its start time and the placement of its first frame (rel
	// identity) the geometry carried from the frame before
	private val carries = HashMap<Int, Pair<Double, Int>>()
	private var chains = 0
	private var estimate: Estimate? = null
	private var estimatedCount = 0
	private val stable = ArrayList<String>()
	private var follower: Follower? = null
	private val events = ArrayList<String>()
	// this frame's step times (ms), reset per frame
	private var tFirst = 0.0
	private var tRetry = 0.0
	private var tPlace = 0.0
	private var tEstimate = 0.0
	private var tFollow = 0.0

	private inline fun <T> timed(add: (Double) -> Unit, block: () -> T): T {
		val t0 = System.nanoTime()
		val r = block()
		add((System.nanoTime() - t0) / 1e6)
		return r
	}

	// a known focal length (camera intrinsics) instead of estimating it
	fun useFocal(fx: Double) {
		this.fx = fx
		if (stage == Stage.FOCAL)
			stage = Stage.ACQUIRE
	}

	fun reset(keepFocal: Boolean = true) {
		prev = null
		lastGood = null
		costs.clear()
		clearEvidence()
		follower = null
		events.clear()
		stage = if (keepFocal && fx > 0) Stage.ACQUIRE else Stage.FOCAL
	}

	private fun clearEvidence() {
		droppedUntil = -1.0
		pendingDrops.clear()
		carries.clear()
		frames.clear()
		rotations.clear()
		pending.clear()
		chains = 0
		estimate = null
		estimatedCount = 0
		stable.clear()
	}

	private fun event(text: String, time: Double) {
		events.add(String.format("%7.2f s  %s", time, text))
		if (events.size > 12)
			events.removeAt(0)
	}

	// One camera frame, already set upright on the detector. time: capture time in seconds. prior: the local search's
	// start (the frame tracker's pose at this frame) instead of the prediction from the last detections.
	fun process(detector: Detector, frame: FrameInfo, time: Double, prior: Pair<DoubleArray, DoubleArray>? = null, hint: TurnHint? = null): Snapshot {
		val started = System.nanoTime()
		tFirst = 0.0
		tRetry = 0.0
		tPlace = 0.0
		tEstimate = 0.0
		tFollow = 0.0
		if (stage == Stage.FOCAL) {
			// no focal estimate on a frame without a cube: it would fit the scene's lines
			val provisional = settings.provisionalFocal * maxOf(frame.width, frame.height)
			val probe = timed({ tFirst += it }) { detector.detect(frame, provisional, null, settings.workWidth, settings.keepSegments, settings.axisHypotheses) }
			probe.mode = "probe"
			probe.time = time
			if (probe.presence < settings.presenceThreshold)
				return snapshot(time, frame, probe, false, null, started)
			val f = detector.estimateFocal(frame)
			focal = f
			if (f.ok && f.fx > 0) {
				fx = f.fx
				stage = Stage.ACQUIRE
				event(String.format("focal length %.0f px (%.2f × long side, support %.2f, %.0f ms)", f.fx, f.ratio, f.support, f.ms), time)
			}
			else
				event("focal estimate failed", time)
			return snapshot(time, frame, probe, false, null, started)
		}
		val (r, link) = track(detector, frame, time, prior)
		val good = r.ok && r.latticeCost <= settings.fallbackCost && r.presence >= settings.presenceThreshold
		when (stage) {
			Stage.ACQUIRE -> timed({ tEstimate += it }) { acquire(time, r, hint) }
			Stage.FOLLOW -> if (good && fitsWell(r)) {
				val f = follower!!
				val decision = timed({ tFollow += it }) { f.step(r, hint) }
				if (decision == "switch")
					event("turned ${f.moves.last()} → followed", time)
				if (f.misses >= settings.globalAfter) {
					event("no match within two turns on ${f.misses} frames: global search", time)
					follower = null
					clearEvidence()
					stage = Stage.ACQUIRE
				}
			}
			else -> {}
		}
		return snapshot(time, frame, r, good, link, started)
	}

	private fun snapshot(time: Double, frame: FrameInfo, r: Detection?, good: Boolean, link: LinkInfo?, started: Long): Snapshot {
		val snapStarted = System.nanoTime()
		val covered = frames.indices.flatMap { v -> frames[v].samples.filter { it.valid }.map { Cube.perms[rotations[v]][it.facelet] } }.toSet().size
		val follow = follower?.info()
		val chainCount = frames.map { it.chain }.toSet().size
		val net = net(follow)
		val evs = events.toList()
		val steps = StepTimes(tFirst, tRetry, tPlace, tEstimate, tFollow, (System.nanoTime() - snapStarted) / 1e6)
		val presence = r?.presence ?: 0.0
		val fits = r != null && good && fitsWell(r)
		val placement = if (follow != null) follow.placement else (r?.rotation ?: -1)
		return Snapshot(stage, presence >= settings.presenceThreshold, presence, time, frame, fx, focal, r, good, link, frames.size, chainCount, covered, estimate,
			stable.size, follow, net, (System.nanoTime() - started) / 1e6, evs, steps, fits, placement, clusterGap)
	}

	private fun net(follow: FollowInfo?): Net {
		if (follow != null)
			return Net(BooleanArray(54) { true }, IntArray(54) { follow.colours[follow.state[it]] ?: 0xff606060.toInt() },
				DoubleArray(54) { Double.NaN }, DoubleArray(54) { 1.0 }, BooleanArray(54))
		val e = estimate ?: return Net.EMPTY
		if (!e.ok)
			return Net.EMPTY
		// the colour of each cluster: the median of the observations of the facelets it holds
		val letters = CharArray(54) { k -> if (e.legal) e.state!![k] else Cube.FACES[e.probability[k].indices.maxByOrNull { e.probability[k][it] }!!] }
		val obs = observations()
		val colours = HashMap<Char, Int>()
		for (c in Cube.FACES) {
			val labs = (0 until 54).filter { letters[it] == c && e.observed[it] }.flatMap { obs[it] }
			if (labs.isNotEmpty())
				colours[c] = Colors.labToArgb(FloatArray(3) { i -> labs.map { it[i] }.sorted()[labs.size / 2] })
		}
		val known = BooleanArray(54) { e.observed[it] || e.legal }
		return Net(known, IntArray(54) { colours[letters[it]] ?: 0xff606060.toInt() },
			DoubleArray(54) { k -> if (known[k]) e.probability[k].maxOrNull()!! else Double.NaN },
			DoubleArray(54) { k -> if (e.legal) e.confidence[k] else e.probability[k].maxOrNull()!! },
			BooleanArray(54) { !e.observed[it] && e.legal })
	}

	// --- tracking --------------------------------------------------------------------------------------------------

	// prior pose for a frame at `time`: the last good pose moved on at the velocity of the last two linked detections
	private fun predict(time: Double): Pair<DoubleArray, DoubleArray>? {
		val b = prev ?: return null
		val a = b.linkedTo ?: return if (b.mode == "fallback") null else Pair(b.R, b.t)
		val (_, g) = Cube.frameMatch(a.R, b.R)
		val delta = Cube.mul(Cube.transpose(Cube.renamed(a.R, g)), b.R)
		val s = (time - b.time) / maxOf(1e-3, b.time - a.time)
		val step = Cube.log(delta).map { it * s }.toDoubleArray()
		val R = Cube.mul(b.R, Cube.exp(step))
		val t = DoubleArray(3) { b.t[it] + (b.t[it] - a.t[it]) * s }
		return Pair(R, t)
	}

	fun fitsWell(r: Detection): Boolean {
		val recent = costs.takeLast(30)
		var limit = settings.evidenceCost
		if (recent.size >= 3)
			limit = minOf(limit, settings.relativeCost * recent.sorted()[recent.size / 2])
		return r.latticeCost <= limit
	}

	private fun colourDistance(a: Detection, b: Detection, g: Int): Double {
		val la = a.samples.filter { it.valid }.associateBy { it.facelet }
		val d = b.samples.filter { it.valid && Cube.perms[g][it.facelet] in la }.map { s ->
			val o = la.getValue(Cube.perms[g][s.facelet]).lab
			Math.sqrt((0 until 3).sumOf { ((s.lab[it] - o[it]) * (s.lab[it] - o[it])).toDouble() })
		}.sorted()
		return if (d.isEmpty()) 1e9 else d[d.size / 2]
	}

	private fun track(detector: Detector, frame: FrameInfo, time: Double, external: Pair<DoubleArray, DoubleArray>?): Pair<Detection, LinkInfo?> {
		var mode = "wide"
		var r: Detection? = null
		val prior = external ?: predict(time)
		if (prior != null) {
			r = timed({ tFirst += it }) { detector.detect(frame, fx, prior, settings.workWidth, settings.keepSegments) }
			mode = "local"
		}
		if (r == null || !r.ok || r.latticeCost > settings.fallbackCost || r.presence < settings.presenceThreshold) {
			r = timed({ if (prior != null) tRetry += it else tFirst += it }) { detector.detect(frame, fx, null, settings.workWidth, settings.keepSegments, settings.axisHypotheses) }
			mode = if (mode == "local") "fallback" else "wide"
		}
		r.mode = mode
		r.time = time
		val good = r.ok && r.latticeCost <= settings.fallbackCost && r.presence >= settings.presenceThreshold
		if (!good) {
			prev = null
			return Pair(r, null)
		}
		costs.add(r.latticeCost)
		if (costs.size > 200)
			costs.removeAt(0)
		var link: LinkInfo? = null
		var g = 0
		val p = prev
		if (p != null) {
			val m = Cube.frameMatch(p.R, r.R)
			g = m.second
			val colour = colourDistance(p, r, g)
			link = LinkInfo(m.first, colour, m.first <= settings.linkDeg && colour <= settings.agreeColour)
		}
		if (p != null && link!!.linked) {
			r.linkedTo = p
			p.linkedTo = null
			r.chain = p.chain
			r.rotation = p.rotation?.let { Cube.chain(it, g) }
			r.rel = Cube.chain(p.rel, g)
		}
		else {
			chains++
			r.chain = chains
			// the angle links, only the colours differ: maybe a layer turned between the two. The geometry's placement is
			// kept for a turn hint (it comes later, see carryTurned): placed by colour the chain takes the rotation hiding
			// the turn. Detection often fails during the turn itself: then from the last good frame, if recent.
			// (a frame of a chain not placed yet: by its own carried placement)
			val q = p ?: lastGood?.takeIf { time - it.time <= settings.carrySeconds }
			val qRotation = q?.let { f -> f.rotation ?: carries[f.chain]?.let { Cube.chain(it.second, f.rel) } }
			qRotation?.let { qr ->
				val (angle, h) = Cube.frameMatch(q!!.R, r.R)
				if (angle <= settings.linkDeg)
					carries[chains] = Pair(time, Cube.chain(qr, h))
			}
			r.rotation = null
			r.rel = Cube.identity
			pending.clear()
		}
		prev = r
		lastGood = r
		if (stage == Stage.ACQUIRE && fitsWell(r))
			addEvidence(r)
		return Pair(r, link)
	}

	// --- acquire ---------------------------------------------------------------------------------------------------

	private fun addEvidence(r: Detection) {
		val rot = r.rotation
		if (rot != null) {
			frames.add(r)
			rotations.add(rot)
		}
		else {
			pending.add(r)
			val anchor = pending[0]
			val placed = timed({ tPlace += it }) { place(anchor) }
			if (placed != null) {
				val start = Cube.chainStart(placed, anchor.rel)
				for (v in pending) {
					v.rotation = Cube.chain(start, v.rel)
					frames.add(v)
					rotations.add(v.rotation!!)
				}
				pending.clear()
			}
			else
				pending.removeAt(0)
		}
		while (frames.size > settings.maxEvidence) {
			frames.removeAt(0)
			rotations.removeAt(0)
			estimatedCount = minOf(estimatedCount, frames.size - 1)
		}
	}

	// a chain's frame placed by colour against the evidence so far (one frame per placed chain)
	private fun place(r: Detection): Int? {
		if (frames.isEmpty())
			return Cube.identity
		val anchors = ArrayList<Detection>()
		val rots = ArrayList<Int>()
		val seen = HashSet<Int>()
		for (v in frames.indices)
			if (seen.add(frames[v].chain)) {
				anchors.add(frames[v])
				rots.add(rotations[v])
			}
		anchors.add(r)
		rots.add(-1)
		val e = Engine.estimate(anchors, rots.toIntArray(), legal = false, partial = true, reliability = false)
		val g = e.rotations.last()
		return if (g >= 0) g else null
	}

	// the face of the state's frame whose outward normal (camera frame) is `n`, for a detection placed by g
	private fun faceOf(r: Detection, g: Int, n: DoubleArray): Char? {
		for (fv in 0 until 6) {
			val axis = fv / 2
			val sign = if (fv % 2 == 1) 1.0 else -1.0
			if ((0 until 3).sumOf { sign * r.R[it * 3 + axis] * n[it] } > 0.8)
				return Cube.FACES[Cube.perms[g][fv * 9 + 4] / 9]
		}
		return null
	}

	private var droppedUntil = -1.0     // the end of the last turn whose layer's evidence was dropped
	var clusterGap = Double.NaN         // the last estimate's (minClusterGap)
		private set

	// the closest two letters' colours of a state: per letter the median of its facelets' kept observations
	private fun gapOf(state: String): Double {
		val obs = observations()
		val med = Cube.FACES.map { c ->
			val labs = (0 until 54).filter { state[it] == c }.flatMap { obs[it] }
			if (labs.isEmpty()) null else DoubleArray(3) { i -> labs.map { it[i].toDouble() }.sorted()[labs.size / 2] * (if (i == 0) 0.5 else 1.0) }
		}
		var gap = Double.POSITIVE_INFINITY
		for (a in 0 until 6)
			for (b in a + 1 until 6) {
				val x = med[a] ?: continue
				val y = med[b] ?: continue
				gap = minOf(gap, Math.sqrt((0 until 3).sumOf { (x[it] - y[it]) * (x[it] - y[it]) }))
			}
		return gap
	}

	// A layer the tracker saw turn (a hint) is checked by colour before its older evidence goes: the tracker also signals
	// in whole-cube turns and blur, and a dropped layer costs the lock seconds. Once enough of its facelets are seen
	// before and after the turn, it is dropped if enough of them changed colour, else the hint is discarded.
	private class PendingDrop(val face: Char, val layer: List<Int>, val end: Double, val made: Double)
	// debugging: per hint taken (detection time, hint's start / end, normal in that detection's camera frame, face)
	class HintRecord(val time: Double, val start: Double, val end: Double, val normal: DoubleArray, val face: Char)
	val hintLog = ArrayList<HintRecord>()
	private val pendingDrops = ArrayList<PendingDrop>()

	// A chain begun while a hinted layer turned (its link broken by colour only) is placed by the geometry carried over
	// the turn, not by colour: its frames and the later ones of the chain renamed.
	private fun carryTurned(hint: TurnHint, time: Double) {
		val firsts = HashMap<Int, Detection>()
		for (v in frames)
			firsts.putIfAbsent(v.chain, v)
		for ((chain, first) in firsts) {
			val (start, carry) = carries[chain] ?: continue
			if (start < hint.start - 0.1 || start > hint.end + 0.5 || first.rotation == Cube.chain(carry, first.rel))
				continue
			for (v in frames.indices)
				if (frames[v].chain == chain) {
					frames[v].rotation = Cube.chain(carry, frames[v].rel)
					rotations[v] = frames[v].rotation!!
				}
			prev?.takeIf { it.chain == chain }?.let { it.rotation = Cube.chain(carry, it.rel) }
			stable.clear()
			estimatedCount = -1
			event(String.format("chain from %.2f s placed by the geometry over the turn", start), time)
		}
	}

	private fun dropTurned(r: Detection?, hint: TurnHint?, time: Double) {
		if (hint == null || r == null || hint.end <= droppedUntil)
			return
		carryTurned(hint, time)
		val g = r.rotation ?: return
		val face = faceOf(r, g, hint.normal) ?: return
		droppedUntil = hint.end
		hintLog.add(HintRecord(time, hint.start, hint.end, hint.normal, face))
		// the same run seen again (its end grew): the pending check moves on, no second one
		pendingDrops.find { it.face == face && hint.start <= it.end + 0.05 }?.let { p ->
			pendingDrops.remove(p)
			pendingDrops.add(PendingDrop(p.face, p.layer, hint.end, p.made))
			return
		}
		pendingDrops.add(PendingDrop(face, MiniCube.layer(Cube.FACES.indexOf(face)), hint.end, time))
		event("layer $face seen turning: checking its colours", time)
	}

	private fun decideDrops(time: Double) {
		val f = { lab: FloatArray -> doubleArrayOf(lab[0] * 0.5, lab[1].toDouble(), lab[2].toDouble()) }
		val median = { list: List<DoubleArray> -> DoubleArray(3) { c -> list.map { it[c] }.sorted()[list.size / 2] } }
		for (p in pendingDrops.toList()) {
			var seen = 0
			var changed = 0
			for (k in p.layer) {
				val pre = ArrayList<DoubleArray>()
				val post = ArrayList<DoubleArray>()
				for (v in frames.indices) {
					val fr = frames[v]
					if (fr.dropped?.get(k) == true)
						continue
					for (s in fr.samples)
						if (s.valid && Cube.perms[rotations[v]][s.facelet] == k)
							if (fr.time <= p.end) pre.add(f(s.lab)) else if (fr.time > p.end + 0.1) post.add(f(s.lab))
				}
				if (pre.size >= 2 && post.size >= 2) {
					seen++
					val a = median(pre)
					val b = median(post)
					if (Math.sqrt((0 until 3).sumOf { (a[it] - b[it]) * (a[it] - b[it]) }) >= settings.consensusDelta)
						changed++
				}
			}
			if (seen >= settings.dropCheckFacelets) {
				pendingDrops.remove(p)
				if (changed >= settings.dropChangedShare * seen) {
					var n = 0
					for (v in frames)
						if (v.time <= p.end + 0.1) {
							val d = v.dropped ?: BooleanArray(54).also { v.dropped = it }
							for (k in p.layer)
								d[k] = true
							n++
						}
					stable.clear()
					event("layer ${p.face} turned ($changed/$seen facelets changed): its evidence from $n frames before dropped", time)
				}
				else
					event("layer ${p.face}: $changed/$seen facelets changed, no turn: evidence kept", time)
			}
			else if (time - p.made > 6.0) {
				pendingDrops.remove(p)
				event("layer ${p.face}: not seen after the turn signal: evidence kept", time)
			}
		}
	}

	// per sample whether it is evidence: not on a dropped facelet, and within consensusDelta of its facelet's consensus
	// colour — per facelet the chain medians, the one with the most (√size-weighted) others within the delta
	private fun keepMasks() {
		val f = { lab: FloatArray -> doubleArrayOf(lab[0] * 0.5, lab[1].toDouble(), lab[2].toDouble()) }
		val dist = { a: DoubleArray, b: DoubleArray -> Math.sqrt((0 until 3).sumOf { (a[it] - b[it]) * (a[it] - b[it]) }) }
		val byFacelet = HashMap<Int, HashMap<Int, ArrayList<DoubleArray>>>()
		for (v in frames.indices)
			frames[v].samples.forEach { s ->
				val k = Cube.perms[rotations[v]][s.facelet]
				if (s.valid && frames[v].dropped?.get(k) != true)
					byFacelet.getOrPut(k) { HashMap() }.getOrPut(frames[v].chain) { ArrayList() }.add(f(s.lab))
			}
		val mode = HashMap<Int, DoubleArray>()
		for ((k, chains) in byFacelet) {
			val meds = chains.values.map { list -> DoubleArray(3) { c -> list.map { it[c] }.sorted()[list.size / 2] } }
			val w = chains.values.map { Math.sqrt(it.size.toDouble()) }
			mode[k] = meds[meds.indices.maxByOrNull { a -> meds.indices.sumOf { b -> if (dist(meds[a], meds[b]) < settings.consensusDelta) w[b] else 0.0 } }!!]
		}
		for (v in frames.indices) {
			val fr = frames[v]
			fr.keep = BooleanArray(fr.samples.size) { i ->
				val s = fr.samples[i]
				val k = Cube.perms[rotations[v]][s.facelet]
				s.valid && fr.dropped?.get(k) != true && mode[k]?.let { dist(f(s.lab), it) < settings.consensusDelta } ?: false
			}
		}
	}

	private fun acquire(time: Double, r: Detection?, hint: TurnHint?) {
		dropTurned(r, hint, time)
		if (frames.size == estimatedCount || frames.isEmpty())
			return
		estimatedCount = frames.size
		decideDrops(time)
		keepMasks()
		val covered = frames.indices.flatMap { v -> frames[v].samples.filterIndexed { i, s -> s.valid && frames[v].keep?.get(i) != false }
			.map { Cube.perms[rotations[v]][it.facelet] } }.toSet().size
		val e = Engine.estimate(frames, rotations.toIntArray(), legal = covered >= settings.legalFrom, partial = true, reliability = true,
			groups = if (settings.reliabilitySeconds > 0) IntArray(frames.size) { frames[it].chain * 4096 + (frames[it].time / settings.reliabilitySeconds).toInt() % 4096 } else null)
		estimate = e
		clusterGap = if (e.ok && e.legal) gapOf(e.state!!) else Double.NaN
		// a cube's six colours are distinct: two letters of one colour (a finger read as a sticker, two colours merged)
		// is no state to lock, however confident its facelets
		val sure = e.ok && e.legal && e.minConfidence >= settings.lockConfidence && e.coverage >= settings.lockCoverage &&
			clusterGap >= settings.minClusterGap
		if (sure && (stable.isEmpty() || stable.last() == e.state))
			stable.add(e.state!!)
		else {
			stable.clear()
			if (sure)
				stable.add(e.state!!)
		}
		if (stable.size >= settings.lockStable) {
			follower = Follower(settings, e.state!!, e.model, lockColours(e), e.confidence.fold(1.0) { a, c -> a * c })
			stage = Stage.FOLLOW
			event("locked ${e.state}", time)
		}
	}

	// per colour letter the median observed colour, for drawing the followed state
	private fun lockColours(e: Estimate): Map<Char, Int> {
		val labs = HashMap<Char, ArrayList<FloatArray>>()
		for (v in frames.indices)
			for ((i, s) in frames[v].samples.withIndex())
				if (s.valid && frames[v].keep?.get(i) != false)
					labs.getOrPut(e.state!![Cube.perms[rotations[v]][s.facelet]]) { ArrayList() }.add(s.lab)
		return labs.mapValues { (_, list) ->
			val m = FloatArray(3) { c -> list.map { it[c] }.sorted()[list.size / 2] }
			Colors.labToArgb(m)
		}
	}

	// the evidence frames and their placements, a chain's carried placement (host checks)
	fun evidence(): List<Pair<Detection, Int>> = frames.indices.map { Pair(frames[it], rotations[it]) }
	fun carryOf(chain: Int): Int? = carries[chain]?.second

	// the observations behind the current estimate, per common facelet (Lab)
	fun observations(): List<List<FloatArray>> {
		val out = List(54) { ArrayList<FloatArray>() }
		for (v in frames.indices)
			for ((i, s) in frames[v].samples.withIndex())
				if (s.valid && frames[v].keep?.get(i) != false)
					out[Cube.perms[rotations[v]][s.facelet]].add(s.lab)
		return out
	}
}

class Follower(private val settings: Settings, var state: String, private val model: Model, private val colours: Map<Char, Int>,
	private val lockConfidence: Double = 1.0) {
	val moves = ArrayList<String>()
	var misses = 0
		private set
	private var pending: Triple<String, String, Int>? = null
	private var decision = "locked"
	private var cacheState: String? = null
	private var candidates: List<Pair<String, String>> = emptyList()
	private var joined = ""
	private var lastConflicts = 0
	private var lastStateConflicts = 0
	private var lastMargin: Double? = null
	private var scoreMs = 0.0
	private var candidateFaces: List<String> = emptyList()   // per candidate the faces its moves turn
	private var lastR: DoubleArray? = null
	private var lastG = -1
	private var lastHintId = -1L
	private var lastHinted = false

	private fun neighbours(from: String, moves: String): List<Pair<String, String>> {
		val last = moves.split(" ").lastOrNull { it.isNotEmpty() }?.get(0)
		return Cube.MOVES.filter { it[0] != last }.map { Pair(Engine.applyMoves(from, it), (moves + " " + it).trim()) }
	}

	// (state, moves) within two face turns, distinct states, the fewest turns first
	private fun candidates(): List<Pair<String, String>> {
		if (cacheState != state) {
			val seen = hashSetOf(state)
			val out = arrayListOf(Pair(state, ""))
			val first = neighbours(state, "")
			for (c in first + first.flatMap { neighbours(it.first, it.second) })
				if (seen.add(c.first))
					out.add(c)
			candidates = out
			candidateFaces = out.map { (_, m) -> m.split(" ").filter { it.isNotEmpty() }.map { it[0] }.joinToString("") }
			joined = out.joinToString("") { it.first }
			cacheState = state
		}
		return candidates
	}

	// the face (letter of the state's frame) whose outward normal, with the view frame placed by g, is the hint's
	private fun hintedFace(r: Detection, g: Int, hint: DoubleArray): Char? {
		for (fv in 0 until 6) {
			val axis = fv / 2
			val sign = if (fv % 2 == 1) 1.0 else -1.0
			val n = DoubleArray(3) { sign * r.R[it * 3 + axis] }
			if (n[0] * hint[0] + n[1] * hint[1] + n[2] * hint[2] > 0.8)
				return Cube.FACES[Cube.perms[g][fv * 9 + 4] / 9]
		}
		return null
	}

	fun step(r: Detection, hint: TurnHint? = null): String {
		val started = System.nanoTime()
		val cands = candidates()
		val base = DoubleArray(cands.size) { settings.moveCost * cands[it].second.split(" ").count { m -> m.isNotEmpty() } }
		val usable = hint?.takeIf { it.id > lastHintId }
		var best = Double.MAX_VALUE
		var bi = 0
		var bg = 0
		var scores = DoubleArray(0)
		var bestPrior = base
		var bestHinted: Char? = null
		for (g in 0 until 24) {
			val s = Engine.scoreModel(model, r, g, joined)
			val face = usable?.let { hintedFace(r, g, it.normal) }
			val prior = if (face == null) base else DoubleArray(base.size) { if (face in candidateFaces[it]) base[it] - settings.hintBonus else base[it] }
			for (i in s.indices) {
				val v = s[i] + prior[i]
				if (v < best) {
					best = v
					bi = i
					bg = g
					scores = s
					bestPrior = prior
					bestHinted = face
				}
			}
		}
		lastR = r.R
		lastG = bg
		val withPrior = DoubleArray(scores.size) { scores[it] + bestPrior[it] }
		val stay = withPrior[0]
		lastConflicts = Engine.conflicts(model, r, bg, cands[bi].first, settings.conflictNll).size
		lastStateConflicts = Engine.conflicts(model, r, bg, state, settings.conflictNll).size
		scoreMs = (System.nanoTime() - started) / 1e6
		if (lastConflicts > settings.maxConflicts) {
			misses++
			pending = null
			lastMargin = null
			decision = "no match within 2 turns: $lastConflicts stickers conflict ($misses/${settings.globalAfter})"
			return "miss"
		}
		misses = 0
		if (bi == 0) {
			val next = withPrior.drop(1).minOrNull() ?: stay
			lastMargin = next - stay
			pending = null
			decision = String.format("stay (next best +%.0f)", next - stay)
			return "stay"
		}
		val (cand, mv) = cands[bi]
		val margin = stay - best
		lastMargin = margin
		if (margin < settings.switchMargin) {
			pending = null
			decision = String.format("%s? weak (+%.0f)", mv, margin)
			return "weak"
		}
		val p = pending
		val count = if (p != null && p.first == cand) p.third + 1 else 1
		// a turn of the face the tracker saw turn needs no second frame
		val hinted = bestHinted != null && bestHinted in candidateFaces[bi]
		if (count >= settings.confirm || margin >= settings.strongMargin || hinted) {
			state = cand
			moves.add(mv)
			pending = null
			lastHinted = hinted
			if (hinted)
				lastHintId = usable!!.id
			decision = String.format("turned %s (+%.0f%s)", mv, margin, if (hinted) ", turn seen" else "")
			return "switch"
		}
		pending = Triple(cand, mv, count)
		decision = String.format("%s? %d/%d (+%.0f)", mv, count, settings.confirm, margin)
		return "pending"
	}

	fun info() = FollowInfo(state, moves.toList(), decision, candidates.size, lastConflicts, lastStateConflicts, lastMargin, scoreMs, colours,
		lastR, lastG, lastHinted, lockConfidence)
}
