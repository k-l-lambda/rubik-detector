// The application layer: the state_tracker.py / follow_tracker.py prototype as one pipeline over camera frames.
// No Android imports (the host check runs it on rendered videos).
//
//	presence  every frame first: is there a cube at all (the engine's presence score of the detection, ≥
//	          presenceThreshold)? Nothing below it is evidence, followed, or drawn. Before the focal length is known
//	          the check runs with a provisional fx (provisionalFocal × long side; the score holds within ±25 %).
//	focal     the focal length from the first frame with a cube (Detector.estimateFocal), or given (camera intrinsics)
//	acquire   every good frame is evidence: frames linked to the previous one (small rotation up to a renaming,
//	          agreeing colours) form a chain placed through the renamings; a chain's first frame is placed by
//	          colour. Per facelet the Bayesian colour posterior (each frame right with an unknown probability,
//	          uniform prior), from five faces the legal state. Lock when it is observed on all 54 facelets, every
//	          confidence ≥ lockConfidence, the same in lockStable estimates in a row.
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
	var linkDeg: Double = 20.0,
	var agreeColour: Double = 12.0,
	var legalFrom: Int = 45,
	var lockConfidence: Double = 0.99,
	var lockCoverage: Int = 54,
	var lockStable: Int = 3,
	var moveCost: Double = 10.0,
	var switchMargin: Double = 15.0,
	var strongMargin: Double = 80.0,
	var confirm: Int = 2,
	var conflictNll: Double = 4.0,
	var maxConflicts: Int = 2,
	var globalAfter: Int = 3,
	var maxEvidence: Int = 200,
	var workWidth: Int = 1024,
	var keepSegments: Boolean = true,
	var presenceThreshold: Double = 0.5,
	var provisionalFocal: Double = 1.0,
)

enum class Stage { FOCAL, ACQUIRE, FOLLOW }

class LinkInfo(val angle: Double, val colour: Double, val linked: Boolean)

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
)

class Pipeline(val settings: Settings = Settings()) {
	var stage = Stage.FOCAL
		private set
	var fx = 0.0                    // focal length of the upright frame, pixels
	var focal: Focal? = null
	private var prev: Detection? = null
	private val costs = ArrayList<Double>()
	private val frames = ArrayList<Detection>()
	private val rotations = ArrayList<Int>()
	private val pending = ArrayList<Detection>()
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
		costs.clear()
		clearEvidence()
		follower = null
		events.clear()
		stage = if (keepFocal && fx > 0) Stage.ACQUIRE else Stage.FOCAL
	}

	private fun clearEvidence() {
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

	// One camera frame, already set upright on the detector. time: capture time in seconds.
	fun process(detector: Detector, frame: FrameInfo, time: Double): Snapshot {
		val started = System.nanoTime()
		tFirst = 0.0
		tRetry = 0.0
		tPlace = 0.0
		tEstimate = 0.0
		tFollow = 0.0
		if (stage == Stage.FOCAL) {
			// no focal estimate on a frame without a cube: it would fit the scene's lines
			val provisional = settings.provisionalFocal * maxOf(frame.width, frame.height)
			val probe = timed({ tFirst += it }) { detector.detect(frame, provisional, null, settings.workWidth, settings.keepSegments) }
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
		val (r, link) = track(detector, frame, time)
		val good = r.ok && r.latticeCost <= settings.fallbackCost && r.presence >= settings.presenceThreshold
		when (stage) {
			Stage.ACQUIRE -> timed({ tEstimate += it }) { acquire(time) }
			Stage.FOLLOW -> if (good && fitsWell(r)) {
				val f = follower!!
				val decision = timed({ tFollow += it }) { f.step(r) }
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
		return Snapshot(stage, presence >= settings.presenceThreshold, presence, time, frame, fx, focal, r, good, link, frames.size, chainCount, covered, estimate,
			stable.size, follow, net, (System.nanoTime() - started) / 1e6, evs, steps)
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

	private fun track(detector: Detector, frame: FrameInfo, time: Double): Pair<Detection, LinkInfo?> {
		var mode = "wide"
		var r: Detection? = null
		val prior = predict(time)
		if (prior != null) {
			r = timed({ tFirst += it }) { detector.detect(frame, fx, prior, settings.workWidth, settings.keepSegments) }
			mode = "local"
		}
		if (r == null || !r.ok || r.latticeCost > settings.fallbackCost || r.presence < settings.presenceThreshold) {
			r = timed({ if (prior != null) tRetry += it else tFirst += it }) { detector.detect(frame, fx, null, settings.workWidth, settings.keepSegments) }
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
			r.rotation = null
			r.rel = Cube.identity
			pending.clear()
		}
		prev = r
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

	private fun acquire(time: Double) {
		if (frames.size == estimatedCount || frames.isEmpty())
			return
		estimatedCount = frames.size
		val covered = frames.indices.flatMap { v -> frames[v].samples.filter { it.valid }.map { Cube.perms[rotations[v]][it.facelet] } }.toSet().size
		val e = Engine.estimate(frames, rotations.toIntArray(), legal = covered >= settings.legalFrom, partial = true, reliability = true)
		estimate = e
		val sure = e.ok && e.legal && e.minConfidence >= settings.lockConfidence && e.coverage >= settings.lockCoverage
		if (sure && (stable.isEmpty() || stable.last() == e.state))
			stable.add(e.state!!)
		else {
			stable.clear()
			if (sure)
				stable.add(e.state!!)
		}
		if (stable.size >= settings.lockStable) {
			follower = Follower(settings, e.state!!, e.model, lockColours(e))
			stage = Stage.FOLLOW
			event("locked ${e.state}", time)
		}
	}

	// per colour letter the median observed colour, for drawing the followed state
	private fun lockColours(e: Estimate): Map<Char, Int> {
		val labs = HashMap<Char, ArrayList<FloatArray>>()
		for (v in frames.indices)
			for (s in frames[v].samples)
				if (s.valid)
					labs.getOrPut(e.state!![Cube.perms[rotations[v]][s.facelet]]) { ArrayList() }.add(s.lab)
		return labs.mapValues { (_, list) ->
			val m = FloatArray(3) { c -> list.map { it[c] }.sorted()[list.size / 2] }
			Colors.labToArgb(m)
		}
	}

	// the observations behind the current estimate, per common facelet (Lab)
	fun observations(): List<List<FloatArray>> {
		val out = List(54) { ArrayList<FloatArray>() }
		for (v in frames.indices)
			for (s in frames[v].samples)
				if (s.valid)
					out[Cube.perms[rotations[v]][s.facelet]].add(s.lab)
		return out
	}
}

class Follower(private val settings: Settings, var state: String, private val model: Model, private val colours: Map<Char, Int>) {
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
			joined = out.joinToString("") { it.first }
			cacheState = state
		}
		return candidates
	}

	fun step(r: Detection): String {
		val started = System.nanoTime()
		val cands = candidates()
		val prior = DoubleArray(cands.size) { settings.moveCost * cands[it].second.split(" ").count { m -> m.isNotEmpty() } }
		var best = Double.MAX_VALUE
		var bi = 0
		var bg = 0
		var scores = DoubleArray(0)
		for (g in 0 until 24) {
			val s = Engine.scoreModel(model, r, g, joined)
			for (i in s.indices) {
				val v = s[i] + prior[i]
				if (v < best) {
					best = v
					bi = i
					bg = g
					scores = s
				}
			}
		}
		val withPrior = DoubleArray(scores.size) { scores[it] + prior[it] }
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
		if (count >= settings.confirm || margin >= settings.strongMargin) {
			state = cand
			moves.add(mv)
			pending = null
			decision = String.format("turned %s (+%.0f)", mv, margin)
			return "switch"
		}
		pending = Triple(cand, mv, count)
		decision = String.format("%s? %d/%d (+%.0f)", mv, count, settings.confirm, margin)
		return "pending"
	}

	fun info() = FollowInfo(state, moves.toList(), decision, candidates.size, lastConflicts, lastStateConflicts, lastMargin, scoreMs, colours)
}
