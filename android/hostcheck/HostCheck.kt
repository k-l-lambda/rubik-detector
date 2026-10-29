// Host check of the app's pipeline (Engine.kt, Cube.kt, Pipeline.kt unchanged) on a rendered video from klstudio's
// 🎬 (frame*.png + labels.json): the camera timeline of tools/follow_tracker.py — frame k arrives at k / fps + 30 ms,
// the pipeline takes the newest arrived frame whenever it is free, each step costing its measured time × scale.
// Prints the stage changes, the turns followed and the state against the labels. Run: hostcheck/run.sh VIDEO_DIR [scale]
import com.klstudio.rubikdetector.Cube
import com.klstudio.rubikdetector.Detector
import com.klstudio.rubikdetector.Pipeline
import com.klstudio.rubikdetector.Stage
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.imageio.ImageIO

fun main(args: Array<String>) {
	val dir = File(args[0])
	val scale = args.getOrNull(1)?.toDouble() ?: 1.0
	// labels.json (a rendered video) or none (a real one: fps from argument 3, focal length from argument 4 or estimated)
	val labelFile = dir.resolve("labels.json")
	val labels = if (labelFile.exists()) labelFile.readText() else null
	val fps = labels?.let { Regex("\"fps\":\\s*([0-9.]+)").find(it)!!.groupValues[1].toDouble() } ?: (args.getOrNull(2)?.toDouble() ?: 30.0)
	val facelets = labels?.let { l -> Regex("\"facelets\":\\s*\"([LRDUBF]{54})\"").findAll(l).map { it.groupValues[1] }.toList() } ?: emptyList()
	val frames = dir.listFiles { f -> f.name.matches(Regex("frame\\d{4}\\.png")) }!!.sortedBy { it.name }
	val n = frames.size
	val truth = { i: Int -> facelets.getOrNull(1 + i) }   // cube.facelets first, then one per frame
	val arrive = { i: Int -> i / fps + 0.03 }

	val detector = Detector()
	val pipeline = Pipeline()
	args.getOrNull(3)?.toDouble()?.let { pipeline.useFocal(it) }
	var goodFrames = 0
	var lostRuns = 0
	var lostLongest = 0
	var lostRun = 0
	val modes = HashMap<String, Int>()
	// per detection mode: count, total ms of the frame, of the first detection, of the wide retry, engine-internal ms
	class Acc { var n = 0; var frame = 0.0; var rotate = 0.0; var first = 0.0; var retry = 0.0; var place = 0.0; var estimate = 0.0
		var follow = 0.0; var snapshot = 0.0; var engineSeg = 0.0; var enginePos = 0.0; var engineAxes = 0.0; var engineTotal = 0.0 }
	val acc = HashMap<String, Acc>()
	var free = arrive(0)
	var last = -1
	var stage: Stage? = null
	var lastSnapshot: com.klstudio.rubikdetector.Snapshot? = null
	var moves = 0
	var processed = 0
	var exactFollowing = 0
	var following = 0
	val buffer = ByteBuffer.allocateDirect(4096 * 4096 * 4).order(ByteOrder.nativeOrder())
	while (true) {
		var i = Math.floor((free - 0.03) * fps + 1e-9).toInt()
		if (i <= last)
			i = last + 1
		if (i >= n)
			break
		val image = ImageIO.read(frames[i])
		val w = image.width
		val h = image.height
		val rgb = image.getRGB(0, 0, w, h, null, 0, w)
		buffer.clear()
		for (p in rgb) {
			buffer.put((p shr 16 and 0xff).toByte())
			buffer.put((p shr 8 and 0xff).toByte())
			buffer.put((p and 0xff).toByte())
			buffer.put(0xff.toByte())
		}
		val start = maxOf(free, arrive(i))
		val t0 = System.nanoTime()
		val frame = detector.setFrame(buffer, w, h, w * 4, 0)
		val s = pipeline.process(detector, frame, i / fps)
		val ms = (System.nanoTime() - t0) / 1e6
		s.detection?.let { d ->
			val a = acc.getOrPut(d.mode) { Acc() }
			a.n++
			a.frame += ms
			a.rotate += frame.ms
			a.first += s.steps.first
			a.retry += s.steps.retry
			a.place += s.steps.place
			a.estimate += s.steps.estimate
			a.follow += s.steps.follow
			a.snapshot += s.steps.snapshot
			a.engineSeg += d.timings[0]
			a.engineAxes += d.timings[1]
			a.enginePos += d.timings[2]
			a.engineTotal += d.totalMs
		}
		free = start + ms * scale / 1000
		last = i
		processed++
		lastSnapshot = s
		if (s.detection != null && s.stage != Stage.FOCAL) {
			modes.merge(s.detection!!.mode, 1, Int::plus)
			if (s.good) {
				goodFrames++
				if (lostRun > 0)
					lostRuns++
				lostRun = 0
			}
			else {
				lostRun++
				lostLongest = maxOf(lostLongest, lostRun)
			}
		}
		if (s.stage != stage) {
			println(String.format("%6.2f s  frame %4d  stage %s", free, i + 1, s.stage))
			stage = s.stage
		}
		val f = s.follow
		if (f != null) {
			following++
			val t = truth(i)
			val exact = t != null && Cube.faceMap(f.state, t) != null
			if (exact)
				exactFollowing++
			if (f.moves.size > moves) {
				println(String.format("%6.2f s  frame %4d  turned %-6s → state %s", free, i + 1, f.moves.last(),
					if (t == null) "(no labels)" else if (exact) "exact" else "differs from the frame's labels"))
				moves = f.moves.size
			}
		}
		if (s.events.isNotEmpty() && s.events.last().contains("global"))
			println(String.format("%6.2f s  %s", free, s.events.last()))
	}
	println("$processed frames processed of $n; following on $following" +
		(if (labels != null) ", state = the frame's labels on $exactFollowing (the rest: a turn not yet confirmed)" else "") +
		"; focal ${pipeline.fx.toInt()} px")
	lastSnapshot?.let { ls ->
		val e = ls.estimate
		println("end: stage ${ls.stage}, evidence ${ls.evidenceFrames} frames in ${ls.chains} chains, ${ls.covered}/54 facelets observed" +
			(if (e != null) String.format(", estimate legal %b, min confidence %.2f, stable %d", e.legal, e.minConfidence, ls.stable) else ""))
		if (e?.state != null && e.legal)
			println("     state ${e.state}")
		if (e != null && e.ok) {
			val obs = pipeline.observations()
			for (k in 0 until 54)
				if (e.confidence[k] < 0.99) {
					val p = e.probability[k]
					val top = p.indices.sortedByDescending { p[it] }.take(2)
					println(String.format("     facelet %2d (%c%d): state %c, confidence %.2f, posterior %c %.2f / %c %.2f, %d observations, Lab median (%.0f %.0f %.0f)",
						k, Cube.FACES[k / 9], k % 9, e.state!![k], e.confidence[k], Cube.FACES[top[0]], p[top[0]], Cube.FACES[top[1]], p[top[1]], obs[k].size,
						obs[k].map { it[0] }.sorted().getOrElse(obs[k].size / 2) { 0f }, obs[k].map { it[1] }.sorted().getOrElse(obs[k].size / 2) { 0f },
						obs[k].map { it[2] }.sorted().getOrElse(obs[k].size / 2) { 0f }))
				}
		}
	}
	println("time per processed frame by detection mode (ms; first = the first engine detect, retry = the wide one after a failed local):")
	println(String.format("  %-9s %4s %7s %7s %7s %7s %7s %7s %7s %7s | last detect: segments %s position", "mode", "n", "frame", "rotate", "first", "retry", "place", "estim.", "follow", "snap", ""))
	var all = Acc()
	for ((mode, a) in acc.toSortedMap()) {
		println(String.format("  %-9s %4d %7.0f %7.1f %7.0f %7.0f %7.1f %7.1f %7.1f %7.1f | %7.0f %7.0f", mode, a.n, a.frame / a.n, a.rotate / a.n, a.first / a.n,
			a.retry / a.n, a.place / a.n, a.estimate / a.n, a.follow / a.n, a.snapshot / a.n, a.engineSeg / a.n, a.enginePos / a.n))
		all.n += a.n; all.frame += a.frame; all.first += a.first; all.retry += a.retry; all.place += a.place; all.estimate += a.estimate
		all.follow += a.follow; all.snapshot += a.snapshot; all.rotate += a.rotate
	}
	if (all.n > 0)
		println(String.format("  %-9s %4d %7.0f %7.1f %7.0f %7.0f %7.1f %7.1f %7.1f %7.1f", "all", all.n, all.frame / all.n, all.rotate / all.n, all.first / all.n,
			all.retry / all.n, all.place / all.n, all.estimate / all.n, all.follow / all.n, all.snapshot / all.n))
	val tracked = modes.values.sum()
	println(String.format("HUD: good fit on %d/%d processed frames (%.0f%%); %d lost spells, longest %d frames; modes %s",
		goodFrames, tracked, 100.0 * goodFrames / maxOf(1, tracked), lostRuns, lostLongest, modes))
	detector.close()
}
