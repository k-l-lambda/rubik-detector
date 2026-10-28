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
	val labels = dir.resolve("labels.json").readText()
	val fps = Regex("\"fps\":\\s*([0-9.]+)").find(labels)!!.groupValues[1].toDouble()
	val facelets = Regex("\"facelets\":\\s*\"([LRDUBF]{54})\"").findAll(labels).map { it.groupValues[1] }.toList()
	val frames = dir.listFiles { f -> f.name.matches(Regex("frame\\d{4}\\.png")) }!!.sortedBy { it.name }
	val n = frames.size
	val truth = { i: Int -> facelets[1 + i] }   // cube.facelets first, then one per frame
	val arrive = { i: Int -> i / fps + 0.03 }

	val detector = Detector()
	val pipeline = Pipeline()
	var free = arrive(0)
	var last = -1
	var stage: Stage? = null
	var moves = 0
	var processed = 0
	var exactFollowing = 0
	var following = 0
	val buffer = ByteBuffer.allocateDirect(1024 * 768 * 4).order(ByteOrder.nativeOrder())
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
		free = start + ms * scale / 1000
		last = i
		processed++
		if (s.stage != stage) {
			println(String.format("%6.2f s  frame %4d  stage %s", free, i + 1, s.stage))
			stage = s.stage
		}
		val f = s.follow
		if (f != null) {
			following++
			val exact = Cube.faceMap(f.state, truth(i)) != null
			if (exact)
				exactFollowing++
			if (f.moves.size > moves) {
				println(String.format("%6.2f s  frame %4d  turned %-6s → state %s", free, i + 1, f.moves.last(), if (exact) "exact" else "differs from the frame's labels"))
				moves = f.moves.size
			}
		}
		if (s.events.isNotEmpty() && s.events.last().contains("global"))
			println(String.format("%6.2f s  %s", free, s.events.last()))
	}
	println("$processed frames processed of $n; following on $following, state = the frame's labels on $exactFollowing " +
		"(the rest: a turn not yet confirmed); focal ${pipeline.fx.toInt()} px")
	detector.close()
}
