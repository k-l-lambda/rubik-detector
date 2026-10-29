// Host check of the tracking + background detection loop (Coordinator.kt, unchanged) on a video: the app's two threads
// simulated on the camera timeline — frame k arrives at k / fps + 30 ms; the tracker thread takes the newest arrived
// frame whenever free (CameraX keep-only-latest), the detector thread the newest frame the tracker has pushed whenever
// free; each step costs its measured time × scale, a detection is applied by the tracker once ready. Every camera frame
// is displayed with the newest HUD ready by its arrival. With reference poses (DIR/eval/results.json: tools/real_video.py
// or a render's labels) the HUD's error against the displayed frame's pose, against the detection-only HUD.
//	MAIN=CoordCheckKt hostcheck/run.sh VIDEO_DIR [scale] [OUT_DIR for a HUD video] [fx]
import com.klstudio.rubikdetector.Coordinator
import com.klstudio.rubikdetector.Cube
import com.klstudio.rubikdetector.Detector
import com.klstudio.rubikdetector.Hud
import com.klstudio.rubikdetector.HudMode
import com.klstudio.rubikdetector.Snapshot
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.imageio.ImageIO
import kotlin.math.hypot

private class Ref(val R: DoubleArray, val t: DoubleArray)

// per frame the reference pose (null: none, a poor fit, or a turning layer)
private fun parseRefs(file: File): List<Ref?> {
	if (!file.exists())
		return emptyList()
	val num = "[-0-9.eE+]+"
	return file.readText().split("{\"frame\":").drop(1).map { p ->
		val ok = Regex("\"ok\":\\s*true").containsMatchIn(p)
		val cost = Regex("\"cost\":\\s*($num)").find(p)?.groupValues?.get(1)?.toDouble() ?: 1.0
		val r = Regex("\"R\":\\s*\\[\\[([^\\]]*)\\],\\s*\\[([^\\]]*)\\],\\s*\\[([^\\]]*)\\]\\]").find(p)
		val t = Regex("\"t\":\\s*\\[([^\\]]*)\\]").find(p)
		if (!ok || r == null || t == null || cost > 0.08) null
		else Ref((1..3).flatMap { r.groupValues[it].split(",").map { x -> x.trim().toDouble() } }.toDoubleArray(),
			t.groupValues[1].split(",").map { it.trim().toDouble() }.toDoubleArray())
	}
}

// the pose renamed so the faces turned to the camera are +x +y +z
private fun facing(R: DoubleArray, t: DoubleArray): DoubleArray {
	val c = DoubleArray(3) { k -> -(R[k] * t[0] + R[3 + k] * t[1] + R[6 + k] * t[2]) }
	var best = Cube.rotations[0]
	var score = -1e300
	for (G in Cube.rotations) {
		val s = (0 until 3).minOf { r -> (0 until 3).sumOf { k -> G[k * 3 + r] * c[k] } }
		if (s > score) {
			score = s
			best = G
		}
	}
	return Cube.mul(R, best.map { it.toDouble() }.toDoubleArray())
}

private val CENTRES = (0 until 3).flatMap { a ->
	val b = (a + 1) % 3
	val c = (a + 2) % 3
	(-1..1).flatMap { u -> (-1..1).map { v -> DoubleArray(3).also { it[a] = 1.5; it[b] = u.toDouble(); it[c] = v.toDouble() } } }
}

private fun centres(R: DoubleArray, t: DoubleArray, fx: Double, cx: Double, cy: Double): List<DoubleArray> = CENTRES.map { p ->
	val q = DoubleArray(3) { r -> R[r * 3] * p[0] + R[r * 3 + 1] * p[1] + R[r * 3 + 2] * p[2] + t[r] }
	doubleArrayOf(fx * q[0] / q[2] + cx, fx * q[1] / q[2] + cy)
}

// mean distance of a's sticker centres to the nearest of b's, in b's sticker pitch (both renamed to face the camera)
private fun centreError(Ra: DoubleArray, ta: DoubleArray, Rb: DoubleArray, tb: DoubleArray, fx: Double, cx: Double, cy: Double): Double {
	val a = centres(facing(Ra, ta), ta, fx, cx, cy)
	val b = centres(facing(Rb, tb), tb, fx, cx, cy)
	val d = a.map { p -> b.minOf { q -> hypot(p[0] - q[0], p[1] - q[1]) } }
	val pitch = b.map { p -> b.filter { it !== p }.minOf { q -> hypot(p[0] - q[0], p[1] - q[1]) } }.sorted()[b.size / 2]
	return d.average() / pitch
}

private fun pct(v: List<Double>, q: Double) = if (v.isEmpty()) Double.NaN else v.sorted()[minOf(v.size - 1, (q * v.size).toInt())]

fun main(args: Array<String>) {
	val dir = File(args[0])
	val scale = args.getOrNull(1)?.toDouble() ?: 1.0
	val outDir = args.getOrNull(2)?.takeIf { it.isNotEmpty() && it != "-" }?.let { File(it) }
	val fxGiven = args.getOrNull(3)?.toDoubleOrNull()
	val labelFile = dir.resolve("labels.json")
	val fps = if (labelFile.exists()) Regex("\"fps\":\\s*([0-9.]+)").find(labelFile.readText())!!.groupValues[1].toDouble() else 30.0
	val frames = dir.listFiles { f -> f.name.matches(Regex("frame\\d{4}\\.png")) }!!.sortedBy { it.name }
	val n = frames.size
	val refs = parseRefs(dir.resolve("eval/results.json"))
	val camera = 0.03
	val arrive = { i: Int -> i / fps + camera }

	val ts = com.klstudio.rubikdetector.TrackSettings()
	System.getenv("RING")?.toIntOrNull()?.let { ts.ring = it }
	System.getenv("BLEND")?.toDoubleOrNull()?.let { ts.blend = it }
	System.getenv("CATCHUP")?.toIntOrNull()?.let { ts.catchUpSteps = it }
	val coord = Coordinator(ts)
	System.getenv("WORK_WIDTH")?.toIntOrNull()?.let { coord.pipeline.settings.workWidth = it }
	fxGiven?.let { coord.pipeline.useFocal(it) }
	val detector = Detector()
	var buffer: ByteBuffer? = null
	var w = 0
	var h = 0
	val load = { i: Int ->
		val image = ImageIO.read(frames[i])
		w = image.width
		h = image.height
		val rgb = image.getRGB(0, 0, w, h, null, 0, w)
		if (buffer == null || buffer!!.capacity() < w * h * 4)
			buffer = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
		val b = buffer!!
		b.clear()
		for (p in rgb) {
			b.put((p shr 16 and 0xff).toByte())
			b.put((p shr 8 and 0xff).toByte())
			b.put((p and 0xff).toByte())
			b.put(0xff.toByte())
		}
		b.flip()
		b
	}

	val huds = ArrayList<Pair<Double, Hud>>()                     // (ready, hud)
	val detections = ArrayList<Triple<Double, Long, Snapshot>>()  // (ready, frame, snapshot)
	val pending = ArrayList<Triple<Double, Long, Snapshot>>()
	val pushTime = HashMap<Long, Double>()
	var trackerFree = arrive(0)
	var detectorFree = 0.0
	var trackerLast = -1
	var detectorLast = -1L
	val trackMs = ArrayList<Double>()
	val detectMs = ArrayList<Double>()
	while (true) {
		var ti = Math.floor((trackerFree - camera) * fps + 1e-9).toInt()
		if (ti <= trackerLast)
			ti = trackerLast + 1
		val trackerStart = if (ti < n) maxOf(trackerFree, arrive(ti)) else Double.POSITIVE_INFINITY
		val detectorStart = if (trackerLast >= 0 && trackerLast.toLong() > detectorLast) maxOf(detectorFree, pushTime[trackerLast.toLong()]!!) else Double.POSITIVE_INFINITY
		if (trackerStart.isInfinite() && detectorStart.isInfinite())
			break
		if (trackerStart <= detectorStart) {
			val ready = pending.filter { it.first <= trackerStart }
			pending.removeAll(ready.toSet())
			for ((_, k, s) in ready)
				coord.post(k, s)
			val buf = load(ti)
			coord.push(buf, w, h, w * 4, 0, ti / fps, ti.toLong())
			val t0 = System.nanoTime()
			val hud = coord.trackStep(ti.toLong())
			val ms = (System.nanoTime() - t0) / 1e6
			trackMs.add(ms)
			trackerFree = trackerStart + ms * scale / 1000
			pushTime[ti.toLong()] = trackerStart
			if (hud != null)
				huds.add(Pair(trackerFree, hud))
			trackerLast = ti
		}
		else {
			val di = trackerLast.toLong()
			val t0 = System.nanoTime()
			val snap = coord.detectStep(detector, di)
			val ms = (System.nanoTime() - t0) / 1e6
			detectMs.add(ms)
			detectorFree = detectorStart + ms * scale / 1000
			detectorLast = di
			if (snap != null) {
				pending.add(Triple(detectorFree, di, snap))
				detections.add(Triple(detectorFree, di, snap))
			}
		}
	}

	// display: every camera frame with the newest HUD ready by its arrival
	val fx = coord.pipeline.fx
	var hi = -1
	var di = -1
	val trackedErr = ArrayList<Double>()
	val ownErr = ArrayList<Double>()           // the HUD against its own frame's reference (no display lag)
	val phase = if (labelFile.exists()) Regex("\"phase\":\\s*\"([a-z]+)\"").findAll(labelFile.readText()).map { it.groupValues[1] }.toList() else emptyList()
	val byPhase = HashMap<String, ArrayList<Double>>()
	val detErr = ArrayList<Double>()
	val trackedAge = ArrayList<Double>()
	val detAge = ArrayList<Double>()
	var shownTracked = 0
	var shownDet = 0
	outDir?.let { it.resolve("frames").mkdirs() }
	val outScale = 0.5
	for (i in 0 until n) {
		val t = arrive(i)
		while (hi + 1 < huds.size && huds[hi + 1].first <= t)
			hi++
		while (di + 1 < detections.size && detections[di + 1].first <= t)
			di++
		val hud = if (hi >= 0) huds[hi].second else null
		val det = if (di >= 0) detections[di].third else null
		val ref = refs.getOrNull(i)
		val tracked = hud != null && hud.mode == HudMode.TRACKED && hud.R != null
		if (tracked) {
			shownTracked++
			trackedAge.add((i / fps - hud!!.time) * 1000)
			if (ref != null && fx > 0) {
				val e = centreError(hud.R!!, hud.t!!, ref.R, ref.t, fx, w / 2.0, h / 2.0)
				trackedErr.add(e)
				phase.getOrNull(i)?.let { byPhase.getOrPut(it) { ArrayList() }.add(e) }
			}
			refs.getOrNull(hud.index.toInt())?.let { own -> if (fx > 0) ownErr.add(centreError(hud.R!!, hud.t!!, own.R, own.t, fx, w / 2.0, h / 2.0)) }
		}
		val d = det?.detection
		if (det != null && d != null && det.present && det.good) {
			shownDet++
			detAge.add((i / fps - det.time) * 1000)
			if (ref != null && fx > 0)
				detErr.add(centreError(d.R, d.t, ref.R, ref.t, fx, w / 2.0, h / 2.0))
		}
		if (outDir != null) {
			val src = ImageIO.read(frames[i])
			val img = BufferedImage((w * outScale).toInt(), (h * outScale).toInt(), BufferedImage.TYPE_INT_RGB)
			val g = img.createGraphics()
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
			g.drawImage(src, 0, 0, img.width, img.height, null)
			if (tracked && hud != null) {
				hud.quads.forEachIndexed { q, (_, quad) ->
					val xs = IntArray(4) { (quad[2 * it] * outScale).toInt() }
					val ys = IntArray(4) { (quad[2 * it + 1] * outScale).toInt() }
					val c = hud.colours.getOrElse(q) { 0 }
					if (c != 0) {
						g.color = Color((c shr 16) and 0xff, (c shr 8) and 0xff, c and 0xff, 150)
						g.fillPolygon(xs, ys, 4)
					}
					g.color = Color.WHITE
					g.stroke = BasicStroke(1.5f)
					g.drawPolygon(xs, ys, 4)
				}
				g.color = Color(255, 0, 255)
				hud.track?.points?.forEach { p -> g.fillOval((p[0] * outScale - 2).toInt(), (p[1] * outScale - 2).toInt(), 4, 4) }
			}
			g.font = Font(Font.MONOSPACED, Font.BOLD, 13)
			val lines = listOf(
				String.format("t %5.2f s  frame %d/%d", i / fps, i + 1, n),
				if (hud == null) "starting" else when (hud.mode) {
					HudMode.TRACKED -> String.format("TRACKED  %d inliers  %.1f px  HUD age %.0f ms", hud.track!!.inliers, hud.track.reprojection, (i / fps - hud.time) * 1000)
					HudMode.LOST -> "lost: waiting for a detection"
					HudMode.NONE -> "no cube yet"
				},
				if (hud?.snapshot != null) String.format("detection age %.0f ms  stage %s  presence %.2f", hud.detectionAge * 1000, hud.snapshot.stage, hud.snapshot.presence) else "",
				hud?.events?.lastOrNull() ?: "",
			)
			lines.forEachIndexed { k, line ->
				g.color = Color.BLACK
				g.drawString(line, 7, 17 + 16 * k)
				g.color = Color.WHITE
				g.drawString(line, 6, 16 + 16 * k)
			}
			g.dispose()
			ImageIO.write(img, "png", outDir.resolve(String.format("frames/frame%04d.png", i + 1)))
		}
	}
	println(String.format("%d camera frames; tracker steps %d (%.1f ms mean, p95 %.1f), detections %d (%.0f ms mean); latency scale %.1f",
		n, trackMs.size, trackMs.average(), pct(trackMs, 0.95), detectMs.size, detectMs.average(), scale))
	println(String.format("tracked HUD on %d/%d displayed frames (%.0f%%), age median %.0f ms p90 %.0f; error vs the reference median %.3f p90 %.3f max %.2f cubie (%d with a reference), > 0.3 on %d",
		shownTracked, n, 100.0 * shownTracked / n, pct(trackedAge, 0.5), pct(trackedAge, 0.9), pct(trackedErr, 0.5), pct(trackedErr, 0.9),
		trackedErr.maxOrNull() ?: Double.NaN, trackedErr.size, trackedErr.count { it > 0.3 }))
	println(String.format("detection-only HUD would be on %d/%d (%.0f%%), age median %.0f ms p90 %.0f; error median %.3f p90 %.3f (%d), > 0.3 on %d",
		shownDet, n, 100.0 * shownDet / n, pct(detAge, 0.5), pct(detAge, 0.9), pct(detErr, 0.5), pct(detErr, 0.9), detErr.size, detErr.count { it > 0.3 }))
	println(String.format("  the tracked HUD against its own frame (no display lag): median %.3f p90 %.3f, > 0.3 on %d/%d",
		pct(ownErr, 0.5), pct(ownErr, 0.9), ownErr.count { it > 0.3 }, ownErr.size))
	for ((ph, e) in byPhase.toSortedMap())
		println(String.format("  displayed during %-8s: error median %.3f p90 %.3f (%d)", ph, pct(e, 0.5), pct(e, 0.9), e.size))
	println("  stats: " + coord.stats.toSortedMap())
	val filled = huds.filter { it.second.mode == HudMode.TRACKED }.map { h -> h.second.colours.count { it != 0 } }
	println("  colours filled per tracked HUD: " + filled.groupingBy { it }.eachCount().toSortedMap())
	// the first seconds: where the track came from
	huds.map { it.second }.filter { it.time < 7 && it.mode == HudMode.TRACKED }.groupBy { (it.time * 2).toInt() }.forEach { (_, hs) ->
		val h = hs.first(); val ref = refs.getOrNull(h.index.toInt())
		val e = if (ref != null && fx > 0) centreError(h.R!!, h.t!!, ref.R, ref.t, fx, w / 2.0, h.frame.height / 2.0) else Double.NaN
		println(String.format("    t %.1f frame %d: error vs the reference %.2f, inliers %d, snapshot fits %s good %s cost %.3f", h.time, h.index, e,
			h.track!!.inliers, h.snapshot?.fits, h.snapshot?.good, h.snapshot?.detection?.latticeCost ?: -1.0))
	}
	val last = huds.lastOrNull()?.second
	println("events (last): " + (last?.events?.takeLast(6)?.joinToString(" | ") ?: ""))
	if (outDir != null) {
		val p = ProcessBuilder("ffmpeg", "-loglevel", "error", "-y", "-framerate", fps.toString(), "-i", outDir.resolve("frames/frame%04d.png").path,
			"-c:v", "libx264", "-pix_fmt", "yuv420p", outDir.resolve("hud.mp4").path).inheritIO().start()
		p.waitFor()
		println("→ ${outDir.resolve("hud.mp4")}")
	}
	coord.close()
	detector.close()
}
