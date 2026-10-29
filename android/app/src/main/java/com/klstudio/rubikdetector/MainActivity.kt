// Prototype UI: camera preview with the detection drawn over it, the cube net top right, debug text and controls.
package com.klstudio.rubikdetector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {
	private var session: CameraSession? = null
	private var granted by mutableStateOf(false)

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
		granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
		if (!granted)
			registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }.launch(Manifest.permission.CAMERA)
		val s = CameraSession(this)
		session = s
		setContent {
			MaterialTheme(colorScheme = darkColorScheme()) {
				if (granted)
					DetectorScreen(s, this)
				else
					Box(Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
						Text("Camera permission needed", color = Color.White)
					}
			}
		}
	}

	override fun onDestroy() {
		session?.close()
		super.onDestroy()
	}
}

@Composable
fun DetectorScreen(session: CameraSession, owner: androidx.lifecycle.LifecycleOwner) {
	val snapshot by session.snapshots.collectAsState()
	val hud by session.huds.collectAsState()
	val stats by session.stats.collectAsState()
	var tracking by remember { mutableStateOf(session.tracking) }
	var showSegments by remember { mutableStateOf(false) }
	var showSamples by remember { mutableStateOf(true) }
	var showAxes by remember { mutableStateOf(true) }
	var frozen by remember { mutableStateOf(false) }
	var cameraFocal by remember { mutableStateOf(false) }

	Column(Modifier.fillMaxSize().background(Color.Black)) {
		// the analysis frames are 4:3 upright (3:4 portrait): the preview box has the same shape, so the overlay maps by
		// one scale
		Box(Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
			AndroidView(factory = { ctx ->
				PreviewView(ctx).apply {
					scaleType = PreviewView.ScaleType.FIT_CENTER
					session.bind(owner, this)
				}
			}, modifier = Modifier.fillMaxSize())
			Canvas(Modifier.fillMaxSize()) {
				val s = snapshot
				val h = hud
				if (tracking && h != null && h.mode == HudMode.TRACKED) {
					// the tracked pose every frame; the detection's segments under it if asked
					if (s != null && showSegments)
						drawDetection(s, true, false, false)
					drawTracked(h, showSamples, showAxes, s?.fx ?: 0.0)
				}
				else if (s != null)
					// no track: the last detection, dimmed while tracking is on (it is that old)
					drawDetection(s, showSegments, showSamples, showAxes, if (tracking) 0.45f else 1f)
			}
			snapshot?.let { s ->
				Box(Modifier.align(Alignment.TopEnd).padding(6.dp).background(Color(0xb0101010)).padding(4.dp)) {
					CubeNet(s.net, Modifier.width(150.dp).aspectRatio(4f / 3f))
				}
				Text((if (tracking) trackHeadline(hud) + "\n" else "") + headline(s), color = Color.White, fontSize = 13.sp, fontFamily = FontFamily.Monospace,
					modifier = Modifier.align(Alignment.TopStart).padding(6.dp).background(Color(0x90000000)).padding(4.dp))
			}
		}
		Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
			FilterChip(selected = false, onClick = { session.reset(true) }, label = { Text("reset") })
			FilterChip(selected = false, onClick = { session.reset(false) }, label = { Text("FOV") })
			FilterChip(selected = frozen, onClick = { frozen = !frozen; session.frozen = frozen }, label = { Text("freeze") })
			FilterChip(selected = cameraFocal, onClick = {
				cameraFocal = !cameraFocal
				session.useCameraFocal = cameraFocal
				session.reset(false)
			}, label = { Text("lens fx") })
			FilterChip(selected = false, onClick = { session.dump() }, label = { Text("dump") })
		}
		Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
			FilterChip(selected = showSamples, onClick = { showSamples = !showSamples }, label = { Text("stickers") })
			FilterChip(selected = showSegments, onClick = { showSegments = !showSegments }, label = { Text("segments") })
			FilterChip(selected = showAxes, onClick = { showAxes = !showAxes }, label = { Text("axes") })
			FilterChip(selected = tracking, onClick = { tracking = !tracking; session.tracking = tracking }, label = { Text("track") })
		}
		Text((if (tracking) describeTrack(hud) else "") + (snapshot?.let { describe(it, stats) } ?: "waiting for the camera…"),
			color = Color(0xffd0d0d0), fontSize = 11.sp, fontFamily = FontFamily.Monospace, lineHeight = 13.sp,
			modifier = Modifier.fillMaxWidth().weight(1f).verticalScroll(rememberScrollState()).padding(6.dp))
	}
}

fun headline(s: Snapshot): String {
	val d = s.detection
	val det = when {
		d == null -> ""
		!s.present -> String.format("no cube (presence %.2f)", s.presence)
		s.good -> String.format("%s %.3f  presence %.2f", d.mode, d.latticeCost, s.presence)
		else -> String.format("lost (%.2f)", d.latticeCost)
	}
	return when (s.stage) {
		Stage.FOCAL -> if (s.present) "estimating focal length…" else "looking for a cube…\n$det"
		Stage.ACQUIRE -> String.format("ACQUIRE %d/54 %s\n%s", s.covered, s.estimate?.takeIf { it.legal }?.let { String.format("conf %.2f", it.minConfidence) } ?: "", det)
		Stage.FOLLOW -> "FOLLOW ${s.follow?.moves?.size ?: 0} turns\n$det"
	}
}

// image pixels of the upright analysis frame → canvas
private class Mapping(val scale: Float, val dx: Float, val dy: Float) {
	fun map(x: Float, y: Float) = Offset(dx + x * scale, dy + y * scale)
}

private fun DrawScope.mapping(frame: FrameInfo): Mapping {
	val scale = minOf(size.width / frame.width, size.height / frame.height)
	return Mapping(scale, (size.width - frame.width * scale) / 2, (size.height - frame.height * scale) / 2)
}

private val FAMILY = listOf(Color(0xffff4040), Color(0xff40ff40), Color(0xff4080ff))

fun trackHeadline(h: Hud?): String = when (h?.mode) {
	null, HudMode.NONE -> "no track yet"
	HudMode.LOST -> "track lost: waiting for a detection"
	HudMode.TRACKED -> String.format("TRACKED %d pts %.1f px  detection %.0f ms ago", h.track!!.inliers, h.track.reprojection, h.detectionAge * 1000)
}

fun describeTrack(h: Hud?): String = buildString {
	if (h == null)
		return@buildString
	val tr = h.track
	appendLine(String.format("track %s  frame %d  tracker thread %.1f ms (applying detections %.1f)", h.mode, h.index, h.msTrack, h.msApply))
	if (tr != null)
		appendLine(String.format("  %d points followed, %d inliers, reprojection %.2f px; ms pyramid %.1f flow %.1f pose %.2f",
			tr.tracked, tr.inliers, tr.reprojection, tr.ms[0], tr.ms[1], tr.ms[2]))
	if (!h.detectionAge.isNaN())
		appendLine(String.format("  last detection applied %.0f ms ago", h.detectionAge * 1000))
	if (h.events.isNotEmpty()) {
		appendLine("track events:")
		h.events.reversed().take(6).forEach { appendLine("  $it") }
	}
	appendLine()
}

// the tracked pose: its facelet quads in the last detection's colours, the tracked points, the axes
fun DrawScope.drawTracked(h: Hud, samples: Boolean, axes: Boolean, fx: Double) {
	val m = mapping(h.frame)
	if (samples)
		h.quads.forEachIndexed { q, (_, quad) ->
			val path = Path().apply {
				val p0 = m.map(quad[0], quad[1])
				moveTo(p0.x, p0.y)
				for (i in 1 until 4) {
					val p = m.map(quad[2 * i], quad[2 * i + 1])
					lineTo(p.x, p.y)
				}
				close()
			}
			val c = h.colours.getOrElse(q) { 0 }
			if (c != 0)
				drawPath(path, Color(c).copy(alpha = 0.6f))
			drawPath(path, Color.White, style = Stroke(width = 2f))
		}
	h.track?.points?.forEach { p -> drawCircle(Color(0xffff40ff), 3f, m.map(p[0], p[1])) }
	val R = h.R
	val t = h.t
	if (axes && R != null && t != null && fx > 0) {
		val project = { p: DoubleArray -> m.map((fx * p[0] / p[2] + h.frame.width / 2.0).toFloat(), (fx * p[1] / p[2] + h.frame.height / 2.0).toFloat()) }
		val o = project(t)
		for (a in 0 until 3) {
			val e = DoubleArray(3) { t[it] + 1.5 * R[it * 3 + a] }
			if (e[2] > 0)
				drawLine(FAMILY[a], o, project(e), strokeWidth = 4f)
		}
	}
}

fun DrawScope.drawDetection(s: Snapshot, segments: Boolean, samples: Boolean, axes: Boolean, alpha: Float = 1f) {
	val d = s.detection ?: return
	val m = mapping(s.frame)
	if (segments)
		for (g in d.segments)
			drawLine(if (g.family in 0..2) FAMILY[g.family] else Color(0x80a0a0a0), m.map(g.x1, g.y1), m.map(g.x2, g.y2),
				strokeWidth = if (g.family >= 0) 2f else 1f)
	// nothing of a detection without a cube (only the debug segments above)
	if (!d.ok || !s.present)
		return
	val outline = if (s.good) Color.White else Color(0xffff5050)
	if (samples)
		for (smp in d.samples) {
			if (!smp.valid)
				continue
			val q = smp.quad
			val path = Path().apply {
				val p0 = m.map(q[0], q[1])
				moveTo(p0.x, p0.y)
				for (i in 1 until 4) {
					val p = m.map(q[2 * i], q[2 * i + 1])
					lineTo(p.x, p.y)
				}
				close()
			}
			drawPath(path, Color(Colors.labToArgb(smp.lab)).copy(alpha = 0.6f * alpha))
			drawPath(path, outline.copy(alpha = alpha), style = Stroke(width = 2f))
		}
	if (axes) {
		// the cube's axes from its centre (view frame: x red, y green, z blue), 1.5 cubies long
		val project = { p: DoubleArray -> m.map((s.fx * p[0] / p[2] + s.frame.width / 2.0).toFloat(), (s.fx * p[1] / p[2] + s.frame.height / 2.0).toFloat()) }
		val o = project(d.t)
		for (a in 0 until 3) {
			val e = DoubleArray(3) { d.t[it] + 1.5 * d.R[it * 3 + a] }
			if (e[2] > 0)
				drawLine(FAMILY[a], o, project(e), strokeWidth = 4f)
		}
	}
}

// the cube net (U on top; L F R B; D): cell colour, best posterior component in percent, border = confidence,
// white corner = completed (not observed)
@Composable
fun CubeNet(net: Net, modifier: Modifier) {
	Canvas(modifier) {
		val cell = minOf(size.width / 12, size.height / 9)
		val places = mapOf('U' to Pair(1, 0), 'L' to Pair(0, 1), 'F' to Pair(1, 1), 'R' to Pair(2, 1), 'B' to Pair(3, 1), 'D' to Pair(1, 2))
		val paint = android.graphics.Paint().apply {
			textSize = cell * 0.42f
			textAlign = android.graphics.Paint.Align.CENTER
			isAntiAlias = true
		}
		for ((f, face) in Cube.FACES.withIndex()) {
			val (gx, gy) = places.getValue(face)
			for (i in 0 until 9) {
				val k = f * 9 + i
				val x = (gx * 3 + i % 3) * cell
				val y = (gy * 3 + i / 3) * cell
				val inner = Size(cell - 2, cell - 2)
				if (!net.known[k]) {
					drawRect(Color(0xff404040), Offset(x, y), inner)
					continue
				}
				val colour = Color(net.colour[k])
				drawRect(colour, Offset(x, y), inner)
				val c = net.confidence[k]
				val border = if (c >= 0.99) Color(0xff40c840) else if (c >= 0.9) Color(0xffe6c800) else Color(0xffe62828)
				drawRect(border, Offset(x + 1, y + 1), Size(cell - 4, cell - 4), style = Stroke(width = 2f))
				if (net.inferred[k]) {
					val tri = Path().apply {
						moveTo(x, y)
						lineTo(x + cell / 3, y)
						lineTo(x, y + cell / 3)
						close()
					}
					drawPath(tri, Color.White)
				}
				val p = net.probability[k]
				if (!p.isNaN()) {
					paint.color = if (colour.red + colour.green + colour.blue > 1.5f) android.graphics.Color.BLACK else android.graphics.Color.WHITE
					val label = if (p >= 0.995) "99" else String.format("%d", (p * 100).toInt())
					drawContext.canvas.nativeCanvas.drawText(label, x + cell / 2, y + cell * 0.65f, paint)
				}
			}
		}
	}
}

// everything the pipeline knows about the last frame, as text (the debug panel and the dump files)
fun describe(s: Snapshot, st: Stats): String = buildString {
	appendLine(String.format("stage %s   t %.2f s   process %.0f ms (pipeline %.0f, rotate %.1f)", s.stage, s.time, st.totalMs, s.processMs, st.convertMs))
	appendLine(String.format("  steps ms: detect %.0f + wide retry %.0f, place %.1f, estimate %.1f, follow %.1f, snapshot %.1f",
		s.steps.first, s.steps.retry, s.steps.place, s.steps.estimate, s.steps.follow, s.steps.snapshot))
	appendLine(String.format("frame %dx%d upright (analysis %dx%d, rot %d°)   %.1f fps processed, every %.1f camera frames, age %.0f ms",
		s.frame.width, s.frame.height, st.analysisSize.width, st.analysisSize.height, st.rotation, st.processedFps, st.cameraFramesPerProcessed, st.ageMs))
	val f = s.focal
	appendLine(String.format("focal fx %.0f px (%.2f × long side)%s", s.fx, s.fx / maxOf(s.frame.width, s.frame.height),
		if (f != null) String.format("   estimate: support %.2f, %d solves, %.0f ms", f.support, f.solves, f.ms) else ""))
	if (f != null && f.curve.isNotEmpty())
		appendLine("  coarse curve: " + f.curve.joinToString(" ") { String.format("%.2f:%.2f", it.first, it.second) })
	appendLine("  lens: ${st.intrinsics}" + (st.cameraFx?.let { String.format(" → fx %.0f px", it) } ?: ""))
	val d = s.detection
	if (d != null) {
		appendLine(String.format("detect %s  ok %b  good %b  lattice cost %.3f  support %.2f  segments %d (%d on axes)",
			d.mode, d.ok, s.good, d.latticeCost, d.support, d.segmentCount, d.familySegments))
		appendLine(String.format("  ms: segments %.0f  axes %.0f  position %.0f  sampling %.0f  total %.0f",
			d.timings[0], d.timings[1], d.timings[2], d.timings[3], d.totalMs))
		appendLine(String.format("  pose t (%.2f, %.2f, %.2f) cubies, %d/%d samples valid", d.t[0], d.t[1], d.t[2], d.samples.count { it.valid }, d.samples.size))
		appendLine(String.format("  presence %.2f (%s; face line coverage %.2f %.2f %.2f)", d.presence, if (s.present) "cube" else "no cube",
			d.faceCoverage[0], d.faceCoverage[1], d.faceCoverage[2]))
	}
	s.link?.let { appendLine(String.format("  link to previous: %.1f° (renamed), colour ΔE %.1f → %s", it.angle, it.colour, if (it.linked) "linked" else "new chain")) }
	appendLine(String.format("evidence %d frames in %d chains, %d/54 facelets observed", s.evidenceFrames, s.chains, s.covered))
	s.estimate?.let { e ->
		appendLine(String.format("estimate ok %b  legal %b  coverage %d  min posterior %.3f  min confidence %.3f  %.0f ms  stable %d",
			e.ok, e.legal, e.coverage, e.minPosterior, e.minConfidence, e.ms, s.stable))
		e.state?.let { if (e.legal) appendLine("  ${it.substring(0, 27)}\n  ${it.substring(27)}") }
	}
	s.follow?.let { fo ->
		appendLine("follow ${fo.state.substring(0, 27)}\n       ${fo.state.substring(27)}")
		appendLine("  turns (${fo.moves.size}): ${fo.moves.takeLast(12).joinToString(" ").ifEmpty { "-" }}")
		appendLine(String.format("  %s   candidates %d, conflicts best %d / state %d, scoring %.1f ms",
			fo.decision, fo.candidates, fo.conflicts, fo.stateConflicts, fo.scoreMs))
	}
	st.dumped?.let { appendLine("dumped $it") }
	if (s.events.isNotEmpty()) {
		appendLine("events:")
		s.events.reversed().forEach { appendLine("  $it") }
	}
}
