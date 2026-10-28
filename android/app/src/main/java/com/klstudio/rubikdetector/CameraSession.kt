// CameraX: preview + RGBA analysis frames (keep-only-latest: the pipeline takes the newest frame whenever it is
// free) on one worker thread that owns the Detector and the Pipeline. Publishes a Snapshot per processed frame.
package com.klstudio.rubikdetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File
import java.util.concurrent.Executors

// What the camera says about its lens: the focal length for an upright frame whose long side is `longSide` pixels.
class Intrinsics(val focalMm: Float?, val sensorWidthMm: Float?, val calibration: FloatArray?, val activeWidth: Int?) {
	fun fx(longSide: Int): Double? = when {
		calibration != null && calibration[0] > 0 && activeWidth != null -> calibration[0].toDouble() * longSide / activeWidth
		focalMm != null && sensorWidthMm != null && sensorWidthMm > 0 -> focalMm.toDouble() / sensorWidthMm * longSide
		else -> null
	}

	fun describe(): String = listOfNotNull(
		focalMm?.let { String.format("f %.2f mm", it) },
		sensorWidthMm?.let { String.format("sensor %.2f mm", it) },
		calibration?.let { String.format("calibration fx %.0f px of %d", it[0], activeWidth ?: 0) },
	).joinToString(", ").ifEmpty { "no lens data" }
}

class Stats(
	val analysisSize: Size = Size(0, 0),
	val rotation: Int = 0,
	val processedFps: Double = 0.0,
	val cameraFramesPerProcessed: Double = 0.0,
	val ageMs: Double = 0.0,
	val convertMs: Double = 0.0,
	val totalMs: Double = 0.0,
	val cameraFx: Double? = null,
	val intrinsics: String = "",
	val dumped: String? = null,
)

class CameraSession(private val context: Context) {
	val snapshots = MutableStateFlow<Snapshot?>(null)
	val stats = MutableStateFlow(Stats())
	val pipeline = Pipeline()

	@Volatile var frozen = false
	@Volatile var useCameraFocal = false
	@Volatile private var resetRequest: Boolean? = null   // keep the focal length?
	@Volatile private var dumpRequested = false

	private val worker = Executors.newSingleThreadExecutor()
	private var detector: Detector? = null
	private var intrinsics: Intrinsics? = null
	private var lastTimestamp = 0L
	private var lastProcessed = 0L
	private var frameInterval = 0L

	fun reset(keepFocal: Boolean) {
		resetRequest = keepFocal
	}

	fun dump() {
		dumpRequested = true
	}

	fun bind(owner: LifecycleOwner, previewView: PreviewView) {
		val future = ProcessCameraProvider.getInstance(context)
		future.addListener({
			val provider = future.get()
			val aspect = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
			val preview = Preview.Builder()
				.setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(aspect).build())
				.build()
			preview.setSurfaceProvider(previewView.surfaceProvider)
			val analysis = ImageAnalysis.Builder()
				.setResolutionSelector(ResolutionSelector.Builder()
					.setAspectRatioStrategy(aspect)
					.setResolutionStrategy(ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
					.build())
				.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
				.setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
				.build()
			analysis.setAnalyzer(worker) { image -> analyze(image) }
			provider.unbindAll()
			val camera = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
			intrinsics = readIntrinsics(camera)
		}, ContextCompat.getMainExecutor(context))
	}

	@OptIn(ExperimentalCamera2Interop::class)
	private fun readIntrinsics(camera: Camera): Intrinsics {
		val info = Camera2CameraInfo.from(camera.cameraInfo)
		return Intrinsics(
			info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull(),
			info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.width,
			info.getCameraCharacteristic(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION),
			info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.width(),
		)
	}

	private fun analyze(image: ImageProxy) {
		try {
			val timestamp = image.imageInfo.timestamp
			if (lastTimestamp != 0L && frameInterval == 0L)
				frameInterval = timestamp - lastTimestamp
			lastTimestamp = timestamp
			if (frozen)
				return
			resetRequest?.let { keep ->
				resetRequest = null
				pipeline.reset(keep)
				if (!keep)
					pipeline.fx = 0.0
			}
			val started = System.nanoTime()
			val det = detector ?: Detector().also { detector = it }
			val plane = image.planes[0]
			val rotation = image.imageInfo.rotationDegrees
			val frame = det.setFrame(plane.buffer, image.width, image.height, plane.rowStride, rotation)
			val cameraFx = intrinsics?.fx(maxOf(frame.width, frame.height))
			if (useCameraFocal && cameraFx != null && pipeline.stage == Stage.FOCAL)
				pipeline.useFocal(cameraFx)
			val snapshot = pipeline.process(det, frame, timestamp / 1e9)
			val now = System.nanoTime()
			var dumped: String? = stats.value.dumped
			if (dumpRequested) {
				dumpRequested = false
				dumped = save(image, rotation, snapshot)
			}
			val processedFps = if (lastProcessed != 0L) 1e9 / (now - lastProcessed) else 0.0
			lastProcessed = now
			stats.value = Stats(
				analysisSize = Size(image.width, image.height),
				rotation = rotation,
				processedFps = processedFps,
				cameraFramesPerProcessed = if (frameInterval > 0 && processedFps > 0) 1e9 / frameInterval / processedFps else 0.0,
				// CameraX timestamps are on the sensor clock, usually SystemClock.elapsedRealtimeNanos
				ageMs = (SystemClock.elapsedRealtimeNanos() - timestamp) / 1e6,
				convertMs = frame.ms,
				totalMs = (now - started) / 1e6,
				cameraFx = cameraFx,
				intrinsics = intrinsics?.describe() ?: "",
				dumped = dumped,
			)
			snapshots.value = snapshot
		}
		catch (e: Throwable) {
			Log.e("RubikDetector", "analysis failed", e)
		}
		finally {
			image.close()
		}
	}

	// the frame (upright PNG) and what the pipeline made of it (text) under the app's external files, for offline
	// runs of the Python tools
	private fun save(image: ImageProxy, rotation: Int, s: Snapshot): String {
		val plane = image.planes[0]
		val buffer = plane.buffer
		buffer.rewind()
		val padded = Bitmap.createBitmap(plane.rowStride / 4, image.height, Bitmap.Config.ARGB_8888)
		padded.copyPixelsFromBuffer(buffer)
		val cropped = Bitmap.createBitmap(padded, 0, 0, image.width, image.height)
		val upright = Bitmap.createBitmap(cropped, 0, 0, image.width, image.height, Matrix().apply { postRotate(rotation.toFloat()) }, false)
		val dir = File(context.getExternalFilesDir(null), "dumps").apply { mkdirs() }
		val name = "frame-${System.currentTimeMillis()}"
		File(dir, "$name.png").outputStream().use { upright.compress(Bitmap.CompressFormat.PNG, 100, it) }
		File(dir, "$name.txt").writeText(describe(s, stats.value))
		return File(dir, "$name.png").absolutePath
	}

	fun close() {
		worker.execute {
			detector?.close()
			detector = null
		}
		worker.shutdown()
	}
}
