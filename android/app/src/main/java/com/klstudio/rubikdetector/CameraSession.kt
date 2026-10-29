// CameraX: preview + RGBA analysis frames (keep-only-latest). Tracking on (default): every analysis frame goes to
// the Coordinator's tracker on the analysis thread (a few ms; publishes a Hud per frame), and a second thread detects
// in the background on the newest frame whenever it is free (publishes a Snapshot per detection). Tracking off: the
// detection pipeline alone on the analysis thread, as before (one Snapshot per processed frame).
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
	val huds = MutableStateFlow<Hud?>(null)
	val stats = MutableStateFlow(Stats())
	val coordinator = Coordinator()
	val pipeline get() = coordinator.pipeline
	@Volatile var tracking = true

	@Volatile var frozen = false
	@Volatile var useCameraFocal = false
	@Volatile private var resetRequest: Boolean? = null   // keep the focal length?
	@Volatile private var pipelineReset: Boolean? = null  // tracking: the reset for the detector thread
	@Volatile private var dumpRequested = false

	private val worker = Executors.newSingleThreadExecutor()
	private val detectorThread = Executors.newSingleThreadExecutor()
	private var detector: Detector? = null           // the analysis thread's (tracking off)
	private var backgroundDetector: Detector? = null // the detector thread's (tracking on)
	private val detecting = java.util.concurrent.atomic.AtomicBoolean(false)
	@Volatile private var newest = -1L               // the newest frame pushed to the coordinator
	private var frameIndex = 0L
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
				if (tracking) {
					// the pipeline belongs to the detector thread: it resets it before its next detection
					coordinator.resetTrack()
					pipelineReset = keep
				}
				else {
					pipeline.reset(keep)
					if (!keep)
						pipeline.fx = 0.0
				}
			}
			if (tracking) {
				analyzeTracking(image)
				return
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

	// tracker thread: the frame into the coordinator's ring, tracked; a detection started on it if the detector is free
	private fun analyzeTracking(image: ImageProxy) {
		val started = System.nanoTime()
		val plane = image.planes[0]
		val index = frameIndex++
		coordinator.push(plane.buffer, image.width, image.height, plane.rowStride, image.imageInfo.rotationDegrees, image.imageInfo.timestamp / 1e9, index)
		val hud = coordinator.trackStep(index)
		newest = index
		if (detecting.compareAndSet(false, true))
			detectorThread.execute { detectLoop() }
		val now = System.nanoTime()
		val processedFps = if (lastProcessed != 0L) 1e9 / (now - lastProcessed) else 0.0
		lastProcessed = now
		val cameraFx = hud?.let { intrinsics?.fx(maxOf(it.frame.width, it.frame.height)) }
		var dumped: String? = stats.value.dumped
		if (dumpRequested) {
			dumpRequested = false
			snapshots.value?.let { dumped = save(image, image.imageInfo.rotationDegrees, it) }
		}
		stats.value = Stats(
			analysisSize = Size(image.width, image.height),
			rotation = image.imageInfo.rotationDegrees,
			processedFps = processedFps,
			cameraFramesPerProcessed = if (frameInterval > 0 && processedFps > 0) 1e9 / frameInterval / processedFps else 0.0,
			ageMs = (SystemClock.elapsedRealtimeNanos() - image.imageInfo.timestamp) / 1e6,
			convertMs = hud?.frame?.ms ?: 0.0,
			totalMs = (now - started) / 1e6,
			cameraFx = cameraFx,
			intrinsics = intrinsics?.describe() ?: "",
			dumped = dumped,
		)
		huds.value = hud
	}

	// detector thread: detections on the newest pushed frame until there is no newer one
	private fun detectLoop() {
		try {
			val det = backgroundDetector ?: Detector().also { backgroundDetector = it }
			var done = -1L
			while (tracking && !frozen) {
				val index = newest
				if (index <= done)
					break
				pipelineReset?.let { keep ->
					pipelineReset = null
					pipeline.reset(keep)
					if (!keep)
						pipeline.fx = 0.0
				}
				if (useCameraFocal && pipeline.stage == Stage.FOCAL)
					stats.value.cameraFx?.let { pipeline.useFocal(it) }
				val snapshot = coordinator.detectStep(det, index)
				done = index
				if (snapshot != null) {
					coordinator.post(index, snapshot)
					snapshots.value = snapshot
				}
			}
		}
		catch (e: Throwable) {
			Log.e("RubikDetector", "detection failed", e)
		}
		finally {
			detecting.set(false)
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
		tracking = false
		detectorThread.execute {
			backgroundDetector?.close()
			backgroundDetector = null
		}
		detectorThread.shutdown()
		worker.execute {
			detector?.close()
			detector = null
			coordinator.close()
		}
		worker.shutdown()
	}
}
