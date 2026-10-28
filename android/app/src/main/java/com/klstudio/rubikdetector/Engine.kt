// The detection engine through JNI (src/main/cpp/rubik_jni.cpp). No Android imports: the same file runs in the host
// check (hostcheck/), against a host build of the library.
package com.klstudio.rubikdetector

import java.nio.ByteBuffer

class Sample(val facelet: Int, val valid: Boolean, val lab: FloatArray, val quad: FloatArray)

class Segment(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val weight: Float, val family: Int)

// One detection, plus the tracker's bookkeeping (mode, timestamps, chaining).
class Detection(
	val ok: Boolean,
	val latticeCost: Double,
	val support: Double,
	val segmentCount: Int,
	val familySegments: Int,
	val R: DoubleArray,             // 3×3 row-major, cube axes as columns (camera frame)
	val t: DoubleArray,             // cube centre, camera frame, cubie units
	val timings: DoubleArray,       // ms: segments, axes, position, sampling
	val totalMs: Double,
	val samples: List<Sample>,
	val segments: List<Segment>,
) {
	var mode = "wide"
	var time = 0.0                  // capture time, seconds
	var chain = 0
	var rotation: Int? = null       // placement in the evidence's cube frame
	var rel = 0                     // renaming into its chain's first frame
	var linkedTo: Detection? = null

	// per sample (facelet, valid, L, a, b), as the estimate and scoring calls take them
	fun packed(): DoubleArray {
		val out = DoubleArray(samples.size * 5)
		samples.forEachIndexed { i, s ->
			out[i * 5] = s.facelet.toDouble()
			out[i * 5 + 1] = if (s.valid) 1.0 else 0.0
			out[i * 5 + 2] = s.lab[0].toDouble()
			out[i * 5 + 3] = s.lab[1].toDouble()
			out[i * 5 + 4] = s.lab[2].toDouble()
		}
		return out
	}
}

class Focal(val ok: Boolean, val fx: Double, val fy: Double, val cx: Double, val cy: Double, val ratio: Double, val support: Double,
	val ms: Double, val solves: Int, val curve: List<Pair<Double, Double>>)

class Model(val mean: DoubleArray, val variance: DoubleArray)

class Estimate(
	val ok: Boolean,
	val legal: Boolean,
	val coverage: Int,
	val minPosterior: Double,
	val minConfidence: Double,
	val cost: Double,
	val ms: Double,
	val rotations: IntArray,
	val state: String?,             // 54 letters (L R D U B F), null when not ok
	val probability: Array<DoubleArray>,   // 54 × 6, colours in L R D U B F order
	val confidence: DoubleArray,
	val observed: BooleanArray,
	val model: Model,
)

class FrameInfo(val width: Int, val height: Int, val ms: Double)

object Engine {
	init {
		System.loadLibrary("rubik_jni")
	}

	@JvmStatic external fun nativeCreate(): Long
	@JvmStatic external fun nativeDestroy(handle: Long)
	@JvmStatic external fun nativeSetFrame(handle: Long, buffer: ByteBuffer, width: Int, height: Int, stride: Int, rotation: Int): DoubleArray
	@JvmStatic external fun nativeDetect(handle: Long, width: Int, height: Int, fx: Double, fy: Double, cx: Double, cy: Double,
		prior: DoubleArray?, workWidth: Int, keepSegments: Boolean): DoubleArray
	@JvmStatic external fun nativeEstimateFocal(handle: Long, width: Int, height: Int): DoubleArray
	@JvmStatic external fun nativeEstimate(samples: DoubleArray, counts: IntArray, rotations: IntArray, legal: Boolean, partial: Boolean,
		reliability: Boolean): DoubleArray
	@JvmStatic external fun nativeScoreModel(mean: DoubleArray, variance: DoubleArray, samples: DoubleArray, rotation: Int, states: String): DoubleArray
	@JvmStatic external fun nativeConflicts(mean: DoubleArray, variance: DoubleArray, samples: DoubleArray, rotation: Int, state: String,
		threshold: Double): IntArray
	@JvmStatic external fun nativeApplyMoves(state: String, moves: String): String?
	@JvmStatic external fun nativeRotationTables(): IntArray

	fun estimate(views: List<Detection>, rotations: IntArray, legal: Boolean, partial: Boolean, reliability: Boolean): Estimate {
		val packed = views.map { it.packed() }
		val samples = DoubleArray(packed.sumOf { it.size })
		var at = 0
		for (p in packed) {
			p.copyInto(samples, at)
			at += p.size
		}
		val o = nativeEstimate(samples, IntArray(views.size) { views[it].samples.size }, rotations, legal, partial, reliability)
		var i = 7
		val rots = IntArray(views.size) { o[i++].toInt() }
		val colours = IntArray(54) { o[i++].toInt() }
		val probability = Array(54) { DoubleArray(6) { o[i++] } }
		val confidence = DoubleArray(54) { o[i++] }
		val observed = BooleanArray(54) { o[i++] != 0.0 }
		val mean = DoubleArray(24) { o[i++] }
		val variance = DoubleArray(24) { o[i++] }
		val ok = o[0] != 0.0
		return Estimate(ok, o[1] != 0.0, o[2].toInt(), o[3], o[4], o[5], o[6], rots,
			if (ok) String(CharArray(54) { Cube.FACES[colours[it]] }) else null, probability, confidence, observed, Model(mean, variance))
	}

	fun scoreModel(model: Model, view: Detection, rotation: Int, joinedStates: String): DoubleArray =
		nativeScoreModel(model.mean, model.variance, view.packed(), rotation, joinedStates)

	fun conflicts(model: Model, view: Detection, rotation: Int, state: String, threshold: Double = 4.0): IntArray =
		nativeConflicts(model.mean, model.variance, view.packed(), rotation, state, threshold)

	fun applyMoves(state: String, moves: String): String = nativeApplyMoves(state, moves) ?: throw IllegalArgumentException(moves)
}

// One engine instance (scratch buffers: one per thread) and the frame it works on.
class Detector : AutoCloseable {
	private var handle = Engine.nativeCreate()

	// The analysis frame (RGBA, `stride` bytes per row) turned upright by `rotation` degrees clockwise.
	fun setFrame(buffer: ByteBuffer, width: Int, height: Int, stride: Int, rotation: Int): FrameInfo {
		val o = Engine.nativeSetFrame(handle, buffer, width, height, stride, rotation)
		return FrameInfo(o[0].toInt(), o[1].toInt(), o[2])
	}

	fun detect(frame: FrameInfo, fx: Double, prior: Pair<DoubleArray, DoubleArray>?, workWidth: Int = 1024, keepSegments: Boolean = false): Detection {
		val p = prior?.let { (R, t) -> R + t }
		val o = Engine.nativeDetect(handle, frame.width, frame.height, fx, fx, 0.0, 0.0, p, workWidth, keepSegments)
		var i = 5
		val R = DoubleArray(9) { o[i++] }
		val t = DoubleArray(3) { o[i++] }
		val timings = DoubleArray(4) { o[i++] }
		val total = o[i++]
		val samples = List(o[i++].toInt()) {
			val facelet = o[i++].toInt()
			val valid = o[i++] != 0.0
			val lab = FloatArray(3) { o[i++].toFloat() }
			val quad = FloatArray(8) { o[i++].toFloat() }
			Sample(facelet, valid, lab, quad)
		}
		val segments = List(o[i++].toInt()) {
			val s = Segment(o[i].toFloat(), o[i + 1].toFloat(), o[i + 2].toFloat(), o[i + 3].toFloat(), o[i + 4].toFloat(), o[i + 5].toInt())
			i += 6
			s
		}
		return Detection(o[0] != 0.0, o[1], o[2], o[3].toInt(), o[4].toInt(), R, t, timings, total, samples, segments)
	}

	fun estimateFocal(frame: FrameInfo): Focal {
		val o = Engine.nativeEstimateFocal(handle, frame.width, frame.height)
		val n = o[9].toInt()
		return Focal(o[0] != 0.0, o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8].toInt(), List(n) { Pair(o[10 + 2 * it], o[11 + 2 * it]) })
	}

	override fun close() {
		if (handle != 0L) {
			Engine.nativeDestroy(handle)
			handle = 0L
		}
	}
}
