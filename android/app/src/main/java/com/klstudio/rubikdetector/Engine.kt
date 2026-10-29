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
	val presence: Double,           // is there a cube at all (engine ViewResult::presence; PRESENCE_THRESHOLD)
	val faceCoverage: DoubleArray,  // per cube axis, share of its face's 8 lattice lines with a fitting segment
) {
	var mode = "wide"
	var time = 0.0                  // capture time, seconds
	var chain = 0
	var rotation: Int? = null       // placement in the evidence's cube frame
	var rel = 0                     // renaming into its chain's first frame
	var linkedTo: Detection? = null
	var keep: BooleanArray? = null  // per sample whether it is evidence (null: all valid ones), set by the pipeline
	var dropped: BooleanArray? = null   // per common facelet: dropped from this frame's evidence (its layer turned since)

	// per sample (facelet, valid, L, a, b), as the estimate and scoring calls take them (a sample not kept: invalid)
	fun packed(): DoubleArray {
		val out = DoubleArray(samples.size * 5)
		val k = keep
		samples.forEachIndexed { i, s ->
			out[i * 5] = s.facelet.toDouble()
			out[i * 5 + 1] = if (s.valid && (k == null || k[i])) 1.0 else 0.0
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
	@JvmStatic external fun nativeEstimate(samples: DoubleArray, counts: IntArray, rotations: IntArray, groups: IntArray?, legal: Boolean,
		partial: Boolean, reliability: Boolean): DoubleArray
	@JvmStatic external fun nativeScoreModel(mean: DoubleArray, variance: DoubleArray, samples: DoubleArray, rotation: Int, states: String): DoubleArray
	@JvmStatic external fun nativeConflicts(mean: DoubleArray, variance: DoubleArray, samples: DoubleArray, rotation: Int, state: String,
		threshold: Double): IntArray
	@JvmStatic external fun nativeApplyMoves(state: String, moves: String): String?
	@JvmStatic external fun nativeRotationTables(): IntArray
	@JvmStatic external fun nativeTrackerCreate(): Long
	@JvmStatic external fun nativeTrackerDestroy(handle: Long)
	@JvmStatic external fun nativeTrackerSetFrame(handle: Long, buffer: ByteBuffer, width: Int, height: Int, stride: Int, rotation: Int): DoubleArray
	@JvmStatic external fun nativeTrackerSeed(handle: Long, width: Int, height: Int, fx: Double, pose: DoubleArray)
	@JvmStatic external fun nativeTrackerTrack(handle: Long, width: Int, height: Int): DoubleArray
	@JvmStatic external fun nativeTrackerReseed(handle: Long, pose: DoubleArray): Boolean
	@JvmStatic external fun nativeTrackerReset(handle: Long)
	@JvmStatic external fun nativeProjectFacelets(width: Int, height: Int, fx: Double, pose: DoubleArray, sampleSize: Double): DoubleArray

	// per R U F facelet of a pose (view frame), its quad (sampleSize of the sticker's side) in pixels
	fun projectFacelets(frame: FrameInfo, fx: Double, R: DoubleArray, t: DoubleArray, sampleSize: Double = 0.46): List<Pair<Int, FloatArray>> {
		val o = nativeProjectFacelets(frame.width, frame.height, fx, R + t, sampleSize)
		return List(o.size / 9) { i -> Pair(o[i * 9].toInt(), FloatArray(8) { o[i * 9 + 1 + it].toFloat() }) }
	}

	// groups: per view its reliability group (frames of a chain stand or fall together), null: each view its own
	fun estimate(views: List<Detection>, rotations: IntArray, legal: Boolean, partial: Boolean, reliability: Boolean, groups: IntArray? = null): Estimate {
		val packed = views.map { it.packed() }
		val samples = DoubleArray(packed.sumOf { it.size })
		var at = 0
		for (p in packed) {
			p.copyInto(samples, at)
			at += p.size
		}
		val o = nativeEstimate(samples, IntArray(views.size) { views[it].samples.size }, rotations, groups, legal, partial, reliability)
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

class Track(
	val ok: Boolean,
	val R: DoubleArray,             // facing: the faces turned to the camera are R U F
	val t: DoubleArray,
	val renaming: Int,
	val tracked: Int,
	val inliers: Int,
	val reprojection: Double,
	val ms: DoubleArray,            // pyramid, flow, pose
	val points: List<FloatArray>,
	val turning: Boolean,           // a layer's points don't follow the rigid pose (TrackResult turning-layer signal)
	val turnNormal: DoubleArray,    // that layer's face, outward normal, camera frame
	val turnScore: Double,
)

// The engine's FrameTracker (its own thread, own frame buffer): seed with a detected frame, then track every frame.
class Tracker : AutoCloseable {
	private var handle = Engine.nativeTrackerCreate()

	fun setFrame(buffer: ByteBuffer, width: Int, height: Int, stride: Int, rotation: Int): FrameInfo {
		val o = Engine.nativeTrackerSetFrame(handle, buffer, width, height, stride, rotation)
		return FrameInfo(o[0].toInt(), o[1].toInt(), o[2])
	}

	fun seed(frame: FrameInfo, fx: Double, R: DoubleArray, t: DoubleArray) = Engine.nativeTrackerSeed(handle, frame.width, frame.height, fx, R + t)

	// a new pose for the frame tracked last
	fun reseed(R: DoubleArray, t: DoubleArray): Boolean = Engine.nativeTrackerReseed(handle, R + t)

	fun reset() = Engine.nativeTrackerReset(handle)

	fun track(frame: FrameInfo): Track {
		val o = Engine.nativeTrackerTrack(handle, frame.width, frame.height)
		val n = o[25].toInt()
		return Track(o[0] != 0.0, o.copyOfRange(1, 10), o.copyOfRange(10, 13), o[13].toInt(), o[14].toInt(), o[15].toInt(), o[16],
			o.copyOfRange(17, 20), List(n) { floatArrayOf(o[26 + 2 * it].toFloat(), o[27 + 2 * it].toFloat()) },
			o[20] != 0.0, o.copyOfRange(21, 24), o[24])
	}

	override fun close() {
		if (handle != 0L) {
			Engine.nativeTrackerDestroy(handle)
			handle = 0L
		}
	}
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
		val presence = o[i++]
		val coverage = DoubleArray(3) { o[i++] }
		return Detection(o[0] != 0.0, o[1], o[2], o[3].toInt(), o[4].toInt(), R, t, timings, total, samples, segments, presence, coverage)
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
