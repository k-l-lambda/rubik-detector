// The reconstructed cube of the HUD: 26 cubies with their stickers, orthographic, at an orientation (the tracked
// pose's, so it turns with the real cube), one layer optionally turned part way (a twist being played). Facelets in
// the view frame of that orientation, geometry as the engine's cube tables (faces L R D U B F, rows / columns seen
// from outside). No Android imports: polygons for any canvas.
package com.klstudio.rubikdetector

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

// a layer turned by `angle` (radians, right-handed about +axis of the view frame): the cubies at `sign` along `axis`
class MiniTwist(val axis: Int, val sign: Int, val angle: Double)

class MiniView(
	val R: DoubleArray,             // view frame → camera (x right, y down, z away)
	val colours: IntArray,          // per view facelet (54), ARGB
	val twist: MiniTwist?,
	val letters: CharArray? = null, // per view facelet the state's letter shown (debugging; '?' none)
)

class MiniPoly(val points: FloatArray, val argb: Int, val outline: Boolean)

object MiniCube {
	private val AXIS = arrayOf(intArrayOf(0, -1), intArrayOf(0, 1), intArrayOf(1, -1), intArrayOf(1, 1), intArrayOf(2, -1), intArrayOf(2, 1))
	private val FRAMES = arrayOf(
		arrayOf(intArrayOf(0, 0, 1), intArrayOf(0, -1, 0)),
		arrayOf(intArrayOf(0, 0, -1), intArrayOf(0, -1, 0)),
		arrayOf(intArrayOf(1, 0, 0), intArrayOf(0, 0, -1)),
		arrayOf(intArrayOf(1, 0, 0), intArrayOf(0, 0, 1)),
		arrayOf(intArrayOf(-1, 0, 0), intArrayOf(0, -1, 0)),
		arrayOf(intArrayOf(1, 0, 0), intArrayOf(0, -1, 0)),
	)
	const val GREY = 0xff7a7a7a.toInt()
	private const val BODY = 0xff1c1c1c.toInt()

	// per facelet: outward normal, right, down, its cubie's lattice position
	private val normal = Array(54) { k -> DoubleArray(3).also { it[AXIS[k / 9][0]] = AXIS[k / 9][1].toDouble() } }
	private val right = Array(54) { k -> DoubleArray(3) { FRAMES[k / 9][0][it].toDouble() } }
	private val down = Array(54) { k -> DoubleArray(3) { FRAMES[k / 9][1][it].toDouble() } }
	private val cubie = Array(54) { k ->
		IntArray(3) { i -> Math.round(normal[k][i] + (k % 3 - 1) * right[k][i] + ((k % 9) / 3 - 1) * down[k][i]).toInt() }
	}

	// a colour's saturation by confidence: none (a sixth, chance) grey, full the colour
	fun withConfidence(argb: Int, confidence: Double): Int {
		val c = ((confidence - 1.0 / 6) / (5.0 / 6)).coerceIn(0.0, 1.0)
		val mix = { shift: Int -> ((argb shr shift and 0xff) * c + (GREY shr shift and 0xff) * (1 - c)).toInt() }
		return (0xff shl 24) or (mix(16) shl 16) or (mix(8) shl 8) or mix(0)
	}

	// a view with R U F towards the viewer, U up
	fun defaultR(): DoubleArray {
		val d = doubleArrayOf(-1.0, -1.0, -1.0).map { it / sqrt(3.0) }
		val upDot = d[1]
		val up = listOf(-upDot * d[0], 1 - upDot * d[1], -upDot * d[2]).let { u -> val n = sqrt(u.sumOf { it * it }); u.map { it / n } }
		val y = up.map { -it }
		val x = listOf(y[1] * d[2] - y[2] * d[1], y[2] * d[0] - y[0] * d[2], y[0] * d[1] - y[1] * d[0])
		return (x + y + d).toDoubleArray()
	}

	// the layer and right-handed angle about +axis for a face turn of the view frame's face `face` (0..5, L R D U B F)
	// by `move`'s suffix ("", "'", "2"): X is clockwise seen from X
	fun twistFor(face: Int, suffix: String, progress: Double): MiniTwist {
		val q = when (suffix) { "'" -> -1.0; "2" -> 2.0; else -> 1.0 }
		val sign = AXIS[face][1]
		return MiniTwist(AXIS[face][0], sign, sign * -Math.PI / 2 * q * progress)
	}

	private fun rotate(p: DoubleArray, axis: Int, angle: Double): DoubleArray {
		val c = cos(angle)
		val s = sin(angle)
		val (i, j) = when (axis) { 0 -> Pair(1, 2); 1 -> Pair(2, 0); else -> Pair(0, 1) }
		val out = p.copyOf()
		out[i] = c * p[i] - s * p[j]
		out[j] = s * p[i] + c * p[j]
		return out
	}

	// where facelet k lands when its layer is turned (for checking the turn's sense against Engine.applyMoves)
	fun landing(k: Int, twist: MiniTwist): Int {
		if (cubie[k][twist.axis] != twist.sign)
			return k
		val c = rotate(cubie[k].map { it.toDouble() }.toDoubleArray(), twist.axis, twist.angle).map { Math.round(it).toInt() }
		val n = rotate(normal[k], twist.axis, twist.angle).map { Math.round(it).toInt() }
		return (0 until 54).first { j -> (0 until 3).all { cubie[j][it] == c[it] && Math.round(normal[j][it]).toInt() == n[it] } }
	}

	fun polygons(view: MiniView): List<MiniPoly> {
		val R = view.R
		val toCamera = { p: DoubleArray -> DoubleArray(3) { r -> R[r * 3] * p[0] + R[r * 3 + 1] * p[1] + R[r * 3 + 2] * p[2] } }
		val tw = view.twist
		val place = { p: DoubleArray, pos: IntArray -> toCamera(if (tw != null && pos[tw.axis] == tw.sign) rotate(p, tw.axis, tw.angle) else p) }
		class Cubie(val pos: IntArray, val depth: Double)
		val cubies = ArrayList<Cubie>()
		for (x in -1..1)
			for (y in -1..1)
				for (z in -1..1) {
					if (x == 0 && y == 0 && z == 0)
						continue
					val pos = intArrayOf(x, y, z)
					cubies.add(Cubie(pos, place(doubleArrayOf(x.toDouble(), y.toDouble(), z.toDouble()), pos)[2]))
				}
		val out = ArrayList<MiniPoly>()
		val half = 0.47
		for (c in cubies.sortedByDescending { it.depth }) {
			val centre = c.pos.map { it.toDouble() }
			for (a in 0 until 3)
				for (s in intArrayOf(-1, 1)) {
					val n = DoubleArray(3).also { it[a] = s.toDouble() }
					if (place(n, c.pos)[2] >= 0)
						continue   // turned away
					val u = (a + 1) % 3
					val v = (a + 2) % 3
					val corner = { du: Double, dv: Double, inset: Double ->
						val p = DoubleArray(3) { centre[it] }
						p[a] += s * half
						p[u] += du * inset
						p[v] += dv * inset
						place(p, c.pos)
					}
					val quad = { inset: Double ->
						val pts = listOf(corner(-1.0, -1.0, inset), corner(1.0, -1.0, inset), corner(1.0, 1.0, inset), corner(-1.0, 1.0, inset))
						FloatArray(8) { i -> pts[i / 2][i % 2].toFloat() }
					}
					out.add(MiniPoly(quad(half), BODY, false))
					// the sticker on an outer face
					if (c.pos[a] == s) {
						val k = (0 until 54).first { j -> cubie[j].contentEquals(c.pos) && Math.round(normal[j][a]).toInt() == s }
						out.add(MiniPoly(quad(half * 0.82), view.colours[k], true))
					}
				}
		}
		return out
	}
}
