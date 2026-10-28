// Cube tables (from the engine) and the small rotation algebra the tracker needs. No Android imports.
package com.klstudio.rubikdetector

import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

object Cube {
	const val FACES = "LRDUBF"

	// the 24 whole-cube rotations (3×3 row-major) and their facelet permutations: a view whose frame is renamed by g
	// has its facelet k at the other's facelet perms[g][k]
	val rotations: Array<IntArray>
	val perms: Array<IntArray>
	private val permIndex: Map<List<Int>, Int>
	val identity: Int

	init {
		val t = Engine.nativeRotationTables()
		rotations = Array(24) { g -> IntArray(9) { t[g * 9 + it] } }
		perms = Array(24) { g -> IntArray(54) { t[24 * 9 + g * 54 + it] } }
		permIndex = perms.withIndex().associate { (g, p) -> p.toList() to g }
		identity = permIndex.getValue((0 until 54).toList())
	}

	// placement of b, given a's placement and b's renaming g into a's frame
	fun chain(rotA: Int, g: Int): Int = permIndex.getValue(List(54) { perms[rotA][perms[g][it]] })

	// the placement of a chain's first frame, given a frame of it placed at `rot` and renamed into the first by `rel`
	fun chainStart(rot: Int, rel: Int): Int {
		val inverse = IntArray(54)
		perms[rel].forEachIndexed { k, j -> inverse[j] = k }
		return permIndex.getValue(List(54) { perms[rot][inverse[it]] })
	}

	// (angle in degrees, g): the rotation between two poses up to a renaming of the view frame
	fun frameMatch(Ra: DoubleArray, Rb: DoubleArray): Pair<Double, Int> {
		var best = Pair(1e9, -1)
		for (g in 0 until 24) {
			val RaG = mul(Ra, rotations[g].map { it.toDouble() }.toDoubleArray())
			val angle = angleBetween(RaG, Rb)
			if (angle < best.first)
				best = Pair(angle, g)
		}
		return best
	}

	fun renamed(R: DoubleArray, g: Int): DoubleArray = mul(R, rotations[g].map { it.toDouble() }.toDoubleArray())

	// degrees between two rotations
	fun angleBetween(A: DoubleArray, B: DoubleArray): Double {
		var trace = 0.0
		for (i in 0 until 3)
			for (k in 0 until 3)
				trace += A[k * 3 + i] * B[k * 3 + i]
		return Math.toDegrees(acos(((trace - 1) / 2).coerceIn(-1.0, 1.0)))
	}

	fun mul(A: DoubleArray, B: DoubleArray): DoubleArray = DoubleArray(9) { i ->
		val r = i / 3
		val c = i % 3
		A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]
	}

	fun transpose(A: DoubleArray): DoubleArray = DoubleArray(9) { A[(it % 3) * 3 + it / 3] }

	// rotation vector (axis × angle) of a rotation matrix and back
	fun log(R: DoubleArray): DoubleArray {
		val angle = acos(((R[0] + R[4] + R[8] - 1) / 2).coerceIn(-1.0, 1.0))
		if (angle < 1e-9)
			return DoubleArray(3)
		val s = 2 * sin(angle)
		return doubleArrayOf((R[7] - R[5]) / s * angle, (R[2] - R[6]) / s * angle, (R[3] - R[1]) / s * angle)
	}

	fun exp(v: DoubleArray): DoubleArray {
		val angle = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
		if (angle < 1e-12)
			return doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
		val x = v[0] / angle
		val y = v[1] / angle
		val z = v[2] / angle
		val c = cos(angle)
		val s = sin(angle)
		val C = 1 - c
		return doubleArrayOf(
			c + x * x * C, x * y * C - z * s, x * z * C + y * s,
			y * x * C + z * s, c + y * y * C, y * z * C - x * s,
			z * x * C - y * s, z * y * C + x * s, c + z * z * C)
	}

	// the outer face turns, as the engine's applyMoves takes them
	val MOVES = "RLUDFB".flatMap { f -> listOf("$f", "$f'", "${f}2") }

	// compareStates of the engine up to a whole-cube rotation, by renaming: the face renaming that makes `a` equal `b`
	fun faceMap(a: String, b: String): Map<Char, Char>? {
		for (g in 0 until 24) {
			val raw = CharArray(54) { a[perms[g][it]] }
			val names = (0 until 6).associate { raw[it * 9 + 4] to FACES[it] }
			if (String(CharArray(54) { names.getValue(raw[it]) }) == b)
				return FACES.associateWith { names.getValue(it) }
		}
		return null
	}
}
