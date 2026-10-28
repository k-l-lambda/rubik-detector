// Colour conversions for drawing. No Android imports (ARGB ints).
package com.klstudio.rubikdetector

import kotlin.math.pow
import kotlin.math.roundToInt

object Colors {
	// OpenCV float CIE L*a*b* (L 0–100, D65) → opaque sRGB as ARGB
	fun labToArgb(L: Double, a: Double, b: Double): Int {
		val fy = (L + 16) / 116
		val fx = fy + a / 500
		val fz = fy - b / 200
		val inv = { f: Double -> if (f > 6.0 / 29) f * f * f else 3 * (6.0 / 29) * (6.0 / 29) * (f - 4.0 / 29) }
		val X = 0.950456 * inv(fx)
		val Y = inv(fy)
		val Z = 1.088754 * inv(fz)
		val r = 3.240479 * X - 1.53715 * Y - 0.498535 * Z
		val g = -0.969256 * X + 1.875991 * Y + 0.041556 * Z
		val bl = 0.055648 * X - 0.204043 * Y + 1.057311 * Z
		val gamma = { c: Double ->
			val v = if (c <= 0.0031308) 12.92 * c else 1.055 * c.pow(1 / 2.4) - 0.055
			(v.coerceIn(0.0, 1.0) * 255).roundToInt()
		}
		return (0xff shl 24) or (gamma(r) shl 16) or (gamma(g) shl 8) or gamma(bl)
	}

	fun labToArgb(lab: FloatArray): Int = labToArgb(lab[0].toDouble(), lab[1].toDouble(), lab[2].toDouble())
}
