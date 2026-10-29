// MiniCube's twist geometry against the engine: turning a move's layer through the full animation angle carries every
// facelet's colour to where Engine.applyMoves puts it. Run: MAIN=MiniCubeCheckKt hostcheck/run.sh -
import com.klstudio.rubikdetector.Cube
import com.klstudio.rubikdetector.Engine
import com.klstudio.rubikdetector.MiniCube

fun main() {
	val solved = Cube.FACES.flatMap { c -> List(9) { c } }.joinToString("")
	val state = Engine.applyMoves(solved, "R U2 F' L D B2 R' U F2 D' L2 B")
	var failures = 0
	for (move in Cube.MOVES) {
		val after = Engine.applyMoves(state, move)
		val tw = MiniCube.twistFor(Cube.FACES.indexOf(move[0]), move.substring(1), 1.0)
		val shown = CharArray(54) { '?' }
		for (k in 0 until 54)
			shown[MiniCube.landing(k, tw)] = state[k]
		if (String(shown) != after) {
			failures++
			println("$move: turned layer ${String(shown)} vs applyMoves $after")
		}
	}
	// a picture: the default view, solved colours, and an R turn part way
	val argb = mapOf('L' to 0xffff8000.toInt(), 'R' to 0xffc00000.toInt(), 'D' to 0xffffff00.toInt(), 'U' to 0xffffffff.toInt(), 'B' to 0xff0050ff.toInt(), 'F' to 0xff00a040.toInt())
	val img = java.awt.image.BufferedImage(600, 300, java.awt.image.BufferedImage.TYPE_INT_RGB)
	val g = img.createGraphics()
	g.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING, java.awt.RenderingHints.VALUE_ANTIALIAS_ON)
	g.color = java.awt.Color(40, 40, 40)
	g.fillRect(0, 0, 600, 300)
	for ((n, tw) in listOf(null, MiniCube.twistFor(1, "", 0.3)).withIndex()) {
		val view = com.klstudio.rubikdetector.MiniView(MiniCube.defaultR(), IntArray(54) { k -> if (k % 9 == 2 && n == 0) MiniCube.withConfidence(argb.getValue(solved[k]), 0.5) else argb.getValue(solved[k]) }, tw)
		for (poly in MiniCube.polygons(view)) {
			val xs = IntArray(4) { (150 + 300 * n + poly.points[2 * it] * 50).toInt() }
			val ys = IntArray(4) { (150 + poly.points[2 * it + 1] * 50).toInt() }
			g.color = java.awt.Color(poly.argb)
			g.fillPolygon(xs, ys, 4)
			if (poly.outline) {
				g.color = java.awt.Color(0, 0, 0, 150)
				g.stroke = java.awt.BasicStroke(0.5f)
				g.drawPolygon(xs, ys, 4)
			}
		}
	}
	javax.imageio.ImageIO.write(img, "png", java.io.File("/tmp/mini.png"))
	println(if (failures == 0) "MiniCube twists = applyMoves on all ${Cube.MOVES.size} face turns" else "$failures failures")
}
