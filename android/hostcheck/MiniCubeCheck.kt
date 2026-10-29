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
	println(if (failures == 0) "MiniCube twists = applyMoves on all ${Cube.MOVES.size} face turns" else "$failures failures")
}
