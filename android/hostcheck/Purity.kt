// Scoring of a state estimate against a cube's known colours (the real test video's GAN cube; for checking only,
// never used by the pipeline): per state letter the share of its kept observations nearest each colour.
import com.klstudio.rubikdetector.Cube
import com.klstudio.rubikdetector.Pipeline

private val PROTO = mapOf("white" to doubleArrayOf(78.0, -2.0, 2.0), "yellow" to doubleArrayOf(70.0, -12.0, 52.0), "green" to doubleArrayOf(50.0, -40.0, 14.0),
	"blue" to doubleArrayOf(23.0, 10.0, -30.0), "maroon" to doubleArrayOf(28.0, 22.0, 12.0), "orange" to doubleArrayOf(55.0, 31.0, 38.0), "skin" to doubleArrayOf(49.0, 6.0, 16.0))

private fun nearest(lab: FloatArray) = PROTO.minByOrNull { (_, p) ->
	val d = doubleArrayOf((lab[0] - p[0]) * 0.5, lab[1] - p[1], lab[2] - p[2]); d.sumOf { it * it } }!!.key

fun purity(pipeline: Pipeline, state: String): String {
	val obs = pipeline.observations()
	val per = Cube.FACES.associateWith { c -> (0 until 54).filter { state[it] == c }.flatMap { obs[it] }.map { nearest(it) } }
	val tops = per.mapValues { (_, v) -> v.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: "-" }
	val share = per.entries.filter { it.value.isNotEmpty() }.map { (c, v) -> v.count { it == tops[c] }.toDouble() / v.size }.average()
	val distinct = tops.values.filter { it != "skin" && it != "-" }.toSet().size
	return String.format("letters' dominant colours %d/6 distinct (%s), purity %.0f%%", distinct,
		Cube.FACES.map { "$it:${tops[it]}" }.joinToString(" "), share * 100)
}
