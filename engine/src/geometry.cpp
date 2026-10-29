// Port of rubik_detector/geometry.py (LineGeometry, position_wide), plus a local search for tracked frames.
#include <algorithm>
#include <cmath>
#include <numeric>

#include "internal.hpp"
#include "lsd.hpp"
#include "optimize.hpp"

namespace rubik {

namespace {

const double PI = 3.141592653589793;

// eigen-decomposition of a symmetric 3×3 (cyclic Jacobi): eigenvalues ascending, eigenvectors as rows
void eigenSym (const M3& A, double values[3], V3 vectors[3]) {
	double a[3][3], v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			a[i][j] = A[i * 3 + j];
	for (int sweep = 0; sweep < 50; sweep++) {
		double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
		if (off < 1e-300)
			break;
		for (int p = 0; p < 2; p++) {
			for (int q = p + 1; q < 3; q++) {
				if (std::fabs(a[p][q]) < 1e-300)
					continue;
				double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
				double t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
				double c = 1 / std::sqrt(t * t + 1), s = t * c;
				for (int k = 0; k < 3; k++) {
					double akp = a[k][p], akq = a[k][q];
					a[k][p] = c * akp - s * akq;
					a[k][q] = s * akp + c * akq;
				}
				for (int k = 0; k < 3; k++) {
					double apk = a[p][k], aqk = a[q][k];
					a[p][k] = c * apk - s * aqk;
					a[q][k] = s * apk + c * aqk;
				}
				for (int k = 0; k < 3; k++) {
					double vkp = v[k][p], vkq = v[k][q];
					v[k][p] = c * vkp - s * vkq;
					v[k][q] = s * vkp + c * vkq;
				}
			}
		}
	}
	int order[3] = {0, 1, 2};
	std::sort(order, order + 3, [&](int i, int j) { return a[i][i] < a[j][j]; });
	for (int i = 0; i < 3; i++) {
		values[i] = a[order[i]][order[i]];
		vectors[i] = {v[0][order[i]], v[1][order[i]], v[2][order[i]]};
	}
}

// nearest orthonormal matrix (U·Vᵀ of the SVD), rows in and out
M3 orthonormal (const M3& D) {
	// DᵀD = V S² Vᵀ; U Vᵀ = D V S⁻¹ Vᵀ
	M3 DtD{};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				DtD[i * 3 + j] += D[k * 3 + i] * D[k * 3 + j];
	double s2[3];
	V3 v[3];
	eigenSym(DtD, s2, v);
	M3 inv{};
	for (int e = 0; e < 3; e++) {
		double s = 1 / std::sqrt(std::max(s2[e], 1e-300));
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				inv[i * 3 + j] += v[e][i] * s * v[e][j];
	}
	return matmul(D, inv);
}

V3 rayOf (const Intrinsics& K, double x, double y) { return {(x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.0}; }

V3 row (const M3& m, int r) { return {m[r * 3], m[r * 3 + 1], m[r * 3 + 2]}; }
void setRow (M3& m, int r, V3 v) {
	m[r * 3] = v.x;
	m[r * 3 + 1] = v.y;
	m[r * 3 + 2] = v.z;
}

double modPositive (double a, double b) {
	double m = std::fmod(a, b);
	return m < 0 ? m + b : m;
}

}

// --- segments ----------------------------------------------------------------------------------------------------

Segments detectSegments (const Gray8& gray, const LabImage& lab, const Gray8 planes[3]) {
	(void)planes;
	const int w = gray.width, h = gray.height;
	// LSD over gray and Lab L*, a*, b* as uint8 (L·2.55, a·2 + 128, b·2 + 128, clipped and truncated)
	Gray8 chans[3] = {Gray8(w, h), Gray8(w, h), Gray8(w, h)};
	const size_t n = size_t(w) * h;
	for (size_t i = 0; i < n; i++) {
		const float* p = lab.data.data() + i * 3;
		float v[3] = {p[0] * 2.55f, p[1] * 2 + 128, p[2] * 2 + 128};
		for (int c = 0; c < 3; c++)
			chans[c].data[i] = uint8_t(std::min(std::max(v[c], 0.0f), 255.0f));
	}
	Segments out;
	const double minLength = 0.012 * std::max(w, h);
	const Gray8* sources[4] = {&gray, &chans[0], &chans[1], &chans[2]};
	for (const Gray8* src : sources) {
		for (const LsdSegment& s : detectLineSegments(*src)) {
			double x1 = s.x1, y1 = s.y1, x2 = s.x2, y2 = s.y2;
			double length = std::hypot(x2 - x1, y2 - y1);
			if (!(length >= minLength))
				continue;
			out.s.push_back({x1, y1, x2, y2});
			out.w.push_back(length);
		}
	}
	// contrast: colour-feature distance 2.5 px to either side, median of 7 points along the segment
	for (size_t i = 0; i < out.s.size(); i++) {
		const auto& s = out.s[i];
		double length = out.w[i];
		double dx = (s[2] - s[0]) / length, dy = (s[3] - s[1]) / length;
		double nx = -dy, ny = dx;
		double d[7];
		for (int k = 0; k < 7; k++) {
			double t = k == 6 ? 0.85 : 0.15 + k * ((0.85 - 0.15) / 6);
			double px = s[0] + t * (s[2] - s[0]), py = s[1] + t * (s[3] - s[1]);
			auto at = [&](double x, double y) {
				int xi = std::min(std::max(int(std::nearbyint(x)), 0), w - 1);
				int yi = std::min(std::max(int(std::nearbyint(y)), 0), h - 1);
				return weigh(lab.at(xi, yi));
			};
			auto a = at(px + 2.5 * nx, py + 2.5 * ny), b = at(px - 2.5 * nx, py - 2.5 * ny);
			double s2 = 0;
			for (int j = 0; j < 4; j++)
				s2 += (a[j] - b[j]) * (a[j] - b[j]);
			d[k] = std::sqrt(s2);
		}
		std::nth_element(d, d + 3, d + 7);
		out.w[i] = length * d[3];
	}
	out.family.assign(out.s.size(), -1);
	return out;
}

void planeNormals (const Intrinsics& K, Segments& segs) {
	segs.normals.resize(segs.s.size());
	for (size_t i = 0; i < segs.s.size(); i++) {
		V3 n = cross(rayOf(K, segs.s[i][0], segs.s[i][1]), rayOf(K, segs.s[i][2], segs.s[i][3]));
		segs.normals[i] = (1 / std::sqrt(dot(n, n))) * n;
	}
}

// --- axes --------------------------------------------------------------------------------------------------------

namespace {

const double AXIS_TOL = 0.026176948307873153;   // sin(1.5°)

void assignFamilies (Segments& segs, const M3& D, double tol) {
	for (size_t i = 0; i < segs.normals.size(); i++) {
		double best = 1e300;
		int arg = -1;
		for (int a = 0; a < 3; a++) {
			double r = std::fabs(dot(segs.normals[i], row(D, a)));
			if (r < best) {
				best = r;
				arg = a;
			}
		}
		segs.family[i] = best < tol ? arg : -1;
	}
}

void alternate (Segments& segs, M3& D, double tol) {
	for (int it = 0; it < 4; it++) {
		assignFamilies(segs, D, tol);
		for (int a = 0; a < 3; a++) {
			M3 M{};
			int count = 0;
			for (size_t i = 0; i < segs.normals.size(); i++) {
				if (segs.family[i] != a)
					continue;
				count++;
				const V3& n = segs.normals[i];
				for (int r = 0; r < 3; r++)
					for (int c = 0; c < 3; c++)
						M[r * 3 + c] += n[r] * segs.w[i] * n[c];
			}
			if (count >= 2) {
				double values[3];
				V3 vectors[3];
				eigenSym(M, values, vectors);
				setRow(D, a, vectors[0]);
			}
		}
		D = orthonormal(D);
	}
}

}

void refineAxes (Segments& segs, M3& D) {
	for (double tol : {AXIS_TOL * 2, AXIS_TOL * 1.5, AXIS_TOL})
		alternate(segs, D, tol);
	assignFamilies(segs, D, AXIS_TOL);
}

bool cubeAxes (Segments& segs, uint64_t seed, M3& D) {
	std::vector<M3> c = cubeAxesCandidates(segs, seed, 1);
	if (c.empty())
		return false;
	D = c[0];
	refineAxes(segs, D);
	return true;
}

std::vector<M3> cubeAxesCandidates (const Segments& segs, uint64_t seed, int count, double minDeg) {
	const size_t n = segs.normals.size();
	if (n < 6)
		return {};
	std::vector<double> cdf(n);
	std::partial_sum(segs.w.begin(), segs.w.end(), cdf.begin());
	const double total = cdf.back();
	std::mt19937_64 rng(seed ? seed : 0x5eed5eedULL);
	std::uniform_real_distribution<double> uniform(0.0, total);
	auto draw = [&]() {
		return size_t(std::upper_bound(cdf.begin(), cdf.end(), uniform(rng)) - cdf.begin());
	};
	const int HYPOTHESES = 3000;
	std::vector<std::pair<double, M3>> scored;
	for (int hyp = 0; hyp < HYPOTHESES; hyp++) {
		size_t i = std::min(draw(), n - 1), j = std::min(draw(), n - 1), k = std::min(draw(), n - 1);
		V3 d1 = cross(segs.normals[i], segs.normals[j]);
		V3 d2 = cross(d1, segs.normals[k]);
		double l1 = std::sqrt(dot(d1, d1)), l2 = std::sqrt(dot(d2, d2));
		if (!(l1 > 1e-6) || !(l2 > 1e-6))
			continue;
		d1 = (1 / l1) * d1;
		d2 = (1 / l2) * d2;
		V3 d3 = cross(d1, d2);
		double score = 0;
		for (size_t s = 0; s < n; s++) {
			const V3& nn = segs.normals[s];
			double m = 0;
			for (const V3& d : {d1, d2, d3}) {
				double r = std::fabs(dot(nn, d)) / AXIS_TOL;
				m = std::max(m, 1 - r * r);
			}
			score += m * segs.w[s];
		}
		scored.push_back({score, {d1.x, d1.y, d1.z, d2.x, d2.y, d2.z, d3.x, d3.y, d3.z}});
	}
	// best first (ties: the earlier hypothesis, as the single-best search kept), then distinct ones
	std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	std::vector<M3> out;
	for (const auto& [score, D] : scored) {
		if (int(out.size()) >= count)
			break;
		bool same = false;
		for (const M3& E : out)
			same = same || sameFrame(D, E, minDeg);
		if (!same)
			out.push_back(D);
	}
	return out;
}

bool sameFrame (const M3& D, const M3& E, double deg) {
	const double cosMin = std::cos(deg * PI / 180);
	for (int a = 0; a < 3; a++) {
		double m = 0;
		for (int b = 0; b < 3; b++)
			m = std::max(m, std::fabs(dot(row(D, a), row(E, b))));
		if (m < cosMin)
			return false;
	}
	return true;
}

double presenceOf (const Intrinsics& K, const M3& R, V3 t, double cost, const LatticeInput& in, double longSide, double minExtent, double maxExtent) {
	std::array<double, 3> c = faceCoverage(R, t, in);
	std::sort(c.begin(), c.end());
	double u0 = 1e300, u1 = -1e300, v0 = 1e300, v1 = -1e300;
	for (int corner = 0; corner < 8; corner++) {
		V3 q = mul(R, V3{corner & 1 ? 1.5 : -1.5, corner & 2 ? 1.5 : -1.5, corner & 4 ? 1.5 : -1.5}) + t;
		if (!(q.z > 0))
			return 0;
		double u = K.fx * q.x / q.z + K.cx, v = K.fy * q.y / q.z + K.cy;
		u0 = std::min(u0, u);
		u1 = std::max(u1, u);
		v0 = std::min(v0, v);
		v1 = std::max(v1, v);
	}
	double extent = std::max(u1 - u0, v1 - v0) / longSide;
	return extent >= 0.75 * minExtent && extent <= maxExtent ? (c[1] + c[2]) / 2 - cost : 0.0;
}

M3 signAxes (const M3& Din, const Intrinsics& K, double u, double v, Segments& segs) {
	M3 D = Din;
	V3 ray = rayOf(K, u, v);
	for (int a = 0; a < 3; a++)
		if (dot(row(D, a), ray) > 0)
			setRow(D, a, -1 * row(D, a));
	M3 R = {D[0], D[3], D[6], D[1], D[4], D[7], D[2], D[5], D[8]};   // Dᵀ: axes as columns
	if (det(R) < 0) {
		for (int r = 0; r < 3; r++)
			std::swap(R[r * 3], R[r * 3 + 1]);
		for (int& f : segs.family)
			f = f == 0 ? 1 : (f == 1 ? 0 : f);
	}
	return R;
}

// --- position ----------------------------------------------------------------------------------------------------

LatticeInput latticeInput (const Intrinsics& K, const Segments& segs) {
	LatticeInput in;
	for (size_t i = 0; i < segs.s.size(); i++) {
		if (segs.family[i] < 0)
			continue;
		in.rays1.push_back(rayOf(K, segs.s[i][0], segs.s[i][1]));
		in.rays2.push_back(rayOf(K, segs.s[i][2], segs.s[i][3]));
		in.fam.push_back(segs.family[i]);
		in.w.push_back(segs.w[i]);
	}
	return in;
}

double latticeCost (const M3& R, V3 t, const LatticeInput& in, double tol, double outside) {
	V3 tc = mulT(R, t);   // centre in the cube frame
	if (std::max({tc.x, tc.y, tc.z}) > -1.5 - 0.3)
		return outside;
	double total = 0, wsum = 0;
	for (size_t i = 0; i < in.fam.size(); i++) {
		const int a = in.fam[i];
		const V3 q1 = mulT(R, in.rays1[i]), q2 = mulT(R, in.rays2[i]);   // rays in the cube frame
		double best = outside;
		for (int b = 0; b < 3; b++) {
			if (b == a)
				continue;
			const int c = 3 - a - b;
			auto hit = [&](const V3& q, double& s, double& pa, double& pc) {
				double den = q[b];
				den = std::fabs(den) < 1e-9 ? 1e-9 : den;
				s = (1.5 + tc[b]) / den;
				pa = s * q[a] - tc[a];
				pc = s * q[c] - tc[c];
			};
			double s1, pa1, pc1, s2, pa2, pc2;
			hit(q1, s1, pa1, pc1);
			hit(q2, s2, pa2, pc2);
			double pc = (pc1 + pc2) / 2;
			double lo = std::min(pa1, pa2), hi = std::max(pa1, pa2);
			bool inside = s1 > 0 && s2 > 0 && std::fabs(pc) <= 1.6 && lo >= -1.65 && hi <= 1.65;
			double phase = std::fabs(modPositive(pc, 1.0) - 0.5);
			double r = std::min(std::max(phase / tol, 0.0), 1.0);
			double cst = inside ? r * r : outside;
			best = std::min(best, cst);
		}
		total += best * in.w[i];
		wsum += in.w[i];
	}
	return total / wsum;
}

std::array<double, 3> faceCoverage (const M3& R, V3 t, const LatticeInput& in, double tol) {
	V3 tc = mulT(R, t);
	bool hit[3][3][4] = {};   // face b, line direction a, line index
	for (size_t i = 0; i < in.fam.size(); i++) {
		const int a = in.fam[i];
		const V3 q1 = mulT(R, in.rays1[i]), q2 = mulT(R, in.rays2[i]);
		double best = 1.0;
		int bestFace = -1, bestLine = 0;
		for (int b = 0; b < 3; b++) {
			if (b == a)
				continue;
			const int c = 3 - a - b;
			auto at = [&](const V3& q, double& s, double& pa, double& pc) {
				double den = q[b];
				den = std::fabs(den) < 1e-9 ? 1e-9 : den;
				s = (1.5 + tc[b]) / den;
				pa = s * q[a] - tc[a];
				pc = s * q[c] - tc[c];
			};
			double s1, pa1, pc1, s2, pa2, pc2;
			at(q1, s1, pa1, pc1);
			at(q2, s2, pa2, pc2);
			double pc = (pc1 + pc2) / 2;
			bool inside = s1 > 0 && s2 > 0 && std::fabs(pc) <= 1.6 && std::min(pa1, pa2) >= -1.65 && std::max(pa1, pa2) <= 1.65;
			double phase = std::fabs(modPositive(pc, 1.0) - 0.5);
			double r = std::min(phase / tol, 1.0);
			double cst = inside ? r * r : 1.0;
			if (cst < best) {
				best = cst;
				bestFace = b;
				bestLine = std::min(3, std::max(0, int(std::floor(pc + 2.0))));
			}
		}
		if (bestFace >= 0 && best < 0.25)
			hit[bestFace][a][bestLine] = true;
	}
	std::array<double, 3> out{};
	for (int b = 0; b < 3; b++) {
		int n = 0;
		for (int a = 0; a < 3; a++)
			if (a != b)
				for (int k = 0; k < 4; k++)
					n += hit[b][a][k];
		out[b] = n / 8.0;
	}
	return out;
}

namespace {

struct PoseSearch {
	const Intrinsics& K;
	const LatticeInput& in;
	// log-depth range the search may use: outside it a pose costs as if no segment fit. Without it the refinement can
	// run to a lattice several times the cube's size (a pitch of several cubies) on which a few long segments happen to
	// fit, at a low cost averaged over the few assigned — seen on real video, accepted at cost 0.04.
	double lzLo = -1e300, lzHi = 1e300;

	V3 tOf (const Vec& x) const {
		double z = std::exp(x[2]);
		return {z * (x[0] - K.cx) / K.fx, z * (x[1] - K.cy) / K.fy, z};
	}

	Vec xOf (V3 t) const { return {K.fx * t.x / t.z + K.cx, K.fy * t.y / t.z + K.cy, std::log(t.z)}; }

	double cost (const Vec& x, const M3& R, double tol = 0.35) const {
		if (x[2] < lzLo || x[2] > lzHi)
			return 1.0;
		return latticeCost(R, tOf(x), in, tol);
	}

	// Nelder–Mead simplices sized to the problem: a quarter cubie in (u, v, log Z), a degree in rotation
	std::vector<Vec> simplex (const Vec& x, bool rotation) const {
		double pitch = K.fx / std::exp(x[2]);
		Vec x0 = x;
		std::vector<double> steps = {pitch / 4, pitch / 4, 0.25 / 4.9};
		if (rotation) {
			x0.insert(x0.end(), {0.0, 0.0, 0.0});
			steps.insert(steps.end(), 3, PI / 180);
		}
		std::vector<Vec> sim(x0.size() + 1, x0);
		for (size_t i = 0; i < x0.size(); i++)
			sim[i + 1][i] += steps[i];
		return sim;
	}

	Vec refine3 (const Vec& x, const M3& R, double tol) const {
		return nelderMead([&](const Vec& y) { return cost(y, R, tol); }, simplex(x, false), 0.05, 1e-5).x;
	}

	PoseFit refine6 (const M3& R, const Vec& x, double xatol = 1e-4, double fatol = 1e-6) const {
		auto res = nelderMead([&](const Vec& y) {
			return cost({y[0], y[1], y[2]}, matmul(rodrigues({y[3], y[4], y[5]}), R));
		}, simplex(x, true), xatol, fatol, 3000);
		return {matmul(rodrigues({res.x[3], res.x[4], res.x[5]}), R), tOf({res.x[0], res.x[1], res.x[2]}), res.fun};
	}

	// the lattice repeats every cubie: shifts by ±1 along each axis (pairs: also along two axes at once), with the
	// refined axes
	PoseFit shiftCheck (PoseFit best, bool pairs = false) const {
		std::vector<V3> shifts;
		for (int a = 0; a < 3; a++)
			for (int sgn : {-1, 1}) {
				V3 d{0, 0, 0};
				d[a] = sgn;
				shifts.push_back(d);
			}
		if (pairs)
			for (int a = 0; a < 3; a++)
				for (int b = a + 1; b < 3; b++)
					for (int sa : {-1, 1})
						for (int sb : {-1, 1}) {
							V3 d{0, 0, 0};
							d[a] = sa;
							d[b] = sb;
							shifts.push_back(d);
						}
		bool moved = true;
		while (moved) {
			moved = false;
			for (const V3& d : shifts) {
				{
					Vec s = xOf(best.t + mul(best.R, d));
					Vec y = refine3(s, best.R, 0.35);
					double cy = cost(y, best.R);
					if (cy < best.cost - 1e-3) {
						best = refine6(best.R, y);
						moved = true;
					}
				}
			}
		}
		return best;
	}
};

}

namespace {

LatticeInput heaviest (const LatticeInput& in, size_t count) {
	std::vector<size_t> order(in.w.size());
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return in.w[a] > in.w[b]; });
	LatticeInput top;
	for (size_t i = 0; i < std::min(order.size(), count); i++) {
		top.rays1.push_back(in.rays1[order[i]]);
		top.rays2.push_back(in.rays2[order[i]]);
		top.fam.push_back(in.fam[order[i]]);
		top.w.push_back(in.w[order[i]]);
	}
	return top;
}

// From coarse cells (u, v, log Z): rank by the coarse cost (wide phase tolerance, heaviest segments), refine
// `starts` distinct ones (tolerance 0.5 → 0.35), jointly refine the best `joint` with a small rotation (loose),
// polish the winner, check ±1 cubie shifts.
PoseFit fromCells (const Intrinsics& K, const M3& R, const LatticeInput& in, const std::vector<Vec>& cells, int starts, int joint,
	const FramingPrior& effort) {
	const size_t COARSE_SEGMENTS = effort.coarseSegments > 0 ? size_t(effort.coarseSegments) : in.w.size();
	const double DEPTH_MARGIN = 0.15;   // log depth beyond the cells' range (±16 %)
	LatticeInput top = heaviest(in, COARSE_SEGMENTS);
	PoseSearch search{K, in};
	PoseSearch coarse{K, top};
	if (!cells.empty()) {
		double lo = 1e300, hi = -1e300;
		for (const Vec& c : cells) {
			lo = std::min(lo, c[2]);
			hi = std::max(hi, c[2]);
		}
		search.lzLo = coarse.lzLo = lo - DEPTH_MARGIN;
		search.lzHi = coarse.lzHi = hi + DEPTH_MARGIN;
	}
	std::vector<double> costs(cells.size());
	for (size_t i = 0; i < cells.size(); i++)
		costs[i] = coarse.cost(cells[i], R, 0.5);
	std::vector<size_t> ranked(cells.size());
	std::iota(ranked.begin(), ranked.end(), 0);
	std::stable_sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) { return costs[a] < costs[b]; });

	// distinct starts: more than half a cubie apart in the cube frame
	std::vector<size_t> chosen;
	for (size_t i : ranked) {
		V3 t = search.tOf(cells[i]);
		bool distinct = true;
		for (size_t j : chosen) {
			V3 d = mulT(R, t - search.tOf(cells[j]));
			distinct = distinct && std::sqrt(dot(d, d)) > 0.5;
		}
		if (distinct)
			chosen.push_back(i);
		if (int(chosen.size()) == starts)
			break;
	}
	std::vector<std::pair<double, Vec>> fits;
	for (size_t i : chosen) {
		Vec x = cells[i];
		for (double tol : {0.5, 0.35})
			x = search.refine3(x, R, tol);
		fits.push_back({search.cost(x, R), x});
	}
	std::stable_sort(fits.begin(), fits.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
	std::vector<Vec> kept;
	for (const auto& f : fits) {
		bool distinct = true;
		for (const Vec& y : kept) {
			V3 d = mulT(R, search.tOf(f.second) - search.tOf(y));
			distinct = distinct && std::sqrt(dot(d, d)) > 0.3;
		}
		if (distinct)
			kept.push_back(f.second);
	}
	if (kept.empty())
		return {R, cells.empty() ? V3{0, 0, 1} : search.tOf(cells[ranked[0]]), 1.0};

	PoseFit best{R, {0, 0, 0}, 1e300};
	for (size_t i = 0; i < std::min(kept.size(), size_t(joint)); i++) {
		PoseFit f = search.refine6(R, kept[i], 1e-2, 1e-4);
		if (f.cost < best.cost)
			best = f;
	}
	best = search.refine6(best.R, search.xOf(best.t));
	return search.shiftCheck(best, effort.shiftPairs);
}

}

PoseFit positionWide (const Intrinsics& K, const M3& R, const Segments& segs, int width, int height, const FramingPrior& prior, int starts, int joint) {
	LatticeInput in = latticeInput(K, segs);
	const double L = std::max(width, height);
	// per depth (log step 0.1), (u, v) steps of a third of the cubie pitch, over the framing prior
	std::vector<Vec> cells;
	double lz0 = std::log(K.fx * 4.9 / (prior.maxExtent * L)), lz1 = std::log(K.fx * 4.9 / (prior.minExtent * L));
	for (int iz = 0;; iz++) {
		double lz = lz0 + 0.1 * iz;
		if (lz > lz1 + 1e-9)
			break;
		double step = K.fx / std::exp(lz) / 3;
		double u0 = width * (1 - prior.centreShare) / 2, u1 = width * (1 + prior.centreShare) / 2;
		double v0 = height * (1 - prior.centreShare) / 2, v1 = height * (1 + prior.centreShare) / 2;
		int nu = int(std::ceil((u1 + 1e-9 - u0) / step)), nv = int(std::ceil((v1 + 1e-9 - v0) / step));
		for (int iv = 0; iv < nv; iv++)
			for (int iu = 0; iu < nu; iu++)
				cells.push_back({u0 + iu * step, v0 + iv * step, lz});
	}
	return fromCells(K, R, in, cells, prior.starts > 0 ? prior.starts : starts, prior.joint > 0 ? prior.joint : joint, prior);
}

PoseFit positionLocal (const Intrinsics& K, const M3& R, V3 t, const Segments& segs, const FramingPrior& effort) {
	LatticeInput in = latticeInput(K, segs);
	PoseSearch search{K, in};
	// the prior's neighbourhood: ±2/3 cubie in (u, v) at thirds of a cubie, depth ±10 % in log steps of 0.05
	Vec x0 = search.xOf(t);
	double step = K.fx / std::exp(x0[2]) / 3;
	std::vector<Vec> cells;
	for (int iz = -2; iz <= 2; iz++)
		for (int iv = -2; iv <= 2; iv++)
			for (int iu = -2; iu <= 2; iu++)
				cells.push_back({x0[0] + iu * step, x0[1] + iv * step, x0[2] + 0.05 * iz});
	return fromCells(K, R, in, cells, 3, 2, effort);
}

std::vector<std::pair<int, std::array<double, 8>>> faceletQuads (const Intrinsics& K, const M3& R, V3 t, double sampleSize) {
	const double h = sampleSize / 2;
	const double offsets[4][2] = {{-h, -h}, {h, -h}, {h, h}, {-h, h}};
	std::vector<std::pair<int, std::array<double, 8>>> out;
	for (int f : {1, 3, 5}) {
		for (int k = f * 9; k < f * 9 + 9; k++) {
			std::array<double, 8> q{};
			for (int i = 0; i < 4; i++) {
				V3 P = mul(R, faceletPoint(k, 1.5, offsets[i][0], offsets[i][1])) + t;
				q[2 * i] = K.fx * P.x / P.z + K.cx;
				q[2 * i + 1] = K.fy * P.y / P.z + K.cy;
			}
			out.push_back({k, q});
		}
	}
	return out;
}

}
