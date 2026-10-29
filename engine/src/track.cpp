// Frame-to-frame tracking between detections: the visible lattice vertices (the gap crossings of the faces turned to
// the camera) followed with pyramidal Lucas–Kanade (forward–backward checked), the pose refined from them by robust
// Gauss–Newton (Cauchy weights, then the inliers) from the previous pose. Grey images at the input resolution.
#include <algorithm>
#include <chrono>
#include <cmath>

#include "internal.hpp"

namespace rubik {

namespace {

double ms (std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// bilinear sample with clamped coordinates
inline float at (const FloatImage& im, float x, float y) {
	x = std::min(std::max(x, 0.0f), float(im.width - 1) - 1e-3f);
	y = std::min(std::max(y, 0.0f), float(im.height - 1) - 1e-3f);
	int x0 = int(x), y0 = int(y);
	float fx = x - x0, fy = y - y0;
	const float* r0 = im.data.data() + size_t(y0) * im.width + x0;
	const float* r1 = r0 + im.width;
	return (r0[0] * (1 - fx) + r0[1] * fx) * (1 - fy) + (r1[0] * (1 - fx) + r1[1] * fx) * fy;
}

FloatImage downsample (const FloatImage& src) {
	// 1-4-6-4-1 blur, then every second pixel
	const int w = src.width, h = src.height, W = (w + 1) / 2, H = (h + 1) / 2;
	FloatImage tmp(W, h), out(W, H);
	const float k[5] = {1 / 16.f, 4 / 16.f, 6 / 16.f, 4 / 16.f, 1 / 16.f};
	for (int y = 0; y < h; y++) {
		const float* row = src.data.data() + size_t(y) * w;
		float* dst = tmp.data.data() + size_t(y) * W;
		for (int X = 0; X < W; X++) {
			int x = 2 * X;
			float s = 0;
			for (int j = -2; j <= 2; j++)
				s += k[j + 2] * row[std::min(std::max(x + j, 0), w - 1)];
			dst[X] = s;
		}
	}
	for (int Y = 0; Y < H; Y++) {
		int y = 2 * Y;
		float* dst = out.data.data() + size_t(Y) * W;
		for (int X = 0; X < W; X++) {
			float s = 0;
			for (int j = -2; j <= 2; j++)
				s += k[j + 2] * tmp.data[size_t(std::min(std::max(y + j, 0), h - 1)) * W + X];
			dst[X] = s;
		}
	}
	return out;
}

// Scharr derivatives / 32 (intensity per pixel), replicated border
void gradients (const FloatImage& im, FloatImage& gx, FloatImage& gy) {
	const int w = im.width, h = im.height;
	gx = FloatImage(w, h);
	gy = FloatImage(w, h);
	auto px = [&](int x, int y) { return im.data[size_t(std::min(std::max(y, 0), h - 1)) * w + std::min(std::max(x, 0), w - 1)]; };
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			float a = px(x - 1, y - 1), b = px(x, y - 1), c = px(x + 1, y - 1);
			float d = px(x - 1, y), f = px(x + 1, y);
			float g = px(x - 1, y + 1), i = px(x, y + 1), j = px(x + 1, y + 1);
			gx.data[size_t(y) * w + x] = (3 * (c - a) + 10 * (f - d) + 3 * (j - g)) / 32.f;
			gy.data[size_t(y) * w + x] = (3 * (g - a) + 10 * (i - b) + 3 * (j - c)) / 32.f;
		}
}

// 6×6 symmetric positive (semi-)definite solve by Cholesky with a small ridge; false if singular
bool solve6 (std::array<double, 36> A, std::array<double, 6> b, std::array<double, 6>& x) {
	for (int i = 0; i < 6; i++)
		A[i * 7] += 1e-9 * (1 + A[i * 7]);
	for (int j = 0; j < 6; j++) {
		double d = A[j * 6 + j];
		for (int k = 0; k < j; k++)
			d -= A[j * 6 + k] * A[j * 6 + k];
		if (!(d > 0))
			return false;
		d = std::sqrt(d);
		A[j * 6 + j] = d;
		for (int i = j + 1; i < 6; i++) {
			double s = A[i * 6 + j];
			for (int k = 0; k < j; k++)
				s -= A[i * 6 + k] * A[j * 6 + k];
			A[i * 6 + j] = s / d;
		}
	}
	for (int i = 0; i < 6; i++) {
		double s = b[i];
		for (int k = 0; k < i; k++)
			s -= A[i * 6 + k] * x[k];
		x[i] = s / A[i * 6 + i];
	}
	for (int i = 5; i >= 0; i--) {
		double s = x[i];
		for (int k = i + 1; k < 6; k++)
			s -= A[k * 6 + i] * x[k];
		x[i] = s / A[i * 6 + i];
	}
	return true;
}

// the 56 lattice vertices on the cube's surface (the gap crossings of the six faces, cubie units)
const std::vector<V3>& surfaceVertices () {
	static const std::vector<V3> v = [] {
		std::vector<V3> out;
		const double g[4] = {-1.5, -0.5, 0.5, 1.5};
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
					if (std::fabs(g[i]) == 1.5 || std::fabs(g[j]) == 1.5 || std::fabs(g[k]) == 1.5)
						out.push_back({g[i], g[j], g[k]});
		return out;
	}();
	return v;
}

}

FloatImage::FloatImage (int w, int h) : width(w), height(h), data(size_t(w) * size_t(h)) {}

GreyPyramid buildPyramid (const Image& image, int levels) {
	GreyPyramid p;
	const int ch = (image.format == PixelFormat::BGR || image.format == PixelFormat::RGB) ? 3 : 4;
	const int stride = image.stride ? image.stride : image.width * ch;
	const bool rgb = image.format == PixelFormat::RGB || image.format == PixelFormat::RGBA;
	FloatImage base(image.width, image.height);
	for (int y = 0; y < image.height; y++) {
		const uint8_t* row = image.data + size_t(y) * stride;
		float* dst = base.data.data() + size_t(y) * image.width;
		for (int x = 0; x < image.width; x++) {
			const uint8_t* q = row + x * ch;
			float r = rgb ? q[0] : q[2], g = q[1], b = rgb ? q[2] : q[0];
			dst[x] = 0.299f * r + 0.587f * g + 0.114f * b;
		}
	}
	p.levels.push_back(std::move(base));
	for (int l = 1; l < levels && p.levels.back().width >= 32 && p.levels.back().height >= 32; l++)
		p.levels.push_back(downsample(p.levels.back()));
	for (const FloatImage& im : p.levels) {
		p.gx.emplace_back();
		p.gy.emplace_back();
		gradients(im, p.gx.back(), p.gy.back());
	}
	return p;
}

bool lucasKanade (const GreyPyramid& a, const GreyPyramid& b, float x, float y, float& ox, float& oy, const TrackOptions& o) {
	const int L = int(std::min(a.levels.size(), b.levels.size()));
	const int r = o.window / 2;
	float dx = 0, dy = 0;   // flow at the current level (pixels of that level)
	std::vector<float> T, Gx, Gy;
	for (int l = L - 1; l >= 0; l--) {
		const float s = 1.0f / float(1 << l);
		const float px = x * s, py = y * s;
		const FloatImage& A = a.levels[l];
		const FloatImage& B = b.levels[l];
		if (px < 0 || py < 0 || px > A.width - 1 || py > A.height - 1)
			return false;
		// template and its gradient matrix
		T.clear();
		Gx.clear();
		Gy.clear();
		double gxx = 0, gxy = 0, gyy = 0;
		for (int j = -r; j <= r; j++)
			for (int i = -r; i <= r; i++) {
				float ix = px + i, iy = py + j;
				float gx = at(a.gx[l], ix, iy), gy = at(a.gy[l], ix, iy);
				T.push_back(at(A, ix, iy));
				Gx.push_back(gx);
				Gy.push_back(gy);
				gxx += gx * gx;
				gxy += gx * gy;
				gyy += gy * gy;
			}
		const double n = double(T.size());
		const double tr = gxx + gyy, dt = gxx * gyy - gxy * gxy;
		const double minEig = (tr - std::sqrt(std::max(0.0, tr * tr - 4 * dt))) / 2 / n;
		if (minEig < o.minEigen || !(dt > 1e-12))
			return false;
		for (int it = 0; it < o.iterations; it++) {
			double bx = 0, by = 0;
			int k = 0;
			for (int j = -r; j <= r; j++)
				for (int i = -r; i <= r; i++, k++) {
					float e = at(B, px + dx + i, py + dy + j) - T[k];
					bx += e * Gx[k];
					by += e * Gy[k];
				}
			float ux = float(-(gyy * bx - gxy * by) / dt), uy = float(-(gxx * by - gxy * bx) / dt);
			dx += ux;
			dy += uy;
			if (ux * ux + uy * uy < o.epsilon * o.epsilon)
				break;
		}
		if (l > 0) {
			dx *= 2;
			dy *= 2;
		}
	}
	ox = x + dx;
	oy = y + dy;
	return ox >= 0 && oy >= 0 && ox <= b.levels[0].width - 1 && oy <= b.levels[0].height - 1;
}

// Robust pose refinement: Gauss–Newton on the rotation vector (left increment) and translation, Cauchy-weighted, then
// a few unweighted steps on the inliers. Returns the inlier mask and the median inlier reprojection error.
bool refinePose (const Intrinsics& K, const std::vector<V3>& object, const std::vector<std::array<double, 2>>& image, M3& R, V3& t,
	const TrackOptions& o, std::vector<bool>& inlier, double& medianError) {
	const size_t n = object.size();
	inlier.assign(n, false);
	if (n < size_t(o.minPoints))
		return false;
	auto residuals = [&](const M3& Rm, V3 tv, std::vector<double>& res) {
		res.resize(2 * n);
		for (size_t i = 0; i < n; i++) {
			V3 q = mul(Rm, object[i]) + tv;
			if (!(q.z > 1e-6)) {
				res[2 * i] = res[2 * i + 1] = 1e6;
				continue;
			}
			res[2 * i] = K.fx * q.x / q.z + K.cx - image[i][0];
			res[2 * i + 1] = K.fy * q.y / q.z + K.cy - image[i][1];
		}
	};
	std::vector<double> res;
	auto step = [&](bool robust, const std::vector<bool>* only) {
		std::array<double, 36> H{};
		std::array<double, 6> g{};
		residuals(R, t, res);
		for (size_t i = 0; i < n; i++) {
			if (only && !(*only)[i])
				continue;
			V3 q = mul(R, object[i]) + t;
			if (!(q.z > 1e-6))
				continue;
			double ex = res[2 * i], ey = res[2 * i + 1];
			double w = 1;
			if (robust) {
				double e2 = (ex * ex + ey * ey) / (o.robustScale * o.robustScale);
				w = 1 / (1 + e2);
			}
			const double iz = 1 / q.z, iz2 = iz * iz;
			// ∂(u, v)/∂q
			const double du[3] = {K.fx * iz, 0, -K.fx * q.x * iz2};
			const double dv[3] = {0, K.fy * iz, -K.fy * q.y * iz2};
			// ∂q/∂ω = −[R·P]×, ∂q/∂t = I
			V3 rp = mul(R, object[i]);
			const double Jw[3][3] = {{0, rp.z, -rp.y}, {-rp.z, 0, rp.x}, {rp.y, -rp.x, 0}};
			double Ju[6], Jv[6];
			for (int c = 0; c < 3; c++) {
				Ju[c] = du[0] * Jw[0][c] + du[1] * Jw[1][c] + du[2] * Jw[2][c];
				Jv[c] = dv[0] * Jw[0][c] + dv[1] * Jw[1][c] + dv[2] * Jw[2][c];
				Ju[3 + c] = du[c];
				Jv[3 + c] = dv[c];
			}
			for (int r = 0; r < 6; r++) {
				g[r] -= w * (Ju[r] * ex + Jv[r] * ey);
				for (int c = 0; c < 6; c++)
					H[r * 6 + c] += w * (Ju[r] * Ju[c] + Jv[r] * Jv[c]);
			}
		}
		std::array<double, 6> d;
		if (!solve6(H, g, d))
			return false;
		R = matmul(rodrigues({d[0], d[1], d[2]}), R);
		t = t + V3{d[3], d[4], d[5]};
		return true;
	};
	for (int it = 0; it < o.robustIterations; it++)
		if (!step(true, nullptr))
			return false;
	auto classify = [&]() {
		residuals(R, t, res);
		int count = 0;
		for (size_t i = 0; i < n; i++) {
			inlier[i] = std::hypot(res[2 * i], res[2 * i + 1]) <= o.inlierPixels;
			count += inlier[i];
		}
		return count;
	};
	if (classify() < o.minPoints)
		return false;
	for (int it = 0; it < 3; it++)
		if (!step(false, &inlier))
			return false;
	int count = classify();
	std::vector<double> e;
	for (size_t i = 0; i < n; i++)
		if (inlier[i])
			e.push_back(std::hypot(res[2 * i], res[2 * i + 1]));
	if (e.empty())
		return false;
	std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
	medianError = e[e.size() / 2];
	return count >= o.minPoints;
}

Pose facingPose (const Pose& pose, int* gOut, int near) {
	M3 R;
	std::copy(pose.R.begin(), pose.R.end(), R.begin());
	V3 c = -1 * mulT(R, {pose.t[0], pose.t[1], pose.t[2]});   // camera in the cube frame
	const auto& G = rotations();
	int best = 0;
	double bestScore = -1e300;
	for (int g = 0; g < 24; g++) {
		// camera in the renamed frame: Gᵀ c; its smallest component as large as possible (all faces R U F turned to it)
		V3 cg{0, 0, 0};
		for (int r = 0; r < 3; r++)
			for (int k = 0; k < 3; k++)
				cg[r] += G[g][k * 3 + r] * c[k];
		double score = std::min({cg.x, cg.y, cg.z});
		if (near >= 0) {
			// ties (the same signs, another order of the axes) to the renaming nearest the previous one
			double agree = 0;
			for (int i = 0; i < 9; i++)
				agree += G[g][i] * G[near][i];
			score += 1e-6 * agree;
		}
		if (score > bestScore) {
			bestScore = score;
			best = g;
		}
	}
	M3 Gm;
	for (int i = 0; i < 9; i++)
		Gm[i] = G[best][i];
	M3 Rn = matmul(R, Gm);
	Pose out = pose;
	std::copy(Rn.begin(), Rn.end(), out.R.begin());
	if (gOut)
		*gOut = best;
	return out;
}

// --- FrameTracker --------------------------------------------------------------------------------------------------

struct FrameTracker::State {
	GreyPyramid prev;
	bool hasPrev = false;
	Intrinsics K{};
	M3 R{};
	V3 t{0, 0, 0};
	bool tracking = false;
	struct Point {
		V3 object;
		float x, y;
	};
	std::vector<Point> points;
	int width = 0, height = 0;
	int renaming = -1;   // the last output's
};

FrameTracker::FrameTracker (const TrackOptions& o) : options(o), state(std::make_unique<State>()) {}
FrameTracker::~FrameTracker () = default;
FrameTracker::FrameTracker (FrameTracker&&) noexcept = default;
FrameTracker& FrameTracker::operator= (FrameTracker&&) noexcept = default;

namespace {

// visible vertices of the pose not already tracked (within half a pitch of a tracked one in the image)
template <class Points>
void topUp (Points& points, const Intrinsics& K, const M3& R, V3 t, int width, int height, size_t want) {
	std::vector<V3> keepObjects;
	for (const auto& p : points)
		keepObjects.push_back(p.object);
	V3 camInCube = -1 * mulT(R, t);
	for (const V3& v : surfaceVertices()) {
		if (points.size() >= want)
			break;
		bool tracked = false;
		for (const V3& k : keepObjects)
			tracked = tracked || (std::fabs(k.x - v.x) + std::fabs(k.y - v.y) + std::fabs(k.z - v.z) < 1e-6);
		if (tracked)
			continue;
		// on a face turned to the camera (some face through it, at ±1.5, whose plane the camera is in front of)
		bool visible = false;
		for (int a = 0; a < 3; a++) {
			double c = v[a];
			if (std::fabs(c) != 1.5)
				continue;
			visible = visible || (c > 0 ? camInCube[a] > 1.5 : camInCube[a] < -1.5);
		}
		if (!visible)
			continue;
		V3 q = mul(R, v) + t;
		if (!(q.z > 0))
			continue;
		double u = K.fx * q.x / q.z + K.cx, w = K.fy * q.y / q.z + K.cy;
		if (u < 4 || w < 4 || u > width - 5 || w > height - 5)
			continue;
		points.push_back({v, float(u), float(w)});
	}
}

}

void FrameTracker::seed (const Image& image, const Camera& camera, const Pose& pose) {
	State& S = *state;
	auto t0 = std::chrono::steady_clock::now();
	Camera c = camera;
	if (!(c.cx > 0) || !(c.cy > 0)) {
		c.cx = image.width / 2.0;
		c.cy = image.height / 2.0;
	}
	if (!(c.fy > 0))
		c.fy = c.fx;
	S.K = {c.fx, c.fy, c.cx, c.cy};
	S.prev = buildPyramid(image, options.levels);
	S.hasPrev = true;
	S.width = image.width;
	S.height = image.height;
	std::copy(pose.R.begin(), pose.R.end(), S.R.begin());
	S.t = {pose.t[0], pose.t[1], pose.t[2]};
	S.points.clear();
	S.renaming = -1;
	topUp(S.points, S.K, S.R, S.t, S.width, S.height, 64);
	S.tracking = !S.points.empty();
	(void)ms(t0);
}

void FrameTracker::reset () {
	state->tracking = false;
	state->points.clear();
}

bool FrameTracker::tracking () const { return state->tracking; }

TrackResult FrameTracker::track (const Image& image) {
	State& S = *state;
	TrackResult out;
	auto t0 = std::chrono::steady_clock::now();
	GreyPyramid cur = buildPyramid(image, options.levels);
	out.msPyramid = ms(t0);
	if (!S.tracking || !S.hasPrev || image.width != S.width || image.height != S.height) {
		S.prev = std::move(cur);
		S.hasPrev = true;
		S.width = image.width;
		S.height = image.height;
		out.error = S.tracking ? "frame size changed" : "not seeded";
		S.tracking = false;
		return out;
	}
	t0 = std::chrono::steady_clock::now();
	std::vector<State::Point> moved;
	std::vector<V3> object;
	std::vector<std::array<double, 2>> pixels;
	for (const auto& p : S.points) {
		float x1, y1, xb, yb;
		if (!lucasKanade(S.prev, cur, p.x, p.y, x1, y1, options))
			continue;
		if (!lucasKanade(cur, S.prev, x1, y1, xb, yb, options))
			continue;
		if (std::hypot(xb - p.x, yb - p.y) > options.forwardBackward)
			continue;
		moved.push_back({p.object, x1, y1});
		object.push_back(p.object);
		pixels.push_back({x1, y1});
	}
	out.msFlow = ms(t0);
	out.tracked = int(moved.size());
	t0 = std::chrono::steady_clock::now();
	M3 R = S.R;
	V3 t = S.t;
	std::vector<bool> inlier;
	double median = 0;
	bool ok = refinePose(S.K, object, pixels, R, t, options, inlier, median);
	out.msPose = ms(t0);
	S.prev = std::move(cur);
	if (!ok) {
		S.tracking = false;
		out.error = "too few points follow the cube";
		return out;
	}
	S.R = R;
	S.t = t;
	S.points.clear();
	for (size_t i = 0; i < moved.size(); i++)
		if (inlier[i])
			S.points.push_back(moved[i]);
	out.inliers = int(S.points.size());
	out.reprojection = median;
	// lost vertices back from the new pose (re-seeded where they now project), so the count doesn't run down
	if (out.inliers < options.topUpBelow)
		topUp(S.points, S.K, S.R, S.t, S.width, S.height, size_t(options.topUpTo));
	Pose tracked;
	std::copy(S.R.begin(), S.R.end(), tracked.R.begin());
	tracked.t = {S.t.x, S.t.y, S.t.z};
	out.pose = facingPose(tracked, &out.renaming, S.renaming);
	S.renaming = out.renaming;
	for (const auto& p : S.points)
		out.points.push_back({p.x, p.y});
	out.ok = true;
	return out;
}

}
