// Detector: one image → pose and facelet colours (rubik_detector.geometry.LineGeometry + colour sampling).
#include <chrono>
#include <algorithm>
#include <cmath>

#include "internal.hpp"

namespace rubik {

struct Detector::Scratch {
	Gray8 planes[3];
};

Detector::Detector () : scratch(new Scratch) {}
Detector::~Detector () = default;
Detector::Detector (Detector&&) noexcept = default;
Detector& Detector::operator= (Detector&&) noexcept = default;

namespace {

double ms (std::chrono::steady_clock::time_point since) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

bool validImage (const Image& image) {
	return image.data && image.width >= 16 && image.height >= 16;
}

// working-size intrinsics: the image is scaled by s
Intrinsics scaled (const Camera& c, double s) { return {c.fx * s, c.fy * s, c.cx * s, c.cy * s}; }

M3 poseR (const Pose& p) {
	M3 R;
	std::copy(p.R.begin(), p.R.end(), R.begin());
	return R;
}

std::vector<FaceletSample> sampleAt (const LabImage& lab, const Intrinsics& K, const M3& R, V3 t, double sampleSize, double s) {
	std::vector<FaceletSample> out;
	for (const auto& [k, q] : faceletQuads(K, R, t, sampleSize)) {
		FaceletSample fs;
		fs.facelet = k;
		fs.valid = sampleQuad(lab, q, fs.lab, fs.pixels);
		for (int i = 0; i < 8; i++)
			fs.quad[i] = float(q[i] / s);
		out.push_back(fs);
	}
	return out;
}

}

ViewResult Detector::detect (const Image& image, const Camera& camera, const DetectOptions& options) {
	ViewResult out;
	if (!validImage(image)) {
		out.error = "bad image";
		return out;
	}
	auto t0 = std::chrono::steady_clock::now();
	workingPlanes(image, options.workWidth, scratch->planes);
	const double s = double(scratch->planes[0].width) / image.width;
	const Intrinsics K = scaled(camera, s);
	const int W = scratch->planes[0].width, H = scratch->planes[0].height;
	LabImage lab = toLab(scratch->planes);
	Segments segs = detectSegments(toGray(scratch->planes), lab, scratch->planes);
	out.segmentCount = int(segs.s.size());
	out.timings.segments = ms(t0);
	if (segs.s.size() < 12) {
		out.error = std::to_string(segs.s.size()) + " segments";
		return out;
	}
	planeNormals(K, segs);

	t0 = std::chrono::steady_clock::now();
	M3 R;
	V3 t{0, 0, 0};
	if (options.search == Search::Wide) {
		M3 D;
		if (!cubeAxes(segs, options.seed, D)) {
			out.error = "no cube axes";
			return out;
		}
		// signs: each axis' face turned to the camera, the cube taken at the weighted centroid of family segments
		double cu = 0, cv = 0, ws = 0;
		for (size_t i = 0; i < segs.s.size(); i++)
			if (segs.family[i] >= 0) {
				cu += segs.w[i] * (segs.s[i][0] + segs.s[i][2]) / 2;
				cv += segs.w[i] * (segs.s[i][1] + segs.s[i][3]) / 2;
				ws += segs.w[i];
			}
		if (ws <= 0) {
			out.error = "no segment along the cube axes";
			return out;
		}
		R = signAxes(D, K, cu / ws, cv / ws, segs);
	}
	else {
		// families and axes from the prior, signed towards the camera at the prior centre
		M3 P = poseR(options.prior);
		M3 D = {P[0], P[3], P[6], P[1], P[4], P[7], P[2], P[5], P[8]};
		refineAxes(segs, D);
		V3 pt{options.prior.t[0], options.prior.t[1], options.prior.t[2]};
		if (!(pt.z > 0)) {
			out.error = "prior behind the camera";
			return out;
		}
		R = signAxes(D, K, K.fx * pt.x / pt.z + K.cx, K.fy * pt.y / pt.z + K.cy, segs);
		t = pt;
	}
	for (int f : segs.family)
		out.familySegments += f >= 0;
	out.timings.axes = ms(t0);

	t0 = std::chrono::steady_clock::now();
	if (out.familySegments < 6) {
		out.error = "too few segments along the cube axes";
		return out;
	}
	FramingPrior framing{options.minExtent, options.maxExtent, options.centreShare};
	PoseFit fit = options.search == Search::Wide ? positionWide(K, R, segs, W, H, framing) : positionLocal(K, R, t, segs);
	out.timings.position = ms(t0);

	t0 = std::chrono::steady_clock::now();
	out.samples = sampleAt(lab, K, fit.R, fit.t, options.sampleSize, s);
	out.timings.sampling = ms(t0);
	std::copy(fit.R.begin(), fit.R.end(), out.pose.R.begin());
	out.pose.t = {fit.t.x, fit.t.y, fit.t.z};
	out.latticeCost = fit.cost;
	if (options.keepSegments)
		for (size_t i = 0; i < segs.s.size(); i++)
			out.segments.push_back({float(segs.s[i][0] / s), float(segs.s[i][1] / s), float(segs.s[i][2] / s), float(segs.s[i][3] / s),
				float(segs.w[i]), segs.family[i]});
	out.ok = true;
	return out;
}

std::vector<FaceletSample> Detector::sample (const Image& image, const Camera& camera, const Pose& pose, int workWidth, double sampleSize) {
	if (!validImage(image))
		return {};
	workingPlanes(image, workWidth, scratch->planes);
	const double s = double(scratch->planes[0].width) / image.width;
	const Intrinsics K = scaled(camera, s);
	const M3 R = poseR(pose);
	const V3 t{pose.t[0], pose.t[1], pose.t[2]};
	// Lab only where the patches are (the full-frame conversion would dominate a tracked frame)
	std::vector<std::array<int, 4>> boxes;
	for (const auto& [k, q] : faceletQuads(K, R, t, sampleSize)) {
		(void)k;
		double x0 = q[0], x1 = q[0], y0 = q[1], y1 = q[1];
		for (int i = 1; i < 4; i++) {
			x0 = std::min(x0, q[2 * i]);
			x1 = std::max(x1, q[2 * i]);
			y0 = std::min(y0, q[2 * i + 1]);
			y1 = std::max(y1, q[2 * i + 1]);
		}
		if (!(x1 - x0 < 1e5 && y1 - y0 < 1e5))   // a patch projected through the camera plane: skip
			continue;
		boxes.push_back({int(std::floor(x0)) - 1, int(std::floor(y0)) - 1, int(std::ceil(x1)) + 2, int(std::ceil(y1)) + 2});
	}
	LabImage lab = toLabWithin(scratch->planes, boxes);
	return sampleAt(lab, K, R, t, sampleSize, s);
}

}
