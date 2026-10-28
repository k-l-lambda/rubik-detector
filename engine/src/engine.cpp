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

namespace {

// everything after segment detection, for one camera: axes, signs, position
struct Solve {
	bool ok = false;
	std::string error;
	PoseFit fit;
	Segments segs;
	int familySegments = 0;
	double msAxes = 0, msPosition = 0;
	// share of all segment weight the lattice explains: Σ_family w·(1 - cost_i) / Σ_all w. Comparable across cameras,
	// unlike the lattice cost, which averages over the family segments only: a wrong focal length assigns fewer
	// segments to the axes, and those few can fit well
	double support = 0;
};

Solve solveFor (const Segments& base, const Intrinsics& K, int W, int H, const DetectOptions& options, int starts = 8, int joint = 3) {
	Solve out;
	out.segs = base;
	Segments& segs = out.segs;
	planeNormals(K, segs);
	auto t0 = std::chrono::steady_clock::now();
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
	out.msAxes = ms(t0);
	if (out.familySegments < 6) {
		out.error = "too few segments along the cube axes";
		return out;
	}
	t0 = std::chrono::steady_clock::now();
	FramingPrior framing{options.minExtent, options.maxExtent, options.centreShare, options.coarseSegments, options.shiftPairs, options.starts, options.joint};
	out.fit = options.search == Search::Wide ? positionWide(K, R, segs, W, H, framing, starts, joint) : positionLocal(K, R, t, segs, framing);
	out.msPosition = ms(t0);
	double all = 0, fam = 0;
	for (size_t i = 0; i < segs.w.size(); i++) {
		all += segs.w[i];
		if (segs.family[i] >= 0)
			fam += segs.w[i];
	}
	out.support = all > 0 ? (1 - out.fit.cost) * fam / all : 0;
	out.ok = true;
	return out;
}

}

namespace {

// an image at a working width, up to its segments (camera independent)
struct Prepared {
	bool ok = false;
	std::string error;
	double s = 1;          // working px per input px
	int W = 0, H = 0;
	LabImage lab;
	Segments segs;
	double msSegments = 0;
};

Prepared prepare (const Image& image, int workWidth, Gray8 planes[3]) {
	Prepared p;
	if (!validImage(image)) {
		p.error = "bad image";
		return p;
	}
	auto t0 = std::chrono::steady_clock::now();
	// never upscaled: interpolated pixels add no lines, only time (a 720-wide video frame at 1024: 2× the pixels)
	workingPlanes(image, workWidth > 0 ? std::min(workWidth, image.width) : workWidth, planes);
	p.s = double(planes[0].width) / image.width;
	p.W = planes[0].width;
	p.H = planes[0].height;
	p.lab = toLab(planes);
	p.segs = detectSegments(toGray(planes), p.lab, planes);
	p.msSegments = ms(t0);
	if (p.segs.s.size() < 12) {
		p.error = std::to_string(p.segs.s.size()) + " segments";
		return p;
	}
	p.ok = true;
	return p;
}

Camera completed (Camera c, const Image& image) {
	if (!(c.cx > 0) || !(c.cy > 0)) {
		c.cx = image.width / 2.0;
		c.cy = image.height / 2.0;
	}
	if (!(c.fy > 0))
		c.fy = c.fx;
	return c;
}

}

ViewResult Detector::detect (const Image& image, const Camera& camera, const DetectOptions& options) {
	ViewResult out;
	if (!(camera.fx > 0)) {
		out.error = "unknown focal length: estimate it first (Detector::estimateFocal)";
		return out;
	}
	Prepared p = prepare(image, options.workWidth, scratch->planes);
	out.segmentCount = int(p.segs.s.size());
	out.timings.segments = p.msSegments;
	if (!p.ok) {
		out.error = p.error;
		return out;
	}
	const Camera cam = completed(camera, image);
	const Intrinsics K = scaled(cam, p.s);
	Solve best = solveFor(p.segs, K, p.W, p.H, options);
	out.camera = cam;
	out.familySegments = best.familySegments;
	out.timings.axes = best.msAxes;
	out.timings.position = best.msPosition;
	out.support = best.support;
	if (!best.ok) {
		out.error = best.error;
		return out;
	}
	const PoseFit& fit = best.fit;
	auto t0 = std::chrono::steady_clock::now();
	out.samples = sampleAt(p.lab, K, fit.R, fit.t, options.sampleSize, p.s);
	out.timings.sampling = ms(t0);
	std::copy(fit.R.begin(), fit.R.end(), out.pose.R.begin());
	out.pose.t = {fit.t.x, fit.t.y, fit.t.z};
	out.latticeCost = fit.cost;
	if (options.keepSegments)
		for (size_t i = 0; i < best.segs.s.size(); i++)
			out.segments.push_back({float(best.segs.s[i][0] / p.s), float(best.segs.s[i][1] / p.s), float(best.segs.s[i][2] / p.s),
				float(best.segs.s[i][3] / p.s), float(best.segs.w[i]), best.segs.family[i]});
	out.ok = true;
	return out;
}

double Detector::poseCost (const Image& image, const Camera& camera, const Pose& pose, int workWidth) {
	Prepared p = prepare(image, workWidth, scratch->planes);
	if (!p.ok)
		return 1.0;
	const Intrinsics K = scaled(completed(camera, image), p.s);
	Segments segs = p.segs;
	planeNormals(K, segs);
	M3 P = poseR(pose);
	M3 D = {P[0], P[3], P[6], P[1], P[4], P[7], P[2], P[5], P[8]};
	refineAxes(segs, D);
	// keep the given axes (refineAxes only assigns the families here) and signs
	LatticeInput in = latticeInput(K, segs);
	return latticeCost(P, {pose.t[0], pose.t[1], pose.t[2]}, in);
}

FocalEstimate Detector::estimateFocal (const std::vector<Image>& images, const Camera& hint, const FocalOptions& options) {
	FocalEstimate out;
	auto t0 = std::chrono::steady_clock::now();
	if (images.empty()) {
		out.error = "no image";
		return out;
	}
	for (const Image& im : images)
		if (im.width != images[0].width || im.height != images[0].height) {
			out.error = "the images must have one size (one camera)";
			return out;
		}
	const Image& first = images[0];
	const double L = std::max(first.width, first.height);
	DetectOptions solveOptions;
	solveOptions.seed = options.seed;
	solveOptions.minExtent = options.minExtent;
	solveOptions.maxExtent = options.maxExtent;
	solveOptions.centreShare = options.centreShare;
	auto cameraAt = [&](double logRatio) {
		Camera c = hint;
		c.fx = std::exp(logRatio) * L;
		c.fy = 0;
		return completed(c, first);
	};
	// coarse: the whole range, log-spaced, at the coarse width, each a quick wide solve (3 starts, 1 joint refinement:
	// enough to rank the focal lengths); its pose per image is kept as the fine stage's prior
	std::vector<Prepared> coarse;
	for (const Image& im : images)
		coarse.push_back(prepare(im, options.coarseWidth, scratch->planes));
	if (std::none_of(coarse.begin(), coarse.end(), [](const Prepared& p) { return p.ok; })) {
		out.error = coarse[0].error;
		return out;
	}
	const int steps = std::max(3, options.coarseSteps);
	const double lo = std::log(options.focalMin), hi = std::log(options.focalMax), step = (hi - lo) / (steps - 1);
	int bestI = -1;
	double bestV = 1e300;
	std::vector<std::pair<bool, PoseFit>> priors(images.size(), {false, {}});
	for (int i = 0; i < steps; i++) {
		double x = lo + i * step, total = 0;
		std::vector<std::pair<bool, PoseFit>> poses(images.size(), {false, {}});
		for (size_t j = 0; j < coarse.size(); j++) {
			if (!coarse[j].ok) {
				total += 1;
				continue;
			}
			Solve r = solveFor(coarse[j].segs, scaled(cameraAt(x), coarse[j].s), coarse[j].W, coarse[j].H, solveOptions, 3, 1);
			out.solves++;
			total += r.ok ? 1 - r.support : 1;
			poses[j] = {r.ok, r.fit};
		}
		double v = total / coarse.size();
		out.curve.push_back({std::exp(x), 1 - v});
		if (v < bestV) {
			bestV = v;
			bestI = i;
			priors = poses;
		}
	}
	coarse.clear();
	const double xBest = lo + bestI * step;

	// fine: golden section at the working width between the coarse best's neighbours, each a local solve from the
	// coarse pose; at focal k·f the same projection is the centre (x, y, k·z)
	std::vector<Prepared> fine;
	for (const Image& im : images)
		fine.push_back(prepare(im, options.workWidth, scratch->planes));
	auto objective = [&](double x) {
		double total = 0;
		for (size_t j = 0; j < fine.size(); j++) {
			if (!fine[j].ok || !priors[j].first) {
				total += 1;
				continue;
			}
			DetectOptions local = solveOptions;
			local.search = Search::Local;
			const PoseFit& p = priors[j].second;
			std::copy(p.R.begin(), p.R.end(), local.prior.R.begin());
			local.prior.t = {p.t.x, p.t.y, p.t.z * std::exp(x - xBest)};
			Solve r = solveFor(fine[j].segs, scaled(cameraAt(x), fine[j].s), fine[j].W, fine[j].H, local);
			out.solves++;
			total += r.ok ? 1 - r.support : 1;
		}
		return total / fine.size();
	};
	double a = lo + std::max(bestI - 1, 0) * step, b = lo + std::min(bestI + 1, steps - 1) * step;
	const double g = 0.6180339887498949;
	double x1 = b - g * (b - a), x2 = a + g * (b - a);
	double f1 = objective(x1), f2 = objective(x2);
	for (int it = 0; it < options.refineSteps; it++) {
		if (f1 < f2) {
			b = x2;
			x2 = x1;
			f2 = f1;
			x1 = b - g * (b - a);
			f1 = objective(x1);
		}
		else {
			a = x1;
			x1 = x2;
			f1 = f2;
			x2 = a + g * (b - a);
			f2 = objective(x2);
		}
	}
	double x = f1 < f2 ? x1 : x2;
	out.support = 1 - std::min(f1, f2);
	out.camera = cameraAt(x);
	out.ratio = std::exp(x);
	out.ms = ms(t0);
	out.ok = out.support > 0;
	if (!out.ok)
		out.error = "no lattice fit at any focal length";
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
