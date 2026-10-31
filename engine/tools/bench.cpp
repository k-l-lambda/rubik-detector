// Per-step timing of a detection on raw RGBA frames (no image decoder in the engine: tools/to_rgba.py writes them),
// for comparing machines (desktop vs a phone over adb, pinned to a core with taskset).
//
//	bench FX FILE.rgba WIDTH HEIGHT [FILE.rgba WIDTH HEIGHT ...]
//
// Per frame (best of 3): the segment stage's parts (working planes, Lab, gray, LSD per source, the whole stage) and
// whole wide detections with 1, 2 and 8 cube-axes hypotheses.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <vector>

#include "internal.hpp"
#include "lsd.hpp"
#include "rubik/engine.h"

using namespace rubik;

namespace {

double best (int repeats, const std::function<void()>& f) {
	double out = 1e300;
	for (int r = 0; r < repeats; r++) {
		auto t0 = std::chrono::steady_clock::now();
		f();
		out = std::min(out, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
	}
	return out;
}

}

int main (int argc, char** argv) {
	if (argc < 5 || (argc - 2) % 3) {
		std::fprintf(stderr, "usage: bench FX FILE.rgba WIDTH HEIGHT [...]\n");
		return 2;
	}
	const double fx = std::atof(argv[1]);
	Detector detector;
	for (int a = 2; a + 2 < argc; a += 3) {
		const int w = std::atoi(argv[a + 1]), h = std::atoi(argv[a + 2]);
		std::vector<uint8_t> data(size_t(w) * h * 4);
		std::ifstream in(argv[a], std::ios::binary);
		if (!in.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()))) {
			std::fprintf(stderr, "%s: short read\n", argv[a]);
			return 1;
		}
		Image image{data.data(), w, h, w * 4, PixelFormat::RGBA};
		Gray8 planes[3];
		LabImage lab;
		Gray8 gray;
		const double tPlanes = best(3, [&] { workingPlanes(image, std::min(1024, w), planes); });
		const double tLab = best(3, [&] { lab = toLab(planes); });
		const double tGray = best(3, [&] { gray = toGray(planes); });
		// the LSD sources as detectSegments makes them
		const int W = planes[0].width, H = planes[0].height;
		Gray8 chans[3] = {Gray8(W, H), Gray8(W, H), Gray8(W, H)};
		for (size_t i = 0; i < size_t(W) * H; i++) {
			const float* p = lab.data.data() + i * 3;
			float v[3] = {p[0] * 2.55f, p[1] * 2 + 128, p[2] * 2 + 128};
			for (int c = 0; c < 3; c++)
				chans[c].data[i] = uint8_t(std::min(std::max(v[c], 0.0f), 255.0f));
		}
		const Gray8* sources[4] = {&gray, &chans[0], &chans[1], &chans[2]};
		double tLsd[4];
		size_t nLsd[4];
		for (int s = 0; s < 4; s++)
			tLsd[s] = best(3, [&] { nLsd[s] = detectLineSegments(*sources[s]).size(); });
		Segments segs;
		const double tSegs = best(3, [&] { segs = detectSegments(gray, lab, planes); });
		std::printf("%s %dx%d (work %dx%d)\n  planes %.1f  lab %.1f  gray %.1f  lsd gray %.1f (%zu)  L %.1f (%zu)  a %.1f (%zu)  b %.1f (%zu)  detectSegments %.1f (%zu)  ms\n",
			argv[a], w, h, W, H, tPlanes, tLab, tGray, tLsd[0], nLsd[0], tLsd[1], nLsd[1], tLsd[2], nLsd[2], tLsd[3], nLsd[3], tSegs, segs.s.size());
		for (int hyp : {1, 2, 8}) {
			DetectOptions o;
			o.axisHypotheses = hyp;
			ViewResult r;
			const double t = best(3, [&] { r = detector.detect(image, Camera{fx, fx, 0, 0}, o); });
			std::printf("  wide, %d hypotheses: %.0f ms (segments %.0f axes %.0f position %.0f sampling %.0f)  ok %d cost %.3f presence %.2f\n",
				hyp, t, r.timings.segments, r.timings.axes, r.timings.position, r.timings.sampling, int(r.ok), r.latticeCost, r.presence);
		}
	}
	return 0;
}
