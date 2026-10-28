// Internal stages of the engine (ports of the rubik_detector Python modules). Working-image pixels throughout.
#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "cube.hpp"
#include "imgproc.hpp"
#include "rubik/engine.h"

namespace rubik {

using M3 = std::array<double, 9>;   // row-major

inline V3 mul (const M3& m, V3 v) { return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z, m[6] * v.x + m[7] * v.y + m[8] * v.z}; }
inline V3 mulT (const M3& m, V3 v) { return {m[0] * v.x + m[3] * v.y + m[6] * v.z, m[1] * v.x + m[4] * v.y + m[7] * v.z, m[2] * v.x + m[5] * v.y + m[8] * v.z}; }
inline V3 column (const M3& m, int c) { return {m[c], m[3 + c], m[6 + c]}; }
M3 matmul (const M3& a, const M3& b);
M3 rodrigues (V3 r);    // cv2.Rodrigues: rotation vector → matrix
double det (const M3& m);

struct Intrinsics {
	double fx, fy, cx, cy;
};

// --- imaging (imaging.cpp) ----------------------------------------------------------------------------------------

struct LabImage {
	int width = 0, height = 0;
	std::vector<float> data;   // interleaved L a b
	const float* at (int x, int y) const { return data.data() + (size_t(y) * width + x) * 3; }
};

// B, G, R planes of the input at the working size (INTER_AREA as cv2.resize)
void workingPlanes (const Image& image, int workWidth, Gray8 planes[3]);
LabImage toLab (const Gray8 planes[3]);
// only inside the boxes (x0, y0, x1, y1; exclusive), for sampling at a known pose
LabImage toLabWithin (const Gray8 planes[3], const std::vector<std::array<int, 4>>& boxes);
Gray8 toGray (const Gray8 planes[3]);
// median Lab inside a quad (4 corners), as colors.sample; returns false if it covers no pixel inside the image
bool sampleQuad (const LabImage& lab, const std::array<double, 8>& quad, std::array<float, 3>& out, int& pixels);

// the colour feature of colors.weigh
std::array<double, 4> weigh (const float lab[3]);
constexpr double HUE_SCALE = 60.0, C_SAT = 25.0, LOGC_SCALE = 18.0, L_WEIGHT = 0.3;

// --- geometry (geometry.cpp) --------------------------------------------------------------------------------------

struct Segments {
	std::vector<std::array<double, 4>> s;   // x1 y1 x2 y2
	std::vector<double> w;                  // length × contrast
	std::vector<V3> normals;                // interpretation planes
	std::vector<int> family;
};

Segments detectSegments (const Gray8& gray, const LabImage& lab, const Gray8 planes[3]);
void planeNormals (const Intrinsics& K, Segments& segs);
// three orthogonal directions (rows of D) and families, RANSAC (as geometry.axes)
bool cubeAxes (Segments& segs, uint64_t seed, M3& D);
// the families of the segments for given axes (rows of D), and the alternating refinement of the axes from them
void refineAxes (Segments& segs, M3& D);
// axes signed towards the camera (seen at image point `towards`), as LineGeometry: returns R (axes as columns)
M3 signAxes (const M3& D, const Intrinsics& K, double u, double v, Segments& segs);

struct LatticeInput {
	std::vector<V3> rays1, rays2;   // family segments only
	std::vector<int> fam;
	std::vector<double> w;
};

LatticeInput latticeInput (const Intrinsics& K, const Segments& segs);
double latticeCost (const M3& R, V3 t, const LatticeInput& in, double tol = 0.35, double outside = 1.0);

struct PoseFit {
	M3 R;
	V3 t;
	double cost;
};

struct FramingPrior {
	double minExtent = 0.2, maxExtent = 0.9, centreShare = 0.8;
};

// starts / joint: distinct starts refined, of them jointly refined with a rotation (8, 3; fewer for a quick ranking)
PoseFit positionWide (const Intrinsics& K, const M3& R, const Segments& segs, int width, int height, const FramingPrior& prior,
	int starts = 8, int joint = 3);
// a tracked frame: joint refinement from a prior pose (and the ±1 cubie shift check)
PoseFit positionLocal (const Intrinsics& K, const M3& R, V3 t, const Segments& segs);

// sampled-part quads of the R, U, F facelets (view frame) at a pose
std::vector<std::pair<int, std::array<double, 8>>> faceletQuads (const Intrinsics& K, const M3& R, V3 t, double sampleSize);

// --- colour classification (classify.cpp) -------------------------------------------------------------------------

struct Obs {
	int facelet;          // common frame
	int view, face;       // the view and its own-frame face (lighting group)
	std::array<double, 4> x;
	double weight;
};

std::array<int, 54> classifyBalanced (const std::vector<Obs>& obs, int iterations = 6);

struct Likelihood {
	std::array<std::array<double, 6>, 54> cost;
	ColourModel model;
};

Likelihood likelihood (const std::vector<Obs>& obs, const std::array<int, 54>& labels);
std::array<std::array<double, 6>, 54> posterior (const std::array<std::array<double, 6>, 54>& cost);

// --- legal states (legal.cpp) -------------------------------------------------------------------------------------

struct LegalResult {
	std::array<int, 54> labels;
	double total;
	bool exact;
};

LegalResult bestLegal (const std::array<std::array<double, 6>, 54>& cost, int limit = 20000);
bool isLegal (const std::array<int, 54>& labels);
// assignment minimizing Σ C[r][perm[r]] (square), as scipy.optimize.linear_sum_assignment
std::vector<int> assignment (const std::vector<std::vector<double>>& C);

}
