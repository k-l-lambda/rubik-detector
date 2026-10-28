// Rubik's cube detection engine: C++17, no dependencies (no OpenCV).
//
// Port of rubik_detector (Python, the reference) split into stages an application can drive separately:
//
//	Detector::detect      one photo / video frame → cube pose and the 27 visible facelet colours (view frame)
//	Detector::sample      colours at a known pose (a tracked frame, a pose from elsewhere)
//	Evidence              observations of any number of views, aligned into one cube frame (colour matching,
//	                      or a rotation the caller knows from tracking)
//	estimateState         the most probable legal state from the evidence, with per-facelet confidence and the
//	                      fitted colour model
//	scoreStates           how well candidate states (e.g. the current state after each possible turn) explain a
//	                      new view, under that colour model
//	Turn / applyTurn      layer turns on a 54-facelet state
//
// Conventions (shared with the Python package and klstudio's sample generator): cube frame x → R, y → U, z → F,
// unit = cubie pitch, the cube spans ±1.5. Faces in order L R D U B F; facelet k = face·9 + row·3 + col, rows and
// columns as seen from outside the face. A state is 54 letters naming the face whose centre carries that colour.
// Camera: pinhole, cube → camera pose x_cam = R·x_cube + t, image u = fx·x/z + cx. A view's own frame names the
// three faces it sees R, U, F (facelets 9–17, 27–35, 45–53); which physical faces those are is up to fusion.
//
// Thread safety: Detector is not (it keeps scratch buffers); use one per thread. Everything else is value types and
// free functions.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rubik {

// --- input -----------------------------------------------------------------------------------------------------

enum class PixelFormat { BGR, RGB, BGRA, RGBA };

struct Image {
	const uint8_t* data = nullptr;
	int width = 0;
	int height = 0;
	int stride = 0;          // bytes per row; 0 = width · channels
	PixelFormat format = PixelFormat::BGR;
};

struct Camera {
	double fx = 0, fy = 0, cx = 0, cy = 0;   // pixels of the Image passed with it
	// detect(): fx <= 0 = unknown focal length, estimated (DetectOptions::focal*); cx, cy <= 0 = the image centre;
	// fy <= 0 = fx (square pixels)
};

struct Pose {
	std::array<double, 9> R{};   // row-major, cube axes as columns (camera frame)
	std::array<double, 3> t{};   // cube centre, camera frame, cubie units
};

// --- per view ----------------------------------------------------------------------------------------------------

struct FaceletSample {
	int facelet = -1;                 // view frame: one of the R, U, F facelets
	std::array<float, 3> lab{};       // median CIE L*a*b* of the sampled patch (OpenCV float scale: L 0–100)
	std::array<float, 8> quad{};      // sampled patch corners, input image pixels (x0 y0 … x3 y3)
	int pixels = 0;                   // pixels in the patch (0: sampled at the centre only)
	bool valid = false;               // false: the patch is outside the image
};

struct Segment {
	float x1, y1, x2, y2;   // input image pixels
	float weight;           // length × colour contrast
	int family;             // cube axis 0–2 (view frame), -1 = none
};

struct Timings {
	double segments = 0, axes = 0, position = 0, sampling = 0;   // ms
};

enum class Search {
	Wide,    // no prior: RANSAC for the axes, scan of the whole framing prior for the position
	Local,   // around DetectOptions::prior (a tracked previous frame): axes and position refined from it
};

struct DetectOptions {
	Search search = Search::Wide;
	Pose prior;                       // Search::Local: start pose (input camera)
	int workWidth = 1024;             // the image is resized (INTER_AREA) to this width first; 0 = as given
	double sampleSize = 0.46;         // sampled share of a sticker's side
	bool keepSegments = false;        // fill ViewResult::segments (debug overlays)
	uint64_t seed = 0;                // RANSAC
	// framing prior of the wide search: cube outline share of the long image side, centre share of each side
	double minExtent = 0.2, maxExtent = 0.9, centreShare = 0.8;
	// unknown focal length (Camera::fx <= 0): the one whose lattice fit explains the largest share of all segment
	// weight (not the lowest lattice cost, which ignores the segments left out of the axes); fx / long side in [focalMin, focalMax] (≈ 90°–20° of
	// horizontal view on the long side), focalSteps log-spaced, then focalRefine golden-section steps around the
	// best. Each step is a full axes + position solve (~0.4 s); estimate once per camera and pass it from then on.
	double focalMin = 0.45, focalMax = 2.6;
	int focalSteps = 8, focalRefine = 4;
};

struct ViewResult {
	bool ok = false;
	std::string error;
	Pose pose;                        // input camera
	double latticeCost = 1;           // mean segment cost at the pose (0 perfect … 1 no support); lower = better fit
	int segmentCount = 0;             // all detected / in an axis family
	int familySegments = 0;
	Camera camera;                    // the camera used (input pixels): as given, or with the estimated focal length
	bool focalEstimated = false;
	int focalTried = 0;               // solves run by the focal search
	std::vector<FaceletSample> samples;   // the 27 visible facelets (view frame)
	std::vector<Segment> segments;        // if keepSegments
	Timings timings;
};

class Detector {
public:
	Detector ();
	~Detector ();
	Detector (Detector&&) noexcept;
	Detector& operator= (Detector&&) noexcept;

	ViewResult detect (const Image& image, const Camera& camera, const DetectOptions& options = {});
	// colours of the visible facelets at a known pose (input camera), no geometry search; workWidth and sampleSize as
	// in DetectOptions (keep them equal to the detection's for comparable colours)
	std::vector<FaceletSample> sample (const Image& image, const Camera& camera, const Pose& pose, int workWidth = 1024, double sampleSize = 0.46);

private:
	struct Scratch;
	std::unique_ptr<Scratch> scratch;
};

// Whole-cube rotations relate view frames: a view placed by rotation g puts its facelet k at facelet
// rotationPerm(g)[k] of the common frame.
//
// relativeRotation(from, to): the rotation placing frame `to` into frame `from`, for two poses of the same cube at
// (nearly) the same moment — consecutive tracked frames, whose view frames are renamed whenever another trio of
// faces turns towards the camera. Chain them with composeRotations along a track:
//	g_total = composeRotations(g_total, relativeRotation(previous, current))
int relativeRotation (const Pose& from, const Pose& to);
int composeRotations (int a, int b);   // b first, then a
int inverseRotation (int g);

// --- evidence and state ------------------------------------------------------------------------------------------

struct Observation {
	int view;                    // Evidence view id
	int viewFace;                // face index (L R D U B F) in that view's own frame (its lighting group)
	std::array<float, 3> lab;
};

class Evidence {
public:
	// Adds a view's samples (view frame). rotation: the rotation placing this view's frame into the common one if
	// the caller knows it (relativeRotation chained over tracked frames), or -1 to find it by colour (fuse).
	// Returns the view id.
	int addView (const std::vector<FaceletSample>& samples, int rotation = -1);
	void removeView (int id);
	void clear ();

	// Rotations of the views added without one, by colour. Views with a given rotation are the common frame; with
	// none, the first view is (identity). Up to two views to place: every combination of rotations is tried (the
	// Python reference's three-photo search); more: each is aligned in turn to the frame built so far. Faces
	// are compared by pattern (each view face's mean colour taken out), at least 9 overlapping facelets.
	// requireAllFaces: only combinations covering all six faces (the three-photo protocol). Returns false if
	// nothing qualifies; cost: mean feature distance of the overlapping facelets (lower = better).
	bool fuse (double* cost = nullptr, bool requireAllFaces = true);

	int views () const;
	int rotation (int view) const;          // -1 until placed
	bool complete () const;                 // all 54 facelets observed by placed views
	// common-frame facelet → observations of the placed views
	std::vector<std::vector<Observation>> observations () const;

private:
	struct View {
		int id;
		std::vector<FaceletSample> samples;
		int rotation;
		bool fixed;
	};
	std::vector<View> list;
	int nextId = 0;
};

// Colour model fitted to a labelled state: per colour a mean and diagonal variance of the colour feature, per
// (view, view face) a lighting offset. Features are [hue·w(C), log C*, L*] (see colors.py).
struct ColourModel {
	std::array<std::array<double, 4>, 6> mean{};
	std::array<std::array<double, 4>, 6> var{};
	std::vector<std::pair<std::pair<int, int>, std::array<double, 4>>> offsets;   // ((view, view face), offset)
};

struct EstimateOptions {
	bool legal = true;           // most probable legal state (else the balanced 9-per-colour clustering)
	int iterations = 5;
	int swapRounds = 3;
};

struct StateEstimate {
	bool ok = false;
	std::string error;
	std::string state;                       // 54 letters
	bool legal = false;
	bool exact = true;                       // the legal search finished within its enumeration limit
	double cost = 0;                         // self-consistent negative log-likelihood
	std::array<double, 54> posterior{};      // independent posterior of each chosen colour: a per-facelet confidence
	double minPosterior = 0;
	std::vector<int> corrected;              // facelets legality moved off their own most likely colour
	std::array<std::array<double, 6>, 54> nll{};   // per facelet and colour, under the model
	ColourModel model;
};

StateEstimate estimateState (const Evidence& evidence, const EstimateOptions& options = {});

// Negative log-likelihood of a view's samples (view frame, placed by `rotation` into the frame of the states) under
// each candidate state and the model: a fresh lighting offset per face of the view is fitted per candidate.
// Lower = better; compare candidates by differences.
std::vector<double> scoreStates (const ColourModel& model, const std::vector<FaceletSample>& samples, int rotation,
	const std::vector<std::string>& states);

// --- cube --------------------------------------------------------------------------------------------------------

// The 24 whole-cube rotations (integer 3×3, row-major) and their facelet permutations.
const std::array<std::array<int, 9>, 24>& rotations ();
const std::array<int, 54>& rotationPerm (int g);
int identityRotation ();
// pred up to a whole-cube rotation: wrong facelets of the best one (and the rotation, if asked)
int compareStates (const std::string& pred, const std::string& truth, int* rotation = nullptr);

struct Turn {
	int axis = 0;         // 0 x (R), 1 y (U), 2 z (F)
	int layers = 4;       // bit 0: layer -1, bit 1: layer 0, bit 2: layer +1
	int quarters = 1;     // quarter turns, clockwise seen from the +axis side
};

// Standard notation: U D L R F B (optionally w for two layers), M E S, x y z, with ' or 2. Throws on bad input.
Turn parseTurn (const std::string& move);
// The 18 face turns U U' U2 … B2, and these as notation.
const std::vector<std::string>& faceTurnNames ();
std::string applyTurn (const std::string& state, const Turn& turn);
std::string applyMoves (const std::string& state, const std::string& moves);   // space-separated

}
