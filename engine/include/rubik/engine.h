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
	// cx, cy <= 0 = the image centre; fy <= 0 = fx (square pixels). detect() needs fx: for an unknown camera,
	// Detector::estimateFocal first, once per camera (and zoom), then reuse it
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
	int workWidth = 1024;             // the image is resized (INTER_AREA) to this width first (never up); 0 = as given
	double sampleSize = 0.46;         // sampled share of a sticker's side
	bool keepSegments = false;        // fill ViewResult::segments (debug overlays)
	uint64_t seed = 0;                // RANSAC
	// framing prior of the wide search: cube outline share of the long image side, centre share of each side
	double minExtent = 0.2, maxExtent = 0.9, centreShare = 0.8;
	// search effort: segments ranking the coarse cells (the heaviest; 0 = all), shift check also over pairs of axes,
	// wide-search starts / joint refinements (0 = 8 / 3). 16 / 5: on a real hand-held video 72 % of frames fit
	// (cost ≤ 0.15) vs 69 % with 8 / 3, for +4 % time there but +15 % on the synthetic photos (position 390 → 447 ms)
	int coarseSegments = 300;
	bool shiftPairs = false;
	int starts = 0, joint = 0;
	// Search::Local: detect only in the crop around the prior cube's projection, grown by this share of its size on
	// each side (0: the whole frame; a crop over 80 % of the frame is not made). On a real video 0.3 took a local
	// detection from 219 to 134 ms (the crop 58 % of the pixels), same cost and pose.
	double roiMargin = 0.3;
	// Search::Wide: the RANSAC's best distinct cube-axes triples tried (1: only the best), each fitted, the one with the
	// highest presence kept; stops at the first with presence ≥ axisGoodPresence (the usual case: one fit). A desk of
	// rectangles (papers, keyboards, planks, shelves) can win the RANSAC: on the realism renders 8 located 7 of 13 views
	// the best triple had put on the desk (4: 4, 16: 8)
	int axisHypotheses = 8;
	double axisGoodPresence = 0.6;
};

struct ViewResult {
	bool ok = false;
	std::string error;
	Pose pose;                        // input camera
	double latticeCost = 1;           // mean segment cost at the pose (0 perfect … 1 no support); lower = better fit
	int segmentCount = 0;             // all detected / in an axis family
	int familySegments = 0;
	Camera camera;                    // the camera used (input pixels, principal point / fy completed)
	double support = 0;               // share of all segment weight the lattice explains (0–1; see FocalEstimate)
	std::vector<FaceletSample> samples;   // the 27 visible facelets (view frame)
	std::vector<Segment> segments;        // if keepSegments
	Timings timings;
	// Is there a cube at all? faceCoverage: per cube axis, the share of the 8 lattice lines of its face (turned to the
	// camera) with a fitting segment on them. presence = mean coverage of the best two faces − latticeCost, 0 when the
	// projected cube is implausibly small or large (outside [0.75 · minExtent, maxExtent] of the long side). A flat grid
	// (a notebook, tiles, a keyboard) covers one face at most; a cube two or three. On 5 generated scenes without a cube
	// (30 views with flips, crops, rotations) presence ≤ 0.45 at any fx within ±25 %; on the synthetic photos ≥ 0.8; on
	// a real hand-held video 63–67 % of frames ≥ 0.5 (the rest: blur, the hand over the cube, fits off by a cubie).
	// Suggested threshold 0.5 (PRESENCE_THRESHOLD).
	std::array<double, 3> faceCoverage{};
	double presence = 0;
	std::array<int, 4> roi{0, 0, 0, 0};   // the crop detected in (x, y, width, height; input pixels), 0 = the whole frame
};

constexpr double PRESENCE_THRESHOLD = 0.5;

// Focal length of an unknown camera, a step of its own before detection (and stable after it): the focal length
// whose lattice fit explains the largest share of all segment weight, averaged over the images given (several views
// from one camera steady it). Not the lowest lattice cost: that averages over the segments assigned to the axes,
// and a wrong focal length assigns few that fit well. Coarse: coarseSteps log-spaced fx / long side over
// [focalMin, focalMax] (≈ 96°–21° horizontal view on the long side) at coarseWidth, which narrows the range; fine:
// refineSteps golden-section steps between the coarse best's neighbours at workWidth. Each step is an axes +
// position solve per image.
struct FocalOptions {
	double focalMin = 0.45, focalMax = 2.6;
	int coarseSteps = 8;
	int coarseWidth = 512;
	int refineSteps = 4;
	int workWidth = 1024;
	uint64_t seed = 0;
	double minExtent = 0.2, maxExtent = 0.9, centreShare = 0.8;
};

struct FocalEstimate {
	bool ok = false;
	std::string error;
	Camera camera;          // estimated fx = fy; cx, cy from the hint or the image centre
	double ratio = 0;       // fx / long image side
	double support = 0;     // mean share of segment weight explained at it
	std::vector<std::pair<double, double>> curve;   // coarse scan: (fx / long side, support)
	int solves = 0;
	double ms = 0;
};

class Detector {
public:
	Detector ();
	~Detector ();
	Detector (Detector&&) noexcept;
	Detector& operator= (Detector&&) noexcept;

	ViewResult detect (const Image& image, const Camera& camera, const DetectOptions& options = {});
	// images: one or more photos from the same camera (same size); hint: principal point (cx, cy) if known
	FocalEstimate estimateFocal (const std::vector<Image>& images, const Camera& hint = {}, const FocalOptions& options = {});
	// colours of the visible facelets at a known pose (input camera), no geometry search; workWidth and sampleSize as
	// in DetectOptions (keep them equal to the detection's for comparable colours)
	std::vector<FaceletSample> sample (const Image& image, const Camera& camera, const Pose& pose, int workWidth = 1024, double sampleSize = 0.46);
	// debugging: the lattice cost of a given pose (segments assigned to its axes as a local search would, no search)
	double poseCost (const Image& image, const Camera& camera, const Pose& pose, int workWidth = 1024);

private:
	ViewResult detectIn (const Image& image, const Camera& camera, const DetectOptions& options, int extentLong);
	struct Scratch;
	std::unique_ptr<Scratch> scratch;
};

// --- tracking between detections --------------------------------------------------------------------------------

struct TrackOptions {
	int levels = 4;               // pyramid levels (each half the size)
	int window = 21;              // Lucas–Kanade window, pixels (odd)
	int iterations = 30;
	float epsilon = 0.01f;        // stop when an update is below this (pixels)
	double minEigen = 1e-3;       // smallest eigenvalue of the window's gradient matrix / its size: flat patches are lost
	float forwardBackward = 1.0f; // forward then backward flow must return within this (pixels)
	double robustScale = 3.0;     // Cauchy scale of the pose refinement (pixels)
	int robustIterations = 10;
	double inlierPixels = 4.0;    // reprojection error of an inlier
	int minPoints = 6;            // fewer inliers: the track is lost
	int topUpBelow = 24, topUpTo = 40;   // re-seed visible vertices from the tracked pose when the inliers fall below
	// turning-layer signal: per outer layer, its vertices' outliers (followed, not fitting the rigid pose) and lost
	// points (× lostWeight) minus the others' (the cut between the layer and the middle excluded) minus half its
	// inliers; a layer is turning when its score reaches turnMinScore with ≥ 3 outliers and beats the next by turnMargin
	double turnMinScore = 4, turnMargin = 2, lostWeight = 0.5;
};

struct TrackResult {
	bool ok = false;
	std::string error;
	Pose pose;                    // input camera, renamed so the faces turned to the camera are R U F (facingPose)
	int renaming = 0;             // the rotation g it was renamed by since seed(): pose.R = seedFrame.R · rotations()[g]
	int tracked = 0;              // points followed by the flow (forward–backward consistent)
	int inliers = 0;              // of them consistent with the pose
	double reprojection = 0;      // median inlier reprojection error, pixels
	std::vector<std::array<float, 2>> points;   // the points carried to the next frame (inliers, re-seeded ones), pixels
	double msPyramid = 0, msFlow = 0, msPose = 0;
	// a layer turning (the rigid pose holds on the rest): the outward normal of its face in the camera frame, its score
	// and its outlier / other-outlier counts (TrackOptions::turnMinScore). Evidence, not a decision: a hand over one
	// layer can look the same.
	bool turning = false;
	std::array<double, 3> turnNormal{0, 0, 0};
	double turnScore = 0, turnIn = 0, turnOut = 0;
};

// Follows the cube from frame to frame between detections: seed() with a frame and its detected pose, then track()
// every following frame (same size and camera). The visible lattice vertices (gap crossings of the faces turned to the
// camera) are followed by pyramidal Lucas–Kanade with a forward–backward check, the pose refined from them by robust
// Gauss–Newton from the previous one; inliers below topUpBelow are topped up with visible vertices at the new pose.
// A few ms per frame where a detection takes 150–600: an app tracks every frame and detects in the background, re-
// seeding from each detection. The pose is rigid: a turning layer's points are outliers (a turn shows as fewer
// inliers). Not thread-safe; one per thread, independent of a Detector.
class FrameTracker {
public:
	explicit FrameTracker (const TrackOptions& options = {});
	~FrameTracker ();
	FrameTracker (FrameTracker&&) noexcept;
	FrameTracker& operator= (FrameTracker&&) noexcept;

	void seed (const Image& image, const Camera& camera, const Pose& pose);
	// a new pose for the frame tracked last (a detection moved on to it): fresh points, no new pyramid; false if
	// there is no such frame yet
	bool reseed (const Pose& pose);
	TrackResult track (const Image& image);
	void reset ();
	bool tracking () const;

	TrackOptions options;

private:
	struct State;
	std::unique_ptr<State> state;
};

// The pose renamed so that the faces turned to the camera are its R, U, F (the view frame sample() and detection
// assume): a cube turned past a face edge keeps its frame otherwise. g: the rotation applied (pose.R · rotations()[g]);
// of the renamings that do it, the one nearest `near` (a rotation index into the original frame, -1: any).
Pose facingPose (const Pose& pose, int* g = nullptr, int near = -1);

// The sampled-part quads (sampleSize of the side, as Detector::sample) of the R, U, F facelets of a pose, projected:
// per facelet (view frame) four corners in the camera's pixels. For drawing a tracked pose.
std::vector<std::pair<int, std::array<double, 8>>> projectFacelets (const Camera& camera, const Pose& pose, double sampleSize = 0.46);

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
	double weight;               // the view's weight
	int group;                   // the view's reliability group (EstimateOptions::reliability)
};

class Evidence {
public:
	// Adds a view's samples (view frame). rotation: the rotation placing this view's frame into the common one if
	// the caller knows it (relativeRotation chained over tracked frames), or -1 to find it by colour (fuse).
	// weight: the view's share in the colour likelihood (e.g. 1/n for each of n frames of one pose, so that a pose
	// held for many frames counts once and a single bad frame among them only by its share). group: views whose
	// evidence stands or falls together (EstimateOptions::reliability; -1: the view on its own). Returns the view id.
	int addView (const std::vector<FaceletSample>& samples, int rotation = -1, double weight = 1.0, int group = -1);
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
		double weight;
		int group;
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
	// partial evidence: facelets not observed are allowed. The colours are clustered over the observed ones (a
	// colour whose centre is unseen seeded by the observation farthest from the seen ones); with legal, the hidden
	// facelets are completed by the legal search (five faces leave the sixth's corners and edges to the pieces'
	// colours and the parities: usually one completion), their probability from how much forbidding the completed
	// colour costs.
	bool partial = false;
	// per-observation outlier probability in the colour likelihood (0: none, the photo protocol). Accumulating many
	// video frames, a few may be mis-located or mis-placed: with ε > 0 such a frame cannot veto a colour.
	double outlier = 0;
	// unknown reliability: the evidence of each group of views (by default each view) about a facelet is right with
	// an unknown probability r and otherwise says nothing (any colour alike); r uniform on [0, 1] and integrated out,
	// per facelet: P(c) ∝ ∫ Π_groups (r·L_g(c) + (1 − r)/6) dr, L_g the group's own posterior (uniform prior × Π p(x|c)
	// over its observations). One decisive group gives 7/12, two agreeing ~0.90, three ~0.98, four ~0.997 — the
	// rule of succession for six colours — however sharp each one's colour. The legal state and its confidence
	// are then taken under these posteriors.
	bool reliability = false;
};

struct StateEstimate {
	bool ok = false;
	std::string error;
	std::string state;                       // 54 letters
	bool legal = false;
	bool exact = true;                       // the legal search finished within its enumeration limit
	double cost = 0;                         // self-consistent negative log-likelihood
	std::array<double, 54> posterior{};      // cluster posterior of each chosen colour, colour evidence only (0: not observed)
	// with legal: per facelet the posterior of the chosen colour under the legal constraint, uniform prior over legal
	// states (the colour evidence of the other facelets counts too); without: = posterior
	std::array<double, 54> confidence{};
	double minConfidence = 0;
	double minPosterior = 0;                 // over the observed facelets
	std::vector<int> corrected;              // facelets legality moved off their own most likely colour
	std::array<std::array<double, 6>, 54> nll{};   // per facelet and colour, under the model (0 where not observed)
	// per facelet the posterior of each colour, uniform prior × Π over its observations p(x | colour) (rows sum to
	// 1); a hidden facelet (partial): from the legal completion
	std::array<std::array<double, 6>, 54> probability{};
	std::array<bool, 54> observed{};
	int coverage = 0;                        // observed facelets
	ColourModel model;
};

StateEstimate estimateState (const Evidence& evidence, const EstimateOptions& options = {});

// Negative log-likelihood of a view's samples (view frame, placed by `rotation` into the frame of the states) under
// each candidate state and the model: a fresh lighting offset per face of the view is fitted per candidate.
// Lower = better; compare candidates by differences.
std::vector<double> scoreStates (const ColourModel& model, const std::vector<FaceletSample>& samples, int rotation,
	const std::vector<std::string>& states);

// The view's samples that contradict a state: their colour under the state costs more than `threshold` nll above
// their most likely colour (lighting offsets fitted to the state, as scoreStates). Returns their view facelets.
std::vector<int> conflicts (const ColourModel& model, const std::vector<FaceletSample>& samples, int rotation,
	const std::string& state, double threshold = 4.0);

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
