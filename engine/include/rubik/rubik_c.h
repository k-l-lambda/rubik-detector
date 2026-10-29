// Flat C ABI over the C++ engine (ctypes, JNI or Swift without C++ interop). Arrays are caller-allocated.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define RK_API __declspec(dllexport)
#else
#define RK_API __attribute__((visibility("default")))
#endif

typedef struct rk_detector rk_detector;

// one facelet sample: facelet, valid, pixels, L a b, quad (8)
typedef struct {
	int32_t facelet;
	int32_t valid;
	int32_t pixels;
	float lab[3];
	float quad[8];
} rk_sample;

typedef struct {
	int32_t search;          // 0 wide, 1 local (prior)
	double prior[12];        // R (9, row-major) then t (3)
	int32_t work_width;      // 0 = as given
	double sample_size;
	uint64_t seed;
	double min_extent, max_extent, centre_share;
	int32_t coarse_segments, shift_pairs, starts, joint;   // DetectOptions search effort
	double roi_margin;       // DetectOptions::roiMargin
	int32_t axis_hypotheses; // DetectOptions::axisHypotheses
	double axis_good_presence;
} rk_options;

typedef struct {
	int32_t ok;
	char error[96];
	double pose[12];         // R then t
	double lattice_cost;
	int32_t segment_count, family_segments;
	double camera[4];        // fx fy cx cy used (principal point / fy completed)
	double support;          // share of segment weight the lattice explains
	double ms_segments, ms_axes, ms_position, ms_sampling;
	int32_t sample_count;
	rk_sample samples[27];
	double presence;         // ViewResult::presence (threshold RK_PRESENCE_THRESHOLD)
	double face_coverage[3];
	int32_t roi[4];          // ViewResult::roi
} rk_view;

#define RK_PRESENCE_THRESHOLD 0.5

RK_API void rk_default_options (rk_options* options);
RK_API rk_detector* rk_detector_new (void);
RK_API void rk_detector_free (rk_detector* detector);
// format: 0 BGR, 1 RGB, 2 BGRA, 3 RGBA; K: fx fy cx cy in pixels of this image (cx, cy <= 0: centre; fx must be known — rk_estimate_focal)
RK_API void rk_detect (rk_detector* detector, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const rk_options* options, rk_view* out);

// rk_detect, also writing up to `capacity` segments (x1 y1 x2 y2 weight family, 6 floats each) to `segments`;
// returns the number written (debug overlays)
RK_API int32_t rk_detect_segments (rk_detector* detector, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const rk_options* options, rk_view* out, float* segments, int32_t capacity);

// Detector::poseCost (debugging)
RK_API double rk_pose_cost (rk_detector* detector, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const double pose[12], int32_t work_width);

// colours at a known pose (R then t, input camera); returns the number of samples written to out (27)
RK_API int32_t rk_sample_at (rk_detector* detector, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const double pose[12], int32_t work_width, double sample_size, rk_sample out[27]);

typedef struct {
	int32_t ok;
	char error[96];
	char state[55];
	int32_t legal, exact;
	double cost, min_posterior;
	double posterior[54];
	int32_t corrected[54];
	int32_t corrected_count;
	double fusion_cost;
	int32_t rotations[16];   // per view, in the order given
} rk_state;

// Fuse up to 16 views' samples (rotations: -1 = find by colour) and estimate the state. legal: 1 legal MAP, 0 balanced.
RK_API void rk_estimate (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	int32_t require_all_faces, int32_t legal, rk_state* out);

typedef struct {
	int32_t ok;
	char error[96];
	char state[55];
	int32_t legal, exact;
	double cost, min_posterior;
	double posterior[54];
	double probability[54][6];   // per facelet cluster posterior, colours in face order L R D U B F
	double confidence[54];       // per facelet the chosen colour's posterior under the legal constraint
	double min_confidence;
	int32_t observed[54];
	int32_t coverage;
	int32_t corrected[54];
	int32_t corrected_count;
	double fusion_cost;
	double model_mean[6][4];     // the fitted colour model (ColourModel), for rk_score_model
	double model_var[6][4];
} rk_state_ex;

typedef struct {
	int32_t require_all_faces;   // fusion: only rotations covering all six faces
	int32_t legal;               // 1 legal MAP (with partial: hidden facelets completed), 0 balanced clustering
	int32_t partial;             // allow facelets not observed
	double outlier;              // per-observation outlier probability (0 = none)
	int32_t reliability;         // unknown reliability per group, integrated out (EstimateOptions::reliability)
} rk_estimate_options;

// rk_estimate for any number of views (video frames): weights and reliability groups per view (NULL = 1, each view
// its own), rotations_out per view (NULL = not wanted).
RK_API void rk_estimate_ex (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	const double* weights, const int32_t* groups, const rk_estimate_options* options, rk_state_ex* out, int32_t* rotations_out);

// Score candidate states (n_states × 54 letters) against one view's samples placed by `rotation`, under a colour
// model kept from an estimate (rk_state_ex model_mean / model_var): negative log-likelihoods, lower = better
// (scoreStates). Returns 0 on bad input.
RK_API int32_t rk_score_model (const double mean[6][4], const double var[6][4], const rk_sample* samples, int32_t count,
	int32_t rotation, int32_t n_states, const char* states, double* scores);

// conflicts(): the samples (view facelets, written to out, up to 27) contradicting `state` placed by `rotation`;
// returns their number, -1 on bad input
RK_API int32_t rk_conflicts (const double mean[6][4], const double var[6][4], const rk_sample* samples, int32_t count,
	int32_t rotation, const char* state, double threshold, int32_t out[27]);

// Fuse and estimate as rk_estimate, then score candidate states (n_states × 54 letters, common frame) against the
// samples of view `view` under the fitted colour model: scores[i] = negative log-likelihood (lower = better).
RK_API void rk_score_states (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	int32_t require_all_faces, int32_t view, int32_t n_states, const char* states, double* scores, rk_state* out);

typedef struct {
	int32_t ok;
	char error[96];
	double camera[4];        // fx fy cx cy
	double ratio, support, ms;
	int32_t solves;
	int32_t curve_count;
	double curve[32][2];     // coarse scan: fx / long side, support
} rk_focal;

// Focal length of an unknown camera from 1–16 same-size images (data[i], strides[i]); hint: cx cy (NULL = centre);
// focal_range: min, max of fx / long side (NULL = defaults); steps: coarse_width, coarse_steps, refine_steps,
// work_width (NULL or 0 = defaults).
RK_API void rk_estimate_focal (rk_detector* detector, int32_t count, const uint8_t* const* data, int32_t width, int32_t height,
	const int32_t* strides, int32_t format, const double hint[2], const double focal_range[2], const int32_t steps[4], rk_focal* out);

// --- tracking between detections (FrameTracker) ---

typedef struct rk_tracker rk_tracker;

typedef struct {
	int32_t ok;
	char error[96];
	double pose[12];         // R then t (facing: the faces turned to the camera are R U F)
	int32_t renaming;        // TrackResult::renaming
	int32_t tracked, inliers;
	double reprojection;
	double ms_pyramid, ms_flow, ms_pose;
	int32_t point_count;
	float points[64][2];
	int32_t turning;         // TrackResult turning-layer signal
	double turn_normal[3];
	double turn_score, turn_in, turn_out;
} rk_track_result;

RK_API rk_tracker* rk_tracker_new (void);
RK_API void rk_tracker_free (rk_tracker* tracker);
RK_API void rk_tracker_seed (rk_tracker* tracker, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const double pose[12]);
RK_API void rk_tracker_track (rk_tracker* tracker, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	rk_track_result* out);

// 54-letter state after space-separated moves; returns 0 on bad input
RK_API int32_t rk_apply_moves (const char* state, const char* moves, char out[55]);
RK_API int32_t rk_compare_states (const char* pred, const char* truth);

#ifdef __cplusplus
}
#endif
