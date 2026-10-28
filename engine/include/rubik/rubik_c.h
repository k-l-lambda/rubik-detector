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
	double focal_min, focal_max;     // unknown focal length (K[0] <= 0): fx / long side search range
	int32_t focal_steps, focal_refine;
} rk_options;

typedef struct {
	int32_t ok;
	char error[96];
	double pose[12];         // R then t
	double lattice_cost;
	int32_t segment_count, family_segments;
	double camera[4];        // fx fy cx cy used (estimated if K[0] <= 0)
	int32_t focal_estimated, focal_tried;
	double ms_segments, ms_axes, ms_position, ms_sampling;
	int32_t sample_count;
	rk_sample samples[27];
} rk_view;

RK_API void rk_default_options (rk_options* options);
RK_API rk_detector* rk_detector_new (void);
RK_API void rk_detector_free (rk_detector* detector);
// format: 0 BGR, 1 RGB, 2 BGRA, 3 RGBA; K: fx fy cx cy in pixels of this image (fx <= 0: estimated; cx, cy <= 0: centre)
RK_API void rk_detect (rk_detector* detector, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const rk_options* options, rk_view* out);

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

// Fuse and estimate as rk_estimate, then score candidate states (n_states × 54 letters, common frame) against the
// samples of view `view` under the fitted colour model: scores[i] = negative log-likelihood (lower = better).
RK_API void rk_score_states (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	int32_t require_all_faces, int32_t view, int32_t n_states, const char* states, double* scores, rk_state* out);

// 54-letter state after space-separated moves; returns 0 on bad input
RK_API int32_t rk_apply_moves (const char* state, const char* moves, char out[55]);
RK_API int32_t rk_compare_states (const char* pred, const char* truth);

#ifdef __cplusplus
}
#endif
