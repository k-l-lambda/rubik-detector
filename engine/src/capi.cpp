#include <cstring>
#include <stdexcept>

#include "rubik/engine.h"
#include "rubik/rubik_c.h"

using namespace rubik;

struct rk_detector {
	Detector detector;
};

namespace {

void copyError (char* dst, const std::string& msg) {
	std::strncpy(dst, msg.c_str(), 95);
	dst[95] = 0;
}

}

extern "C" {

void rk_default_options (rk_options* o) {
	DetectOptions d;
	std::memset(o, 0, sizeof(*o));
	o->search = 0;
	o->work_width = d.workWidth;
	o->sample_size = d.sampleSize;
	o->seed = d.seed;
	o->min_extent = d.minExtent;
	o->max_extent = d.maxExtent;
	o->centre_share = d.centreShare;
	o->coarse_segments = d.coarseSegments;
	o->shift_pairs = d.shiftPairs;
	o->starts = d.starts;
	o->joint = d.joint;
	o->roi_margin = d.roiMargin;
}

double rk_pose_cost (rk_detector* d, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const double pose[12], int32_t workWidth) {
	try {
		Pose p;
		std::copy(pose, pose + 9, p.R.begin());
		std::copy(pose + 9, pose + 12, p.t.begin());
		return d->detector.poseCost({data, width, height, stride, PixelFormat(format)}, {K[0], K[1], K[2], K[3]}, p, workWidth);
	}
	catch (const std::exception&) {
		return 1.0;
	}
}

rk_detector* rk_detector_new (void) { return new rk_detector; }
void rk_detector_free (rk_detector* d) { delete d; }

void rk_detect (rk_detector* d, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const rk_options* o, rk_view* out) {
	std::memset(out, 0, sizeof(*out));
	Image image{data, width, height, stride, PixelFormat(format)};
	Camera camera{K[0], K[1], K[2], K[3]};
	DetectOptions options;
	options.search = o->search ? Search::Local : Search::Wide;
	std::copy(o->prior, o->prior + 9, options.prior.R.begin());
	std::copy(o->prior + 9, o->prior + 12, options.prior.t.begin());
	options.workWidth = o->work_width;
	options.sampleSize = o->sample_size;
	options.seed = o->seed;
	options.minExtent = o->min_extent;
	options.maxExtent = o->max_extent;
	options.centreShare = o->centre_share;
	options.coarseSegments = o->coarse_segments;
	options.shiftPairs = o->shift_pairs != 0;
	options.starts = o->starts;
	options.joint = o->joint;
	options.roiMargin = o->roi_margin;
	try {
		ViewResult r = d->detector.detect(image, camera, options);
		out->ok = r.ok;
		copyError(out->error, r.error);
		std::copy(r.pose.R.begin(), r.pose.R.end(), out->pose);
		std::copy(r.pose.t.begin(), r.pose.t.end(), out->pose + 9);
		out->lattice_cost = r.latticeCost;
		out->segment_count = r.segmentCount;
		out->family_segments = r.familySegments;
		out->camera[0] = r.camera.fx;
		out->camera[1] = r.camera.fy;
		out->camera[2] = r.camera.cx;
		out->camera[3] = r.camera.cy;
		out->support = r.support;
		out->ms_segments = r.timings.segments;
		out->ms_axes = r.timings.axes;
		out->ms_position = r.timings.position;
		out->ms_sampling = r.timings.sampling;
		out->presence = r.presence;
		std::copy(r.faceCoverage.begin(), r.faceCoverage.end(), out->face_coverage);
		std::copy(r.roi.begin(), r.roi.end(), out->roi);
		out->sample_count = int32_t(std::min<size_t>(r.samples.size(), 27));
		for (int i = 0; i < out->sample_count; i++) {
			const FaceletSample& s = r.samples[i];
			out->samples[i] = {s.facelet, s.valid, s.pixels, {s.lab[0], s.lab[1], s.lab[2]}, {}};
			std::copy(s.quad.begin(), s.quad.end(), out->samples[i].quad);
		}
	}
	catch (const std::exception& e) {
		out->ok = 0;
		copyError(out->error, e.what());
	}
}

int32_t rk_detect_segments (rk_detector* d, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const rk_options* o, rk_view* out, float* segments, int32_t capacity) {
	int32_t written = 0;
	std::memset(out, 0, sizeof(*out));
	Image image{data, width, height, stride, PixelFormat(format)};
	Camera camera{K[0], K[1], K[2], K[3]};
	DetectOptions options;
	options.search = o->search ? Search::Local : Search::Wide;
	std::copy(o->prior, o->prior + 9, options.prior.R.begin());
	std::copy(o->prior + 9, o->prior + 12, options.prior.t.begin());
	options.workWidth = o->work_width;
	options.sampleSize = o->sample_size;
	options.seed = o->seed;
	options.minExtent = o->min_extent;
	options.maxExtent = o->max_extent;
	options.centreShare = o->centre_share;
	options.coarseSegments = o->coarse_segments;
	options.shiftPairs = o->shift_pairs != 0;
	options.starts = o->starts;
	options.joint = o->joint;
	options.roiMargin = o->roi_margin;
	options.keepSegments = true;
	try {
		ViewResult r = d->detector.detect(image, camera, options);
		out->ok = r.ok;
		copyError(out->error, r.error);
		std::copy(r.pose.R.begin(), r.pose.R.end(), out->pose);
		std::copy(r.pose.t.begin(), r.pose.t.end(), out->pose + 9);
		out->lattice_cost = r.latticeCost;
		out->segment_count = r.segmentCount;
		out->family_segments = r.familySegments;
		out->camera[0] = r.camera.fx;
		out->camera[1] = r.camera.fy;
		out->camera[2] = r.camera.cx;
		out->camera[3] = r.camera.cy;
		out->support = r.support;
		out->ms_segments = r.timings.segments;
		out->ms_axes = r.timings.axes;
		out->ms_position = r.timings.position;
		out->ms_sampling = r.timings.sampling;
		out->presence = r.presence;
		std::copy(r.faceCoverage.begin(), r.faceCoverage.end(), out->face_coverage);
		std::copy(r.roi.begin(), r.roi.end(), out->roi);
		for (const Segment& g : r.segments) {
			if (written >= capacity)
				break;
			float* p = segments + 6 * written++;
			p[0] = g.x1; p[1] = g.y1; p[2] = g.x2; p[3] = g.y2; p[4] = g.weight; p[5] = float(g.family);
		}
		out->sample_count = int32_t(std::min<size_t>(r.samples.size(), 27));
		for (int i = 0; i < out->sample_count; i++) {
			const FaceletSample& s = r.samples[i];
			out->samples[i] = {s.facelet, s.valid, s.pixels, {s.lab[0], s.lab[1], s.lab[2]}, {}};
			std::copy(s.quad.begin(), s.quad.end(), out->samples[i].quad);
		}
	}
	catch (const std::exception& e) {
		out->ok = 0;
		copyError(out->error, e.what());
	}
	return written;
}

void rk_estimate_focal (rk_detector* d, int32_t count, const uint8_t* const* data, int32_t width, int32_t height,
	const int32_t* strides, int32_t format, const double hint[2], const double range[2], const int32_t steps[4], rk_focal* out) {
	std::memset(out, 0, sizeof(*out));
	try {
		std::vector<Image> images;
		for (int i = 0; i < count; i++)
			images.push_back({data[i], width, height, strides ? strides[i] : 0, PixelFormat(format)});
		Camera h;
		if (hint) {
			h.cx = hint[0];
			h.cy = hint[1];
		}
		FocalOptions o;
		if (range && range[0] > 0 && range[1] > range[0]) {
			o.focalMin = range[0];
			o.focalMax = range[1];
		}
		if (steps) {
			if (steps[0] > 0)
				o.coarseWidth = steps[0];
			if (steps[1] > 0)
				o.coarseSteps = std::min(steps[1], 32);
			if (steps[2] > 0)
				o.refineSteps = steps[2];
			if (steps[3] > 0)
				o.workWidth = steps[3];
		}
		FocalEstimate e = d->detector.estimateFocal(images, h, o);
		out->ok = e.ok;
		copyError(out->error, e.error);
		out->camera[0] = e.camera.fx;
		out->camera[1] = e.camera.fy;
		out->camera[2] = e.camera.cx;
		out->camera[3] = e.camera.cy;
		out->ratio = e.ratio;
		out->support = e.support;
		out->ms = e.ms;
		out->solves = e.solves;
		out->curve_count = int32_t(std::min<size_t>(e.curve.size(), 32));
		for (int i = 0; i < out->curve_count; i++) {
			out->curve[i][0] = e.curve[i].first;
			out->curve[i][1] = e.curve[i].second;
		}
	}
	catch (const std::exception& ex) {
		out->ok = 0;
		copyError(out->error, ex.what());
	}
}

int32_t rk_sample_at (rk_detector* d, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const double pose[12], int32_t workWidth, double sampleSize, rk_sample out[27]) {
	try {
		Pose p;
		std::copy(pose, pose + 9, p.R.begin());
		std::copy(pose + 9, pose + 12, p.t.begin());
		auto r = d->detector.sample({data, width, height, stride, PixelFormat(format)}, {K[0], K[1], K[2], K[3]}, p, workWidth, sampleSize);
		int n = int(std::min<size_t>(r.size(), 27));
		for (int i = 0; i < n; i++) {
			out[i] = {r[i].facelet, r[i].valid, r[i].pixels, {r[i].lab[0], r[i].lab[1], r[i].lab[2]}, {}};
			std::copy(r[i].quad.begin(), r[i].quad.end(), out[i].quad);
		}
		return n;
	}
	catch (const std::exception&) {
		return 0;
	}
}

namespace {

// the shared part of rk_estimate and rk_score_states; returns the estimate (out filled), evidence and view ids
StateEstimate estimateFlat (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	int32_t requireAllFaces, int32_t legal, rk_state* out, Evidence& evidence, std::vector<int>& ids,
	std::vector<std::vector<FaceletSample>>& lists) {
	std::memset(out, 0, sizeof(*out));
	if (views < 1 || views > 16)
		throw std::invalid_argument("1 to 16 views");
	const rk_sample* p = samples;
	for (int v = 0; v < views; v++) {
		std::vector<FaceletSample> list;
		for (int i = 0; i < counts[v]; i++, p++) {
			FaceletSample s;
			s.facelet = p->facelet;
			s.valid = p->valid != 0;
			s.pixels = p->pixels;
			s.lab = {p->lab[0], p->lab[1], p->lab[2]};
			std::copy(p->quad, p->quad + 8, s.quad.begin());
			list.push_back(s);
		}
		lists.push_back(list);
		ids.push_back(evidence.addView(list, rotations ? rotations[v] : -1));
	}
	double cost = 0;
	if (!evidence.fuse(&cost, requireAllFaces != 0))
		throw std::runtime_error("no rotation of the views covers all six faces");
	out->fusion_cost = cost;
	for (int v = 0; v < views; v++)
		out->rotations[v] = evidence.rotation(ids[v]);
	EstimateOptions options;
	options.legal = legal != 0;
	StateEstimate e = estimateState(evidence, options);
	out->ok = e.ok;
	copyError(out->error, e.error);
	if (!e.ok)
		return e;
	std::memcpy(out->state, e.state.c_str(), 54);
	out->state[54] = 0;
	out->legal = e.legal;
	out->exact = e.exact;
	out->cost = e.cost;
	out->min_posterior = e.minPosterior;
	std::copy(e.posterior.begin(), e.posterior.end(), out->posterior);
	out->corrected_count = int32_t(e.corrected.size());
	std::copy(e.corrected.begin(), e.corrected.end(), out->corrected);
	return e;
}

}

void rk_estimate (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	int32_t requireAllFaces, int32_t legal, rk_state* out) {
	try {
		Evidence evidence;
		std::vector<int> ids;
		std::vector<std::vector<FaceletSample>> lists;
		estimateFlat(views, samples, counts, rotations, requireAllFaces, legal, out, evidence, ids, lists);
	}
	catch (const std::exception& ex) {
		out->ok = 0;
		copyError(out->error, ex.what());
	}
}

void rk_score_states (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	int32_t requireAllFaces, int32_t view, int32_t nStates, const char* states, double* scores, rk_state* out) {
	try {
		Evidence evidence;
		std::vector<int> ids;
		std::vector<std::vector<FaceletSample>> lists;
		StateEstimate e = estimateFlat(views, samples, counts, rotations, requireAllFaces, 1, out, evidence, ids, lists);
		if (!e.ok)
			return;
		if (view < 0 || view >= views)
			throw std::invalid_argument("view out of range");
		std::vector<std::string> list;
		for (int i = 0; i < nStates; i++)
			list.emplace_back(states + 54 * i, 54);
		std::vector<double> r = scoreStates(e.model, lists[view], evidence.rotation(ids[view]), list);
		std::copy(r.begin(), r.end(), scores);
	}
	catch (const std::exception& ex) {
		out->ok = 0;
		copyError(out->error, ex.what());
	}
}

void rk_estimate_ex (int32_t views, const rk_sample* samples, const int32_t* counts, const int32_t* rotations,
	const double* weights, const int32_t* groups, const rk_estimate_options* options, rk_state_ex* out, int32_t* rotationsOut) {
	try {
		std::memset(out, 0, sizeof(*out));
		if (views < 1)
			throw std::invalid_argument("no views");
		Evidence evidence;
		std::vector<int> ids;
		const rk_sample* p = samples;
		for (int v = 0; v < views; v++) {
			std::vector<FaceletSample> list;
			for (int i = 0; i < counts[v]; i++, p++) {
				FaceletSample s;
				s.facelet = p->facelet;
				s.valid = p->valid != 0;
				s.pixels = p->pixels;
				s.lab = {p->lab[0], p->lab[1], p->lab[2]};
				std::copy(p->quad, p->quad + 8, s.quad.begin());
				list.push_back(s);
			}
			ids.push_back(evidence.addView(list, rotations ? rotations[v] : -1, weights ? weights[v] : 1.0, groups ? groups[v] : -1));
		}
		double cost = 0;
		if (!evidence.fuse(&cost, options->require_all_faces != 0))
			throw std::runtime_error("no rotation of the views covers all six faces");
		out->fusion_cost = cost;
		if (rotationsOut)
			for (int v = 0; v < views; v++)
				rotationsOut[v] = evidence.rotation(ids[v]);
		EstimateOptions o;
		o.legal = options->legal != 0;
		o.partial = options->partial != 0;
		o.outlier = options->outlier;
		o.reliability = options->reliability != 0;
		StateEstimate e = estimateState(evidence, o);
		out->ok = e.ok;
		copyError(out->error, e.error);
		out->coverage = e.coverage;
		for (int k = 0; k < 54; k++)
			out->observed[k] = e.observed[k];
		if (!e.ok)
			return;
		std::memcpy(out->state, e.state.c_str(), 54);
		out->state[54] = 0;
		out->legal = e.legal;
		out->exact = e.exact;
		out->cost = e.cost;
		out->min_posterior = e.minPosterior;
		std::copy(e.posterior.begin(), e.posterior.end(), out->posterior);
		for (int k = 0; k < 54; k++)
			std::copy(e.probability[k].begin(), e.probability[k].end(), out->probability[k]);
		std::copy(e.confidence.begin(), e.confidence.end(), out->confidence);
		for (int c = 0; c < 6; c++)
			for (int d = 0; d < 4; d++) {
				out->model_mean[c][d] = e.model.mean[c][d];
				out->model_var[c][d] = e.model.var[c][d];
			}
		out->min_confidence = e.minConfidence;
		out->corrected_count = int32_t(e.corrected.size());
		std::copy(e.corrected.begin(), e.corrected.end(), out->corrected);
	}
	catch (const std::exception& ex) {
		out->ok = 0;
		copyError(out->error, ex.what());
	}
}

int32_t rk_score_model (const double mean[6][4], const double var[6][4], const rk_sample* samples, int32_t count,
	int32_t rotation, int32_t nStates, const char* states, double* scores) {
	try {
		ColourModel model;
		for (int c = 0; c < 6; c++)
			for (int d = 0; d < 4; d++) {
				model.mean[c][d] = mean[c][d];
				model.var[c][d] = var[c][d];
			}
		std::vector<FaceletSample> list;
		for (int i = 0; i < count; i++) {
			FaceletSample s;
			s.facelet = samples[i].facelet;
			s.valid = samples[i].valid != 0;
			s.pixels = samples[i].pixels;
			s.lab = {samples[i].lab[0], samples[i].lab[1], samples[i].lab[2]};
			std::copy(samples[i].quad, samples[i].quad + 8, s.quad.begin());
			list.push_back(s);
		}
		std::vector<std::string> candidates;
		for (int i = 0; i < nStates; i++)
			candidates.emplace_back(states + 54 * i, 54);
		std::vector<double> r = scoreStates(model, list, rotation, candidates);
		std::copy(r.begin(), r.end(), scores);
		return 1;
	}
	catch (const std::exception&) {
		return 0;
	}
}

int32_t rk_conflicts (const double mean[6][4], const double var[6][4], const rk_sample* samples, int32_t count,
	int32_t rotation, const char* state, double threshold, int32_t out[27]) {
	try {
		ColourModel model;
		for (int c = 0; c < 6; c++)
			for (int d = 0; d < 4; d++) {
				model.mean[c][d] = mean[c][d];
				model.var[c][d] = var[c][d];
			}
		std::vector<FaceletSample> list;
		for (int i = 0; i < count; i++) {
			FaceletSample s;
			s.facelet = samples[i].facelet;
			s.valid = samples[i].valid != 0;
			s.lab = {samples[i].lab[0], samples[i].lab[1], samples[i].lab[2]};
			list.push_back(s);
		}
		std::vector<int> r = conflicts(model, list, rotation, std::string(state, 54), threshold);
		for (size_t i = 0; i < r.size() && i < 27; i++)
			out[i] = r[i];
		return int32_t(r.size());
	}
	catch (const std::exception&) {
		return -1;
	}
}

struct rk_tracker {
	FrameTracker tracker;
};

rk_tracker* rk_tracker_new (void) { return new rk_tracker; }
void rk_tracker_free (rk_tracker* t) { delete t; }

void rk_tracker_seed (rk_tracker* t, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format,
	const double K[4], const double pose[12]) {
	try {
		Pose p;
		std::copy(pose, pose + 9, p.R.begin());
		std::copy(pose + 9, pose + 12, p.t.begin());
		t->tracker.seed({data, width, height, stride, PixelFormat(format)}, {K[0], K[1], K[2], K[3]}, p);
	}
	catch (const std::exception&) {
		t->tracker.reset();
	}
}

void rk_tracker_track (rk_tracker* t, const uint8_t* data, int32_t width, int32_t height, int32_t stride, int32_t format, rk_track_result* out) {
	std::memset(out, 0, sizeof(*out));
	try {
		TrackResult r = t->tracker.track({data, width, height, stride, PixelFormat(format)});
		out->ok = r.ok;
		copyError(out->error, r.error);
		std::copy(r.pose.R.begin(), r.pose.R.end(), out->pose);
		std::copy(r.pose.t.begin(), r.pose.t.end(), out->pose + 9);
		out->renaming = r.renaming;
		out->tracked = r.tracked;
		out->inliers = r.inliers;
		out->reprojection = r.reprojection;
		out->ms_pyramid = r.msPyramid;
		out->ms_flow = r.msFlow;
		out->ms_pose = r.msPose;
		out->turning = r.turning;
		std::copy(r.turnNormal.begin(), r.turnNormal.end(), out->turn_normal);
		out->turn_score = r.turnScore;
		out->turn_in = r.turnIn;
		out->turn_out = r.turnOut;
		out->point_count = int32_t(std::min<size_t>(r.points.size(), 64));
		for (int i = 0; i < out->point_count; i++) {
			out->points[i][0] = r.points[i][0];
			out->points[i][1] = r.points[i][1];
		}
	}
	catch (const std::exception& e) {
		out->ok = 0;
		copyError(out->error, e.what());
	}
}

int32_t rk_apply_moves (const char* state, const char* moves, char out[55]) {
	try {
		std::string r = applyMoves(state, moves);
		std::memcpy(out, r.c_str(), 54);
		out[54] = 0;
		return 1;
	}
	catch (const std::exception&) {
		return 0;
	}
}

int32_t rk_compare_states (const char* pred, const char* truth) {
	try {
		return compareStates(pred, truth);
	}
	catch (const std::exception&) {
		return -1;
	}
}

}
