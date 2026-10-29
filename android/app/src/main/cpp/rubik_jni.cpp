// JNI bridge: the engine's C++ API (include/rubik/engine.h) for app/.../Engine.kt. Results go back as flat double
// arrays (layouts documented next to each function and parsed in Engine.kt), so no JNI object construction is needed.
#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rubik/engine.h"

using namespace rubik;

namespace {

struct Handle {
	Detector detector;
	std::vector<uint8_t> upright;   // the camera frame turned upright
};

Handle* handle (jlong h) {
	return reinterpret_cast<Handle*>(h);
}

// The analysis frame (RGBA, sensor orientation) turned by `rotation` degrees clockwise into `out` (tightly packed).
// Returns the upright width and height.
std::pair<int, int> rotate (const uint8_t* src, int width, int height, int stride, int rotation, std::vector<uint8_t>& out) {
	rotation = ((rotation % 360) + 360) % 360;
	const int w = rotation % 180 ? height : width, h = rotation % 180 ? width : height;
	out.resize(size_t(w) * h * 4);
	auto* dst = reinterpret_cast<uint32_t*>(out.data());
	for (int y = 0; y < height; y++) {
		const auto* row = reinterpret_cast<const uint32_t*>(src + size_t(y) * stride);
		for (int x = 0; x < width; x++) {
			int u, v;
			switch (rotation) {
				case 90: u = height - 1 - y; v = x; break;
				case 180: u = width - 1 - x; v = height - 1 - y; break;
				case 270: u = y; v = width - 1 - x; break;
				default: u = x; v = y; break;
			}
			dst[size_t(v) * w + u] = row[x];
		}
	}
	return {w, h};
}

std::vector<FaceletSample> readSamples (const double* p, int count) {
	// per sample: facelet, valid, L, a, b
	std::vector<FaceletSample> out(count);
	for (int i = 0; i < count; i++, p += 5) {
		out[i].facelet = int(p[0]);
		out[i].valid = p[1] != 0;
		out[i].lab = {float(p[2]), float(p[3]), float(p[4])};
	}
	return out;
}

ColourModel readModel (JNIEnv* env, jdoubleArray mean, jdoubleArray var) {
	ColourModel model;
	jdouble* m = env->GetDoubleArrayElements(mean, nullptr);
	jdouble* v = env->GetDoubleArrayElements(var, nullptr);
	for (int c = 0; c < 6; c++)
		for (int d = 0; d < 4; d++) {
			model.mean[c][d] = m[c * 4 + d];
			model.var[c][d] = v[c * 4 + d];
		}
	env->ReleaseDoubleArrayElements(mean, m, JNI_ABORT);
	env->ReleaseDoubleArrayElements(var, v, JNI_ABORT);
	return model;
}

jdoubleArray toArray (JNIEnv* env, const std::vector<double>& v) {
	jdoubleArray out = env->NewDoubleArray(jsize(v.size()));
	env->SetDoubleArrayRegion(out, 0, jsize(v.size()), v.data());
	return out;
}

double msSince (std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}

extern "C" {

JNIEXPORT jlong JNICALL Java_com_klstudio_rubikdetector_Engine_nativeCreate (JNIEnv*, jclass) {
	return reinterpret_cast<jlong>(new Handle());
}

JNIEXPORT void JNICALL Java_com_klstudio_rubikdetector_Engine_nativeDestroy (JNIEnv*, jclass, jlong h) {
	delete handle(h);
}

// Turns the RGBA frame upright and keeps it for the following calls on it. Returns {width, height, ms}.
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeSetFrame (JNIEnv* env, jclass, jlong h, jobject buffer,
	jint width, jint height, jint stride, jint rotation) {
	auto t0 = std::chrono::steady_clock::now();
	const auto* src = static_cast<const uint8_t*>(env->GetDirectBufferAddress(buffer));
	auto [w, ht] = rotate(src, width, height, stride, rotation, handle(h)->upright);
	return toArray(env, {double(w), double(ht), msSince(t0)});
}

// Detection on the kept frame. prior: R (9) t (3) for a local search, or null for wide.
// Out: ok, latticeCost, support, segmentCount, familySegments, R(9), t(3), timings(4: segments axes position
// sampling), totalMs, sampleCount, per sample (facelet, valid, L, a, b, quad 8) = 13, segmentCount2, per segment
// (x1, y1, x2, y2, weight, family) = 6, presence, faceCoverage (3).
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeDetect (JNIEnv* env, jclass, jlong h, jint width, jint height,
	jdouble fx, jdouble fy, jdouble cx, jdouble cy, jdoubleArray prior, jint workWidth, jboolean keepSegments) {
	Handle* H = handle(h);
	auto t0 = std::chrono::steady_clock::now();
	Image image{H->upright.data(), width, height, width * 4, PixelFormat::RGBA};
	Camera camera{fx, fy, cx, cy};
	DetectOptions options;
	options.workWidth = workWidth;
	options.keepSegments = keepSegments;
	if (prior) {
		jdouble* p = env->GetDoubleArrayElements(prior, nullptr);
		options.search = Search::Local;
		std::copy(p, p + 9, options.prior.R.begin());
		std::copy(p + 9, p + 12, options.prior.t.begin());
		env->ReleaseDoubleArrayElements(prior, p, JNI_ABORT);
	}
	std::vector<double> out;
	try {
		ViewResult r = H->detector.detect(image, camera, options);
		out = {double(r.ok), r.latticeCost, r.support, double(r.segmentCount), double(r.familySegments)};
		out.insert(out.end(), r.pose.R.begin(), r.pose.R.end());
		out.insert(out.end(), r.pose.t.begin(), r.pose.t.end());
		out.insert(out.end(), {r.timings.segments, r.timings.axes, r.timings.position, r.timings.sampling, msSince(t0)});
		out.push_back(double(r.samples.size()));
		for (const FaceletSample& s : r.samples) {
			out.insert(out.end(), {double(s.facelet), double(s.valid), s.lab[0], s.lab[1], s.lab[2]});
			out.insert(out.end(), s.quad.begin(), s.quad.end());
		}
		out.push_back(double(r.segments.size()));
		for (const Segment& s : r.segments)
			out.insert(out.end(), {s.x1, s.y1, s.x2, s.y2, s.weight, double(s.family)});
		out.push_back(r.presence);
		out.insert(out.end(), r.faceCoverage.begin(), r.faceCoverage.end());
	}
	catch (const std::exception&) {
		out.assign(5 + 12 + 5 + 2 + 4, 0.0);
		out[1] = 1;
	}
	return toArray(env, out);
}

// Focal length from the kept frame. Out: ok, fx, fy, cx, cy, ratio, support, ms, solves, curveCount, (ratio, support)…
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeEstimateFocal (JNIEnv* env, jclass, jlong h, jint width, jint height) {
	Handle* H = handle(h);
	Image image{H->upright.data(), width, height, width * 4, PixelFormat::RGBA};
	std::vector<double> out;
	try {
		FocalEstimate f = H->detector.estimateFocal({image});
		out = {double(f.ok), f.camera.fx, f.camera.fy, f.camera.cx, f.camera.cy, f.ratio, f.support, f.ms, double(f.solves), double(f.curve.size())};
		for (const auto& c : f.curve)
			out.insert(out.end(), {c.first, c.second});
	}
	catch (const std::exception&) {
		out.assign(10, 0.0);
	}
	return toArray(env, out);
}

// Fusion and state estimation over views. samples: per view `counts[v]` samples of 5 (facelet, valid, L, a, b);
// rotations: per view, -1 = place by colour.
// Out: ok, legal, coverage, minPosterior, minConfidence, cost, ms, rotations (views), state (54 colour indices, -1
// none), probability (54 × 6), confidence (54), observed (54), model mean (24), model var (24).
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeEstimate (JNIEnv* env, jclass, jdoubleArray samples, jintArray counts,
	jintArray rotations, jboolean legal, jboolean partial, jboolean reliability) {
	auto t0 = std::chrono::steady_clock::now();
	const jsize views = env->GetArrayLength(counts);
	jdouble* s = env->GetDoubleArrayElements(samples, nullptr);
	jint* c = env->GetIntArrayElements(counts, nullptr);
	jint* r = env->GetIntArrayElements(rotations, nullptr);
	Evidence evidence;
	std::vector<int> ids;
	const double* p = s;
	for (int v = 0; v < views; v++) {
		ids.push_back(evidence.addView(readSamples(p, c[v]), r[v]));
		p += 5 * c[v];
	}
	env->ReleaseDoubleArrayElements(samples, s, JNI_ABORT);
	env->ReleaseIntArrayElements(counts, c, JNI_ABORT);
	env->ReleaseIntArrayElements(rotations, r, JNI_ABORT);
	std::vector<double> out(7, 0.0);
	try {
		double cost = 0;
		bool fused = evidence.fuse(&cost, false);
		for (int v = 0; v < views; v++)
			out.push_back(fused ? evidence.rotation(ids[v]) : -1);
		EstimateOptions options;
		options.legal = legal;
		options.partial = partial;
		options.reliability = reliability;
		StateEstimate e = fused ? estimateState(evidence, options) : StateEstimate{};
		out[0] = e.ok;
		out[1] = e.legal;
		out[2] = e.coverage;
		out[3] = e.minPosterior;
		out[4] = e.minConfidence;
		out[5] = e.cost;
		for (int k = 0; k < 54; k++)
			out.push_back(e.ok ? double(std::strchr("LRDUBF", e.state[k]) - "LRDUBF") : -1);
		for (int k = 0; k < 54; k++)
			out.insert(out.end(), e.probability[k].begin(), e.probability[k].end());
		out.insert(out.end(), e.confidence.begin(), e.confidence.end());
		for (int k = 0; k < 54; k++)
			out.push_back(e.observed[k]);
		for (int k = 0; k < 6; k++)
			out.insert(out.end(), e.model.mean[k].begin(), e.model.mean[k].end());
		for (int k = 0; k < 6; k++)
			out.insert(out.end(), e.model.var[k].begin(), e.model.var[k].end());
	}
	catch (const std::exception&) {
		out.resize(7 + views + 54 + 54 * 6 + 54 + 54 + 48, 0.0);
	}
	out[6] = msSince(t0);
	return toArray(env, out);
}

// Candidate states (54 letters each, concatenated) scored for one view's samples placed by `rotation` under a kept
// colour model: negative log-likelihoods.
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeScoreModel (JNIEnv* env, jclass, jdoubleArray mean, jdoubleArray var,
	jdoubleArray samples, jint rotation, jstring states) {
	ColourModel model = readModel(env, mean, var);
	jdouble* s = env->GetDoubleArrayElements(samples, nullptr);
	std::vector<FaceletSample> view = readSamples(s, env->GetArrayLength(samples) / 5);
	env->ReleaseDoubleArrayElements(samples, s, JNI_ABORT);
	const char* chars = env->GetStringUTFChars(states, nullptr);
	std::string all(chars);
	env->ReleaseStringUTFChars(states, chars);
	std::vector<std::string> list;
	for (size_t i = 0; i + 54 <= all.size(); i += 54)
		list.push_back(all.substr(i, 54));
	try {
		return toArray(env, scoreStates(model, view, rotation, list));
	}
	catch (const std::exception&) {
		return toArray(env, std::vector<double>(list.size(), 1e9));
	}
}

// The view facelets whose samples contradict `state` (conflicts()).
JNIEXPORT jintArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeConflicts (JNIEnv* env, jclass, jdoubleArray mean, jdoubleArray var,
	jdoubleArray samples, jint rotation, jstring state, jdouble threshold) {
	ColourModel model = readModel(env, mean, var);
	jdouble* s = env->GetDoubleArrayElements(samples, nullptr);
	std::vector<FaceletSample> view = readSamples(s, env->GetArrayLength(samples) / 5);
	env->ReleaseDoubleArrayElements(samples, s, JNI_ABORT);
	const char* chars = env->GetStringUTFChars(state, nullptr);
	std::string st(chars);
	env->ReleaseStringUTFChars(state, chars);
	std::vector<int> r;
	try {
		r = conflicts(model, view, rotation, st, threshold);
	}
	catch (const std::exception&) {
		r.assign(27, -1);
	}
	jintArray out = env->NewIntArray(jsize(r.size()));
	env->SetIntArrayRegion(out, 0, jsize(r.size()), reinterpret_cast<const jint*>(r.data()));
	return out;
}

JNIEXPORT jstring JNICALL Java_com_klstudio_rubikdetector_Engine_nativeApplyMoves (JNIEnv* env, jclass, jstring state, jstring moves) {
	const char* a = env->GetStringUTFChars(state, nullptr);
	const char* b = env->GetStringUTFChars(moves, nullptr);
	std::string r;
	try {
		r = applyMoves(a, b);
	}
	catch (const std::exception&) {
	}
	env->ReleaseStringUTFChars(state, a);
	env->ReleaseStringUTFChars(moves, b);
	return r.empty() ? nullptr : env->NewStringUTF(r.c_str());
}

// --- tracking between detections ---

struct TrackerHandle {
	FrameTracker tracker;
	std::vector<uint8_t> upright;
};

JNIEXPORT jlong JNICALL Java_com_klstudio_rubikdetector_Engine_nativeTrackerCreate (JNIEnv*, jclass) {
	return reinterpret_cast<jlong>(new TrackerHandle());
}

JNIEXPORT void JNICALL Java_com_klstudio_rubikdetector_Engine_nativeTrackerDestroy (JNIEnv*, jclass, jlong h) {
	delete reinterpret_cast<TrackerHandle*>(h);
}

// Turns the RGBA frame upright into the tracker's own buffer. Returns {width, height, ms}.
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeTrackerSetFrame (JNIEnv* env, jclass, jlong h, jobject buffer,
	jint width, jint height, jint stride, jint rotation) {
	auto t0 = std::chrono::steady_clock::now();
	auto* T = reinterpret_cast<TrackerHandle*>(h);
	const auto* src = static_cast<const uint8_t*>(env->GetDirectBufferAddress(buffer));
	auto [w, ht] = rotate(src, width, height, stride, rotation, T->upright);
	return toArray(env, {double(w), double(ht), msSince(t0)});
}

// Seeds on the frame set last, with a pose (R 9, t 3) and fx (square pixels, principal point the centre).
JNIEXPORT void JNICALL Java_com_klstudio_rubikdetector_Engine_nativeTrackerSeed (JNIEnv* env, jclass, jlong h, jint width, jint height, jdouble fx,
	jdoubleArray pose) {
	auto* T = reinterpret_cast<TrackerHandle*>(h);
	jdouble* p = env->GetDoubleArrayElements(pose, nullptr);
	Pose q;
	std::copy(p, p + 9, q.R.begin());
	std::copy(p + 9, p + 12, q.t.begin());
	env->ReleaseDoubleArrayElements(pose, p, JNI_ABORT);
	T->tracker.seed({T->upright.data(), width, height, width * 4, PixelFormat::RGBA}, {fx, fx, 0, 0}, q);
}

JNIEXPORT void JNICALL Java_com_klstudio_rubikdetector_Engine_nativeTrackerReset (JNIEnv*, jclass, jlong h) {
	reinterpret_cast<TrackerHandle*>(h)->tracker.reset();
}

// A new pose (R 9, t 3) for the frame tracked last; false if none yet.
JNIEXPORT jboolean JNICALL Java_com_klstudio_rubikdetector_Engine_nativeTrackerReseed (JNIEnv* env, jclass, jlong h, jdoubleArray pose) {
	auto* T = reinterpret_cast<TrackerHandle*>(h);
	jdouble* p = env->GetDoubleArrayElements(pose, nullptr);
	Pose q;
	std::copy(p, p + 9, q.R.begin());
	std::copy(p + 9, p + 12, q.t.begin());
	env->ReleaseDoubleArrayElements(pose, p, JNI_ABORT);
	return T->tracker.reseed(q);
}

// Tracks into the frame set last. Out: ok, R (9), t (3), renaming, tracked, inliers, reprojection, ms pyramid, flow,
// pose, pointCount, (x, y)…
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeTrackerTrack (JNIEnv* env, jclass, jlong h, jint width, jint height) {
	auto* T = reinterpret_cast<TrackerHandle*>(h);
	TrackResult r = T->tracker.track({T->upright.data(), width, height, width * 4, PixelFormat::RGBA});
	std::vector<double> out = {double(r.ok)};
	out.insert(out.end(), r.pose.R.begin(), r.pose.R.end());
	out.insert(out.end(), r.pose.t.begin(), r.pose.t.end());
	out.insert(out.end(), {double(r.renaming), double(r.tracked), double(r.inliers), r.reprojection, r.msPyramid, r.msFlow, r.msPose,
		double(r.points.size())});
	for (const auto& p : r.points)
		out.insert(out.end(), {double(p[0]), double(p[1])});
	return toArray(env, out);
}

// Facelet quads of a pose (R 9, t 3) for drawing: per R U F facelet (view frame) facelet, then 4 corners (8).
JNIEXPORT jdoubleArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeProjectFacelets (JNIEnv* env, jclass, jint width, jint height,
	jdouble fx, jdoubleArray pose) {
	jdouble* p = env->GetDoubleArrayElements(pose, nullptr);
	Pose q;
	std::copy(p, p + 9, q.R.begin());
	std::copy(p + 9, p + 12, q.t.begin());
	env->ReleaseDoubleArrayElements(pose, p, JNI_ABORT);
	std::vector<double> out;
	for (const auto& [k, quad] : projectFacelets({fx, fx, width / 2.0, height / 2.0}, q)) {
		out.push_back(k);
		out.insert(out.end(), quad.begin(), quad.end());
	}
	return toArray(env, out);
}

// The 24 cube rotations (9 ints each, row-major) followed by their facelet permutations (54 each).
JNIEXPORT jintArray JNICALL Java_com_klstudio_rubikdetector_Engine_nativeRotationTables (JNIEnv* env, jclass) {
	std::vector<jint> out;
	for (const auto& m : rotations())
		out.insert(out.end(), m.begin(), m.end());
	for (int g = 0; g < 24; g++) {
		const auto& p = rotationPerm(g);
		out.insert(out.end(), p.begin(), p.end());
	}
	jintArray arr = env->NewIntArray(jsize(out.size()));
	env->SetIntArrayRegion(arr, 0, jsize(out.size()), out.data());
	return arr;
}

}
