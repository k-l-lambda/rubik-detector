"""ctypes binding of the engine's C ABI (include/rubik/rubik_c.h). Build first: cmake -S engine -B engine/build && cmake --build engine/build"""
import ctypes, os

import numpy as np

LIB = os.environ.get("RUBIK_ENGINE_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "librubik_engine.so"))


class Sample(ctypes.Structure):
	_fields_ = [("facelet", ctypes.c_int32), ("valid", ctypes.c_int32), ("pixels", ctypes.c_int32),
		("lab", ctypes.c_float * 3), ("quad", ctypes.c_float * 8)]


class Options(ctypes.Structure):
	_fields_ = [("search", ctypes.c_int32), ("prior", ctypes.c_double * 12), ("work_width", ctypes.c_int32),
		("sample_size", ctypes.c_double), ("seed", ctypes.c_uint64),
		("min_extent", ctypes.c_double), ("max_extent", ctypes.c_double), ("centre_share", ctypes.c_double),
		("coarse_segments", ctypes.c_int32), ("shift_pairs", ctypes.c_int32),
		("starts", ctypes.c_int32), ("joint", ctypes.c_int32)]


class View(ctypes.Structure):
	_fields_ = [("ok", ctypes.c_int32), ("error", ctypes.c_char * 96), ("pose", ctypes.c_double * 12),
		("lattice_cost", ctypes.c_double), ("segment_count", ctypes.c_int32), ("family_segments", ctypes.c_int32),
		("camera", ctypes.c_double * 4), ("support", ctypes.c_double),
		("ms_segments", ctypes.c_double), ("ms_axes", ctypes.c_double), ("ms_position", ctypes.c_double), ("ms_sampling", ctypes.c_double),
		("sample_count", ctypes.c_int32), ("samples", Sample * 27)]


class Focal(ctypes.Structure):
	_fields_ = [("ok", ctypes.c_int32), ("error", ctypes.c_char * 96), ("camera", ctypes.c_double * 4), ("ratio", ctypes.c_double),
		("support", ctypes.c_double), ("ms", ctypes.c_double), ("solves", ctypes.c_int32), ("curve_count", ctypes.c_int32),
		("curve", (ctypes.c_double * 2) * 32)]


class State(ctypes.Structure):
	_fields_ = [("ok", ctypes.c_int32), ("error", ctypes.c_char * 96), ("state", ctypes.c_char * 55),
		("legal", ctypes.c_int32), ("exact", ctypes.c_int32), ("cost", ctypes.c_double), ("min_posterior", ctypes.c_double),
		("posterior", ctypes.c_double * 54), ("corrected", ctypes.c_int32 * 54), ("corrected_count", ctypes.c_int32),
		("fusion_cost", ctypes.c_double), ("rotations", ctypes.c_int32 * 16)]


class StateEx(ctypes.Structure):
	_fields_ = [("ok", ctypes.c_int32), ("error", ctypes.c_char * 96), ("state", ctypes.c_char * 55),
		("legal", ctypes.c_int32), ("exact", ctypes.c_int32), ("cost", ctypes.c_double), ("min_posterior", ctypes.c_double),
		("posterior", ctypes.c_double * 54), ("probability", (ctypes.c_double * 6) * 54), ("confidence", ctypes.c_double * 54), ("min_confidence", ctypes.c_double),
		("observed", ctypes.c_int32 * 54),
		("coverage", ctypes.c_int32), ("corrected", ctypes.c_int32 * 54), ("corrected_count", ctypes.c_int32), ("fusion_cost", ctypes.c_double),
		("model_mean", (ctypes.c_double * 4) * 6), ("model_var", (ctypes.c_double * 4) * 6)]


class EstimateOptions(ctypes.Structure):
	_fields_ = [("require_all_faces", ctypes.c_int32), ("legal", ctypes.c_int32), ("partial", ctypes.c_int32), ("outlier", ctypes.c_double), ("reliability", ctypes.c_int32)]


_lib = ctypes.CDLL(LIB)
_lib.rk_detector_new.restype = ctypes.c_void_p
_lib.rk_detector_free.argtypes = [ctypes.c_void_p]
_lib.rk_default_options.argtypes = [ctypes.POINTER(Options)]
_lib.rk_detect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
	ctypes.POINTER(ctypes.c_double * 4), ctypes.POINTER(Options), ctypes.POINTER(View)]
_lib.rk_estimate.argtypes = [ctypes.c_int32, ctypes.POINTER(Sample), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
	ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(State)]
_lib.rk_score_states.argtypes = [ctypes.c_int32, ctypes.POINTER(Sample), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
	ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(State)]
_lib.rk_sample_at.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
	ctypes.POINTER(ctypes.c_double * 4), ctypes.POINTER(ctypes.c_double * 12), ctypes.c_int32, ctypes.c_double, ctypes.POINTER(Sample * 27)]
_lib.rk_sample_at.restype = ctypes.c_int32
_lib.rk_estimate_focal.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32, ctypes.c_int32,
	ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32 * 4), ctypes.POINTER(Focal)]
_lib.rk_estimate_ex.argtypes = [ctypes.c_int32, ctypes.POINTER(Sample), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
	ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(EstimateOptions), ctypes.POINTER(StateEx), ctypes.c_void_p]
_lib.rk_score_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Sample), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
	ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
_lib.rk_score_model.restype = ctypes.c_int32
_lib.rk_conflicts.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Sample), ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p,
	ctypes.c_double, ctypes.POINTER(ctypes.c_int32 * 27)]
_lib.rk_conflicts.restype = ctypes.c_int32
_lib.rk_detect_segments.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
	ctypes.POINTER(ctypes.c_double * 4), ctypes.POINTER(Options), ctypes.POINTER(View), ctypes.c_void_p, ctypes.c_int32]
_lib.rk_detect_segments.restype = ctypes.c_int32
_lib.rk_pose_cost.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
	ctypes.POINTER(ctypes.c_double * 4), ctypes.POINTER(ctypes.c_double * 12), ctypes.c_int32]
_lib.rk_pose_cost.restype = ctypes.c_double
_lib.rk_apply_moves.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_lib.rk_compare_states.argtypes = [ctypes.c_char_p, ctypes.c_char_p]


def view_dict(v):
	samples = [{"facelet": s.facelet, "valid": bool(s.valid), "pixels": s.pixels, "lab": list(s.lab), "quad": np.array(s.quad).reshape(4, 2)}
		for s in v.samples[:v.sample_count]]
	return {"ok": bool(v.ok), "error": v.error.decode(), "R": np.array(v.pose[:9]).reshape(3, 3), "t": np.array(v.pose[9:]),
		"lattice_cost": v.lattice_cost, "segments": v.segment_count, "family_segments": v.family_segments,
		"camera": list(v.camera), "support": v.support,
		"ms": {"segments": v.ms_segments, "axes": v.ms_axes, "position": v.ms_position, "sampling": v.ms_sampling},
		"samples": samples, "_raw": v}


class Detector:
	def __init__(self):
		self._d = _lib.rk_detector_new()

	def __del__(self):
		if getattr(self, "_d", None):
			_lib.rk_detector_free(self._d)

	def detect(self, image_bgr, K, work_width=1024, prior=None, seed=0, keep_segments=False, **effort):
		"""image_bgr: (H, W, 3) uint8 (cv2.imread); K: 3×3 for this image. prior: (R, t) for a local search.
		effort: coarse_segments, shift_pairs, starts, joint (DetectOptions search effort).
		keep_segments: also return "segments_list" (x1 y1 x2 y2 weight family per row, input pixels)."""
		img = np.ascontiguousarray(image_bgr, np.uint8)
		o = Options()
		_lib.rk_default_options(ctypes.byref(o))
		o.work_width = work_width
		o.seed = seed
		for key, value in effort.items():
			setattr(o, key, int(value))
		if prior is not None:
			o.search = 1
			o.prior[:] = list(np.asarray(prior[0], float).ravel()) + list(np.asarray(prior[1], float))
		k = (ctypes.c_double * 4)(K[0][0], K[1][1], K[0][2], K[1][2])
		v = View()
		if keep_segments:
			buf = np.zeros((8192, 6), np.float32)
			n = _lib.rk_detect_segments(self._d, img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], 0, ctypes.byref(k), ctypes.byref(o),
				ctypes.byref(v), buf.ctypes.data, len(buf))
			out = view_dict(v)
			out["segments_list"] = buf[:n].copy()
			return out
		_lib.rk_detect(self._d, img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], 0, ctypes.byref(k), ctypes.byref(o), ctypes.byref(v))
		return view_dict(v)

	def pose_cost(self, image_bgr, K, R, t, work_width=1024):
		"""debugging: the lattice cost of a given pose, no search"""
		img = np.ascontiguousarray(image_bgr, np.uint8)
		k = (ctypes.c_double * 4)(K[0][0], K[1][1], K[0][2], K[1][2])
		pose = (ctypes.c_double * 12)(*(list(np.asarray(R, float).ravel()) + list(np.asarray(t, float))))
		return _lib.rk_pose_cost(self._d, img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], 0, ctypes.byref(k), ctypes.byref(pose), work_width)

	def estimate_focal(self, images_bgr, coarse_width=0, coarse_steps=0, refine_steps=0, work_width=0):
		# Focal length from one or more same-size BGR images of one camera: camera (fx fy cx cy), ratio (fx / long
		# side), support, the coarse curve, solves, ms.
		imgs = [np.ascontiguousarray(i, np.uint8) for i in images_bgr]
		ptrs = (ctypes.c_void_p * len(imgs))(*[i.ctypes.data for i in imgs])
		strides = (ctypes.c_int32 * len(imgs))(*[i.strides[0] for i in imgs])
		steps = (ctypes.c_int32 * 4)(coarse_width, coarse_steps, refine_steps, work_width)
		out = Focal()
		_lib.rk_estimate_focal(self._d, len(imgs), ptrs, imgs[0].shape[1], imgs[0].shape[0], strides, 0, None, None, ctypes.byref(steps), ctypes.byref(out))
		return {"ok": bool(out.ok), "error": out.error.decode(), "camera": list(out.camera), "ratio": out.ratio, "support": out.support,
			"curve": [tuple(out.curve[i]) for i in range(out.curve_count)], "solves": out.solves, "ms": out.ms}

	def sample(self, image_bgr, K, R, t, work_width=1024, sample_size=0.46):
		img = np.ascontiguousarray(image_bgr, np.uint8)
		k = (ctypes.c_double * 4)(K[0][0], K[1][1], K[0][2], K[1][2])
		pose = (ctypes.c_double * 12)(*(list(np.asarray(R, float).ravel()) + list(np.asarray(t, float))))
		out = (Sample * 27)()
		n = _lib.rk_sample_at(self._d, img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], 0, ctypes.byref(k), ctypes.byref(pose),
			work_width, sample_size, ctypes.byref(out))
		return [{"facelet": s.facelet, "valid": bool(s.valid), "pixels": s.pixels, "lab": list(s.lab), "quad": np.array(s.quad).reshape(4, 2)} for s in out[:n]]


def _flat(views):
	counts = (ctypes.c_int32 * len(views))(*[v["_raw"].sample_count for v in views])
	flat = (Sample * sum(counts))()
	i = 0
	for v in views:
		for s in v["_raw"].samples[:v["_raw"].sample_count]:
			flat[i] = s
			i += 1
	return counts, flat


def score_states(views, view, states, rotations=None, require_all_faces=True):
	"""Scores (negative log-likelihood, lower = better) of candidate states for view `view`'s samples, under the
	colour model fitted by the estimate of all `views`."""
	counts, flat = _flat(views)
	rot = (ctypes.c_int32 * len(views))(*(rotations or [-1] * len(views)))
	scores = (ctypes.c_double * len(states))()
	out = State()
	_lib.rk_score_states(len(views), flat, counts, rot, int(require_all_faces), view, len(states), "".join(states).encode(), scores, ctypes.byref(out))
	if not out.ok:
		raise RuntimeError(out.error.decode())
	return list(scores)


def estimate(views, rotations=None, require_all_faces=True, legal=True, weights=None, partial=False, outlier=0.0, groups=None, reliability=False):
	"""views: list of view_dict results (their raw samples are reused), any number. weights: per view (None = 1).
	partial: facelets not observed allowed (legal: completed by the legal search). outlier: per-observation outlier
	probability. reliability: each group's evidence (groups: per view, None = each view its own) right with an unknown
	probability, uniform prior, integrated out (EstimateOptions::reliability). Returns, besides the state, per facelet `probability` (6 colours, face order L R D U B F) and
	`observed`."""
	counts, flat = _flat(views)
	rot = (ctypes.c_int32 * len(views))(*(rotations or [-1] * len(views)))
	w = (ctypes.c_double * len(views))(*weights) if weights is not None else None
	gr = (ctypes.c_int32 * len(views))(*groups) if groups is not None else None
	rot_out = (ctypes.c_int32 * len(views))()
	options = EstimateOptions(int(require_all_faces), int(legal), int(partial), outlier, int(reliability))
	out = StateEx()
	_lib.rk_estimate_ex(len(views), flat, counts, rot, w, gr, ctypes.byref(options), ctypes.byref(out), rot_out)
	return {"ok": bool(out.ok), "error": out.error.decode(), "state": out.state.decode(), "legal": bool(out.legal), "exact": bool(out.exact),
		"cost": out.cost, "min_posterior": out.min_posterior, "posterior": list(out.posterior), "corrected": list(out.corrected[:out.corrected_count]),
		"probability": np.array(out.probability), "confidence": list(out.confidence), "min_confidence": out.min_confidence, "observed": [bool(x) for x in out.observed], "coverage": out.coverage,
		"fusion_cost": out.fusion_cost, "rotations": list(rot_out),
		"model": {"mean": np.array(out.model_mean), "var": np.array(out.model_var)}}


def score_model(model, view, rotation, states):
	"""Negative log-likelihoods (lower = better) of candidate states (common frame) for one view's samples placed by
	`rotation`, under a colour model kept from an estimate (estimate(...)["model"])."""
	mean = np.ascontiguousarray(model["mean"], dtype=np.float64)
	var = np.ascontiguousarray(model["var"], dtype=np.float64)
	raw = view["_raw"]
	scores = (ctypes.c_double * len(states))()
	if not _lib.rk_score_model(mean.ctypes.data, var.ctypes.data, raw.samples, raw.sample_count, rotation, len(states),
			"".join(states).encode(), scores):
		raise ValueError("bad input")
	return np.array(scores)


def conflicts(model, view, rotation, state, threshold=4.0):
	"""view facelets whose sample contradicts `state` (placed by `rotation`): its colour there costs > threshold nll
	above its most likely colour under the model"""
	mean = np.ascontiguousarray(model["mean"], dtype=np.float64)
	var = np.ascontiguousarray(model["var"], dtype=np.float64)
	raw = view["_raw"]
	out = (ctypes.c_int32 * 27)()
	n = _lib.rk_conflicts(mean.ctypes.data, var.ctypes.data, raw.samples, raw.sample_count, rotation, state.encode(), threshold, ctypes.byref(out))
	if n < 0:
		raise ValueError("bad input")
	return list(out[:n])


def apply_moves(state, moves):
	buf = ctypes.create_string_buffer(55)
	if not _lib.rk_apply_moves(state.encode(), moves.encode(), buf):
		raise ValueError(moves)
	return buf.value.decode()


def compare_states(pred, truth):
	return _lib.rk_compare_states(pred.encode(), truth.encode())
