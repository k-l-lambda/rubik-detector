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
		("focal_min", ctypes.c_double), ("focal_max", ctypes.c_double), ("focal_steps", ctypes.c_int32), ("focal_refine", ctypes.c_int32)]


class View(ctypes.Structure):
	_fields_ = [("ok", ctypes.c_int32), ("error", ctypes.c_char * 96), ("pose", ctypes.c_double * 12),
		("lattice_cost", ctypes.c_double), ("segment_count", ctypes.c_int32), ("family_segments", ctypes.c_int32),
		("camera", ctypes.c_double * 4), ("focal_estimated", ctypes.c_int32), ("focal_tried", ctypes.c_int32),
		("ms_segments", ctypes.c_double), ("ms_axes", ctypes.c_double), ("ms_position", ctypes.c_double), ("ms_sampling", ctypes.c_double),
		("sample_count", ctypes.c_int32), ("samples", Sample * 27)]


class State(ctypes.Structure):
	_fields_ = [("ok", ctypes.c_int32), ("error", ctypes.c_char * 96), ("state", ctypes.c_char * 55),
		("legal", ctypes.c_int32), ("exact", ctypes.c_int32), ("cost", ctypes.c_double), ("min_posterior", ctypes.c_double),
		("posterior", ctypes.c_double * 54), ("corrected", ctypes.c_int32 * 54), ("corrected_count", ctypes.c_int32),
		("fusion_cost", ctypes.c_double), ("rotations", ctypes.c_int32 * 16)]


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
_lib.rk_apply_moves.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_lib.rk_compare_states.argtypes = [ctypes.c_char_p, ctypes.c_char_p]


def view_dict(v):
	samples = [{"facelet": s.facelet, "valid": bool(s.valid), "pixels": s.pixels, "lab": list(s.lab), "quad": np.array(s.quad).reshape(4, 2)}
		for s in v.samples[:v.sample_count]]
	return {"ok": bool(v.ok), "error": v.error.decode(), "R": np.array(v.pose[:9]).reshape(3, 3), "t": np.array(v.pose[9:]),
		"lattice_cost": v.lattice_cost, "segments": v.segment_count, "family_segments": v.family_segments,
		"camera": list(v.camera), "focal_estimated": bool(v.focal_estimated), "focal_tried": v.focal_tried,
		"ms": {"segments": v.ms_segments, "axes": v.ms_axes, "position": v.ms_position, "sampling": v.ms_sampling},
		"samples": samples, "_raw": v}


class Detector:
	def __init__(self):
		self._d = _lib.rk_detector_new()

	def __del__(self):
		if getattr(self, "_d", None):
			_lib.rk_detector_free(self._d)

	def detect(self, image_bgr, K, work_width=1024, prior=None, seed=0):
		"""image_bgr: (H, W, 3) uint8 (cv2.imread); K: 3×3 for this image, or None: focal length estimated, principal
		point at the centre. prior: (R, t) for a local search."""
		img = np.ascontiguousarray(image_bgr, np.uint8)
		o = Options()
		_lib.rk_default_options(ctypes.byref(o))
		o.work_width = work_width
		o.seed = seed
		if prior is not None:
			o.search = 1
			o.prior[:] = list(np.asarray(prior[0], float).ravel()) + list(np.asarray(prior[1], float))
		k = (ctypes.c_double * 4)(*((0, 0, 0, 0) if K is None else (K[0][0], K[1][1], K[0][2], K[1][2])))
		v = View()
		_lib.rk_detect(self._d, img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], 0, ctypes.byref(k), ctypes.byref(o), ctypes.byref(v))
		return view_dict(v)

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


def estimate(views, rotations=None, require_all_faces=True, legal=True):
	"""views: list of view_dict results (their raw samples are reused)."""
	counts = (ctypes.c_int32 * len(views))(*[v["_raw"].sample_count for v in views])
	flat = (Sample * sum(counts))()
	i = 0
	for v in views:
		for s in v["_raw"].samples[:v["_raw"].sample_count]:
			flat[i] = s
			i += 1
	rot = (ctypes.c_int32 * len(views))(*(rotations or [-1] * len(views)))
	out = State()
	_lib.rk_estimate(len(views), flat, counts, rot, int(require_all_faces), int(legal), ctypes.byref(out))
	return {"ok": bool(out.ok), "error": out.error.decode(), "state": out.state.decode(), "legal": bool(out.legal), "exact": bool(out.exact),
		"cost": out.cost, "min_posterior": out.min_posterior, "posterior": list(out.posterior), "corrected": list(out.corrected[:out.corrected_count]),
		"fusion_cost": out.fusion_cost, "rotations": list(out.rotations[:len(views)])}


def apply_moves(state, moves):
	buf = ctypes.create_string_buffer(55)
	if not _lib.rk_apply_moves(state.encode(), moves.encode(), buf):
		raise ValueError(moves)
	return buf.value.decode()


def compare_states(pred, truth):
	return _lib.rk_compare_states(pred.encode(), truth.encode())
