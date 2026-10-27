"""One group end to end: per view geometry → sticker colours; fusion of the three views; classification."""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import colors, cube, fusion, geometry, legal


def geometry_error(pred_quads, truth_quads):
	"""Sticker centre error in cubie units: predicted and true visible centres matched one to one (the frames are not
	aligned yet), distance / median distance between neighbouring true centres. Also counts of predicted stickers
	and true ones."""
	pc = np.array([np.mean(q, axis=0) for q in pred_quads.values()])
	tc = np.array([np.mean(q, axis=0) for q in truth_quads.values()])
	if not len(pc) or not len(tc):
		return {"mean": None, "max": None}
	d = np.linalg.norm(pc[:, None] - tc[None], axis=2)
	r, c = linear_sum_assignment(d)
	dt = np.linalg.norm(tc[:, None] - tc[None], axis=2)
	pitch = np.median(np.sort(dt, axis=1)[:, 1])
	e = d[r, c] / pitch
	return {"mean": float(e.mean()), "max": float(e.max()), "px": float(d[r, c].mean()), "pitch_px": float(pitch)}


def run(scenario, geom="line", classifier="legal"):
	out = {"group": scenario.group, "kind": scenario.kind, "geometry": geom, "classifier": classifier, "views": []}
	stage = geometry.oracle if geom == "oracle" else geometry.LineGeometry()
	obs, results = [], []
	t_all = time.perf_counter()
	for view in scenario.views:
		t0 = time.perf_counter()
		rec = {"name": view.name}
		try:
			g = stage(view)
		except geometry.GeometryError as e:
			rec["error"] = str(e)
			out["views"].append(rec)
			results.append(None)
			continue
		lab = colors.to_lab(view.image)
		samples = {k: colors.sample(lab, q) for k, q in g["quads"].items()}
		samples = {k: s for k, s in samples.items() if s is not None}
		obs.append(samples)
		rec["geometry_error"] = geometry_error(g["quads"], view.quads)
		rec["seconds"] = time.perf_counter() - t0
		rec["times"] = g["debug"].get("times", {})
		if "cost" in g["debug"]:
			rec["lattice_cost"] = g["debug"]["cost"]
		out["views"].append(rec)
		results.append(g)
	out["results"] = results	# for debug drawing; dropped before saving

	if len(obs) < 3:
		out.update(state=None, wrong=None, error="a view failed")
		return out
	try:
		t0 = time.perf_counter()
		observations, rotations, cost = fusion.fuse(obs)
		out["fusion"] = {"rotations": rotations, "cost": cost, "seconds": time.perf_counter() - t0}
		if len(observations) < 54:
			raise ValueError(f"fused views cover {len(observations)} facelets")
		t0 = time.perf_counter()
		if classifier == "legal":
			state, info = colors.classify_legal(observations)
		else:
			state, info = colors.classify(observations), {}
		info["seconds"] = time.perf_counter() - t0
		info["legal"] = legal.is_legal(legal.to_labels(state))
		out["classification"] = info
	except ValueError as e:
		out.update(state=None, wrong=None, error=str(e))
		return out
	wrong, g = cube.compare(state, scenario.state)
	out.update(state=state, wrong=int(wrong), seconds=time.perf_counter() - t_all)
	return out
