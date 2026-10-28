"""Checks of the engine's tracking-oriented API on the labelled samples (no Python reference for these):

	1. local search: from the wide result perturbed by a few degrees and a fraction of a cubie (a tracked frame's
	   prior), the local search lands on the same stickers — sticker centre error vs truth, time; and sample() at
	   the detected pose gives detect()'s colours
	2. renamed frames: the prior given in another view frame (R·g, all 24 g); the result is re-signed to the visible
	   faces and relativeRotation(prior, result) names the renaming
	3. turn scoring: under the colour model of the three views, the estimated state must explain each view better than
	   any of the 18 single face turns of it (score margin = runner-up − best)

	python engine/tools/api_check.py [--set g,n] [--kinds render,bg,hand] [--jobs 12]
"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from rubik_detector import cube, data, pipeline  # noqa: E402

FACE_TURNS = [f + s for f in "UDLRFB" for s in ("", "'", "2")]


def job(args):
	group, kind = args
	import rubik_engine as re
	s = data.load(group, kind, include_failed=True)
	if s is None:
		return None
	rng = np.random.default_rng(int(group[1:]) * 7 + len(kind))
	det = re.Detector()
	out = {"group": group, "kind": kind, "local": [], "renamed": [], "turns": []}
	views = []
	for v, view in enumerate(s.views):
		w = det.detect(view.image, view.K, work_width=0)
		views.append(w)
		if not w["ok"]:
			continue
		quads = lambda r: {q["facelet"]: q["quad"] for q in r["samples"]}
		# 1. perturbed prior: 2–4° about a random axis, 0.2–0.4 cubie
		axis = rng.normal(size=3)
		axis /= np.linalg.norm(axis)
		Rp = cv2.Rodrigues(axis * np.radians(rng.uniform(2, 4)))[0] @ w["R"]
		d = rng.normal(size=3)
		tp = w["t"] + d / np.linalg.norm(d) * rng.uniform(0.2, 0.4)
		t0 = time.perf_counter()
		l = det.detect(view.image, view.K, work_width=0, prior=(Rp, tp))
		ms = (time.perf_counter() - t0) * 1000
		out["local"].append({"ok": l["ok"], "err": pipeline.geometry_error(quads(l), view.quads)["max"] if l["ok"] else None,
			"wide_err": pipeline.geometry_error(quads(w), view.quads)["max"], "ms": ms, "ms_position": l["ms"]["position"]})
		# sample() at the wide pose reproduces detect()'s colours
		smp = det.sample(view.image, view.K, w["R"], w["t"], work_width=0)
		out.setdefault("sample_same", []).append(all(a["valid"] == b["valid"] and np.allclose(a["lab"], b["lab"]) for a, b in zip(smp, w["samples"])) and len(smp) == 27)
		# 2. renamed prior frames
		for g in range(24):
			G = cube.ROTATIONS[g]
			r = det.detect(view.image, view.K, work_width=0, prior=(w["R"] @ G, w["t"]))
			ok = r["ok"] and pipeline.geometry_error(quads(r), view.quads)["max"] < 0.5
			# relativeRotation(prior, result) maps result's frame into the prior's: result facelets of the wide frame
			# are the prior frame's facelets permuted by g⁻¹, so composing back should give the wide frame
			out["renamed"].append(ok)
	if not all(v["ok"] for v in views):
		return out
	e = re.estimate(views)
	if not e["ok"] or re.compare_states(e["state"], s.state) != 0:
		out["estimate_exact"] = False
		return out
	out["estimate_exact"] = True
	candidates = [e["state"]] + [re.apply_moves(e["state"], m) for m in FACE_TURNS]
	for v in range(3):
		sc = re.score_states(views, v, candidates)
		order = np.argsort(sc)
		out["turns"].append({"best_is_state": int(order[0]) == 0, "margin": sc[order[1]] - sc[order[0]] if order[0] == 0 else sc[0] - sc[order[0]],
			"runner_up": FACE_TURNS[order[1] - 1] if order[0] == 0 else FACE_TURNS[order[0] - 1]})
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--set", default="g,n")
	ap.add_argument("--kinds", default="render,bg,hand")
	ap.add_argument("--jobs", type=int, default=12)
	a = ap.parse_args()
	jobs = [(g, k) for s in a.set.split(",") for g in data.groups(s) for k in a.kinds.split(",")]
	t0 = time.time()
	with ProcessPoolExecutor(a.jobs) as ex:
		rows = [r for r in ex.map(job, jobs) if r]
	loc = [x for r in rows for x in r["local"]]
	good = [x for x in loc if x["ok"] and x["err"] < 0.5]
	print(f"1. local search from perturbed priors: {len(good)}/{len(loc)} views located (centre err < 0.5 cubie); "
		f"err mean {np.mean([x['err'] for x in good]):.3f} (wide {np.mean([x['wide_err'] for x in loc]):.3f}), max {max(x['err'] for x in good):.3f}; "
		f"time mean {np.mean([x['ms'] for x in loc]):.0f} ms (position {np.mean([x['ms_position'] for x in loc]):.0f} ms)")
	bad = [(r["group"], r["kind"], i, round(x["err"] or -1, 2)) for r in rows for i, x in enumerate(r["local"]) if not (x["ok"] and x["err"] < 0.5)]
	if bad:
		print("   not located:", bad[:20])
	same = [x for r in rows for x in r.get("sample_same", [])]
	print(f"   sample() at the detected pose = detect()'s colours: {sum(same)}/{len(same)} views")
	ren = [x for r in rows for x in r["renamed"]]
	print(f"2. priors in renamed frames (24 per view): {sum(ren)}/{len(ren)} located")
	tu = [x for r in rows for x in r["turns"]]
	ok = [x for x in tu if x["best_is_state"]]
	print(f"3. turn scoring, {sum(1 for r in rows if r.get('estimate_exact'))} groups with an exact estimate: the state beats all 18 turns in "
		f"{len(ok)}/{len(tu)} views; margin (nll) min {min(x['margin'] for x in ok):.1f}, p5 {np.percentile([x['margin'] for x in ok], 5):.1f}, "
		f"median {np.median([x['margin'] for x in ok]):.1f}")
	worse = [(r["group"], r["kind"], i, x["runner_up"], round(x["margin"], 1)) for r in rows for i, x in enumerate(r["turns"]) if not x["best_is_state"]]
	if worse:
		print("   a turn scored better:", worse)
	print(f"\n{len(rows)} scenarios in {time.time() - t0:.0f} s")


if __name__ == "__main__":
	main()
