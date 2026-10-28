"""Focal length estimation (Camera fx <= 0) on samples whose true K is known: per view the estimated fx / true fx,
the sticker centre error vs truth with the estimated camera, and per group the state from three views each
estimated on its own.

	python engine/tools/focal_check.py [--set g,n] [--kinds render,bg,hand] [--limit 20] [--jobs 12]
"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from rubik_detector import data, pipeline  # noqa: E402


def job(args):
	group, kind = args
	import rubik_engine as re
	s = data.load(group, kind, include_failed=True)
	if s is None:
		return None
	det = re.Detector()
	out = {"group": group, "kind": kind, "views": []}
	views = []
	for view in s.views:
		t0 = time.perf_counter()
		r = det.detect(view.image, None, work_width=0)
		ms = (time.perf_counter() - t0) * 1000
		rec = {"ok": r["ok"], "ratio": r["camera"][0] / view.K[0, 0], "ms": ms, "tried": r["focal_tried"]}
		if r["ok"]:
			rec["err"] = pipeline.geometry_error({q["facelet"]: q["quad"] for q in r["samples"]}, view.quads)["max"]
		out["views"].append(rec)
		views.append(r)
	if all(v["ok"] for v in views):
		e = re.estimate(views)
		out["wrong"] = re.compare_states(e["state"], s.state) if e["ok"] else None
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--set", default="g,n")
	ap.add_argument("--kinds", default="render,bg,hand")
	ap.add_argument("--limit", type=int, default=20)
	ap.add_argument("--jobs", type=int, default=12)
	a = ap.parse_args()
	jobs = [(g, k) for s in a.set.split(",") for g in data.groups(s)[:a.limit] for k in a.kinds.split(",")]
	t0 = time.time()
	with ProcessPoolExecutor(a.jobs) as ex:
		rows = [r for r in ex.map(job, jobs) if r]
	for kind in a.kinds.split(","):
		rs = [r for r in rows if r["kind"] == kind]
		vs = [v for r in rs for v in r["views"]]
		ratios = np.array([v["ratio"] for v in vs if v["ok"]])
		err = np.array([v["err"] for v in vs if v["ok"]])
		print(f"{kind:6s} {len(rs)} groups, {len(vs)} views: fx estimate / true median {np.median(ratios):.3f}, "
			f"|log ratio| p50 {np.median(np.abs(np.log(ratios))):.3f} p95 {np.percentile(np.abs(np.log(ratios)), 95):.3f} max {np.abs(np.log(ratios)).max():.3f}; "
			f"views located (< 0.5 cubie) {int((err < 0.5).sum())}/{len(vs)}; groups exact {sum(r.get('wrong') == 0 for r in rs)}/{len(rs)}; "
			f"{np.mean([v['ms'] for v in vs]):.0f} ms/view ({np.mean([v['tried'] for v in vs]):.0f} solves)")
		bad = [(r["group"], i + 1, round(v["ratio"], 2), round(v.get("err", -1), 2)) for r in rs for i, v in enumerate(r["views"]) if not v["ok"] or v["err"] >= 0.5]
		if bad:
			print("   not located (group, view, fx ratio, err):", bad)
		wrong = [(r["group"], r.get("wrong")) for r in rs if r.get("wrong") != 0]
		if wrong:
			print("   groups not exact:", wrong)
	print(f"\n{len(rows)} scenarios in {time.time() - t0:.0f} s")


if __name__ == "__main__":
	main()
