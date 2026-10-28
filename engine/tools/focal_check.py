"""Detector::estimateFocal on samples whose true K is known: fx error estimated from each view alone and from the
three views of a group together (one camera); with the joint estimate, the sticker centre error vs truth and the
group's state.

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
	group, kind, coarse_width = args
	import rubik_engine as re
	s = data.load(group, kind, include_failed=True)
	if s is None:
		return None
	det = re.Detector()
	out = {"group": group, "kind": kind, "views": []}
	# per view on its own, and the three views together (one camera)
	for view in s.views:
		f = det.estimate_focal([view.image], coarse_width=coarse_width, work_width=view.image.shape[1])
		out["views"].append({"ok": f["ok"], "ratio": f["camera"][0] / view.K[0, 0], "ms": f["ms"], "solves": f["solves"]})
	f = det.estimate_focal([v.image for v in s.views], coarse_width=coarse_width, work_width=s.views[0].image.shape[1])
	out["group_ratio"], out["group_ms"] = f["camera"][0] / s.views[0].K[0, 0], f["ms"]
	K = np.array([[f["camera"][0], 0, f["camera"][2]], [0, f["camera"][1], f["camera"][3]], [0, 0, 1]])
	views = []
	for v, view in enumerate(s.views):
		r = det.detect(view.image, K, work_width=0)
		views.append(r)
		out["views"][v]["err"] = pipeline.geometry_error({q["facelet"]: q["quad"] for q in r["samples"]}, view.quads)["max"] if r["ok"] else 99
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
	ap.add_argument("--coarse-width", type=int, default=512)
	a = ap.parse_args()
	jobs = [(g, k, a.coarse_width) for s in a.set.split(",") for g in data.groups(s)[:a.limit] for k in a.kinds.split(",")]
	t0 = time.time()
	with ProcessPoolExecutor(a.jobs) as ex:
		rows = [r for r in ex.map(job, jobs) if r]
	print(f"coarse width {a.coarse_width}")
	for kind in a.kinds.split(","):
		rs = [r for r in rows if r["kind"] == kind]
		vs = [v for r in rs for v in r["views"]]
		single = np.abs(np.log([v["ratio"] for v in vs]))
		joint = np.abs(np.log([r["group_ratio"] for r in rs]))
		print(f"{kind:6s} {len(rs)} groups: |log fx error| single view p50 {np.median(single):.3f} p95 {np.percentile(single, 95):.3f} max {single.max():.3f} "
			f"({np.mean([v['ms'] for v in vs]):.0f} ms, {np.mean([v['solves'] for v in vs]):.0f} solves); three views p50 {np.median(joint):.3f} max {joint.max():.3f} "
			f"({np.mean([r['group_ms'] for r in rs]):.0f} ms); with the joint estimate: views located {sum(v['err'] < 0.5 for v in vs)}/{len(vs)}, "
			f"groups exact {sum(r.get('wrong') == 0 for r in rs)}/{len(rs)}")
		bad = [(r["group"], round(r["group_ratio"], 3), r.get("wrong")) for r in rs if r.get("wrong") != 0]
		if bad:
			print("   not exact (group, fx ratio, wrong):", bad)
	print(f"\n{len(rows)} scenarios in {time.time() - t0:.0f} s")


if __name__ == "__main__":
	main()
