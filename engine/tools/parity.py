"""Parity of the C++ engine with the Python reference over the labelled samples.

	python engine/tools/parity.py [--set g,n] [--kinds render,bg,hand] [--reference runs/v2-*] [--mode working|file] [--jobs 12]

working: the engine gets the exact working image and K the Python pipeline uses (data.load, 1024 wide; no resize in
the engine). file: the original image file and its K, the engine resizes itself (INTER_AREA) — checks that path.
Per scenario: engine state vs truth (wrong facelets up to a whole-cube rotation), per view the sticker centre error
vs truth (pipeline.geometry_error), engine timings; against a reference run's results.jsonl (same scenarios, Python
line geometry): exact both / only one, and the per-view geometry error difference.
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from rubik_detector import cube, data, pipeline  # noqa: E402


def job(args):
	group, kind, mode = args
	import rubik_engine as re
	s = data.load(group, kind, include_failed=True)
	if s is None:
		return None
	det = re.Detector()
	views, out = [], {"group": group, "kind": kind, "views": []}
	for v, view in enumerate(s.views):
		if mode == "file":
			path = f"{data.ROOT}/samples/{group}/view{v + 1}.png" if kind == "render" else f"{data.ROOT}/samples/{group}/edit/view{v + 1}-{kind}.jpg"
			img = cv2.imread(path)
			f = img.shape[1] / view.image.shape[1]	# working K → file K
			K = view.K * np.array([[f], [f], [1]])
			r = det.detect(img, K, work_width=view.image.shape[1])
			quads = {q["facelet"]: q["quad"] / f for q in r["samples"]}
		else:
			r = det.detect(view.image, view.K, work_width=0)
			quads = {q["facelet"]: q["quad"] for q in r["samples"]}
		rec = {"ok": r["ok"], "error": r["error"], "ms": r["ms"], "segments": r["segments"], "lattice_cost": r["lattice_cost"]}
		if r["ok"]:
			rec["geometry_error"] = pipeline.geometry_error(quads, view.quads)
		out["views"].append(rec)
		views.append(r)
	if not all(v["ok"] for v in views):
		out.update(wrong=None, error="a view failed")
		return out
	e = re.estimate(views)
	if not e["ok"]:
		out.update(wrong=None, error=e["error"])
		return out
	out.update(wrong=re.compare_states(e["state"], s.state), min_posterior=e["min_posterior"], legal=e["legal"], rotations=e["rotations"])
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--set", default="r,g,n,a")
	ap.add_argument("--kinds", default="render,bg,hand")
	ap.add_argument("--reference", default="runs/v2-*-1028-1109")
	ap.add_argument("--mode", default="working", choices=("working", "file"))
	ap.add_argument("--jobs", type=int, default=12)
	ap.add_argument("--out", default=None)
	a = ap.parse_args()
	jobs = [(g, k, a.mode) for s in a.set.split(",") for g in data.groups(s) for k in a.kinds.split(",")]
	ref = {}
	for path in glob.glob(f"{a.reference}/results.jsonl"):
		for line in open(path):
			r = json.loads(line)
			if r["geometry"] == "line":
				ref[(r["group"], r["kind"])] = r
	t0 = time.time()
	with ProcessPoolExecutor(a.jobs) as ex:
		rows = [r for r in ex.map(job, jobs) if r]
	if a.out:
		with open(a.out, "w") as fh:
			for r in rows:
				fh.write(json.dumps(r) + "\n")

	print(f"{'set kind':10s} {'groups':>6s} {'engine exact':>13s} {'python exact':>13s} {'both':>5s} {'only engine':>11s} {'only python':>11s} "
		f"{'geom err mean/max engine':>25s} {'python':>14s}")
	keys = sorted({(r["group"][0], r["kind"]) for r in rows})
	for sk in keys:
		rs = [r for r in rows if (r["group"][0], r["kind"]) == sk]
		ee = {r["group"] for r in rs if r.get("wrong") == 0}
		pp = {r["group"] for r in rs if ref.get((r["group"], r["kind"]), {}).get("wrong") == 0}
		have = [r for r in rs if (r["group"], r["kind"]) in ref]
		ge = [v["geometry_error"] for r in rs for v in r["views"] if "geometry_error" in v and v["geometry_error"]["max"] is not None]
		gp = [v["geometry_error"] for r in have for v in ref[(r["group"], r["kind"])]["views"] if v.get("geometry_error", {}).get("max") is not None]
		print(f"{sk[0] + ' ' + sk[1]:10s} {len(rs):6d} {len(ee):6d}/{len(rs):<6d} {len(pp):6d}/{len(have):<6d} {len(ee & pp):5d} {len(ee - pp):11d} {len(pp - ee):11d} "
			f"{np.mean([g['mean'] for g in ge]):12.3f}/{max(g['max'] for g in ge):<12.3f} "
			f"{(np.mean([g['mean'] for g in gp]) if gp else float('nan')):6.3f}/{(max(g['max'] for g in gp) if gp else float('nan')):<6.3f}")
	ms = {k: [v["ms"][k] for r in rows for v in r["views"] if v["ok"]] for k in ("segments", "axes", "position", "sampling")}
	print("\nengine per view (ms): " + ", ".join(f"{k} mean {np.mean(x):.0f} p95 {np.percentile(x, 95):.0f}" for k, x in ms.items())
		+ f"; total mean {np.mean([sum(v['ms'].values()) for r in rows for v in r['views'] if v['ok']]):.0f}")
	bad = [r for r in rows if r.get("wrong") != 0]
	if bad:
		print("\nengine not exact:")
		for r in bad:
			p = ref.get((r["group"], r["kind"]), {})
			print(f"  {r['group']} {r['kind']:6s} wrong={r.get('wrong')} {r.get('error', '')} python wrong={p.get('wrong')} "
				f"view max err {[round(v['geometry_error']['max'], 2) if 'geometry_error' in v else v.get('error') for v in r['views']]}")
	print(f"\n{len(rows)} scenarios in {time.time() - t0:.0f} s")


if __name__ == "__main__":
	main()
