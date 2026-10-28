"""Per-frame detection on a real video (no labels): every frame (or every --step-th) detected on its own (wide), in
parallel, with the focal length estimated once from a few frames; per frame ok, lattice cost, support, segments,
timings; overlays for eyeballing.

	python engine/tools/real_video.py FRAMES_DIR [--step 1] [--jobs 12] [--fx F] [--work-width 1024] [--out DIR]
	                                   [--sheet N] [--extra key=value ...]

FRAMES_DIR: frame0001.png … (e.g. ffmpeg -i VID.mp4 -vsync 0 DIR/frame%04d.png; ffmpeg applies the rotation tag).
Writes OUT/results.json (per frame) and OUT/overlay/frameNNNN.jpg for every --sheet-th frame (sticker quads in their
measured colour, white = cost ≤ 0.15, red otherwise; segments by axis family), OUT/sheet-*.jpg contact sheets.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

FAMILY = [(60, 60, 255), (60, 255, 60), (255, 120, 60)]


def lab_to_bgr(lab):
	px = np.array([[lab]], np.float32)
	return tuple(int(c) for c in np.clip(cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0] * 255, 0, 255))


def overlay(img, r, good):
	out = img.copy()
	for s in r.get("segments", []):
		c = FAMILY[s["family"]] if s["family"] >= 0 else (150, 150, 150)
		cv2.line(out, (int(s["x1"]), int(s["y1"])), (int(s["x2"]), int(s["y2"])), c, 2 if s["family"] >= 0 else 1, cv2.LINE_AA)
	if r["ok"]:
		layer = out.copy()
		for s in r["samples"]:
			if s["valid"]:
				cv2.fillConvexPoly(layer, np.rint(s["quad"]).astype(np.int32), lab_to_bgr(s["lab"]))
		out = cv2.addWeighted(layer, 0.6, out, 0.4, 0)
		for s in r["samples"]:
			if s["valid"]:
				cv2.polylines(out, [np.rint(s["quad"]).astype(np.int32)], True, (255, 255, 255) if good else (40, 40, 255), 2, cv2.LINE_AA)
	return out


def job(args):
	path, fx, work_width, keep, extra = args
	import rubik_engine as re
	img = cv2.imread(path)
	h, w = img.shape[:2]
	K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
	t0 = time.perf_counter()
	r = re.Detector().detect(img, K, work_width=work_width, keep_segments=keep, **{k: int(v) for k, v in extra.items()})
	segs = r.pop("segments_list", None)
	ms = (time.perf_counter() - t0) * 1000
	out = {"frame": int(os.path.basename(path)[5:9]), "ok": r["ok"], "cost": r["lattice_cost"], "support": r.get("support", 0),
		"segments": r["segments"], "family": r["family_segments"], "ms": ms, "timings": r["ms"],
		"R": np.asarray(r["R"]).tolist() if r["ok"] else None, "t": np.asarray(r["t"]).tolist() if r["ok"] else None,
		"samples": [{"facelet": s["facelet"], "valid": s["valid"], "lab": list(s["lab"]), "quad": np.asarray(s["quad"]).tolist()} for s in r["samples"]] if r["ok"] else []}
	if keep:
		out["segments_list"] = [{"x1": float(s[0]), "y1": float(s[1]), "x2": float(s[2]), "y2": float(s[3]), "family": int(s[5])} for s in segs]
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("frames")
	ap.add_argument("--step", type=int, default=1)
	ap.add_argument("--jobs", type=int, default=12)
	ap.add_argument("--fx", type=float, default=0)
	ap.add_argument("--work-width", type=int, default=1024)
	ap.add_argument("--sheet", type=int, default=30, help="overlay every N-th processed frame")
	ap.add_argument("--good-cost", type=float, default=0.15)
	ap.add_argument("--effort", default="", help="detect options, e.g. starts=16,joint=5,shift_pairs=1")
	ap.add_argument("--out", default=None)
	a = ap.parse_args()
	effort = {k: float(v) for k, v in (x.split("=") for x in a.effort.split(",") if x)}
	import rubik_engine as re
	files = sorted(f for f in os.listdir(a.frames) if f.startswith("frame") and f.endswith(".png"))[::a.step]
	out = a.out or os.path.join(a.frames, "eval")
	os.makedirs(os.path.join(out, "overlay"), exist_ok=True)
	fx = a.fx
	if not fx:
		picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, 4)]
		f = re.Detector().estimate_focal([cv2.imread(os.path.join(a.frames, p)) for p in picks])
		fx = f["camera"][0]
		print(f"focal: fx {fx:.0f} from {picks} (support {f['support']:.2f})")
	keep = set(files[::a.sheet])
	t0 = time.time()
	with ProcessPoolExecutor(a.jobs) as ex:
		rows = list(ex.map(job, [(os.path.join(a.frames, f), fx, a.work_width, f in keep, effort) for f in files], chunksize=4))
	print(f"{len(rows)} frames in {time.time() - t0:.0f} s")
	for r in rows:
		if "segments_list" in r:
			img = cv2.imread(os.path.join(a.frames, f"frame{r['frame']:04d}.png"))
			o = overlay(img, {"ok": r["ok"], "samples": [dict(s, quad=np.array(s["quad"])) for s in r["samples"]], "segments": r.pop("segments_list")},
				r["ok"] and r["cost"] <= a.good_cost)
			cv2.putText(o, f"{r['frame']} cost {r['cost']:.3f} sup {r['support']:.2f}", (10, 40), 0, 1.1, (0, 255, 255), 3)
			cv2.imwrite(os.path.join(out, "overlay", f"frame{r['frame']:04d}.jpg"), o, [cv2.IMWRITE_JPEG_QUALITY, 80])
	json.dump({"fx": fx, "frames": rows}, open(os.path.join(out, "results.json"), "w"))
	ok = [r for r in rows if r["ok"]]
	good = [r for r in ok if r["cost"] <= a.good_cost]
	costs = np.array([r["cost"] for r in ok])
	def extent(r):
		Rm, t = np.array(r["R"]), np.array(r["t"])
		h, w = cv2.imread(os.path.join(a.frames, files[0])).shape[:2]
		Kc = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
		pts = [Kc @ (Rm @ np.array([x, y, z]) + t) for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)]
		if min(p[2] for p in pts) <= 0:
			return 9
		uv = np.array([p[:2] / p[2] for p in pts])
		return max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / max(w, h)
	print(f"accepted (cost ≤ {a.good_cost}) with a projected extent > 0.9 (oversized lattice): {sum(extent(r) > 0.9 for r in good)}")
	print(f"ok {len(ok)}/{len(rows)}, cost ≤ {a.good_cost}: {len(good)}/{len(rows)} ({len(good) / len(rows):.0%}); "
		f"cost p25/50/75 {np.percentile(costs, 25):.3f}/{np.median(costs):.3f}/{np.percentile(costs, 75):.3f}; "
		f"ms mean {np.mean([r['ms'] for r in rows]):.0f}")
	print(f"→ {out}")


if __name__ == "__main__":
	main()
