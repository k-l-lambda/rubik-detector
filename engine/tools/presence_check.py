"""Measurements for a cube-presence confidence: the engine's detection on positives (a real video's frames, the
synthetic photos and their edits) and on negatives (scenes with no cube, ~/data/rubik-renderer/negatives, plus flips,
crops and small rotations of them), per view:

	cost        lattice cost (mean over the axis-family segments)
	support     share of all segment weight the lattice explains
	family      share of segment weight on the three axis families
	extent      projected cube outline / long image side
	faces       faces whose lattice is covered: of a face's 8 lines (4 per direction), at least 5 with a fitting
	            segment (cost < 0.25) on them — a grid on one plane (a notebook, tiles) covers one face only
	coverage    mean line coverage of the best two faces

	python engine/tools/presence_check.py [--video DIR] [--fx 1070] [--jobs 12] [--out FILE]
Writes the per-view table (JSON) and prints each feature's distribution per set.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

NEG = os.path.expanduser("~/data/rubik-renderer/negatives")


def line_coverage(R, t, K, segs, tol=0.35):
	"""per face (axis b, the face at +1.5 along it) the share of its 8 lattice lines with a fitting segment on them"""
	Ki = np.linalg.inv(K)
	tc = R.T @ t
	hit = np.zeros((3, 3, 4))   # face b, line direction a, line index
	for s in segs:
		a = int(s[5])
		if a < 0:
			continue
		q1, q2 = R.T @ (Ki @ [s[0], s[1], 1]), R.T @ (Ki @ [s[2], s[3], 1])
		best = (1.0, None, None)
		for b in range(3):
			if b == a:
				continue
			c = 3 - a - b
			out = []
			for q in (q1, q2):
				den = q[b] if abs(q[b]) > 1e-9 else 1e-9
				sc = (1.5 + tc[b]) / den
				out.append((sc, sc * q[a] - tc[a], sc * q[c] - tc[c]))
			(s1, pa1, pc1), (s2, pa2, pc2) = out
			pc = (pc1 + pc2) / 2
			inside = s1 > 0 and s2 > 0 and abs(pc) <= 1.6 and min(pa1, pa2) >= -1.65 and max(pa1, pa2) <= 1.65
			phase = abs((pc % 1.0) - 0.5)
			r = min(phase / tol, 1.0)
			cost = r * r if inside else 1.0
			if cost < best[0]:
				best = (cost, b, int(np.clip(np.floor(pc + 2.0), 0, 3)))
		if best[0] < 0.25:
			hit[best[1], a, best[2]] = 1
	faces = []
	for b in range(3):
		faces.append(sum(hit[b, a].sum() for a in range(3) if a != b) / 8)
	return faces


def measure(job):
	kind, name, path, K, transform = job
	import rubik_engine as re
	img = cv2.imread(path)
	if transform:
		img = apply(img, transform)
	h, w = img.shape[:2]
	K = np.array(K, float) if K is not None else None
	if K is None:
		fx = 1070 * max(w, h) / 1280
		K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
	r = re.Detector().detect(img, K, keep_segments=True)
	row = {"set": kind, "name": name + (f"/{transform}" if transform else ""), "ok": r["ok"], "cost": r["lattice_cost"], "support": r["support"],
		"presence": r["presence"]}
	segs = r["segments_list"]
	wsum = segs[:, 4].sum() if len(segs) else 0
	row["family"] = float(segs[segs[:, 5] >= 0, 4].sum() / wsum) if wsum > 0 else 0.0
	row["extent"], row["faces"], row["coverage"] = 9.0, 0, 0.0
	if r["ok"]:
		pts = [K @ (r["R"] @ np.array([x, y, z]) + r["t"]) for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)]
		if min(p[2] for p in pts) > 0:
			uv = np.array([p[:2] / p[2] for p in pts])
			row["extent"] = float(max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / max(w, h))
		cov = sorted(line_coverage(r["R"], r["t"], K, segs), reverse=True)
		row["faces"] = int(sum(c >= 5 / 8 for c in cov))
		row["coverage"] = float((cov[0] + cov[1]) / 2)
	return row


def apply(img, transform):
	h, w = img.shape[:2]
	if transform == "flip":
		return img[:, ::-1].copy()
	if transform.startswith("crop"):
		f = float(transform[4:])
		ch, cw = int(h * f), int(w * f)
		y0, x0 = (h - ch) // 2, (w - cw) // 2
		return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
	if transform.startswith("rot"):
		M = cv2.getRotationMatrix2D((w / 2, h / 2), float(transform[3:]), 1.0)
		return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
	raise ValueError(transform)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--video", default=os.path.expanduser("~/data/rubik-renderer/real/vid225115"))
	ap.add_argument("--video-step", type=int, default=2)
	ap.add_argument("--jobs", type=int, default=12)
	ap.add_argument("--out", default=os.path.expanduser("~/data/rubik-renderer/negatives/presence.json"))
	a = ap.parse_args()
	from rubik_detector import data
	jobs = []
	for f in sorted(os.listdir(NEG)):
		if f.endswith(".png"):
			for t in ("", "flip", "crop0.7", "crop0.5", "rot8", "rot-8"):
				jobs.append(("negative", f, os.path.join(NEG, f), None, t))
	frames = sorted(x for x in os.listdir(a.video) if x.startswith("frame") and x.endswith(".png"))[::a.video_step]
	jobs += [("video", f, os.path.join(a.video, f), None, "") for f in frames]
	for s in ("g", "n", "a"):
		for g in data.groups(s)[:20]:
			for kind in ("render", "bg", "hand"):
				smp = data.load(g, kind)
				if smp is None:
					continue
				for v, view in enumerate(smp.views):
					path = f"/tmp/presence-{g}-{kind}-{v}.png"
					cv2.imwrite(path, view.image)
					jobs.append(("photo-" + kind, f"{g}/{v}", path, view.K.tolist(), ""))
	with ProcessPoolExecutor(a.jobs) as ex:
		rows = list(ex.map(measure, jobs, chunksize=2))
	json.dump(rows, open(a.out, "w"), indent=0)
	sets = sorted({r["set"] for r in rows})
	for r in rows:
		r["prototype"] = (r["coverage"] - r["cost"]) * (1.0 if 0.15 <= r["extent"] <= 0.9 else 0.0)
	d = np.abs([r["presence"] - r["prototype"] for r in rows])
	print(f"engine presence vs this script's coverage − cost: max |diff| {d.max():.4f}, > 0.01 on {np.sum(d > 0.01)}/{len(rows)}")
	for s in sets:
		v = np.array([r["presence"] for r in rows if r["set"] == s])
		print(f"  presence {s:12s} n {len(v):4d}  min {v.min():5.2f}  p5 {np.percentile(v, 5):5.2f}  p50 {np.median(v):5.2f}  max {v.max():5.2f}  ≥ 0.5: {np.mean(v >= 0.5):.0%}")
	for key in ("cost", "support", "family", "extent", "coverage", "faces"):
		print(f"{key}:")
		for s in sets:
			v = np.array([r[key] for r in rows if r["set"] == s])
			print(f"  {s:12s} n {len(v):4d}  p5 {np.percentile(v, 5):6.2f}  p25 {np.percentile(v, 25):6.2f}  p50 {np.median(v):6.2f}  "
				f"p75 {np.percentile(v, 75):6.2f}  p95 {np.percentile(v, 95):6.2f}")
	print(f"→ {a.out}")


if __name__ == "__main__":
	main()
