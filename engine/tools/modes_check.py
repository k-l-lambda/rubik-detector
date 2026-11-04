"""Colour modes inside a facelet's sample quad: how often a quad holds two colours (a highlight, a finger, a seam), and
whether the area-weighted modes would read the sticker better than the quad's median. On rendered videos (true state
and poses known) and, prevalence only, on real frames.

	python engine/tools/modes_check.py [--videos v103,v104] [--step 6] [--real DIR,...]

Modes: 2-means over the quad's pixels in (L/2, a, b); two modes if their centres are ≥ --split apart and the smaller
holds ≥ --minor of the pixels, else one (the median). Per colour letter the reference is the median of the medians of
the samples that truly show it. A sample reads right if its median's nearest reference is the true colour; with modes,
if the larger mode's is ("major"), or if any mode of ≥ 0.3 share is ("any").
"""
import argparse, glob, json, os, sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

import rubik_engine as re  # noqa: E402
from rubik_detector import cube  # noqa: E402
from placement_check import VIDEOS, true_placement  # noqa: E402

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)


def feature(lab):
	lab = np.asarray(lab, float)
	return np.column_stack([lab[..., 0] * 0.5, lab[..., 1], lab[..., 2]]) if lab.ndim == 2 else np.array([lab[0] * 0.5, lab[1], lab[2]])


def modes(lab_img, quad, split, minor):
	H, W = lab_img.shape[:2]
	m = np.zeros((H, W), np.uint8)
	cv2.fillPoly(m, [np.asarray(quad).astype(np.int32)], 1)
	px = lab_img[m == 1].reshape(-1, 3)
	if len(px) < 8:
		return [(1.0, np.median(px, axis=0) if len(px) else np.zeros(3))]
	X = feature(px).astype(np.float32)
	_, lab_k, c = cv2.kmeans(X, 2, None, CRIT, 3, cv2.KMEANS_PP_CENTERS)
	lab_k = lab_k.ravel()
	share = np.bincount(lab_k, minlength=2) / len(X)
	if np.linalg.norm(c[0] - c[1]) >= split and share.min() >= minor:
		return sorted(((float(share[i]), np.median(px[lab_k == i], axis=0)) for i in range(2)), key=lambda t: -t[0])
	return [(1.0, np.median(px, axis=0))]


def to_lab(img):
	return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def video_samples(name, step, split, minor):
	d = json.load(open(os.path.join(VIDEOS, name, "labels.json")))
	views = d["views"]
	fx = d["intrinsics"]["fx"]
	files = sorted(f for f in os.listdir(os.path.join(VIDEOS, name)) if f.startswith("frame") and f[-4:] in (".png", ".jpg"))
	det = re.Detector()
	out = []   # (true letter, median, modes)
	for i in range(0, len(files), step):
		img = cv2.imread(os.path.join(VIDEOS, name, files[i]))
		h, w = img.shape[:2]
		v = det.detect(img, [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
		if not (v["ok"] and v["lattice_cost"] <= 0.09 and v["presence"] >= 0.5) or views[i].get("phase") in ("twist",):
			continue
		angle, g = true_placement(np.array(views[i]["rotation"]).reshape(3, 3), v["R"])
		if angle > 10:
			continue
		truth = views[i]["facelets"]
		lab = to_lab(img)
		for s in v["samples"]:
			if s["valid"]:
				ms = modes(lab, s["quad"], split, minor)
				m = np.zeros((h, w), np.uint8)
				cv2.fillPoly(m, [s["quad"].astype(np.int32)], 1)
				out.append((truth[cube.PERMS[g][s["facelet"]]], np.median(lab[m == 1].reshape(-1, 3), axis=0), ms))
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--videos", default="v103,v104")
	ap.add_argument("--step", type=int, default=6)
	ap.add_argument("--split", type=float, default=15.0)
	ap.add_argument("--minor", type=float, default=0.15)
	ap.add_argument("--real", default="")
	a = ap.parse_args()
	for name in a.videos.split(","):
		rows = video_samples(name, a.step, a.split, a.minor)
		ref = {c: np.median([feature(m) for t, m, _ in rows if t == c], axis=0) for c in cube.FACES}
		nearest = lambda lab: min(cube.FACES, key=lambda c: np.linalg.norm(feature(lab) - ref[c]))
		two = [r for r in rows if len(r[2]) == 2]
		med_ok = sum(nearest(m) == t for t, m, _ in rows)
		major_ok = sum(nearest(ms[0][1]) == t for t, _, ms in rows)
		any_ok = sum(any(nearest(l) == t for sh, l in ms if sh >= 0.3) for t, _, ms in rows)
		print(f"{name}: {len(rows)} samples, two-mode quads {len(two)} ({100 * len(two) / max(1, len(rows)):.1f} %)")
		print(f"  read right: median {med_ok} ({100 * med_ok / len(rows):.2f} %), major mode {major_ok}, any mode ≥ 0.3 {any_ok}")
		if two:
			tm = sum(nearest(m) == t for t, m, _ in two)
			tj = sum(nearest(ms[0][1]) == t for t, _, ms in two)
			ta = sum(any(nearest(l) == t for sh, l in ms if sh >= 0.3) for t, _, ms in two)
			print(f"  on the two-mode quads: median {tm}/{len(two)}, major {tj}, any ≥ 0.3 {ta}")
			bright = sum(1 for _, _, ms in two if ms[1][1][0] > ms[0][1][0] + 15 or ms[0][1][0] > ms[1][1][0] + 15)
			print(f"    of which a mode ≥ 15 L brighter than the other: {bright}")
	for d in filter(None, a.real.split(",")):
		files = sorted(glob.glob(os.path.join(d, "*.png")))
		det = re.Detector()
		n = two = 0
		for f in files:
			img = cv2.imread(f)
			h, w = img.shape[:2]
			v = det.detect(img, [[1037, 0, w / 2], [0, 1037, h / 2], [0, 0, 1]])
			if not (v["ok"] and v["lattice_cost"] <= 0.15 and v["presence"] >= 0.5):
				continue
			lab = to_lab(img)
			for s in v["samples"]:
				if s["valid"]:
					n += 1
					two += len(modes(lab, s["quad"], a.split, a.minor)) == 2
		print(f"real {d}: {len(files)} frames, {n} samples, two-mode quads {two} ({100 * two / max(1, n):.1f} %)")


if __name__ == "__main__":
	main()
