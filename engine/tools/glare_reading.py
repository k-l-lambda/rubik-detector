"""Ways to read a sticker's colour from its quad's pixels under glare, against true colours: the rendered studio videos
(per sticker its glare share) and the hand-labelled real frames (label_quads.py).

	python engine/tools/glare_reading.py [--videos v105,v106] [--labels ~/data/rubik-renderer/real/labels] [--step 6]

Readings (all medians, of the quad's pixels):
	median      all pixels (the engine now)
	chroma P    the P % with the highest chroma (glare is pale: lightness up, chroma down)
	dark P      the P % with the lowest lightness
	glare-cut   pixels whose chroma ≥ a share of the quad's 90th-percentile chroma, or whose lightness ≤ the quad's
	            median + 10 (a white sticker keeps its pixels: all of low chroma)
A reading is right if its nearest reference is the true colour; references per source and colour from the median
readings of the low-glare samples (video) / of the other single-mode labelled quads (real), the engine's feature.
"""
import argparse, json, os, sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

import rubik_engine as re  # noqa: E402
from rubik_detector import cube  # noqa: E402
from placement_check import VIDEOS, true_placement  # noqa: E402
from label_quads import FACES, COLOURS, feature, quad_pixels, to_lab  # noqa: E402


def readings(px):
	L = px[:, 0]
	C = np.hypot(px[:, 1], px[:, 2])
	out = {"median": np.median(px, axis=0)}
	for p in (50, 30):
		k = max(3, int(len(px) * p / 100))
		out[f"chroma {p}"] = np.median(px[np.argsort(-C)[:k]], axis=0)
		out[f"dark {p}"] = np.median(px[np.argsort(L)[:k]], axis=0)
	c90 = np.percentile(C, 90)
	keep = (C >= 0.6 * c90) | (L <= np.median(L) + 10)
	out["glare-cut"] = np.median(px[keep], axis=0) if keep.sum() >= 3 else out["median"]
	return out


def video_rows(name, step):
	d = json.load(open(os.path.join(VIDEOS, name, "labels.json")))
	fx = d["intrinsics"]["fx"]
	det = re.Detector()
	rows = []
	for i in range(0, len(d["views"]), step):
		vw = d["views"][i]
		if vw.get("phase") == "twist":
			continue
		img = cv2.imread(os.path.join(VIDEOS, name, f"frame{i + 1:04d}.jpg"))
		h, w = img.shape[:2]
		v = det.detect(img, [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
		if not (v["ok"] and v["lattice_cost"] <= 0.15 and v["presence"] >= 0.5):
			continue
		angle, g = true_placement(np.array(vw["rotation"]).reshape(3, 3), v["R"])
		if angle > 10:
			continue
		lab = to_lab(img)
		st = {s["index"]: s for s in vw["stickers"]}
		for s in v["samples"]:
			k = cube.PERMS[g][s["facelet"]]
			if s["valid"] and st[k]["visible"]:
				rows.append((name, vw["facelets"][k], st[k].get("glare", {}).get("share", 0.0), readings(quad_pixels(lab, s["quad"]))))
	return rows


def real_rows(folder):
	labels = json.load(open(os.path.join(folder, "labels.json")))
	rows = []
	for name, faces in labels.items():
		meta = json.load(open(os.path.join(folder, name + ".json")))
		lab = to_lab(cv2.imread(meta["source"]))
		for face, (base, _) in FACES.items():
			for i, letter in enumerate(faces[face]):
				q = meta["quads"].get(str(base + i))
				if q is not None and letter in COLOURS:
					rows.append((name, letter, 0.0, readings(quad_pixels(lab, q))))
	return rows


def score(rows, method, per_frame):
	right = 0
	by_glare = {}
	for r in rows:
		src, letter, glare, rd = r
		refs = {}
		for c in set(x[1] for x in rows):
			pool = [x for x in rows if x[1] == c and x[2] < 0.1 and x is not r and (x[0] == src if per_frame else True)]
			if not pool and per_frame:
				pool = [x for x in rows if x[1] == c and x[2] < 0.1 and x is not r]
			if pool:
				refs[c] = np.median([feature(x[3]["median"]) for x in pool], axis=0)
		got = min(refs, key=lambda c: np.linalg.norm(feature(rd[method]) - refs[c]))
		ok = got == letter
		right += ok
		band = "none" if glare < 0.1 else ("partial" if glare < 0.6 else "full")
		b = by_glare.setdefault(band, [0, 0])
		b[0] += ok
		b[1] += 1
	return right / len(rows), {k: f"{100 * v[0] / v[1]:.1f} % of {v[1]}" for k, v in by_glare.items()}


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--videos", default="v105,v106")
	ap.add_argument("--labels", default=os.path.expanduser("~/data/rubik-renderer/real/labels"))
	ap.add_argument("--step", type=int, default=6)
	a = ap.parse_args()
	sets = {name: (video_rows(name, a.step), False) for name in a.videos.split(",")}
	if a.labels and os.path.exists(os.path.join(a.labels, "labels.json")):
		sets["real (hand labels)"] = (real_rows(a.labels), True)
	methods = list(sets[next(iter(sets))][0][0][3].keys())
	for name, (rows, per_frame) in sets.items():
		print(f"{name}: {len(rows)} samples")
		for m in methods:
			acc, bands = score(rows, m, per_frame)
			print(f"   {m:10s}: {100 * acc:6.2f} %   {bands}")


if __name__ == "__main__":
	main()
