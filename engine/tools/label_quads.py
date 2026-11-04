"""Hand labels of facelet colours on real frames, and the reading of each sample quad by its median vs by the colour
modes inside it (in-block clustering), against those labels.

	python engine/tools/label_quads.py select OUT_DIR [--real DIR] [--dumps DIR] [--top 24] [--random 6]
	python engine/tools/label_quads.py tiles OUT_DIR [--frames A,B,...]
	python engine/tools/label_quads.py evaluate OUT_DIR

select: detects the cube on real frames (every 5th of the video, all dumps), keeps good fits, ranks the video frames
by their two-mode quads (at least 1 s apart) and takes the top ones plus a few at random; per frame it writes
FRAME.json (source, fx, quads) and FRAME.png for labelling: the crop as is, and with the R / U / F faces' quads outlined
(cyan / white / magenta) and numbered 0–8 in facelet order.

tiles: per pair of frames a sheet (OUT_DIR/sheet-NN.png): every sample quad on its own, its neighbourhood warped to a
square tile with the quad outlined, in rows R / U / F of 0–8 — labelling the tiles one by one, not by the numbers on
the crop (that misread positions).

labels: OUT_DIR/labels.json, {frame: {"R": "9 letters", "U": ..., "F": ...}} in facelet order, letters W Y O M B G
(M: the red role, maroon on these cubes), x a sticker mostly covered (finger), ? unsure.

evaluate: per labelled quad its median, and its modes (2-means over the quad's pixels in (L/2, a, b): two modes if
their centres are ≥ split apart and the smaller holds ≥ minor of the pixels). Colour references per frame from the
other labelled single-mode quads of that colour (the engine's colour feature), else the source's. Readings: median;
major mode; glare dropped (a mode ≥ 15 L brighter with under half the other's chroma is left out); and the share of
the quad given to the true colour (modes whose nearest reference is it, by pixel share).
"""
import argparse, glob, json, os, random, sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import rubik_engine as re  # noqa: E402

REAL = os.path.expanduser("~/data/rubik-renderer/real/vid225115")
DUMPS = os.path.join(HERE, "..", "..", "dump.local")
FACES = {"R": (9, (255, 255, 0)), "U": (27, (255, 255, 255)), "F": (45, (255, 0, 255))}
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
COLOURS = "WYOMBG"


def dump_fx(name):
	# per session: its focal length (the 11-01 session's estimate was off; the lens's value)
	stamp = int(name.split("-")[1][:7])
	return {1793199: 1041, 1793371: 1037, 1793523: 1004, 1793759: 964}.get(stamp, 1037)


def to_lab(img):
	return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def quad_pixels(lab, quad):
	H, W = lab.shape[:2]
	m = np.zeros((H, W), np.uint8)
	cv2.fillPoly(m, [np.asarray(quad).astype(np.int32)], 1)
	return lab[m == 1].reshape(-1, 3)


def modes(px, split, minor, k=2):
	if len(px) < 8:
		return [(1.0, np.median(px, axis=0))]
	X = np.column_stack([px[:, 0] * 0.5, px[:, 1], px[:, 2]]).astype(np.float32)
	_, lab_k, c = cv2.kmeans(X, k, None, CRIT, 3, cv2.KMEANS_PP_CENTERS)
	lab_k = lab_k.ravel()
	share = np.bincount(lab_k, minlength=k) / len(X)
	order = np.argsort(-share)
	kept = [order[0]]
	for i in order[1:]:
		if share[i] >= minor and all(np.linalg.norm(c[i] - c[j]) >= split for j in kept):
			kept.append(i)
	if len(kept) == 1:
		return [(1.0, np.median(px, axis=0))]
	total = sum(share[i] for i in kept)
	return [(float(share[i] / total), np.median(px[lab_k == i], axis=0)) for i in kept]


def feature(lab):
	# the engine's colour feature (imaging.cpp weigh): hue direction, log chroma, lightness
	L, a, b = lab
	C = np.hypot(a, b)
	h = np.arctan2(b, a)
	sat = min(C / 25, 1)
	return np.array([60 * sat * np.cos(h), 60 * sat * np.sin(h), 18 * np.log1p(C), 0.3 * L])


def detect(img, fx):
	h, w = img.shape[:2]
	v = re.Detector().detect(img, [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
	ok = v["ok"] and v["lattice_cost"] <= 0.15 and v["presence"] >= 0.5
	return v if ok else None


def panel(img, v, path):
	quads = [s["quad"] for s in v["samples"]]
	pts = np.concatenate(quads)
	x0, y0 = pts.min(0)
	x1, y1 = pts.max(0)
	pad = 0.25 * max(x1 - x0, y1 - y0)
	H, W = img.shape[:2]
	x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
	x1, y1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
	crop = img[y0:y1, x0:x1]
	s = 560 / max(crop.shape[:2])
	clean = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
	marked = clean.copy()
	for smp in v["samples"]:
		k = smp["facelet"]
		face = next(f for f, (base, _) in FACES.items() if base <= k < base + 9)
		q = ((smp["quad"] - [x0, y0]) * s).astype(np.int32)
		cv2.polylines(marked, [q], True, (0, 0, 0), 3)
		cv2.polylines(marked, [q], True, FACES[face][1], 1)
		cx, cy = q.mean(0).astype(int)
		cv2.putText(marked, str(k % 9), (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
		cv2.putText(marked, str(k % 9), (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FACES[face][1], 1)
	cv2.putText(marked, "R cyan  U white  F magenta", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
	cv2.putText(marked, "R cyan  U white  F magenta", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
	cv2.imwrite(path, np.hstack([clean, marked]))


def select(a):
	os.makedirs(a.out, exist_ok=True)
	files = sorted(glob.glob(os.path.join(a.real, "frame*.png")))[::5]
	scored = []
	for f in files:
		img = cv2.imread(f)
		v = detect(img, 1070)
		if v is None:
			continue
		lab = to_lab(img)
		two = sum(len(modes(quad_pixels(lab, s["quad"]), 15, 0.15)) == 2 for s in v["samples"] if s["valid"])
		scored.append((two, f))
	index = {f: i for i, f in enumerate(sorted(glob.glob(os.path.join(a.real, "frame*.png"))))}
	chosen = []
	for two, f in sorted(scored, reverse=True):
		if len(chosen) < a.top and all(abs(index[f] - index[g]) >= 30 for g in chosen):
			chosen.append(f)
	rest = [f for _, f in scored if f not in chosen]
	chosen += random.Random(1).sample(rest, min(a.random, len(rest)))
	jobs = [(f, 1070, "real") for f in sorted(chosen)] + [(f, dump_fx(os.path.basename(f)), "dump") for f in sorted(glob.glob(os.path.join(a.dumps, "frame-*.png")))]
	n = 0
	for f, fx, source in jobs:
		img = cv2.imread(f)
		v = detect(img, fx)
		if v is None:
			continue
		name = ("real-" if source == "real" else "") + os.path.splitext(os.path.basename(f))[0]
		json.dump({"source": f, "fx": fx, "kind": source, "quads": {str(s["facelet"]): s["quad"].tolist() for s in v["samples"] if s["valid"]}},
			open(os.path.join(a.out, name + ".json"), "w"))
		panel(img, v, os.path.join(a.out, name + ".png"))
		n += 1
	print(f"{n} frames written to {a.out} ({len(scored)} video frames with a good fit scanned)")


def tile(img, quad, size=96, context=1.7):
	q = np.asarray(quad, np.float32)
	c = q.mean(0)
	src = (c + (q - c) * context).astype(np.float32)
	dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
	M = cv2.getPerspectiveTransform(src, dst)
	t = cv2.warpPerspective(img, M, (size, size))
	inner = cv2.perspectiveTransform(q[None], M)[0].astype(np.int32)
	cv2.polylines(t, [inner], True, (0, 0, 0), 2)
	cv2.polylines(t, [inner], True, (255, 255, 255), 1)
	return t


def tiles(a):
	names = a.frames.split(",") if a.frames else sorted(os.path.splitext(f)[0] for f in os.listdir(a.out) if f.endswith(".json") and f != "labels.json")
	size, head = 96, 18
	blocks = []
	for name in names:
		meta = json.load(open(os.path.join(a.out, name + ".json")))
		img = cv2.imread(meta["source"])
		rows = []
		for face, (base, colour) in FACES.items():
			row = []
			for i in range(9):
				q = meta["quads"].get(str(base + i))
				t = tile(img, q, size) if q is not None else np.zeros((size, size, 3), np.uint8)
				cap = np.full((head, size, 3), 32, np.uint8)
				cv2.putText(cap, f"{face}{i}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
				row.append(np.vstack([cap, t]))
			rows.append(np.hstack(row))
		title = np.full((22, rows[0].shape[1], 3), 0, np.uint8)
		cv2.putText(title, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
		blocks.append(np.vstack([title] + rows))
	for n in range(0, len(blocks), 2):
		cv2.imwrite(os.path.join(a.out, f"sheet-{n // 2:02d}.png"), np.vstack(blocks[n:n + 2]))
	print(f"{(len(blocks) + 1) // 2} sheets: " + ", ".join(f"sheet-{n // 2:02d} = {names[n]}" + (f" + {names[n + 1]}" if n + 1 < len(names) else "") for n in range(0, len(names), 2)))


def evaluate(a):
	labels = json.load(open(os.path.join(a.out, "labels.json")))
	rows = []   # (frame, kind, true letter, px)
	for name, faces in labels.items():
		meta = json.load(open(os.path.join(a.out, name + ".json")))
		lab = to_lab(cv2.imread(meta["source"]))
		for face, (base, _) in FACES.items():
			for i, letter in enumerate(faces[face]):
				q = meta["quads"].get(str(base + i))
				if q is not None:
					rows.append((name, meta["kind"], letter, base + i, quad_pixels(lab, q)))
	print(f"{len(labels)} frames, {len(rows)} labelled quads: " + ", ".join(f"{c} {sum(r[2] == c for r in rows)}" for c in COLOURS + "x?"))
	for split, minor, k in [(15, 0.15, 2), (10, 0.15, 2), (20, 0.15, 2), (15, 0.10, 2), (15, 0.25, 2), (15, 0.15, 3)]:
		data = []
		for name, kind, letter, k_, px in rows:
			data.append((name, kind, letter, k_, np.median(px, axis=0), modes(px, split, minor, k)))
		# references: per frame and colour, the other labelled single-mode quads' medians; else the source's
		def refs(exclude):
			out = {}
			for scope in ("frame", "kind"):
				for c in COLOURS:
					key = exclude[0] if scope == "frame" else exclude[1]
					fs = [feature(d[4]) for d in data if d[2] == c and len(d[5]) == 1 and (d[0] if scope == "frame" else d[1]) == key and d[:4] != exclude[:4]]
					if fs and c not in out:
						out[c] = np.median(fs, axis=0)
			return out
		counts = {"two": 0, "med": 0, "major": 0, "glare": 0, "share": 0.0, "n": 0, "help": 0, "hurt": 0, "med2": 0, "major2": 0, "glare2": 0, "x2": 0, "x": 0}
		for d in data:
			name, kind, letter, k_, med, ms = d
			if letter == "x":
				counts["x"] += 1
				counts["x2"] += len(ms) > 1
				continue
			if letter not in COLOURS:
				continue
			R = refs(d)
			near = lambda lab: min(R, key=lambda c: np.linalg.norm(feature(lab) - R[c]))
			m_ok = near(med) == letter
			j_ok = near(ms[0][1]) == letter
			kept = ms
			if len(ms) == 2:
				(s0, l0), (s1, l1) = ms
				c0, c1 = np.hypot(l0[1], l0[2]), np.hypot(l1[1], l1[2])
				if l1[0] > l0[0] + 15 and c1 < 0.5 * c0:
					kept = [ms[0]]
				elif l0[0] > l1[0] + 15 and c0 < 0.5 * c1:
					kept = [ms[1]]
			g_ok = near(max(kept, key=lambda t: t[0])[1]) == letter
			share = sum(sh for sh, l in ms if near(l) == letter)
			counts["n"] += 1
			counts["med"] += m_ok
			counts["major"] += j_ok
			counts["glare"] += g_ok
			counts["share"] += share
			if len(ms) > 1:
				counts["two"] += 1
				counts["med2"] += m_ok
				counts["major2"] += j_ok
				counts["glare2"] += g_ok
				counts["help"] += (not m_ok) and g_ok
				counts["hurt"] += m_ok and not g_ok
		c = counts
		print(f"  split {split} minor {minor} k {k}: two-mode {c['two']}/{c['n']} ({100 * c['two'] / c['n']:.1f} %); read right — median {c['med']} ({100 * c['med'] / c['n']:.2f} %), "
			f"major {c['major']}, glare dropped {c['glare']} ({100 * c['glare'] / c['n']:.2f} %); on two-mode quads median {c['med2']}, major {c['major2']}, glare dropped {c['glare2']} "
			f"(helped {c['help']}, hurt {c['hurt']}); true-colour share {c['share'] / c['n']:.3f}; covered (x) quads split {c['x2']}/{c['x']}")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("command", choices=["select", "tiles", "evaluate"])
	ap.add_argument("out")
	ap.add_argument("--real", default=REAL)
	ap.add_argument("--dumps", default=DUMPS)
	ap.add_argument("--top", type=int, default=24)
	ap.add_argument("--random", type=int, default=6)
	ap.add_argument("--frames", default="")
	a = ap.parse_args()
	{"select": select, "tiles": tiles, "evaluate": evaluate}[a.command](a)


if __name__ == "__main__":
	main()
