"""Per-frame accuracy of the engine on rendered videos (true poses and states): every --step-th frame detected wide on
its own at the true focal length, compared with its labels.

	python engine/tools/video_accuracy.py [--videos v103,v104,v105,v106] [--step 2] [--jobs 12]

Per frame: accepted (as the app: fit ok, cost ≤ 0.15, presence ≥ 0.5); located (each visible sticker's sampled quad
centre within 0.3 cubie of its true centre, the faces matched to the true pose up to a renaming); per sample the true
colour, read as the nearest of per-colour references (the median feature of the low-glare samples of that colour, the
engine's colour feature). By phase (hold / reorient / twist / settle) and, for the studio videos, by glare share.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

VIDEOS = os.path.expanduser("~/data/rubik-renderer/videos")


def frame_rows(args):
	name, i, fx = args
	import rubik_engine as re
	from rubik_detector import cube
	from placement_check import true_placement
	d = frame_rows.labels.setdefault(name, None) or json.load(open(os.path.join(VIDEOS, name, "labels.json")))
	frame_rows.labels[name] = d
	view = d["views"][i]
	files = frame_rows.files.setdefault(name, sorted(f for f in os.listdir(os.path.join(VIDEOS, name)) if f.startswith("frame") and f[-4:] in (".png", ".jpg")))
	img = cv2.imread(os.path.join(VIDEOS, name, files[i]))
	h, w = img.shape[:2]
	det = frame_rows.det or re.Detector()
	frame_rows.det = det
	v = det.detect(img, [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
	accepted = bool(v["ok"] and v["lattice_cost"] <= 0.15 and v["presence"] >= 0.5)
	out = {"phase": view.get("phase", "hold"), "accepted": accepted, "located": False, "err": None, "samples": []}
	if not v["ok"]:
		return out
	stickers = {s["index"]: s for s in view["stickers"]}
	angle, g = true_placement(np.array(view["rotation"]).reshape(3, 3), v["R"])
	# the pitch in pixels at the cube: fx / depth
	pitch = fx / view["translation"][2]
	errs = []
	for s in v["samples"]:
		k = cube.PERMS[g][s["facelet"]]
		st = stickers[k]
		if not st["visible"]:
			continue
		c = np.asarray(s["quad"]).mean(0)
		e = np.hypot(*(c - np.asarray(st["center"]))) / pitch
		errs.append(e)
		if s["valid"]:
			out["samples"].append((view["facelets"][k], [float(x) for x in s["lab"]], st.get("glare", {}).get("share", 0.0), float(e)))
	out["err"] = float(np.max(errs)) if errs and angle <= 30 else 9.9
	out["located"] = out["err"] < 0.3
	return out


frame_rows.labels = {}
frame_rows.files = {}
frame_rows.det = None


def feature(lab):
	L, a, b = lab
	C = np.hypot(a, b)
	h = np.arctan2(b, a)
	sat = min(C / 25, 1)
	return np.array([60 * sat * np.cos(h), 60 * sat * np.sin(h), 18 * np.log1p(C), 0.3 * L])


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--videos", default="v103,v104,v105,v106")
	ap.add_argument("--step", type=int, default=2)
	ap.add_argument("--jobs", type=int, default=12)
	a = ap.parse_args()
	for name in a.videos.split(","):
		d = json.load(open(os.path.join(VIDEOS, name, "labels.json")))
		fx = d["intrinsics"]["fx"]
		jobs = [(name, i, fx) for i in range(0, len(d["views"]), a.step)]
		with ProcessPoolExecutor(a.jobs) as pool:
			rows = list(pool.map(frame_rows, jobs, chunksize=4))
		studio = bool(d["render"].get("realism", {}).get("studio"))
		n = len(rows)
		acc = [r for r in rows if r["accepted"]]
		loc = [r for r in acc if r["located"]]
		wrong = [r for r in acc if not r["located"]]
		errs = [r["err"] for r in loc]
		print(f"{name}{' (studio)' if studio else ''}: {n} frames; accepted {len(acc)} ({100 * len(acc) / n:.0f} %), located right {len(loc)} ({100 * len(loc) / n:.0f} %), "
			f"accepted but off > 0.3 cubie {len(wrong)} ({100 * len(wrong) / max(1, len(acc)):.1f} % of accepted); worst sticker centre error on located frames p50 {np.median(errs):.3f} p90 {np.percentile(errs, 90):.3f}")
		for ph in ("hold", "reorient", "twist", "settle"):
			sel = [r for r in rows if r["phase"] == ph]
			if sel:
				print(f"    {ph:8s}: {len(sel):4d} frames, accepted {100 * sum(r['accepted'] for r in sel) / len(sel):3.0f} %, located {100 * sum(r['accepted'] and r['located'] for r in sel) / len(sel):3.0f} %, "
					f"accepted but off {sum(r['accepted'] and not r['located'] for r in sel)}")
		# colour: samples of accepted, located frames outside twists
		samples = [s for r in loc if r["phase"] != "twist" for s in r["samples"]]
		ref = {}
		for c in "LRDUBF":
			fs = [feature(l) for t, l, gl, e in samples if t == c and gl < 0.1]
			if fs:
				ref[c] = np.median(fs, axis=0)
		near = lambda lab: min(ref, key=lambda c: np.linalg.norm(feature(lab) - ref[c]))
		right = sum(near(l) == t for t, l, gl, e in samples)
		print(f"    colour: {len(samples)} samples on located frames, nearest reference right {100 * right / len(samples):.2f} %")
		if studio:
			for lo, hi in ((0, 0.1), (0.1, 0.6), (0.6, 1.01)):
				sel = [(t, l) for t, l, gl, e in samples if lo <= gl < hi]
				if sel:
					print(f"      glare {lo:.1f}–{hi:.1f}: {len(sel):5d} samples, right {100 * sum(near(l) == t for t, l in sel) / len(sel):5.1f} %")


if __name__ == "__main__":
	main()
