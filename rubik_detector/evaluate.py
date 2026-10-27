"""Evaluate the pipeline over labelled groups.

	python -m rubik_detector.evaluate [--kinds render,bg,hand] [--set g|n] [--groups g01,g02 | --limit N] [--geometry line,oracle]
	                                  [--jobs 8] [--out runs/NAME] [--debug fail|all|none] [--palette odd|random]
	                                  [--classifier legal,balanced]

Per scenario (group × kind × geometry): wrong facelets up to a whole-cube rotation (0 = solved exactly), per view the
sticker centre error in cubie units, timings. Writes OUT/results.jsonl, OUT/summary.txt and debug overlays
(OUT/debug/<group>-<kind>-<geometry>-view<v>.jpg: truth centres green, predicted sampling quads white, segments by
axis family red / green / blue, unused grey).
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from . import colors, cube, data, pipeline


# non-standard sticker colours for --palette (sRGB): close pairs on purpose — pink/red-ish, two blues of different
# lightness, purple, teal, brown
PALETTES = {
	"odd": ["#ff69b4", "#8a2be2", "#20b2aa", "#8b5a2b", "#87ceeb", "#1f3a93"],
}


def palette_for(name, group):
	"""A named palette, or "random": six colours drawn per group (seeded by it), any hue, saturation 0.35–1,
	value 0.35–1, pairwise ΔE76 ≥ 28 in the lit colour — distinguishable on a lit face, close enough to be hard."""
	if name != "random":
		return PALETTES[name]
	rng = np.random.default_rng(int(group[1:]) + 1000)
	chosen, labs = [], []
	while len(chosen) < 6:
		hsv = np.array([[[rng.uniform(0, 180), rng.uniform(0.35, 1) * 255, rng.uniform(0.35, 1) * 255]]], np.uint8)
		bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
		lab = colors.to_lab(bgr[None, None])[0, 0]
		if all(np.linalg.norm(lab - o) >= 28 for o in labs):
			chosen.append("#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0]))
			labs.append(lab)
	return chosen


def recolour(s, palette):
	# Pixel repaint of a render, kept for quick checks only: its shading comes partly from the original sticker colour
	# (a once-white and a once-blue sticker of one face end up differently bright), coloured light casts are lost and
	# stickerless rounded edges keep the old colour. Real non-standard schemes are rendered: --set n.
	"""Paint the six sticker colours of a scenario with a palette (face letter order LRDUBF), keeping the render's
	shading: per sticker, every pixel's Lab offset from the sticker's median is added to the new colour (lightness
	and chroma offsets scaled with the colour). Only the truth sticker quads change."""
	targets = colors.to_lab(np.array([[[int(c[5:7], 16), int(c[3:5], 16), int(c[1:3], 16)] for c in palette]], np.uint8))[0]
	for view in s.views:
		lab = colors.to_lab(view.image)
		out = lab.copy()
		for k, q in view.quads.items():
			mask = np.zeros(lab.shape[:2], np.uint8)
			cv2.fillConvexPoly(mask, np.rint(q * 4).astype(np.int32), 1, cv2.LINE_8, 2)
			sel = mask > 0
			if sel.sum() < 4:
				continue
			px = lab[sel]
			med = np.median(px, axis=0)
			# the face's lighting: ratio of this sticker's lightness to its colour's lit lightness is unknown here, so
			# keep the sticker's relative brightness by scaling the new colour's lightness with med L / 60
			tgt = targets[cube.FACES.index(s.state[k])].copy()
			f = med[0] / 60.0
			tgt[0] = np.clip(tgt[0] * f, 0, 100)
			tgt[1:] *= min(1.2, f)
			out[sel] = tgt + (px - med)
		out[..., 0] = np.clip(out[..., 0], 0, 100)
		view.image = np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR) * 255, 0, 255).astype(np.uint8)
	return s


def _job(args):
	group, kind, geom, width, debug_dir, debug, palette, classifier = args
	s = data.load(group, kind, width)
	if s is None:
		return None
	if palette:
		s = recolour(s, palette_for(palette, group))
	t0 = time.perf_counter()
	r = pipeline.run(s, geom, classifier)
	r["seconds"] = time.perf_counter() - t0
	results = r.pop("results")
	if debug == "all" or (debug == "fail" and r.get("wrong") != 0):
		for v, (view, g) in enumerate(zip(s.views, results)):
			cv2.imwrite(f"{debug_dir}/{group}-{kind}-{geom}{'-' + palette if palette else ''}-view{v + 1}.jpg", draw(view, g))
	if palette:
		r["kind"] += "+" + palette
	return r


def draw(view, g):
	img = (view.image * 0.6).astype(np.uint8)
	if g is not None:
		dbg = g.get("debug", {})
		if "segments" in dbg:
			colours = [(60, 60, 230), (60, 200, 60), (230, 120, 40)]
			for s, f in zip(dbg["segments"], dbg["families"]):
				c = colours[f] if f >= 0 else (110, 110, 110)
				cv2.line(img, tuple(np.rint(s[:2]).astype(int)), tuple(np.rint(s[2:]).astype(int)), c, 1, cv2.LINE_AA)
		for q in g["quads"].values():
			cv2.polylines(img, [np.rint(q * 4).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA, shift=2)
	for k, q in view.quads.items():
		c = np.mean(q, axis=0)
		cv2.circle(img, tuple(np.rint(c).astype(int)), 3, (0, 0, 255) if k in view.occluded else (0, 255, 0), -1)
	return img


def summarize(rows):
	lines = []
	keys = sorted({(r["kind"], r["geometry"], r.get("classifier", "")) for r in rows})
	lines.append(f"{'kind':12s} {'geometry':8s} {'classifier':10s} {'groups':>6s} {'exact':>6s} {'mean wrong':>10s} {'failed':>6s} "
		f"{'flagged':>8s} {'centre err (cubie) mean/p95/max':>32s} {'s/group':>8s}")
	for kind, geom, cl in keys:
		rs = [r for r in rows if (r["kind"], r["geometry"], r.get("classifier", "")) == (kind, geom, cl)]
		ok = [r for r in rs if r.get("wrong") is not None]
		errs = [v["geometry_error"]["max"] for r in rs for v in r["views"] if v.get("geometry_error", {}).get("max") is not None]
		means = [v["geometry_error"]["mean"] for r in rs for v in r["views"] if v.get("geometry_error", {}).get("mean") is not None]
		# flagged: results the classifier itself marks unsure (a facelet with posterior < 0.5); shown as
		# flagged among wrong / flagged among exact
		flag = lambda r: r.get("classification", {}).get("min_posterior", 1) < 0.5
		fw = sum(flag(r) for r in ok if r["wrong"]); nw = sum(1 for r in ok if r["wrong"])
		fe = sum(flag(r) for r in ok if not r["wrong"]); ne = len(ok) - nw
		lines.append(f"{kind:12s} {geom:8s} {cl:10s} {len(rs):6d} {ne:6d} "
			f"{(np.mean([r['wrong'] for r in ok]) if ok else float('nan')):10.2f} {len(rs) - len(ok):6d} "
			f"{f'{fw}/{nw} {fe}/{ne}':>8s} "
			f"{(np.mean(means) if means else float('nan')):10.3f} {(np.percentile(errs, 95) if errs else float('nan')):10.3f} "
			f"{(max(errs) if errs else float('nan')):10.3f} {np.mean([r['seconds'] for r in rs]):8.2f}")
	lines.append("flagged = min facelet posterior < 0.5: among wrong results / among exact ones (legal classifier)")
	bad = [r for r in rows if r.get("wrong") != 0]
	if bad:
		lines.append("\nnot exact:")
		for r in sorted(bad, key=lambda r: (r["kind"], r["geometry"], r.get("classifier", ""), r["group"])):
			ge = [round(v["geometry_error"]["max"], 2) if v.get("geometry_error", {}).get("max") is not None else v.get("error") for v in r["views"]]
			c = r.get("classification", {})
			lines.append(f"  {r['group']} {r['kind']:10s} {r['geometry']:6s} {r.get('classifier', ''):8s} wrong={r.get('wrong')} "
				f"{r.get('error', '')} min post {c.get('min_posterior', float('nan')):.2f} view max err {ge}")
	return "\n".join(lines)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--kinds", default="render,bg,hand")
	ap.add_argument("--groups", default=None)
	ap.add_argument("--set", default="g", help="sample set: g (standard colour schemes) or n (non-standard)")
	ap.add_argument("--limit", type=int, default=None)
	ap.add_argument("--geometry", default="line,oracle")
	ap.add_argument("--width", type=int, default=1024)
	ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
	ap.add_argument("--out", default=None)
	ap.add_argument("--debug", default="fail", choices=("fail", "all", "none"))
	ap.add_argument("--palette", default=None, choices=sorted(PALETTES) + ["random"], help="repaint the stickers (colour-scheme stress test)")
	ap.add_argument("--classifier", default="legal", help="legal (probabilistic, legal states only) or balanced; comma list to compare")
	a = ap.parse_args()

	groups = a.groups.split(",") if a.groups else data.groups(a.set)[:a.limit]
	out = a.out or f"runs/{time.strftime('%m%d-%H%M%S')}"
	os.makedirs(f"{out}/debug", exist_ok=True)
	jobs = [(g, k, geom, a.width, f"{out}/debug", a.debug, a.palette, cl) for g in groups for k in a.kinds.split(",")
		for geom in a.geometry.split(",") for cl in a.classifier.split(",")]
	t0 = time.time()
	rows = []
	with ProcessPoolExecutor(a.jobs) as ex, open(f"{out}/results.jsonl", "w") as fh:
		for r in ex.map(_job, jobs):
			if r is None:
				continue
			rows.append(r)
			fh.write(json.dumps(r) + "\n")
	text = summarize(rows) + f"\n\n{len(rows)} scenarios in {time.time() - t0:.0f} s, {a.jobs} jobs"
	open(f"{out}/summary.txt", "w").write(text + "\n")
	print(text)
	print(f"\n→ {out}")


if __name__ == "__main__":
	main()
