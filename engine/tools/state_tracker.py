"""Application-layer prototype: continuous cube-state detection on a video, with the engine's real latency.

	python engine/tools/state_tracker.py VIDEO_DIR [--latency-scale 1] [--known-fov] [--out DIR]

Stages, in order, on the timeline of hud_video.py (frame k arrives at k / fps; the detector works on the newest
arrived frame whenever it is free; every step costs its measured time × --latency-scale):

	1. FOV: Detector::estimateFocal on the first frame (or --known-fov: the labels' K); fixed from then on
	2. evidence: tracked detections (local from the last good pose, wide fallback when the lattice cost passes
	   --fallback-cost). A detection becomes evidence only if it is
	     consistent — within --agree-deg of the previous good detection up to a renaming of the view frame, and its
	     27 colours, paired by that renaming, within --agree-colour (median ΔE) of the previous ones (a mid-turn fit
	     can have a low cost and still be wrong, but it rarely agrees with its neighbour; a turn onto a cube
	     symmetry agrees in angle, not in colour), and
	     new — fused with the views so far, it adds at least 9 facelets (a face not seen yet).
	   Kept views are placed in one cube frame by colour fusion, incrementally: each new view aligned to the
	   kept ones, whose placement stays (all faces not required).
	3. state: once the kept views cover all 54 facelets, the most probable legal state (estimateState); re-estimated
	   whenever a view is added.

HUD: the sticker overlay of the latest good result (as hud_video.py), and top right the state panel — the cube net
(U on top; L F R B; D), per facelet the observed colour (or, with a state, the mean colour of its estimated label),
the number of observations, and the confidence (border: green posterior ≥ 0.99, yellow ≥ 0.9, red below), then the
stage, coverage, views, the best state guess and, with truth labels, its error.

Writes OUT/frame0001.png …, OUT/state.mp4, OUT/log.json; prints the timeline of stages and the final result.
"""
import argparse, json, os, subprocess, sys, time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import rubik_engine as re  # noqa: E402
from rubik_detector import cube  # noqa: E402
from hud_video import centre_error, draw, lab_to_bgr  # noqa: E402

FACES = cube.FACES
NET = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}


def frame_match(Ra, Rb):
	"""(angle, g): the rotation between two poses of the cube up to a renaming of the view frame, and the renaming —
	b's facelet k is a's facelet PERMS[g][k]. A whole-cube symmetry (a 90° turn, one corner to another) is a
	renaming too, so the angle alone can't tell a held cube from a turned one: colour_distance adds the colours."""
	best = (1e9, -1)
	for g, G in enumerate(cube.ROTATIONS):
		angle = float(np.degrees(np.arccos(np.clip((np.trace((Ra @ G).T @ Rb) - 1) / 2, -1, 1))))
		best = min(best, (angle, g))
	return best


def colour_distance(a, b, g):
	"""median ΔE76 between b's samples and a's at the facelets the renaming g pairs them with"""
	la = {s["facelet"]: np.array(s["lab"]) for s in a["samples"] if s["valid"]}
	d = [np.linalg.norm(np.array(s["lab"]) - la[cube.PERMS[g][s["facelet"]]]) for s in b["samples"]
		if s["valid"] and cube.PERMS[g][s["facelet"]] in la]
	return float(np.median(d)) if d else 1e9


class Tracker:
	def __init__(self, args, K_truth, truth):
		self.a = args
		self.K = K_truth if args.known_fov else None
		self.truth = truth
		self.det = re.Detector()
		self.prev = None            # last good detection
		self.views = []             # kept evidence views (detection dicts)
		self.rotations = []         # their placement in the common frame
		self.estimate = None

	# --- stage 1 --------------------------------------------------------------------------------------------------

	def calibrate(self, frame):
		f = self.det.estimate_focal([frame])
		c = f["camera"]
		self.K = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]])
		return f

	# --- stage 2 --------------------------------------------------------------------------------------------------

	def detect(self, frame):
		mode, r = "wide", None
		if self.prev is not None:
			r = self.det.detect(frame, self.K, prior=(self.prev["R"], self.prev["t"]))
			mode = "local"
		if r is None or not r["ok"] or r["lattice_cost"] > self.a.fallback_cost:
			r = self.det.detect(frame, self.K)
			mode = "fallback" if mode == "local" else "wide"
		good = r["ok"] and r["lattice_cost"] <= self.a.fallback_cost
		event = {"mode": mode, "good": good, "cost": r["lattice_cost"], "result": r}
		if good and self.prev is not None:
			angle, g = frame_match(self.prev["R"], r["R"])
			colour = colour_distance(self.prev, r, g)
			agree = angle <= self.a.agree_deg and colour <= self.a.agree_colour
			event.update(agree=agree, angle=angle, colour=colour)
			if agree and len(self.views) < 16:
				self.add_view(r, event)
		self.prev = r if good else None
		return event

	def add_view(self, r, event):
		# incremental: the kept views stay where they were placed, only the new one is aligned (re-fusing all of them
		# lets two near-identical views, overlapping on 27 facelets at ~0 cost, outvote their 9-facelet alignment to
		# the rest)
		views = self.views + [r]
		e = re.estimate(views, rotations=(self.rotations + [-1]) if self.views else None, require_all_faces=False)
		rot = e["rotations"]
		if not e["ok"] and e["error"].startswith("no rotation"):
			event["rejected"] = "no colour alignment with the views so far"
			return
		covered = lambda vs, rs: {cube.PERMS[g][s["facelet"]] for v, g in zip(vs, rs) if g >= 0 for s in v["samples"] if s["valid"]}
		if self.views and len(covered(views, rot)) - len(covered(self.views, self.rotations)) < 9:
			return   # nothing new: the same faces again
		self.views = views
		self.rotations = rot
		event["kept"] = len(views)
		# --- stage 3: once the views cover the cube
		if e["ok"]:
			e["wrong"] = re.compare_states(e["state"], self.truth) if self.truth else None
			self.estimate = e
			event["estimate"] = {k: e[k] for k in ("state", "legal", "min_posterior", "wrong")}

	# --- evidence per common facelet ------------------------------------------------------------------------------

	def evidence(self):
		obs = [[] for _ in range(54)]
		for v, g in zip(self.views, self.rotations):
			if g < 0:
				continue
			for s in v["samples"]:
				if s["valid"]:
					obs[cube.PERMS[g][s["facelet"]]].append(s["lab"])
		return obs


def panel(img, tracker, stage, scale):
	"""the state panel, top right"""
	obs = tracker.evidence() if tracker.views else [[] for _ in range(54)]
	e = tracker.estimate
	cell = int(26 * scale)
	pad = int(10 * scale)
	W, H = 4 * 3 * cell + 3 * int(3 * scale), 3 * 3 * cell + 2 * int(3 * scale)
	text_lines = []
	coverage = sum(1 for o in obs if o)
	text_lines.append(f"stage: {stage}")
	text_lines.append(f"evidence: {len(tracker.views)} views, {coverage}/54 facelets, {sum(len(o) for o in obs)} obs")
	if e:
		text_lines.append(f"state: {'legal' if e['legal'] else 'ILLEGAL'}, min posterior {e['min_posterior']:.2f}"
			+ (f", {'exact' if e['wrong'] == 0 else str(e['wrong']) + ' wrong'}" if e.get("wrong") is not None else ""))
		text_lines.append(e["state"][:27])
		text_lines.append(e["state"][27:])
	else:
		text_lines.append("state: " + ("waiting for all faces" if tracker.views else "no evidence yet"))
	line_h = int(20 * scale)
	box_w, box_h = W + 2 * pad, H + 2 * pad + line_h * len(text_lines) + pad
	x0, y0 = img.shape[1] - box_w - pad, pad
	overlay = img.copy()
	cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
	img[:] = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
	# per label mean colour, from the estimate
	label_colour = {}
	if e:
		for c in FACES:
			labs = [lab for k in range(54) if e["state"][k] == c for lab in obs[k]]
			if labs:
				label_colour[c] = lab_to_bgr(np.median(np.array(labs), axis=0))
	for f, face in enumerate(FACES):
		gx, gy = NET[face]
		fx0 = x0 + pad + gx * (3 * cell + int(3 * scale))
		fy0 = y0 + pad + gy * (3 * cell + int(3 * scale))
		for i in range(9):
			k = f * 9 + i
			r_, c_ = divmod(i, 3)
			p0 = (fx0 + c_ * cell, fy0 + r_ * cell)
			p1 = (p0[0] + cell - 2, p0[1] + cell - 2)
			if e:
				colour = label_colour.get(e["state"][k], (90, 90, 90))
			elif obs[k]:
				colour = lab_to_bgr(np.median(np.array(obs[k]), axis=0))
			else:
				colour = (70, 70, 70)
			cv2.rectangle(img, p0, p1, colour, -1)
			if e:
				p = e["posterior"][k]
				border = (60, 200, 60) if p >= 0.99 else ((0, 200, 230) if p >= 0.9 else (40, 40, 230))
				cv2.rectangle(img, p0, p1, border, max(1, round(2 * scale)))
			label = str(len(obs[k])) if obs[k] else "?"
			lum = sum(colour) / 3
			cv2.putText(img, label, (p0[0] + int(8 * scale), p1[1] - int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
				(0, 0, 0) if lum > 128 else (255, 255, 255), max(1, round(scale)), cv2.LINE_AA)
	y = y0 + pad + H + pad + line_h - int(4 * scale)
	for line in text_lines:
		cv2.putText(img, line, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (255, 255, 255), max(1, round(scale)), cv2.LINE_AA)
		y += line_h


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("video")
	ap.add_argument("--latency-scale", type=float, default=1.0)
	ap.add_argument("--known-fov", action="store_true", help="use the labels' K instead of estimating it")
	ap.add_argument("--fallback-cost", type=float, default=0.15)
	ap.add_argument("--agree-deg", type=float, default=6.0)
	ap.add_argument("--agree-colour", type=float, default=12.0, help="median ΔE76 between consecutive detections' paired samples")
	ap.add_argument("--out", default=None)
	a = ap.parse_args()

	labels = json.load(open(os.path.join(a.video, "labels.json")))
	fps, n = labels["video"]["fps"], labels["video"]["frames"]
	k = labels["intrinsics"]
	K_truth = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
	frames = [cv2.imread(os.path.join(a.video, f"frame{i + 1:04d}.png")) for i in range(n)]
	out = a.out or os.path.join(a.video, f"state-x{a.latency_scale:g}" + ("-knownfov" if a.known_fov else ""))
	os.makedirs(out, exist_ok=True)
	tracker = Tracker(a, K_truth, labels["cube"]["facelets"])

	# the timeline: events (frame processed, ready time, what happened)
	events, free = [], 0.0
	if not a.known_fov:
		t0 = time.perf_counter()
		f = tracker.calibrate(frames[0])
		ms = (time.perf_counter() - t0) * 1000
		free = ms * a.latency_scale / 1000
		events.append({"kind": "fov", "frame": 0, "ready": free, "ms": ms, "fx": tracker.K[0, 0], "fx_truth": K_truth[0, 0], "support": f["support"]})
	last_frame = -1
	while True:
		f = min(n - 1, int(np.floor(free * fps + 1e-9)))
		if f <= last_frame:
			f = last_frame + 1
			if f >= n:
				break
			free = f / fps
		t0 = time.perf_counter()
		ev = tracker.detect(frames[f])
		ms = (time.perf_counter() - t0) * 1000
		start = max(free, f / fps)
		ev.update(kind="detect", frame=f, start=start, ready=start + ms * a.latency_scale / 1000, ms=ms,
			snapshot={"views": len(tracker.views), "estimate": tracker.estimate, "rotations": list(tracker.rotations)})
		ev["view_list"] = list(tracker.views)
		events.append(ev)
		free, last_frame = ev["ready"], f
		if free * fps >= n:
			break

	# render: each frame with what was ready by its arrival
	log = {"fps": fps, "latency_scale": a.latency_scale, "events": [], "frames": []}
	for i in range(n):
		t = i / fps
		ready = [e for e in events if e["ready"] <= t]
		fov = next((e for e in events if e["kind"] == "fov"), None)
		dets = [e for e in ready if e["kind"] == "detect"]
		last = dets[-1] if dets else None
		shown = last if last and last["good"] else None
		if fov and fov["ready"] > t:
			stage = f"FOV estimation ({fov['ms'] * a.latency_scale / 1000:.1f} s)"
		elif last is None:
			stage = "detecting"
		elif last["snapshot"]["estimate"]:
			stage = "state estimated"
		else:
			stage = "accumulating evidence"
		# the tracker's state as of the last ready detection
		snap = Tracker(a, K_truth, labels["cube"]["facelets"])
		if last:
			snap.views, snap.rotations, snap.estimate = last["view_list"], last["snapshot"]["rotations"], last["snapshot"]["estimate"]
		info = [f"t {t:4.1f}s  frame {i + 1}/{n}  {labels['views'][i]['phase']}"]
		if shown:
			info.append(f"result of frame {shown['frame'] + 1} ({shown['mode']}), age {(t - shown['frame'] / fps) * 1000:.0f} ms, cost {shown['cost']:.3f}")
		elif last:
			info.append(f"lost (frame {last['frame'] + 1}: cost {last['cost']:.2f})")
		scale = frames[i].shape[1] / 1024
		img = draw(frames[i], shown["result"] if shown else None, info, scale)
		panel(img, snap, stage, scale)
		cv2.imwrite(os.path.join(out, f"frame{i + 1:04d}.png"), img)
		tq = [np.array(s["corners"]) for s in labels["views"][i]["stickers"] if s["visible"]]
		log["frames"].append({"frame": i, "stage": stage, "shows": shown["frame"] if shown else None,
			"error_cubie": centre_error([s["quad"] for s in shown["result"]["samples"] if s["valid"]], tq) if shown else None})
	subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", os.path.join(out, "frame%04d.png"),
		"-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, "state.mp4")], check=True)

	for e in events:
		x = {k_: v for k_, v in e.items() if k_ not in ("result", "snapshot", "view_list")}
		log["events"].append(x)
	json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1, default=float)

	fov = next((e for e in events if e["kind"] == "fov"), None)
	if fov:
		print(f"FOV: fx {fov['fx']:.0f} (truth {fov['fx_truth']:.0f}, {fov['fx'] / fov['fx_truth'] - 1:+.1%}), support {fov['support']:.2f}, "
			f"ready at {fov['ready']:.1f} s")
	dets = [e for e in events if e["kind"] == "detect"]
	print(f"{len(dets)} detections: " + ", ".join(f"{m} {sum(e['mode'] == m for e in dets)}" for m in ("wide", "local", "fallback"))
		+ f"; good {sum(e['good'] for e in dets)}, consistent {sum(bool(e.get('agree')) for e in dets)}")
	print("  consecutive good pairs (frame:angle°/ΔE, + agree): " + " ".join(f"{e['frame'] + 1}:{e['angle']:.0f}/{e['colour']:.0f}{'+' if e['agree'] else '-'}" for e in dets if "angle" in e))
	for e in dets:
		if "kept" in e or "rejected" in e:
			print(f"  {e['ready']:5.1f} s  frame {e['frame'] + 1:2d} ({labels['views'][e['frame']]['phase']}): "
				+ (f"view {e['kept']} kept" if "kept" in e else f"rejected — {e['rejected']}")
				+ (f" → state {'legal' if e['estimate']['legal'] else 'ILLEGAL'}, min posterior {e['estimate']['min_posterior']:.2f}, "
					f"{'exact' if e['estimate']['wrong'] == 0 else str(e['estimate']['wrong']) + ' wrong'}" if "estimate" in e else ""))
	final = tracker.estimate
	print("final: " + (f"{'exact' if final['wrong'] == 0 else str(final['wrong']) + ' wrong'}, min posterior {final['min_posterior']:.2f}" if final else "no state"))
	print(f"→ {out}")


if __name__ == "__main__":
	main()
