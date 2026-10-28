"""Application-layer prototype: continuous cube-state detection on a camera stream, with the engine's real latency.

	python engine/tools/state_tracker.py VIDEO_DIR [--latency-scale 1] [--camera-ms 30] [--out-fps 5] [--known-fov] [--out DIR]

VIDEO_DIR is the camera (klstudio /rubik-samples 🎬, e.g. ?videoFps=30): frame k is captured at k / fps and reaches
the app --camera-ms later. Two workers, each step costing its measured time × --latency-scale:

	detector   1. FOV: Detector::estimateFocal on the first frame (or --known-fov: the labels' K), fixed from then on
	           2. then back to back on the newest frame that has arrived: local detection from the last good pose,
	              wide when there is none or the lattice cost passes --fallback-cost. The prior is the last good pose moved
	              on at the velocity of the last two linked detections (a turn would leave a stale pose behind); none
	              after a fallback without velocity (straight to wide).
	estimator  3. whenever the evidence has grown: the colour clusters and per facelet the Bayesian posterior of its
	              colour, uniform prior. Each piece of evidence (--unit: a frame, or a chain of linked frames) is right
	              with an unknown probability r, uniform on [0, 1], and otherwise says nothing:
	              P(c) ∝ ∫ Π_units (r·L_u(c) + (1 − r)/6) dr, L_u the unit's own posterior from its observations
	              (Student-t colour model). One unit gives 58 % however sharp its colour, two agreeing 90 %, three
	              98 %, four 99.7 % — the rule of succession for six colours. From --legal-from facelets (5 faces) the
	              most probable legal state under these, the hidden facelets completed by the legal search, and per
	              facelet the confidence of the state's colour under the legal constraint.

Placing frames in one cube frame: consecutive good detections within --link-deg of each other up to a renaming of
the view frame, whose colours agree (median ΔE ≤ --agree-colour), form a chain and inherit its placement through the
renaming. A chain counts once --confirm frames agree (default 1); it is then placed by colour against the evidence so far (≥ 9
overlapping facelets; if it can't be, from a later frame). Evidence needs a lattice cost ≤ --evidence-cost and ≤
--relative-cost × the median of the recent good fits (tracking accepts up to --fallback-cost: a mid-turn fit can be
well under it and still wrong).

Output (downsampled to --out-fps to keep it small; the pipeline runs on every camera frame): OUT/frame*.png, OUT/
state.mp4, OUT/log.json. HUD: the stickers of the latest detection ready by the frame's arrival; top right the state
panel — the cube net (U on top; L F R B; D), per facelet the colour of the best state guess (before one: of its
most probable cluster), in percent the best component of its cluster posterior, the border the state's confidence
there under the legal constraint (green ≥ 99 %, yellow ≥ 90 %, red below); corner marks: white = completed, not
observed; red = the legal state overrides the best cluster. Then the stage, the evidence, the best state guess and,
with truth labels, its error.
"""
import argparse, json, os, subprocess, sys, time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import rubik_engine as re  # noqa: E402
from rubik_detector import cube  # noqa: E402
from hud_video import centre_error, draw, lab_to_bgr  # noqa: E402

FACES = cube.FACES
NET = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}
PERM_INDEX = {tuple(p): g for g, p in enumerate(cube.PERMS)}
IDENTITY = PERM_INDEX[tuple(range(54))]


def frame_match(Ra, Rb):
	"""(angle, g): the rotation between two poses of the cube up to a renaming of the view frame, and the renaming —
	b's facelet k is a's facelet PERMS[g][k]"""
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


def chain_rotation(rot_a, g):
	"""placement of b, given a's placement and b's renaming g into a's frame"""
	return PERM_INDEX[tuple(cube.PERMS[rot_a][cube.PERMS[g][k]] for k in range(54))]


class Tracker:
	"""the detector worker's state: pose tracking and the placed evidence"""

	def __init__(self, args, K):
		self.a = args
		self.K = K
		self.det = re.Detector()
		self.prev = None            # last good detection, with its "rotation" (None: its chain not placed yet)
		self.frames = []            # placed evidence: detections
		self.rotations = []
		self.chains = 0
		self.pending = []           # frames of the current chain until it is confirmed and placed
		self.collect = True         # placing frames as evidence (off: tracking only, e.g. once a state is followed)
		self.costs = []             # lattice costs of good detections, for the relative evidence gate

	def fits_well(self, r):
		"""evidence-grade fit: lattice cost ≤ --evidence-cost and ≤ --relative-cost × the median of the recent good
		fits (the cost level depends on the cube and the scene; a fit mid-turn can be good in absolute terms and still
		twice the usual)"""
		recent = self.costs[-30:]
		limit = self.a.evidence_cost
		if len(recent) >= 3:
			limit = min(limit, self.a.relative_cost * float(np.median(recent)))
		return r["lattice_cost"] <= limit

	def calibrate(self, frame):
		f = self.det.estimate_focal([frame])
		c = f["camera"]
		self.K = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]])
		return f

	def place(self, r):
		"""colour alignment of a chain's frame against the evidence so far (one frame per placed chain suffices)"""
		if not self.frames:
			return 0
		anchors, rots, seen = [], [], set()
		for v, g in zip(self.frames, self.rotations):
			if v["chain"] not in seen:
				seen.add(v["chain"])
				anchors.append(v)
				rots.append(g)
		e = re.estimate(anchors + [r], rotations=rots + [-1], require_all_faces=False, partial=True, legal=False)
		return e["rotations"][-1] if e["rotations"][-1] >= 0 and not e["error"].startswith("no rotation") else None

	def predict(self, index):
		"""prior pose for camera frame `index`: the last good pose, moved on at the angular and linear velocity of the
		last two linked detections (None: no prior worth a local search)"""
		b = self.prev
		if b is None:
			return None
		a = b.get("linked_to")
		if a is None:
			# no velocity: after a fallback the motion was fast, a local search from a stale pose would fail again
			return None if b["mode"] == "fallback" else (b["R"], b["t"])
		_, g = frame_match(a["R"], b["R"])
		Ra = a["R"] @ np.array(cube.ROTATIONS[g])
		delta = Ra.T @ b["R"]   # b = Ra · delta (cube frame)
		s = (index - b["frame"]) / max(1, b["frame"] - a["frame"])
		R = b["R"] @ Rotation.from_rotvec(Rotation.from_matrix(delta).as_rotvec() * s).as_matrix()
		t = np.asarray(b["t"]) + (np.asarray(b["t"]) - np.asarray(a["t"])) * s
		return R, t

	def detect(self, frame, index):
		mode, r = "wide", None
		prior = self.predict(index)
		if prior is not None:
			r = self.det.detect(frame, self.K, prior=prior)
			mode = "local"
		if r is None or not r["ok"] or r["lattice_cost"] > self.a.fallback_cost:
			r = self.det.detect(frame, self.K)
			mode = "fallback" if mode == "local" else "wide"
		good = r["ok"] and r["lattice_cost"] <= self.a.fallback_cost
		r["frame"], r["mode"] = index, mode
		if good:
			self.costs.append(r["lattice_cost"])
		event = {"mode": mode, "good": good, "cost": r["lattice_cost"], "result": r}
		if not good:
			self.prev = None
			return event
		linked = False
		if self.prev is not None:
			angle, g = frame_match(self.prev["R"], r["R"])
			colour = colour_distance(self.prev, r, g)
			linked = angle <= self.a.link_deg and colour <= self.a.agree_colour
			event.update(angle=angle, colour=colour, linked=linked)
		if linked:
			r["linked_to"] = self.prev
			self.prev.pop("linked_to", None)   # keep one step of history
			r["chain"] = self.prev["chain"]
			r["rotation"] = chain_rotation(self.prev["rotation"], g) if self.prev["rotation"] is not None else None
			r["rel"] = chain_rotation(self.prev["rel"], g)   # renaming into the chain's first frame
		else:
			self.chains += 1
			r["chain"] = self.chains
			r["rotation"] = None
			r["rel"] = IDENTITY
			self.pending = []
		self.prev = r
		if not self.collect:
			return event
		if not self.fits_well(r):
			event["weak_fit"] = True
			return event   # tracked, not evidence
		if r["rotation"] is not None:
			self.frames.append(r)
			self.rotations.append(r["rotation"])
		else:
			# a chain counts once --confirm of its frames agree (a lone fit mid-turn can be wrong and still look good);
			# then it is placed by colour and its frames placed through their renamings
			self.pending.append(r)
			if len(self.pending) >= self.a.confirm:
				anchor = self.pending[0]
				rot = self.place(anchor)
				event["placed"] = rot is not None
				if rot is not None:
					# the chain's first frame placed: start's facelet j is anchor's facelet inv(rel_anchor)[j]
					inv = np.argsort(cube.PERMS[anchor["rel"]])
					start = PERM_INDEX[tuple(cube.PERMS[rot][inv[j]] for j in range(54))]
					for v in self.pending:
						v["rotation"] = chain_rotation(start, v["rel"])
						self.frames.append(v)
						self.rotations.append(v["rotation"])
					self.pending = []
				else:
					self.pending.pop(0)   # try again from the next frame
		if r["rotation"] is not None:
			event["evidence"] = len(self.frames)
		return event


def estimate(frames, rotations, args):
	"""the estimator worker: clusters, per facelet posteriors, and from --legal-from facelets the legal state"""
	covered = len({cube.PERMS[g][s["facelet"]] for v, g in zip(frames, rotations) for s in v["samples"] if s["valid"]})
	e = re.estimate(frames, rotations=rotations, require_all_faces=False, partial=True, legal=covered >= args.legal_from,
		outlier=args.outlier, reliability=args.unit != "none", groups=[v["chain"] for v in frames] if args.unit == "chain" else None)
	e["covered"] = covered
	e["frames"] = len(frames)
	e["chains"] = len({v["chain"] for v in frames})
	return e


def panel(img, e, obs, stage, truth, scale, colours=None, extra=(), numbers=True):
	"""the state panel, top right. e: the latest estimate (or None); obs: per common facelet its observations' Lab;
	colours: BGR per colour letter instead of the observations' medians; extra: text lines after the stage"""
	cell = int(26 * scale)
	pad = int(10 * scale)
	gap = int(3 * scale)
	W, H = 4 * 3 * cell + 3 * gap, 3 * 3 * cell + 2 * gap
	lines = [f"stage: {stage}"] + list(extra)
	if e and e["ok"]:
		if "frames" in e:
			lines.append(f"evidence: {e['frames']} frames in {e['chains']} chains, {e['covered']}/54 facelets observed")
		probs = e["probability"]
		best = probs.argmax(1)
		pbest = probs.max(1)
		if e["legal"]:
			wrong = re.compare_states(e["state"], truth) if truth else None
			lines.append(f"best state: legal, min confidence {e['min_confidence']:.2f}" + (f", {54 - e['covered']} completed" if e["covered"] < 54 else "")
				+ (f"  [truth: {'exact' if wrong == 0 else str(wrong) + ' wrong'}]" if wrong is not None else ""))
			lines.append(e["state"][:27])
			lines.append(e["state"][27:])
			letters = e["state"]
		else:
			lines.append(f"clusters only (legal state from {54 - 9} facelets)")
			letters = None
	else:
		lines.append("evidence: none yet")
		probs = None
	line_h = int(20 * scale)
	text_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, max(1, round(scale)))[0][0] for line in lines)
	box_w, box_h = max(W, text_w) + 2 * pad, H + 2 * pad + line_h * len(lines) + pad
	x0, y0 = img.shape[1] - box_w - pad, pad
	overlay = img.copy()
	cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
	img[:] = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
	# colour of each cluster: the median Lab of the observations of the facelets it holds
	label_colour = dict(colours or {})
	if probs is not None and not colours:
		lab_of = letters if letters else [FACES[c] for c in best]
		for c in FACES:
			labs = [lab for k in range(54) if lab_of[k] == c for lab in obs[k]]
			if labs:
				label_colour[c] = lab_to_bgr(np.median(np.array(labs), axis=0))
	for f, face in enumerate(FACES):
		gx, gy = NET[face]
		fx0 = x0 + pad + gx * (3 * cell + gap)
		fy0 = y0 + pad + gy * (3 * cell + gap)
		for i in range(9):
			k = f * 9 + i
			r_, c_ = divmod(i, 3)
			p0 = (fx0 + c_ * cell, fy0 + r_ * cell)
			p1 = (p0[0] + cell - 2, p0[1] + cell - 2)
			known = probs is not None and (e["observed"][k] or e["legal"])
			if known:
				letter = letters[k] if letters else FACES[best[k]]
				p = pbest[k]                                  # the cluster posterior's best component
				conf = e["confidence"][k] if letters else p   # the state's colour under the legal constraint
				colour = label_colour.get(letter, (110, 110, 110))
			else:
				colour = (70, 70, 70)
			cv2.rectangle(img, p0, p1, colour, -1)
			if known:
				border = (60, 200, 60) if conf >= 0.99 else ((0, 200, 230) if conf >= 0.9 else (40, 40, 230))
				cv2.rectangle(img, p0, p1, border, max(1, round(2 * scale)))
				if not e["observed"][k]:
					tri = np.array([p0, (p0[0] + cell // 3, p0[1]), (p0[0], p0[1] + cell // 3)], np.int32)
					cv2.fillConvexPoly(img, tri, (255, 255, 255))
				elif letters and FACES[best[k]] != letter:
					# the legal state overrides the best cluster here
					cv2.line(img, (p1[0] - cell // 3, p0[1] + 2), (p1[0] - 2, p0[1] + cell // 3), (40, 40, 230), max(1, round(2 * scale)))
				label = ("99" if p >= 0.995 else f"{int(p * 100):2d}") if numbers else ""
			else:
				label = "?"
			lum = sum(colour) / 3
			cv2.putText(img, label, (p0[0] + int(5 * scale), p1[1] - int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale,
				(0, 0, 0) if lum > 128 else (255, 255, 255), max(1, round(scale)), cv2.LINE_AA)
	y = y0 + pad + H + pad + line_h - int(4 * scale)
	for line in lines:
		cv2.putText(img, line, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (255, 255, 255), max(1, round(scale)), cv2.LINE_AA)
		y += line_h


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("video")
	ap.add_argument("--latency-scale", type=float, default=1.0)
	ap.add_argument("--camera-ms", type=float, default=30.0, help="capture to app")
	ap.add_argument("--out-fps", type=float, default=5.0)
	ap.add_argument("--known-fov", action="store_true", help="use the labels' K instead of estimating it")
	ap.add_argument("--fallback-cost", type=float, default=0.15)
	ap.add_argument("--evidence-cost", type=float, default=0.09, help="lattice cost limit for evidence (tracking: --fallback-cost)")
	ap.add_argument("--relative-cost", type=float, default=1.5, help="evidence: lattice cost ≤ this × the median of recent good fits")
	ap.add_argument("--confirm", type=int, default=1, help="frames of a chain that must agree before it counts (with unknown "
		"reliability a lone frame is weak evidence anyway: 58 %%)")
	ap.add_argument("--link-deg", type=float, default=20.0)
	ap.add_argument("--agree-colour", type=float, default=12.0, help="median ΔE76 between linked detections' paired samples")
	ap.add_argument("--unit", choices=("frame", "chain", "none"), default="frame",
		help="one piece of evidence, right with an unknown probability (uniform prior): a frame, or a chain of linked frames "
		"(its frames stand or fall together); none: the plain product of the colour likelihoods")
	ap.add_argument("--outlier", type=float, default=0.0, help="per-observation outlier share (with --unit none)")
	ap.add_argument("--legal-from", type=int, default=45)
	ap.add_argument("--out", default=None)
	a = ap.parse_args()

	labels = json.load(open(os.path.join(a.video, "labels.json")))
	fps, n = labels["video"]["fps"], labels["video"]["frames"]
	k = labels["intrinsics"]
	K_truth = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
	truth = labels["cube"]["facelets"]
	frame_path = lambda i: os.path.join(a.video, f"frame{i + 1:04d}.png")
	arrive = lambda i: i / fps + a.camera_ms / 1000
	out = a.out or os.path.join(a.video, f"state-x{a.latency_scale:g}" + ("-knownfov" if a.known_fov else ""))
	os.makedirs(out, exist_ok=True)
	tracker = Tracker(a, K_truth if a.known_fov else None)

	# detector worker
	events, free = [], arrive(0)
	if not a.known_fov:
		t0 = time.perf_counter()
		f = tracker.calibrate(cv2.imread(frame_path(0)))
		ms = (time.perf_counter() - t0) * 1000
		events.append({"kind": "fov", "frame": 0, "start": free, "ready": free + ms * a.latency_scale / 1000, "ms": ms,
			"fx": tracker.K[0, 0], "fx_truth": K_truth[0, 0], "support": f["support"]})
		free = events[-1]["ready"]
	last = -1
	while True:
		i = int(np.floor((free - a.camera_ms / 1000) * fps + 1e-9))   # newest arrived
		if i <= last:
			i = last + 1
		if i >= n:
			break
		start = max(free, arrive(i))
		t0 = time.perf_counter()
		ev = tracker.detect(cv2.imread(frame_path(i)), i)
		ms = (time.perf_counter() - t0) * 1000
		ev.update(kind="detect", frame=i, start=start, ready=start + ms * a.latency_scale / 1000, ms=ms, evidence_count=len(tracker.frames))
		events.append(ev)
		free, last = ev["ready"], i
	dets = [e for e in events if e["kind"] == "detect"]

	# estimator worker: whenever free and the evidence ready by then has grown, on all of it
	estimates, free, done = [], 0.0, 0
	grown = [e for e in dets if "evidence" in e]
	while True:
		pending = [e for e in grown if e["evidence"] > done]
		if not pending:
			break
		start = max(free, pending[0]["ready"])
		count = max(e["evidence"] for e in grown if e["ready"] <= start)
		t0 = time.perf_counter()
		est = estimate(tracker.frames[:count], tracker.rotations[:count], a)
		ms = (time.perf_counter() - t0) * 1000
		est.update(start=start, ready=start + ms * a.latency_scale / 1000, ms=ms, count=count)
		estimates.append(est)
		free, done = est["ready"], count

	# output, downsampled: each shown frame with what was ready by its arrival
	step = max(1, int(round(fps / a.out_fps)))
	fov = next((e for e in events if e["kind"] == "fov"), None)
	log = {"fps": fps, "out_fps": fps / step, "latency_scale": a.latency_scale, "camera_ms": a.camera_ms, "frames": [], "estimates": []}
	shown_frames = list(range(0, n, step))
	for j, i in enumerate(shown_frames):
		t = arrive(i)
		ready = [e for e in dets if e["ready"] <= t]
		lastd = ready[-1] if ready else None
		shown = lastd if lastd and lastd["good"] else None
		est = next((e for e in reversed(estimates) if e["ready"] <= t), None)
		if fov and fov["ready"] > t:
			stage = f"FOV estimation ({fov['ms'] * a.latency_scale / 1000:.1f} s)"
		elif est is None:
			stage = "detecting"
		elif est["ok"] and est["legal"]:
			stage = "state estimated" if est["covered"] == 54 else "state inferred (5 faces + legal completion)"
		else:
			stage = "accumulating evidence"
		obs = [[] for _ in range(54)]
		if est:
			for v, g in zip(tracker.frames[:est["count"]], tracker.rotations[:est["count"]]):
				for s in v["samples"]:
					if s["valid"]:
						obs[cube.PERMS[g][s["facelet"]]].append(s["lab"])
		info = [f"t {t:4.2f}s  camera frame {i + 1}/{n}  {labels['views'][i]['phase']}"]
		if shown:
			info.append(f"detection of frame {shown['frame'] + 1} ({shown['mode']}), age {(t - shown['frame'] / fps) * 1000:.0f} ms, cost {shown['cost']:.3f}")
		elif lastd:
			info.append(f"lost (frame {lastd['frame'] + 1}: cost {lastd['cost']:.2f})")
		info.append(f"detections so far {len(ready)}" + (f", estimate age {(t - est['start']) * 1000:.0f} ms" if est else ""))
		frame = cv2.imread(frame_path(i))
		scale = frame.shape[1] / 1024
		img = draw(frame, shown["result"] if shown else None, info, scale)
		panel(img, est, obs, stage, truth, scale)
		cv2.imwrite(os.path.join(out, f"frame{j + 1:04d}.png"), img)
		tq = [np.array(s["corners"]) for s in labels["views"][i]["stickers"] if s["visible"]]
		log["frames"].append({"camera_frame": i, "stage": stage, "shows": shown["frame"] if shown else None,
			"age_ms": (t - shown["frame"] / fps) * 1000 if shown else None,
			"error_cubie": centre_error([s["quad"] for s in shown["result"]["samples"] if s["valid"]], tq) if shown else None})
	subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps / step), "-i", os.path.join(out, "frame%04d.png"),
		"-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, "state.mp4")], check=True)
	for e in estimates:
		wrong = re.compare_states(e["state"], truth) if e["ok"] and e["legal"] else None
		log["estimates"].append({"start": e["start"], "ready": e["ready"], "ms": e["ms"], "frames": e["frames"], "chains": e["chains"],
			"covered": e["covered"], "legal": e["legal"], "min_posterior": e["min_posterior"], "min_confidence": e["min_confidence"],
			"overridden": sum(1 for k in range(54) if e["ok"] and e["legal"] and e["observed"][k] and FACES[int(np.argmax(e["probability"][k]))] != e["state"][k]),
			"wrong": wrong, "error": e["error"]})
	log["detections"] = [{k_: v for k_, v in e.items() if k_ != "result"} for e in dets]
	json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1, default=float)

	# summary
	if fov:
		print(f"FOV: fx {fov['fx']:.0f} (truth {fov['fx_truth']:.0f}, {fov['fx'] / fov['fx_truth'] - 1:+.1%}), support {fov['support']:.2f}, ready at {fov['ready']:.1f} s")
	lat = [e["ready"] - e["frame"] / fps for e in dets]
	print(f"{len(dets)} detections on {n} camera frames ({n / fps:.1f} s): " + ", ".join(f"{m} {sum(e['mode'] == m for e in dets)}" for m in ("wide", "local", "fallback"))
		+ f"; good {sum(e['good'] for e in dets)}, evidence {len(tracker.frames)} frames in {len({v['chain'] for v in tracker.frames})} chains; "
		f"capture→result mean {np.mean(lat) * 1000:.0f} ms")
	ages = [f["age_ms"] for f in log["frames"] if f["age_ms"] is not None]
	for phase in ("hold", "turn"):
		fr = [f for f in log["frames"] if labels["views"][f["camera_frame"]]["phase"] == phase]
		err = [f["error_cubie"] for f in fr if f["error_cubie"] is not None]
		print(f"  {phase}: HUD on {len(err)}/{len(fr)} shown frames, error vs the shown frame mean {np.mean(err) if err else float('nan'):.2f} cubie")
	print(f"  HUD age on shown frames: median {np.median(ages):.0f} ms, p90 {np.percentile(ages, 90):.0f} ms")
	first = {}
	for e in log["estimates"]:
		if e["legal"]:
			key = "54" if e["covered"] == 54 else "45"
			if key not in first:
				first[key] = e
	for key, e in sorted(first.items(), reverse=True):
		print(f"  first legal state from {e['covered']} observed facelets: ready {e['ready']:.1f} s, {e['frames']} frames, "
			f"{'exact' if e['wrong'] == 0 else str(e['wrong']) + ' wrong'}, min confidence {e['min_confidence']:.2f}")
	print(f"  {len(estimates)} estimates, {np.mean([e['ms'] for e in estimates]):.0f} ms mean, max {max(e['ms'] for e in estimates):.0f} ms")
	fin = estimates[-1] if estimates else None
	print("final: " + (f"{'exact' if log['estimates'][-1]['wrong'] == 0 else str(log['estimates'][-1]['wrong']) + ' wrong'}, min confidence {fin['min_confidence']:.2f}, "
		f"{fin['covered']}/54 observed" if fin and fin["legal"] else "no state"))
	print(f"→ {out}")


if __name__ == "__main__":
	main()
