"""Application-layer prototype: follow a cube being turned, after its state has been recognised.

	python engine/tools/follow_tracker.py VIDEO_DIR [--latency-scale 1] [--camera-ms 30] [--out-fps 5] [--out DIR]

Two modes on the camera stream of state_tracker.py (same timeline: frame k arrives --camera-ms after k / fps, one
worker back to back on the newest arrived frame, every step costing its measured time × --latency-scale):

	acquire   state_tracker.py's stages: FOV, evidence of every good frame, Bayesian colour posteriors, the legal
	          state. Once it is legal, observed on --lock-coverage facelets (default all 54: a five-face completion
	          can be confidently wrong while the colour clusters are still unsettled), every facelet's confidence
	          ≥ --lock, and the same in --lock-stable estimates in a row, the state and the colour model are
	          locked: follow.
	follow    the state S is known; each well-fitted frame (lattice cost ≤ --evidence-cost and ≤ --relative-cost × the
	          recent median: mid-turn frames fail it) is matched against the states at most two face turns from S —
	          S, its 18 single turns and 243 double turns (distinct states, the shorter way kept), cached per S. The
	          prior: the cube changes by turns, each costing --move-cost nll. Candidates and the 24 renamings of the
	          view frame are scored together under the locked colour model (score_model, a few ms per frame). The
	          best is a match if at most --max-conflicts of the frame's samples conflict with it (their colour there
	          costs > --conflict-nll above their most likely one, lighting fitted to the candidate: rk_conflicts) —
	          a test of the stickers themselves, independent of how well this view fits overall. A candidate C ≠ S replaces S when it beats S by ≥ --switch-margin on --confirm
	          consecutive frames (C the same each time), or on one frame by ≥ --strong-margin.
	global    no match within two turns on --global-after frames in a row (more turns in one gap than that, or the
	          state lost track of): a global search — acquire again from fresh evidence, as at the start.

HUD: as state_tracker.py — the latest detection's stickers; top right the followed state (cube net in the locked
colours; while acquiring, the evidence panel), the turns recognised so far, the last decision (stay / pending /
switch, with margins) and, with truth labels (a video from klstudio's 🎬 with ?videoTwists=), the error of the shown
state against the frame's own.

Writes OUT/frame*.png, OUT/follow.mp4, OUT/log.json; prints per true turn when (and whether) it was followed.
"""
import argparse, json, os, subprocess, sys, time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

import rubik_engine as re  # noqa: E402
from rubik_detector import cube  # noqa: E402
from hud_video import centre_error, draw, lab_to_bgr  # noqa: E402
from state_tracker import FACES, Tracker, estimate, panel  # noqa: E402

MOVES = [f + m for f in "RLUDFB" for m in ("", "'", "2")]


def face_map(state, truth):
	"""renaming of faces from the state's cube frame to the truth's (the whole-cube rotation that makes them equal), or
	None"""
	for g in range(24):
		raw = [state[cube.PERMS[g][k]] for k in range(54)]
		names = {raw[f * 9 + 4]: FACES[f] for f in range(6)}
		if "".join(names[c] for c in raw) == truth:
			return {x: names[x] for x in FACES}
	return None


def neighbours(state, moves=""):
	"""(state, moves) one face turn away; not the same face again"""
	last = moves.split()[-1][0] if moves else None
	return [(re.apply_moves(state, m), (moves + " " + m).strip()) for m in MOVES if m[0] != last]


class Follower:
	def __init__(self, args, state, model, colours):
		self.a = args
		self.state = state
		self.model = model
		self.colours = colours
		self.moves = []             # recognised turns: (move string, camera frame it was decided on)
		self.pending = None         # (candidate state, moves, count)
		self.fits = []              # S's own per-sample nll on frames it explained
		self.misses = 0
		self.decision = "locked"
		self.cache = (None, None)   # (S, its candidates)

	def candidates(self):
		"""(state, moves) within two face turns of S, distinct states, the fewest turns first"""
		if self.cache[0] != self.state:
			seen = {self.state}
			out = [(self.state, "")]
			first = neighbours(self.state)
			for c in first + [c for state, m in first for c in neighbours(state, m)]:
				if c[0] not in seen:
					seen.add(c[0])
					out.append(c)
			self.cache = (self.state, out)
		return self.cache[1]

	def score(self, r, candidates):
		"""best (score + prior, candidate index, rotation) over the candidates and the 24 view renamings"""
		n = sum(1 for s in r["samples"] if s["valid"])
		states = [c for c, _ in candidates]
		prior = np.array([self.a.move_cost * len(m.split()) for _, m in candidates])
		best = None
		for g in range(24):
			s = re.score_model(self.model, r, g, states) + prior
			i = int(np.argmin(s))
			if best is None or s[i] < best[0]:
				best = (s[i], i, g, s)
		return best, n

	def step(self, r, frame):
		"""one good detection: returns the decision record"""
		candidates = self.candidates()
		(total, i, g, s), n = self.score(r, candidates)
		stay = s[0]
		# a match: the best candidate contradicted by at most --max-conflicts of the frame's samples
		clash = re.conflicts(self.model, r, g, candidates[i][0], self.a.conflict_nll)
		record = {"frame": frame, "rotation": g, "depth": len(candidates[i][1].split()), "per_sample": total / n,
			"conflicts": len(clash), "state_conflicts": len(re.conflicts(self.model, r, g, self.state, self.a.conflict_nll))}
		if len(clash) > self.a.max_conflicts:
			self.misses += 1
			record["decision"] = f"no match ({self.misses})"
			self.pending = None
			self.decision = f"no match within 2 turns: {len(clash)} stickers conflict ({self.misses}/{self.a.global_after})"
			return record
		self.misses = 0
		self.fits.append(total / n)
		if i == 0:
			others = np.delete(s, 0)
			record.update(decision="stay", margin=float(others.min() - stay))
			self.pending = None
			self.decision = f"stay (next best +{others.min() - stay:.0f})"
			return record
		state, moves = candidates[i]
		margin = float(stay - total)
		record.update(candidate=moves, margin=margin)
		if margin < self.a.switch_margin:
			record["decision"] = "weak"
			self.decision = f"{moves}? weak (+{margin:.0f})"
			self.pending = None
			return record
		count = self.pending[2] + 1 if self.pending and self.pending[0] == state else 1
		if count >= self.a.confirm or margin >= self.a.strong_margin:
			self.state = state
			self.moves.append((moves, frame))
			self.pending = None
			record["decision"] = "switch"
			self.decision = f"turned {moves} (+{margin:.0f})"
		else:
			self.pending = (state, moves, count)
			record["decision"] = "pending"
			self.decision = f"{moves}? {count}/{self.a.confirm} (+{margin:.0f})"
		return record


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("video")
	ap.add_argument("--latency-scale", type=float, default=1.0)
	ap.add_argument("--camera-ms", type=float, default=30.0)
	ap.add_argument("--out-fps", type=float, default=5.0)
	ap.add_argument("--known-fov", action="store_true")
	ap.add_argument("--fallback-cost", type=float, default=0.15)
	ap.add_argument("--evidence-cost", type=float, default=0.09)
	ap.add_argument("--relative-cost", type=float, default=1.5)
	ap.add_argument("--confirm", type=int, default=2, help="follow: consecutive frames a new state must win")
	ap.add_argument("--link-deg", type=float, default=20.0)
	ap.add_argument("--agree-colour", type=float, default=12.0)
	ap.add_argument("--unit", default="frame")
	ap.add_argument("--outlier", type=float, default=0.0)
	ap.add_argument("--legal-from", type=int, default=45)
	ap.add_argument("--lock", type=float, default=0.99, help="every facelet's confidence to start following")
	ap.add_argument("--lock-coverage", type=int, default=54, help="observed facelets to start following")
	ap.add_argument("--lock-stable", type=int, default=3, help="consecutive estimates with the same sure state")
	ap.add_argument("--move-cost", type=float, default=10.0, help="prior: nll per face turn")
	ap.add_argument("--switch-margin", type=float, default=15.0)
	ap.add_argument("--strong-margin", type=float, default=80.0)
	ap.add_argument("--conflict-nll", type=float, default=4.0, help="a sample conflicts with a state when its colour there costs this much more than its best")
	ap.add_argument("--max-conflicts", type=int, default=2, help="conflicting samples a match may have (glare, a fingertip)")
	ap.add_argument("--global-after", type=int, default=3, help="frames in a row with no match within two turns: global search")
	ap.add_argument("--out", default=None)
	a = ap.parse_args()
	a.confirm_chain = 1

	labels = json.load(open(os.path.join(a.video, "labels.json")))
	fps, n = labels["video"]["fps"], labels["video"]["frames"]
	k = labels["intrinsics"]
	K_truth = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
	truth_at = lambda i: labels["views"][i].get("facelets", labels["cube"]["facelets"])
	frame_path = lambda i: os.path.join(a.video, f"frame{i + 1:04d}.png")
	arrive = lambda i: i / fps + a.camera_ms / 1000
	out = a.out or os.path.join(a.video, f"follow-x{a.latency_scale:g}")
	os.makedirs(out, exist_ok=True)

	# the tracker takes state_tracker's options; its chain confirmation is 1 (a.confirm is the follower's)
	targs = argparse.Namespace(**vars(a))
	targs.confirm = a.confirm_chain
	tracker = Tracker(targs, K_truth if a.known_fov else None)
	follower = None
	events, free = [], arrive(0)
	if not a.known_fov:
		t0 = time.perf_counter()
		tracker.calibrate(cv2.imread(frame_path(0)))
		ms = (time.perf_counter() - t0) * 1000
		events.append({"kind": "fov", "frame": 0, "ready": free + ms * a.latency_scale / 1000, "ms": ms})
		free = events[-1]["ready"]
	last, est, est_count, stable = -1, None, 0, []
	while True:
		i = int(np.floor((free - a.camera_ms / 1000) * fps + 1e-9))
		if i <= last:
			i = last + 1
		if i >= n:
			break
		start = max(free, arrive(i))
		t0 = time.perf_counter()
		ev = tracker.detect(cv2.imread(frame_path(i)), i)
		ev.update(kind="detect", frame=i, start=start)
		r = ev["result"]
		if follower is None:
			# acquire: estimate on fresh evidence, lock when sure
			if len(tracker.frames) > est_count:
				est = estimate(tracker.frames, tracker.rotations, a)
				est_count = len(tracker.frames)
				sure = est["ok"] and est["legal"] and est["min_confidence"] >= a.lock and est["covered"] >= a.lock_coverage
				stable = stable + [est["state"]] if sure and (not stable or stable[-1] == est["state"]) else ([est["state"]] if sure else [])
				if len(stable) >= a.lock_stable:
					colours = {}
					for c in FACES:
						labs = [s["lab"] for v, g in zip(tracker.frames, tracker.rotations) for s in v["samples"]
							if s["valid"] and est["state"][cube.PERMS[g][s["facelet"]]] == c]
						if labs:
							colours[c] = lab_to_bgr(np.median(np.array(labs), axis=0))
					follower = Follower(a, est["state"], est["model"], colours)
					tracker.collect = False
					ev["locked"] = est["state"]
			ev["stage"] = "acquire"
			ev["estimate"] = est
		elif ev["good"] and tracker.fits_well(r):
			rec = follower.step(r, i)
			ev["follow"] = rec
			if follower.misses >= a.global_after:
				# no match within two turns: global search (acquire from scratch)
				follower, est, est_count, stable = None, None, 0, []
				tracker.frames, tracker.rotations, tracker.pending = [], [], []
				tracker.collect = True
				ev["global"] = True
		ms = (time.perf_counter() - t0) * 1000
		ev.update(ready=start + ms * a.latency_scale / 1000, ms=ms,
			snapshot=None if follower is None else {"state": follower.state, "moves": [m for m, _ in follower.moves],
				"decision": follower.decision, "colours": follower.colours})
		ev["following"] = follower is not None
		events.append(ev)
		free, last = ev["ready"], i
	dets = [e for e in events if e["kind"] == "detect"]
	fov = next((e for e in events if e["kind"] == "fov"), None)

	# output, downsampled
	step = max(1, int(round(fps / a.out_fps)))
	log = {"fps": fps, "out_fps": fps / step, "latency_scale": a.latency_scale, "frames": []}
	for j, i in enumerate(range(0, n, step)):
		t = arrive(i)
		ready = [e for e in dets if e["ready"] <= t]
		lastd = ready[-1] if ready else None
		shown = next((e for e in reversed(ready) if e["good"]), None)
		if shown and lastd is not shown:
			shown = None
		frame = cv2.imread(frame_path(i))
		scale = frame.shape[1] / 1024
		v = labels["views"][i]
		info = [f"t {t:4.2f}s  camera frame {i + 1}/{n}  {v['phase']}" + (f" {v['twist']['move']}" if v.get("twist") else "")]
		if shown:
			info.append(f"detection of frame {shown['frame'] + 1} ({shown['mode']}), age {(t - shown['frame'] / fps) * 1000:.0f} ms, cost {shown['cost']:.3f}")
		elif lastd:
			info.append(f"lost (frame {lastd['frame'] + 1}: cost {lastd['cost']:.2f})")
		img = draw(frame, shown["result"] if shown else None, info, scale)
		snap = lastd["snapshot"] if lastd else None
		truth = truth_at(i)
		if snap:
			wrong = re.compare_states(snap["state"], truth)
			e = {"ok": True, "legal": True, "state": snap["state"], "probability": np.eye(6)[[FACES.index(c) for c in snap["state"]]],
				"observed": [True] * 54, "confidence": [1.0] * 54, "covered": 54, "min_confidence": 1.0}
			moves = snap["moves"]
			extra = [f"turns followed ({len(moves)}): " + (" ".join(moves[-8:]) if moves else "-"), f"last: {snap['decision']}"]
			panel(img, e, None, "follow", truth, scale, colours=snap["colours"], extra=extra, numbers=False)
		else:
			stage = "FOV estimation" if fov and fov["ready"] > t else "acquire"
			est = lastd.get("estimate") if lastd else None
			# the observations behind it (the evidence kept at lock, in order)
			obs = [[] for _ in range(54)]
			if est:
				for vv, g in zip(tracker.frames[:est["frames"]], tracker.rotations[:est["frames"]]):
					for smp in vv["samples"]:
						if smp["valid"]:
							obs[cube.PERMS[g][smp["facelet"]]].append(smp["lab"])
			panel(img, est, obs, stage, truth, scale)
			wrong = None
		cv2.imwrite(os.path.join(out, f"frame{j + 1:04d}.png"), img)
		tq = [np.array(s["corners"]) for s in v["stickers"] if s["visible"]]
		log["frames"].append({"camera_frame": i, "phase": v["phase"], "following": bool(snap), "wrong": wrong,
			"error_cubie": centre_error([s["quad"] for s in shown["result"]["samples"] if s["valid"]], tq) if shown else None})
	subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps / step), "-i", os.path.join(out, "frame%04d.png"),
		"-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, "follow.mp4")], check=True)
	log["detections"] = [{k_: v_ for k_, v_ in e.items() if k_ not in ("result", "snapshot", "estimate")} for e in dets]
	json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))

	# summary
	lock = next((e for e in dets if "locked" in e), None)
	print(f"{len(dets)} detections; locked at {lock['ready']:.1f} s (frame {lock['frame'] + 1}), "
		f"{'exact' if re.compare_states(lock['locked'], truth_at(lock['frame'])) == 0 else 'WRONG'}" if lock else "never locked")
	switches = [(e["follow"]["candidate"], e["frame"], e["ready"]) for e in dets if e.get("follow", {}).get("decision") == "switch"]
	names = face_map(lock["locked"], truth_at(lock["frame"])) if lock else None
	in_video = lambda moves: " ".join(names[m[0]] + m[1:] for m in moves.split()) if names else moves
	for t in labels["video"].get("twists", []):
		end = t["twist"][1] / fps
		hit = next((s for s in switches if s[1] >= t["twist"][0] and re.compare_states(
			next(e["snapshot"]["state"] for e in dets if e["frame"] == s[1]), t["facelets"]) == 0), None)
		print(f"  turn {t['move']:3s} done at {end:5.2f} s: " + (f"followed as {in_video(hit[0])} at {hit[2]:.2f} s (+{(hit[2] - end) * 1000:.0f} ms)" if hit else "NOT followed"))
	print(f"  switches: {len(switches)} — " + " ".join(in_video(m) for m, _, _ in switches) + " (in the video's frame)")
	for e in dets:
		f = e.get("follow", {})
		if f.get("decision") == "switch" or e.get("global") or "locked" in e:
			what = f"switch {in_video(f['candidate'])}" if f.get("decision") == "switch" else "GLOBAL search" if e.get("global") else "lock"
			state = e["snapshot"]["state"] if e.get("snapshot") else None
			wrong = re.compare_states(state, truth_at(e["frame"])) if state else None
			print(f"    {e['ready']:6.2f} s  frame {e['frame'] + 1:4d} ({labels['views'][e['frame']]['phase']:8s}) {what:18s}"
				+ (f" state {'exact' if wrong == 0 else str(wrong) + ' wrong'}" if wrong is not None else ""))
	fr = [f for f in log["frames"] if f["following"]]
	print(f"  shown frames following: {len(fr)}, state exact on {sum(f['wrong'] == 0 for f in fr)} "
		f"(settle/hold: {sum(f['wrong'] == 0 for f in fr if f['phase'] in ('settle', 'hold'))}/{sum(f['phase'] in ('settle', 'hold') for f in fr)})")
	print(f"→ {out}")


if __name__ == "__main__":
	main()
