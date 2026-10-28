"""Overlay a simulated detection HUD on a video, with the engine's real latency.

	python engine/tools/hud_video.py VIDEO_DIR [--latency-scale 1] [--overhead-ms 0] [--fallback-cost 0.15] [--out DIR]

VIDEO_DIR: frames frame0001.png … plus labels.json (klstudio /rubik-samples 🎬: intrinsics, video.fps, per-frame
labels). Timeline: frame k arrives at k / fps. The detector takes the newest frame that has arrived whenever it is
free, and its result is ready after the measured engine time × --latency-scale + --overhead-ms (a phone is slower
than this desktop: try 2–3). Every displayed frame gets the HUD of the latest result ready by its arrival — drawn
at that result's pose, so the overlay lags the cube until the next result.

Policy (as an app would): wide search first; then local from the last good pose, falling back to wide when the
lattice cost passes --fallback-cost. Results whose pose turned more than --new-view-deg since the last kept one
are kept as views (colour fusion, all faces not required); once they cover all 54 facelets the state is estimated.

Writes OUT/hud0001.png …, OUT/hud.mp4 (ffmpeg, the video's fps), OUT/log.json (per detection and per displayed
frame), and prints the lag statistics: with truth labels, per displayed frame the distance between the HUD's sticker
centres and the true ones of that frame (cubie units), against what an instant detector would show.
"""
import argparse, json, os, subprocess, sys, time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import rubik_engine as re  # noqa: E402


def lab_to_bgr(lab):
	px = np.array([[lab]], np.float32)
	return tuple(int(c) for c in np.clip(cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0] * 255, 0, 255))


def rotation_deg(Ra, Rb):
	return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def centre_error(quads, truth):
	"""mean distance between HUD sticker centres and the frame's true visible ones (matched), in cubie units"""
	if not quads or not truth:
		return None
	pc = np.array([np.mean(q, 0) for q in quads])
	tc = np.array([np.mean(q, 0) for q in truth])
	d = np.linalg.norm(pc[:, None] - tc[None], axis=2)
	r, c = linear_sum_assignment(d)
	pitch = np.median(np.sort(np.linalg.norm(tc[:, None] - tc[None], axis=2), axis=1)[:, 1])
	return float(d[r, c].mean() / pitch)


def draw(frame, result, info, scale):
	img = frame.copy()
	if result is not None and result["ok"]:
		overlay = img.copy()
		for s in result["samples"]:
			if not s["valid"]:
				continue
			q = np.rint(s["quad"]).astype(np.int32)
			cv2.fillConvexPoly(overlay, q, lab_to_bgr(s["lab"]))
		img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
		for s in result["samples"]:
			if s["valid"]:
				cv2.polylines(img, [np.rint(s["quad"]).astype(np.int32)], True, (255, 255, 255), max(1, round(scale)), cv2.LINE_AA)
	lines = info
	y = int(24 * scale)
	for line in lines:
		cv2.putText(img, line, (int(10 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), max(2, round(4 * scale)), cv2.LINE_AA)
		cv2.putText(img, line, (int(10 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), max(1, round(1.5 * scale)), cv2.LINE_AA)
		y += int(24 * scale)
	return img


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("video")
	ap.add_argument("--latency-scale", type=float, default=1.0)
	ap.add_argument("--overhead-ms", type=float, default=0.0)
	ap.add_argument("--fallback-cost", type=float, default=0.15)
	ap.add_argument("--new-view-deg", type=float, default=35.0)
	ap.add_argument("--out", default=None)
	a = ap.parse_args()

	labels = json.load(open(os.path.join(a.video, "labels.json")))
	fps = labels["video"]["fps"]
	n = labels["video"]["frames"]
	k = labels["intrinsics"]
	K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
	frames = [cv2.imread(os.path.join(a.video, f"frame{i + 1:04d}.png")) for i in range(n)]
	out = a.out or os.path.join(a.video, f"hud-x{a.latency_scale:g}")
	os.makedirs(out, exist_ok=True)
	truth = labels.get("cube", {}).get("facelets")

	det = re.Detector()
	detections = []          # (frame, ready time, result, mode, ms)
	views, last_kept = [], None
	state = None
	free, prev = 0.0, None
	while True:
		f = min(n - 1, int(np.floor(free * fps + 1e-9)))
		if detections and f <= detections[-1]["frame"]:
			f = detections[-1]["frame"] + 1   # wait for a new frame
			if f >= n:
				break
			free = f / fps
		t0 = time.perf_counter()
		mode = "wide"
		r = None
		if prev is not None:
			r = det.detect(frames[f], K, prior=(prev["R"], prev["t"]))
			mode = "local"
		if r is None or not r["ok"] or r["lattice_cost"] > a.fallback_cost:
			r2 = det.detect(frames[f], K)
			if mode == "local":
				mode = "fallback"
			r = r2
		ms = (time.perf_counter() - t0) * 1000
		ready = max(free, f / fps) + (ms * a.latency_scale + a.overhead_ms) / 1000
		good = r["ok"] and r["lattice_cost"] <= a.fallback_cost
		rec = {"frame": f, "start": max(free, f / fps), "ready": ready, "mode": mode, "ms": ms, "ok": r["ok"], "good": good, "cost": r["lattice_cost"]}
		prev = r if r["ok"] and r["lattice_cost"] <= a.fallback_cost else None
		# views for the state: good fits whose pose turned enough since the last kept one
		if r["ok"] and r["lattice_cost"] <= a.fallback_cost and (last_kept is None or rotation_deg(last_kept["R"], r["R"]) > a.new_view_deg):
			views.append(r)
			last_kept = r
			rec["kept_view"] = len(views)
			if len(views) >= 2:
				e = re.estimate(views, require_all_faces=False)
				if e["ok"]:
					state = {"at": ready, "state": e["state"], "min_posterior": e["min_posterior"], "views": len(views),
						"wrong": re.compare_states(e["state"], truth) if truth else None}
					rec["state"] = state
		rec["result"] = r
		detections.append(rec)
		free = ready
		if free * fps >= n:
			break

	# display: each frame with the latest result ready by its arrival
	log = {"fps": fps, "latency_scale": a.latency_scale, "detections": [], "frames": []}
	errs, instant = [], []
	cur = None
	for i in range(n):
		t = i / fps
		ready = [d for d in detections if d["ready"] <= t]
		last = ready[-1] if ready else None
		# a result that failed the fit test is not shown: the HUD says lost until a good one is ready
		d = last if last and last["good"] else None
		st = None
		for x in detections:
			if x["ready"] <= t and "state" in x:
				st = x["state"]
		tq = [np.array(s["corners"]) for s in labels["views"][i]["stickers"] if s["visible"]]
		hud_q = [s["quad"] for s in d["result"]["samples"] if s["valid"]] if d and d["result"]["ok"] else []
		err = centre_error(hud_q, tq)
		info = [f"t {t:4.1f}s  frame {i + 1}/{n}  {labels['views'][i]['phase']}"]
		if d:
			info.append(f"result of frame {d['frame'] + 1} ({d['mode']}, {d['ms'] * a.latency_scale + a.overhead_ms:.0f} ms), age {(t - d['frame'] / fps) * 1000:.0f} ms, cost {d['cost']:.3f}")
		elif last:
			info.append(f"lost (frame {last['frame'] + 1}: cost {last['cost']:.2f}), searching...")
		else:
			info.append("detecting...")
		info.append(f"views kept {sum(1 for x in detections if x['ready'] <= t and 'kept_view' in x)}"
			+ (f", state {'exact' if st['wrong'] == 0 else str(st['wrong']) + ' wrong'} (min post {st['min_posterior']:.2f})" if st and st.get("wrong") is not None else (", state found" if st else "")))
		scale = frames[i].shape[1] / 1024
		cv2.imwrite(os.path.join(out, f"hud{i + 1:04d}.png"), draw(frames[i], d["result"] if d else None, info, scale))
		log["frames"].append({"frame": i, "shows": d["frame"] if d else None, "error_cubie": err, "phase": labels["views"][i]["phase"]})
		errs.append((labels["views"][i]["phase"], err))
	for d in detections:
		x = {k_: v for k_, v in d.items() if k_ != "result"}
		tq = [np.array(s["corners"]) for s in labels["views"][d["frame"]]["stickers"] if s["visible"]]
		x["own_frame_error"] = centre_error([s["quad"] for s in d["result"]["samples"] if s["valid"]], tq) if d["result"]["ok"] else None
		instant.append((labels["views"][d["frame"]]["phase"], x["own_frame_error"]))
		log["detections"].append(x)
	json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
	subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", os.path.join(out, "hud%04d.png"),
		"-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, "hud.mp4")], check=True)

	modes = {m: sum(1 for d in detections if d["mode"] == m) for m in ("wide", "local", "fallback")}
	lat = [d["ready"] - d["start"] for d in detections]
	print(f"{len(detections)} detections over {n} frames ({n / fps:.1f} s): {modes}; latency mean {np.mean(lat) * 1000:.0f} ms, max {max(lat) * 1000:.0f} ms; "
		f"frames processed every {np.mean(np.diff([d['frame'] for d in detections])):.1f} frames")
	for phase in ("hold", "turn"):
		allp = [x for p, x in errs if p == phase]
		e = [x for x in allp if x is not None]
		own = [x for p, x in instant if p == phase and x is not None]
		if allp:
			print(f"  {phase}: {len(allp)} frames, HUD shown on {len(e)}; on those, error vs the displayed frame's truth mean {np.mean(e) if e else float('nan'):.2f} cubie, "
				f"under 0.3 cubie on {sum(x < 0.3 for x in e)}/{len(allp)} frames; detections on their own frames mean {np.mean(own) if own else float('nan'):.2f}")
	if state:
		print(f"  state: {'exact' if state['wrong'] == 0 else str(state['wrong']) + ' wrong'} from {state['views']} views, ready at {state['at']:.1f} s")
	else:
		print("  state: not found")
	print(f"→ {out}")


if __name__ == "__main__":
	main()
