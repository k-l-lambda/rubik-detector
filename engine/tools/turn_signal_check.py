"""FrameTracker's turning-layer signal against a rendered video's true twists (klstudio 🎬 ?videoTwists=): tracked
through the whole video (re-seeded from the true pose only when lost), per frame the signal's face normal against the
turning face's (twist frames) and false signals elsewhere.

	python engine/tools/turn_signal_check.py [VIDEO_DIR]
"""
import json, os, sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))


def main():
	import rubik_engine as re
	V = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/rubik-renderer/videos/v103")
	L = json.load(open(V + "/labels.json"))
	k = L["intrinsics"]
	K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
	views = L["views"]
	pose = lambda i: (np.array(views[i]["rotation"]).reshape(3, 3), np.array(views[i]["translation"]))
	tr = re.Tracker()
	load = lambda i: cv2.imread(f"{V}/frame{i + 1:04d}.png") if os.path.exists(f"{V}/frame{i + 1:04d}.png") else cv2.imread(f"{V}/frame{i + 1:04d}.jpg")
	tr.seed(load(0), K, *pose(0))
	per_phase, per_twist = {}, {}
	for i in range(1, len(views)):
		r = tr.track(load(i))
		if not r["ok"]:
			tr.seed(load(i), K, *pose(i))
			continue
		v = views[i]
		ph = v["phase"]
		c = per_phase.setdefault(ph, {"frames": 0, "signal": 0, "right": 0})
		c["frames"] += 1
		if r["turning"]:
			c["signal"] += 1
		if ph == "twist":
			tw = v["twist"]
			n = np.zeros(3)
			n[tw["axis"]] = 1 if tw["layer"] == 2 else -1
			true_normal = pose(i)[0] @ n
			right = r["turning"] and float(true_normal @ r["turn_normal"]) > 0.9
			c["right"] += right
			key = (v["hold"], tw["move"])
			t = per_twist.setdefault(key, {"frames": 0, "right": 0, "first": None, "progress": []})
			t["frames"] += 1
			t["right"] += right
			if right and t["first"] is None:
				t["first"] = v["twist"]["progress"]
	for ph, c in sorted(per_phase.items()):
		extra = f", right face {c['right']}/{c['frames']} ({c['right'] / c['frames']:.0%})" if ph == "twist" else ""
		print(f"{ph:9s} {c['frames']:5d} frames: signal on {c['signal']} ({c['signal'] / c['frames']:.0%}){extra}")
	for (h, move), t in sorted(per_twist.items()):
		print(f"  twist {move:3s}: right face on {t['right']}/{t['frames']} frames, first at "
			+ (f"{t['first']:.0%} of the turn" if t["first"] is not None else "never"))


if __name__ == "__main__":
	main()
