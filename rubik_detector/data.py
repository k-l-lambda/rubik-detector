"""Labelled samples from ~/data/rubik-renderer (override with RUBIK_DATA).

	samples/rNNN/view{1,2,3}.jpg + labels.json    renders with phone-photo realism (klstudio /rubik-samples ?realism=1, seed
	                                              5000 + NNN): all colour schemes and builds, 1024×768, 768×1024, 1280×720 or
	                                              720×1280, the cube small and off centre on a cluttered desk, defocus, shake,
	                                              noise, white balance, JPEG
	samples/gNN/view{1,2,3}.png + labels.json     clean renders (seed = NN), 1024×768 — kept only where they have edits
	samples/nNN/…                                 the same with non-standard colour schemes (seed 1000 + NN)
	samples/gNN/edit/view{v}-{bg,hand}.jpg + .json seedream edits of a render: natural background / a hand holding it;
	                                              the .json has the checker's verdict and the sticker corners carried
	                                              over into the edit's pixels (the edit moves the cube a little)

A scenario is one group seen as three photos of one kind ("render", "bg" or "hand"). Images are resized to a working
width (default 1024) and K is scaled with them. Truth per view: the sticker quads (corners in working pixels) and
which facelets show; per group: the 54-letter state. Label conventions: rubik_detector.cube.
"""
import dataclasses, json, os

import cv2
import numpy as np


ROOT = os.path.expanduser(os.environ.get("RUBIK_DATA", "~/data/rubik-renderer"))
KINDS = ("render", "bg", "hand")


@dataclasses.dataclass
class View:
	name: str
	image: np.ndarray		# BGR uint8, working size
	K: np.ndarray			# 3×3, working pixels
	quads: dict			# facelet → (4, 2) sticker corners (truth), visible ones only
	occluded: set			# facelets covered by a finger (hand edits)
	R: np.ndarray = None		# truth pose (renders only: an edit moves the cube)
	t: np.ndarray = None


@dataclasses.dataclass
class Scenario:
	group: str
	kind: str
	state: str			# truth, 54 letters
	views: list
	labels: dict
	verdicts: list = None		# edits: the checker's verdict per view (PASS / WARN / FAIL)


def groups(prefix="g"):
	"""Group names of one sample set: r = realism renders (seed 5000 + NNN); g / n / a = the clean renders kept for their
	edits (standard / non-standard colour schemes, anti-pop)."""
	return sorted(d for d in os.listdir(f"{ROOT}/samples") if d.startswith(prefix) and d[1:].isdigit())


def intrinsics(labels):
	k = labels["intrinsics"]
	return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], float)


def load(group, kind="render", width=1024, include_failed=False):
	"""The scenario, or None if that kind is missing for the group, or an edit failed its check (unless include_failed;
	an edit without transferred sticker labels — no image, or not registered — is always None: no truth)."""
	labels = json.load(open(f"{ROOT}/samples/{group}/labels.json"))
	K0 = intrinsics(labels)
	W0 = labels["image"]["width"]
	views, verdicts = [], []
	for v in range(3):
		lv = labels["views"][v]
		if kind == "render":
			path, scale = f"{ROOT}/samples/{group}/view{v + 1}.png", 1.0
			if not os.path.exists(path):
				path = f"{ROOT}/samples/{group}/view{v + 1}.jpg"
			stickers = [(s["index"], s["corners"]) for s in lv["stickers"] if s["visible"]]
			occluded = set()
		else:
			path = f"{ROOT}/samples/{group}/edit/view{v + 1}-{kind}.jpg"
			meta = f"{ROOT}/samples/{group}/edit/view{v + 1}-{kind}.json"
			if not os.path.exists(path) or not os.path.exists(meta):
				return None
			rec = json.load(open(meta))
			if (rec.get("verdict") == "FAIL" and not include_failed) or "stickers" not in rec:
				return None
			verdicts.append(rec.get("verdict"))
			scale = rec["image_size"][0] / W0	# the record's corners are in the edit's pixels: back to render units
			stickers = [(s["index"], (np.array(s["corners"]) / scale).tolist()) for s in rec["stickers"]]
			occluded = {s["index"] for s in rec["stickers"] if s["occluded"]}
		image = cv2.imread(path)
		f = width / image.shape[1]
		image = cv2.resize(image, (width, round(image.shape[0] * f)), interpolation=cv2.INTER_AREA)
		s = width / W0	# render units → working pixels
		K = K0 * np.array([[s], [s], [1]])
		quads = {k: np.array(c) * s for k, c in stickers}
		pose = dict(R=np.array(lv["rotation"]).reshape(3, 3), t=np.array(lv["translation"])) if kind == "render" else {}
		views.append(View(f"{group}/{kind}/view{v + 1}", image, K, quads, occluded, **pose))
	return Scenario(group, kind, labels["cube"]["facelets"], views, labels, verdicts or None)


def scenarios(kinds=KINDS, groups_=None, width=1024):
	for g in groups_ or groups():
		for kind in kinds:
			s = load(g, kind, width)
			if s:
				yield s
