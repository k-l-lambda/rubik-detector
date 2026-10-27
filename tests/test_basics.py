"""python -m pytest tests  (needs the samples in ~/data/rubik-renderer)"""
import json, os

import numpy as np
import pytest

from rubik_detector import colors, cube, data, fusion, geometry, pipeline


def test_rotation_group():
	assert len(cube.ROTATIONS) == 24
	assert all(round(np.linalg.det(g)) == 1 for g in cube.ROTATIONS)
	assert all(sorted(p) == list(range(54)) for p in cube.PERMS)
	# closed under composition
	keys = {g.tobytes() for g in cube.ROTATIONS}
	assert all((a @ b).tobytes() in keys for a in cube.ROTATIONS for b in cube.ROTATIONS)


def test_compare_up_to_rotation():
	rng = np.random.default_rng(0)
	solved = "".join(f * 9 for f in cube.FACES)
	assert cube.compare(solved, solved)[0] == 0
	state = list(solved)
	ks = rng.choice([k for k in range(54) if k % 9 != 4], 2, replace=False)
	state[ks[0]], state[ks[1]] = "U", "F"	# two wrong stickers
	assert cube.compare("".join(state), solved)[0] == sum(solved[k] != c for k, c in zip(ks, "UF"))
	for g in range(24):
		assert cube.compare(cube.rotate(solved, g), solved)[0] == 0


needs_data = pytest.mark.skipif(not os.path.isdir(f"{data.ROOT}/samples/g01"), reason="no samples")


@needs_data
def test_label_conventions():
	"""Facelet centres through the label pose land on the labelled sticker centres."""
	L = json.load(open(f"{data.ROOT}/samples/g01/labels.json"))
	K = data.intrinsics(L)
	for v in L["views"]:
		R, t = np.array(v["rotation"]).reshape(3, 3), np.array(v["translation"])
		uv = geometry.project(K, R, t, cube.facelet_points(plane=L["render"]["stickerPlane"])[:, 0])
		assert np.abs(uv - np.array([s["center"] for s in v["stickers"]])).max() < 1e-6


@needs_data
@pytest.mark.parametrize("geom", ["oracle", "line"])
def test_g01_render_exact(geom):
	s = data.load("g01", "render")
	r = pipeline.run(s, geom)
	assert r["wrong"] == 0
	assert all(v["geometry_error"]["max"] < 0.25 for v in r["views"])


@needs_data
def test_fusion_finds_identity_with_true_frames():
	s = data.load("g02", "render")
	obs = []
	for v in s.views:
		lab = colors.to_lab(v.image)
		obs.append({k: colors.sample(lab, q) for k, q in geometry.oracle(v)["quads"].items()})
	_, rotations, _ = fusion.fuse(obs)
	assert rotations == [cube.IDENTITY] * 3


@needs_data
def test_legality_of_truth_and_map_recovery():
	from rubik_detector import legal
	rng = np.random.default_rng(3)
	for g in data.groups()[:20]:
		truth = legal.to_labels(json.load(open(f"{data.ROOT}/samples/{g}/labels.json"))["cube"]["facelets"])
		assert legal.is_legal(truth)
		cost = rng.normal(0, 0.3, (54, 6)) + 3.0
		cost[np.arange(54), truth] -= 3.0
		k = rng.choice([k for k in range(54) if k % 9 != 4])
		cost[k, (truth[k] + 1) % 6] -= 4.0	# one sticker clearly misread
		labels, _, _ = legal.best_legal(cost)
		assert (labels == truth).all()
