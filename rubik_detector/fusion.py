"""Putting three views into one cube frame.

Each view's geometry stage reports facelets in its own frame (the line detector names the three faces it sees R, U,
F; nothing in one photo says which physical face is which). The views differ by whole-cube rotations: view 1 is the
reference, and for views 2 and 3 all 24 × 24 rotation pairs are tried; a pair must cover all six faces, and the one
whose overlapping facelets (the faces two views share) agree best in colour wins. Colours of one sticker in two photos
differ by lighting; each face's mean is taken out first, so the shared faces' 3×3 patterns decide.
"""
import itertools

import numpy as np

from . import colors, cube


def fuse(views):
	"""views: list of {facelet: Lab} (each view's own frame). Returns (observations, rotations, cost):
	observations = common-frame facelet → [(view, view-frame face, Lab)], rotations = rotation index per view
	(view 0: identity)."""
	# per face of each photo, its mean feature is subtracted: what is left is the face's pattern, which the same face
	# in another photo shares whatever the lighting (a face of a scrambled cube is never one colour)
	feats = []
	for v in views:
		f = {k: colors.weigh(lab) for k, lab in v.items()}
		for face in {cube.FACE_OF[k] for k in f}:
			ks = [k for k in f if cube.FACE_OF[k] == face]
			m = np.mean([f[k] for k in ks], axis=0)
			for k in ks:
				f[k] = f[k] - m
		feats.append(f)
	faces = [{cube.FACE_OF[k] for k in v} for v in views]
	if len(views) == 1:
		return _gather(views, [cube.IDENTITY]), [cube.IDENTITY], 0.0

	def mapped(v, g):
		return {cube.PERMS[g][k]: f for k, f in feats[v].items()}

	best = (np.inf, None)
	others = list(range(1, len(views)))
	# rotations of each other view, with their face sets (a rotation moves faces as a whole)
	cand = {v: [(g, mapped(v, g), {cube.FACE_OF[cube.PERMS[g][9 * f + 4]] for f in faces[v]}) for g in range(24)] for v in others}
	base = feats[0]

	def overlap_cost(maps):
		total, n = 0.0, 0
		for i in range(len(maps)):
			for j in range(i + 1, len(maps)):
				a, b = maps[i], maps[j]
				for k in a.keys() & b.keys():
					total += np.linalg.norm(a[k] - b[k])
					n += 1
		return total / n if n else np.inf, n

	for combo in itertools.product(*[cand[v] for v in others]):
		covered = set(faces[0])
		for _, _, fs in combo:
			covered |= fs
		if len(covered) < 6:
			continue
		cost, n = overlap_cost([base] + [m for _, m, _ in combo])
		if n >= 9 and cost < best[0]:
			best = (cost, [cube.IDENTITY] + [g for g, _, _ in combo])
	if best[1] is None:
		raise ValueError("no rotation of the views covers all six faces")
	return _gather(views, best[1]), best[1], float(best[0])


def _gather(views, rotations):
	"""common-frame facelet → [(view, view-frame face, Lab)]"""
	obs = {}
	for v, (view, g) in enumerate(zip(views, rotations)):
		for k, lab in view.items():
			obs.setdefault(int(cube.PERMS[g][k]), []).append((v, int(cube.FACE_OF[k]), lab))
	return obs
