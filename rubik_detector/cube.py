"""Facelet conventions, shared with klstudio's sample generator (app/rubikSamples/sampler.ts, inc/cubeN.ts).

Cube frame: x → R, y → U, z → F, unit = cubie pitch, the cube spans ±1.5. Faces in CubeN order LRDUBF
(-X +X -Y +Y -Z +Z). Facelet k = face·9 + row·3 + col, rows and columns as seen from outside the face, along
FRAMES[face] = (right, down). A state is 54 colour labels named by the face whose centre carries that colour, so the
solved cube is "LLLLLLLLLRRRRRRRRR…". Two states describe the same physical cube if one is a whole-cube rotation of
the other (24 of them); `compare` counts wrong facelets up to that.
"""
import itertools

import numpy as np


FACES = "LRDUBF"
AXIS = {"L": (0, -1), "R": (0, 1), "D": (1, -1), "U": (1, 1), "B": (2, -1), "F": (2, 1)}
FRAMES = {
	"L": ((0, 0, 1), (0, -1, 0)),
	"R": ((0, 0, -1), (0, -1, 0)),
	"D": ((1, 0, 0), (0, 0, -1)),
	"U": ((1, 0, 0), (0, 0, 1)),
	"B": ((-1, 0, 0), (0, -1, 0)),
	"F": ((1, 0, 0), (0, -1, 0)),
}


def _normal(face):
	axis, sign = AXIS[face]
	n = np.zeros(3, int)
	n[axis] = sign
	return n


NORMALS = np.array([_normal(f) for f in FACES])
FACE_OF = np.arange(54) // 9
ROW = (np.arange(54) % 9) // 3
COL = np.arange(54) % 3
RIGHTS = np.array([FRAMES[FACES[f]][0] for f in FACE_OF], float)
DOWNS = np.array([FRAMES[FACES[f]][1] for f in FACE_OF], float)
FACE_NORMALS = NORMALS[FACE_OF].astype(float)
CENTRE_FACELETS = np.arange(6) * 9 + 4


def facelet_points(plane=1.5, offsets=((0.0, 0.0),)):
	"""Cube-frame points of every facelet: the centre on a face plane at `plane`, moved by (a, b) along (right, down).
	Returns (54, len(offsets), 3)."""
	centre = plane * FACE_NORMALS + (COL - 1)[:, None] * RIGHTS + (ROW - 1)[:, None] * DOWNS
	return np.stack([centre + a * RIGHTS + b * DOWNS for a, b in offsets], axis=1)


CENTRES = facelet_points()[:, 0]

_POS2 = np.rint(2 * CENTRES).astype(int)
_INDEX = {tuple(p): k for k, p in enumerate(_POS2)}

ROTATIONS = np.array([m for m in (
	np.array([[s[i] if p[i] == j else 0 for j in range(3)] for i in range(3)])
	for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)
) if round(np.linalg.det(m)) == 1])
IDENTITY = int(np.flatnonzero([(g == np.eye(3)).all() for g in ROTATIONS])[0])
# PERMS[g][k]: where facelet k goes under whole-cube rotation g
PERMS = np.array([[_INDEX[tuple(g @ p)] for p in _POS2] for g in ROTATIONS])


def relabel(state):
	"""Name each label by the face its centre sits on."""
	state = list(state)
	names = {state[c]: FACES[f] for f, c in enumerate(CENTRE_FACELETS)}
	if len(names) != 6:
		raise ValueError("the six centres must carry six distinct labels")
	return "".join(names[c] for c in state)


def rotate(state, g):
	"""The state after whole-cube rotation g (labels unchanged)."""
	out = [None] * 54
	for k, c in enumerate(state):
		out[PERMS[g][k]] = c
	return out


def compare(pred, truth):
	"""(wrong facelets, rotation) of the best whole-cube rotation of `pred` onto `truth`."""
	best = (55, None)
	for g in range(len(ROTATIONS)):
		r = relabel(rotate(pred, g))
		wrong = sum(a != b for a, b in zip(r, truth))
		if wrong < best[0]:
			best = (wrong, g)
	return best
