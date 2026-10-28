#include "cube.hpp"

#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

#include "rubik/engine.h"

namespace rubik {

namespace {

// axis and sign of each face, and its (right, down) frame, as cube.py
const int AXIS[6][2] = {{0, -1}, {0, 1}, {1, -1}, {1, 1}, {2, -1}, {2, 1}};
const int FRAMES[6][2][3] = {
	{{0, 0, 1}, {0, -1, 0}},
	{{0, 0, -1}, {0, -1, 0}},
	{{1, 0, 0}, {0, 0, -1}},
	{{1, 0, 0}, {0, 0, 1}},
	{{-1, 0, 0}, {0, -1, 0}},
	{{1, 0, 0}, {0, -1, 0}},
};

CubeTables build () {
	CubeTables T{};
	for (int k = 0; k < 54; k++) {
		int f = k / 9;
		T.faceOf[k] = f;
		T.row[k] = (k % 9) / 3;
		T.col[k] = k % 3;
		V3 n{0, 0, 0};
		n[AXIS[f][0]] = AXIS[f][1];
		T.normal[k] = n;
		T.right[k] = {double(FRAMES[f][0][0]), double(FRAMES[f][0][1]), double(FRAMES[f][0][2])};
		T.down[k] = {double(FRAMES[f][1][0]), double(FRAMES[f][1][1]), double(FRAMES[f][1][2])};
		T.centre[k] = 1.5 * n + (T.col[k] - 1) * T.right[k] + (T.row[k] - 1) * T.down[k];
		for (int i = 0; i < 3; i++)
			T.pos2[k][i] = int(std::lround(2 * T.centre[k][i]));
	}
	// itertools.permutations(range(3)) × product((1, -1), repeat=3), proper rotations only, in that order
	const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
	int g = 0;
	for (const auto& p : perms) {
		for (int s = 0; s < 8; s++) {
			int sign[3] = {(s & 4) ? -1 : 1, (s & 2) ? -1 : 1, (s & 1) ? -1 : 1};
			std::array<int, 9> m{};
			for (int i = 0; i < 3; i++)
				m[i * 3 + p[i]] = sign[i];
			int det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
			if (det == 1)
				T.rotations[g++] = m;
		}
	}
	std::map<std::array<int, 3>, int> index;
	for (int k = 0; k < 54; k++)
		index[T.pos2[k]] = k;
	for (int r = 0; r < 24; r++) {
		const auto& m = T.rotations[r];
		bool identity = true;
		for (int i = 0; i < 9; i++)
			identity = identity && m[i] == (i % 4 == 0 ? 1 : 0);
		if (identity)
			T.identity = r;
		for (int k = 0; k < 54; k++) {
			std::array<int, 3> q{};
			for (int i = 0; i < 3; i++)
				q[i] = m[i * 3] * T.pos2[k][0] + m[i * 3 + 1] * T.pos2[k][1] + m[i * 3 + 2] * T.pos2[k][2];
			T.perms[r][k] = index.at(q);
		}
	}
	return T;
}

}

const CubeTables& tables () {
	static const CubeTables T = build();
	return T;
}

std::string relabel (const std::string& state) {
	std::map<char, char> names;
	for (int f = 0; f < 6; f++)
		names[state[CENTRE_FACELETS[f]]] = FACES[f];
	if (names.size() != 6)
		throw std::invalid_argument("the six centres must carry six distinct labels");
	std::string out(54, '?');
	for (int k = 0; k < 54; k++)
		out[k] = names.count(state[k]) ? names[state[k]] : '?';
	return out;
}

// --- public cube API ---------------------------------------------------------------------------------------------

const std::array<std::array<int, 9>, 24>& rotations () { return tables().rotations; }
const std::array<int, 54>& rotationPerm (int g) { return tables().perms.at(g); }
int identityRotation () { return tables().identity; }

int compareStates (const std::string& pred, const std::string& truth, int* rotation) {
	int best = 55, bestG = -1;
	for (int g = 0; g < 24; g++) {
		std::string rotated(54, '?');
		for (int k = 0; k < 54; k++)
			rotated[tables().perms[g][k]] = pred[k];
		std::string r;
		try {
			r = relabel(rotated);
		}
		catch (const std::invalid_argument&) {
			continue;
		}
		int wrong = 0;
		for (int k = 0; k < 54; k++)
			wrong += r[k] != truth[k];
		if (wrong < best) {
			best = wrong;
			bestG = g;
		}
	}
	if (rotation)
		*rotation = bestG;
	return best;
}

Turn parseTurn (const std::string& move) {
	if (move.empty())
		throw std::invalid_argument("empty move");
	Turn t;
	char c = move[0];
	size_t i = 1;
	bool wide = i < move.size() && move[i] == 'w';
	if (wide)
		i++;
	// face letter: axis, outer layer bit, and the sign of "clockwise seen from that face" about +axis
	struct Def {
		char c;
		int axis, layers, sign;
	};
	const Def defs[] = {
		{'R', 0, 4, 1}, {'L', 0, 1, -1}, {'U', 1, 4, 1}, {'D', 1, 1, -1}, {'F', 2, 4, 1}, {'B', 2, 1, -1},
		{'M', 0, 2, -1}, {'E', 1, 2, -1}, {'S', 2, 2, 1}, {'x', 0, 7, 1}, {'y', 1, 7, 1}, {'z', 2, 7, 1},
	};
	const Def* d = nullptr;
	for (const Def& e : defs)
		if (e.c == c)
			d = &e;
	if (!d || (wide && (d->layers == 2 || d->layers == 7)))
		throw std::invalid_argument("bad move: " + move);
	t.axis = d->axis;
	t.layers = wide ? (d->layers | 2) : d->layers;
	int q = 1;
	if (i < move.size() && move[i] == '2') {
		q = 2;
		i++;
	}
	if (i < move.size() && move[i] == '\'') {
		q = -q;
		i++;
	}
	if (i != move.size())
		throw std::invalid_argument("bad move: " + move);
	t.quarters = ((q * d->sign) % 4 + 4) % 4;
	return t;
}

const std::vector<std::string>& faceTurnNames () {
	static const std::vector<std::string> names = [] {
		std::vector<std::string> out;
		for (const char* f : {"U", "D", "L", "R", "F", "B"})
			for (const char* s : {"", "'", "2"})
				out.push_back(std::string(f) + s);
		return out;
	}();
	return names;
}

std::string applyTurn (const std::string& state, const Turn& turn) {
	const CubeTables& T = tables();
	if (state.size() != 54)
		throw std::invalid_argument("a state has 54 facelets");
	// a quarter turn clockwise seen from +axis is -90° about +axis: v' = n(n·v) - n×v
	std::map<std::array<int, 3>, int> index;
	for (int k = 0; k < 54; k++)
		index[T.pos2[k]] = k;
	std::string cur = state;
	for (int q = 0; q < ((turn.quarters % 4) + 4) % 4; q++) {
		std::string next = cur;
		for (int k = 0; k < 54; k++) {
			const auto& p = T.pos2[k];
			int c = p[turn.axis];
			int layer = c >= 2 ? 2 : (c <= -2 ? 0 : 1);
			if (!(turn.layers & (1 << layer)))
				continue;
			std::array<int, 3> v = p, out{};
			int a = turn.axis, b = (a + 1) % 3, e = (a + 2) % 3;
			// n = unit a: n×v has components (b: -v_e, e: v_b); v' = n(n·v) - n×v
			out[a] = v[a];
			out[b] = v[e];
			out[e] = -v[b];
			next[index.at(out)] = cur[k];
		}
		cur = next;
	}
	return cur;
}

std::string applyMoves (const std::string& state, const std::string& moves) {
	std::string cur = state;
	size_t i = 0;
	while (i < moves.size()) {
		while (i < moves.size() && moves[i] == ' ')
			i++;
		size_t j = i;
		while (j < moves.size() && moves[j] != ' ')
			j++;
		if (j > i)
			cur = applyTurn(cur, parseTurn(moves.substr(i, j - i)));
		i = j;
	}
	return cur;
}

}
