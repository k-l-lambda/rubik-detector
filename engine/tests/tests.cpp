// Unit tests without sample data: cube tables, turns, rotations, the legal search. Parity with the Python
// reference over the samples: tools/parity.py.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "cube.hpp"
#include "internal.hpp"
#include "rubik/engine.h"

using namespace rubik;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

static std::string solved () {
	std::string s;
	for (int f = 0; f < 6; f++)
		s += std::string(9, FACES[f]);
	return s;
}

static std::array<int, 54> labelsOf (const std::string& s) {
	std::array<int, 54> l{};
	for (int k = 0; k < 54; k++)
		l[k] = int(std::string(FACES).find(s[k]));
	return l;
}

int main () {
	const std::string S = solved();

	// rotations: 24, a group, identity found
	CHECK(rotations().size() == 24);
	CHECK(rotationPerm(identityRotation())[17] == 17);
	for (int a = 0; a < 24; a++) {
		CHECK(composeRotations(a, inverseRotation(a)) == identityRotation());
		for (int b = 0; b < 24; b++) {
			int c = composeRotations(a, b);
			for (int k = 0; k < 54; k++)
				CHECK(rotationPerm(c)[k] == rotationPerm(a)[rotationPerm(b)[k]]);
		}
	}

	// turns: orders and the classic identities
	for (const std::string& m : faceTurnNames())
		CHECK(applyMoves(S, m + " " + m + " " + m + " " + m) == S);
	CHECK(applyMoves(S, "R") != S);
	CHECK(applyMoves(S, "R U R' U' R U R' U' R U R' U' R U R' U' R U R' U' R U R' U'") == S);   // (sexy move)⁶
	CHECK(applyMoves(applyMoves(S, "R U F' L2 D B"), "B' D' L2 F U' R'") == S);
	// R: the F face's right column goes up to U
	{
		std::string t = applyMoves(S, "R");
		CHECK(t[3 * 9 + 2] == 'F' && t[3 * 9 + 5] == 'F' && t[3 * 9 + 8] == 'F');
		CHECK(t[5 * 9 + 2] == 'D');
	}
	// x is R M' L': the same state up to a whole-cube rotation, and compareStates sees through it
	CHECK(applyMoves(S, "x") == applyMoves(S, "R M' L'"));
	CHECK(compareStates(applyMoves(S, "x y"), S) == 0);
	CHECK(compareStates(applyMoves(S, "R"), S) == 12);   // three facelets on each of U F D B

	// legality: scrambles are legal; a single twisted corner or swapped edge pair is not
	std::mt19937 rng(1);
	std::string scrambled = S;
	for (int i = 0; i < 40; i++)
		scrambled = applyTurn(scrambled, parseTurn(faceTurnNames()[rng() % 18]));
	CHECK(isLegal(labelsOf(scrambled)));
	{
		auto l = labelsOf(S);
		std::swap(l[0], l[36]);   // two facelets of one corner? not a corner: just an illegal colouring
		CHECK(!isLegal(l));
	}

	// best legal state from noisy costs recovers the scramble with one sticker misread
	{
		auto truth = labelsOf(scrambled);
		std::array<std::array<double, 6>, 54> cost{};
		std::normal_distribution<double> noise(0, 0.3);
		for (int k = 0; k < 54; k++)
			for (int c = 0; c < 6; c++)
				cost[k][c] = noise(rng) + (c == truth[k] ? 0 : 3.0);
		cost[7][(truth[7] + 1) % 6] -= 4.0;
		LegalResult r = bestLegal(cost);
		CHECK(r.labels == truth);
		CHECK(r.exact);
	}

	// relativeRotation: a pose and the same pose in a frame renamed by g
	{
		Pose a;
		a.R = {0.36, 0.48, -0.8, -0.8, 0.6, 0, 0.48, 0.64, 0.6};
		for (int g = 0; g < 24; g++) {
			// R_to = R_from·g
			const auto& G = rotations()[g];
			Pose b;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++) {
					double s = 0;
					for (int k = 0; k < 3; k++)
						s += a.R[i * 3 + k] * G[k * 3 + j];
					b.R[i * 3 + j] = s;
				}
			CHECK(relativeRotation(a, b) == g);
		}
	}

	std::printf(failures ? "%d failures\n" : "all engine unit tests passed\n", failures);
	return failures ? 1 : 0;
}
