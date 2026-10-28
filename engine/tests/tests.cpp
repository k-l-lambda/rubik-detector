// Unit tests without sample data: cube tables, turns, rotations, the legal search. Parity with the Python
// reference over the samples: tools/parity.py.
#include <algorithm>
#include <cmath>
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

	// evidence: one view needs no fusion; it places as the reference frame
	{
		Evidence evidence;
		std::vector<FaceletSample> samples;
		for (int f : {1, 3, 5})
			for (int k = f * 9; k < f * 9 + 9; k++) {
				FaceletSample s;
				s.facelet = k;
				s.valid = true;
				s.lab = {50.0f, float(k), 0.0f};
				samples.push_back(s);
			}
		int id = evidence.addView(samples);
		double cost = -1;
		CHECK(evidence.fuse(&cost, false));
		CHECK(evidence.rotation(id) == identityRotation() && cost == 0);
		CHECK(!evidence.fuse(&cost, true));   // three faces only
		CHECK(!evidence.complete());
	}

	// partial evidence: frames of two views showing five faces, one frame with its colours scrambled; the legal
	// completion recovers the sixth face, the bad frame cannot veto a colour with the outlier share
	{
		const std::string state = applyMoves(S, "R U2 F' L D' B2 R' U F2 D L'");
		const float LAB[6][3] = {{95, 0, 0}, {85, -5, 80}, {45, 65, 45}, {65, 45, 70}, {35, 15, -60}, {55, -60, 40}};
		std::mt19937 rng(7);
		std::normal_distribution<float> noise(0, 2);
		auto view = [&](int g, bool scrambled) {
			std::vector<FaceletSample> samples;
			for (int f : {1, 3, 5})
				for (int k = f * 9; k < f * 9 + 9; k++) {
					int c = int(std::string(FACES).find(state[rotationPerm(g)[k]]));
					if (scrambled)
						c = (c + k) % 6;
					FaceletSample s;
					s.facelet = k;
					s.valid = true;
					s.lab = {LAB[c][0] + noise(rng), LAB[c][1] + noise(rng), LAB[c][2] + noise(rng)};
					samples.push_back(s);
				}
			return samples;
		};
		// a second view sharing one face with the identity's R U F and adding two: five faces
		int second = -1;
		for (int g = 0; g < 24 && second < 0; g++) {
			std::array<bool, 6> seen{};
			for (int f : {1, 3, 5}) {
				seen[f] = true;
				seen[rotationPerm(g)[f * 9 + 4] / 9] = true;
			}
			if (std::count(seen.begin(), seen.end(), true) == 5)
				second = g;
		}
		CHECK(second >= 0);
		Evidence evidence;
		for (int i = 0; i < 4; i++) {
			evidence.addView(view(identityRotation(), false), identityRotation());
			evidence.addView(view(second, false), second);
		}
		evidence.addView(view(second, true), second);
		EstimateOptions options;
		options.partial = true;
		options.outlier = 0.02;
		StateEstimate e = estimateState(evidence, options);
		CHECK(e.ok && e.coverage == 45);
		// observed facelets right; a hidden one either right or an even tie between completions (this scramble leaves a
		// 3-cycle of the hidden face's edges open: each edge shows one colour on the side faces)
		int hidden = 0, ties = 0;
		for (int k = 0; k < 54; k++) {
			double sum = 0;
			for (double p : e.probability[k])
				sum += p;
			CHECK(std::abs(sum - 1) < 1e-9);
			if (e.observed[k])
				CHECK(e.state[k] == state[k]);
			else {
				hidden++;
				CHECK(e.probability[k][labelsOf(e.state)[k]] == e.confidence[k]);
				CHECK(e.state[k] == state[k] || e.confidence[k] <= 0.5 + 1e-9);
				ties += e.confidence[k] <= 0.5 + 1e-9;
			}
		}
		CHECK(hidden == 9 && ties <= 3 && e.minPosterior > 0.99 && e.legal);
		options.partial = false;
		CHECK(!estimateState(evidence, options).ok);
	}

	// unknown reliability: n views agreeing on a decisive colour give P = ∫(r + (1−r)/6)^n / (that + 5·∫((1−r)/6)^n)
	{
		const float LAB[6][3] = {{95, 0, 0}, {85, -5, 80}, {45, 65, 45}, {65, 45, 70}, {35, 15, -60}, {55, -60, 40}};
		const std::string state = applyMoves(S, "R U F");
		auto view = [&]() {
			std::vector<FaceletSample> samples;
			for (int f : {1, 3, 5})
				for (int k = f * 9; k < f * 9 + 9; k++) {
					int c = int(std::string(FACES).find(state[k]));
					FaceletSample s;
					s.facelet = k;
					s.valid = true;
					s.lab = {LAB[c][0] + float(k % 3), LAB[c][1] - float(k % 2), LAB[c][2]};
					samples.push_back(s);
				}
			return samples;
		};
		const double expect[4] = {7.0 / 12, 0.8958333, 0.9810606, 0.9967949};
		Evidence evidence;
		EstimateOptions options;
		options.partial = true;
		options.legal = false;
		options.reliability = true;
		for (int n = 1; n <= 4; n++) {
			evidence.addView(view(), identityRotation());
			StateEstimate e = estimateState(evidence, options);
			CHECK(e.ok && e.coverage == 27);
			for (int k : {9, 13, 31, 50})
				CHECK(std::abs(e.posterior[k] - expect[n - 1]) < 1e-4);
		}
		// the same four frames as one group: one piece of evidence
		Evidence grouped;
		for (int n = 0; n < 4; n++)
			grouped.addView(view(), identityRotation(), 1.0, 0);
		CHECK(std::abs(estimateState(grouped, options).posterior[13] - 7.0 / 12) < 1e-4);
	}

	std::printf(failures ? "%d failures\n" : "all engine unit tests passed\n", failures);
	return failures ? 1 : 0;
}
