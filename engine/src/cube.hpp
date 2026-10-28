// Facelet conventions and the cube's symmetry tables, as rubik_detector/cube.py.
#pragma once

#include <array>
#include <string>

namespace rubik {

constexpr const char* FACES = "LRDUBF";
constexpr int CENTRE_FACELETS[6] = {4, 13, 22, 31, 40, 49};

struct V3 {
	double x, y, z;
	double operator[] (int i) const { return i == 0 ? x : (i == 1 ? y : z); }
	double& operator[] (int i) { return i == 0 ? x : (i == 1 ? y : z); }
};

inline V3 operator+ (V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator- (V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator* (double s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot (V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross (V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

struct CubeTables {
	std::array<int, 54> faceOf, row, col;
	std::array<V3, 54> normal, right, down, centre;   // centre: on the plane at 1.5
	std::array<std::array<int, 3>, 54> pos2;          // rint(2·centre)
	std::array<std::array<int, 9>, 24> rotations;
	std::array<std::array<int, 54>, 24> perms;
	int identity;
};

const CubeTables& tables ();

// facelet k's point on the face plane at `plane`, moved by (a, b) along its (right, down)
inline V3 faceletPoint (int k, double plane, double a, double b) {
	const CubeTables& T = tables();
	return plane * T.normal[k] + (T.col[k] - 1 + a) * T.right[k] + (T.row[k] - 1 + b) * T.down[k];
}

// labels named by the face whose centre carries them; throws if the centres aren't six distinct labels
std::string relabel (const std::string& state);

}
