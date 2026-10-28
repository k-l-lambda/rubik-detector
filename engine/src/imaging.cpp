#include <algorithm>
#include <cmath>
#include <cstring>

#include "internal.hpp"

namespace rubik {

M3 matmul (const M3& a, const M3& b) {
	M3 c{};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
	return c;
}

double det (const M3& m) {
	return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

M3 rodrigues (V3 r) {
	double theta = std::sqrt(dot(r, r));
	if (theta < 2.2204460492503131e-16)
		return {1, 0, 0, 0, 1, 0, 0, 0, 1};
	V3 k = (1 / theta) * r;
	double c = std::cos(theta), s = std::sin(theta), c1 = 1 - c;
	return {
		c + c1 * k.x * k.x, c1 * k.x * k.y - s * k.z, c1 * k.x * k.z + s * k.y,
		c1 * k.y * k.x + s * k.z, c + c1 * k.y * k.y, c1 * k.y * k.z - s * k.x,
		c1 * k.z * k.x - s * k.y, c1 * k.z * k.y + s * k.x, c + c1 * k.z * k.z,
	};
}

void workingPlanes (const Image& image, int workWidth, Gray8 planes[3]) {
	int channels = (image.format == PixelFormat::BGR || image.format == PixelFormat::RGB) ? 3 : 4;
	bool bgr = image.format == PixelFormat::BGR || image.format == PixelFormat::BGRA;
	int index[3] = {bgr ? 0 : 2, 1, bgr ? 2 : 0};
	size_t rowBytes = size_t(image.width) * channels;
	size_t stride = image.stride > 0 ? size_t(image.stride) : rowBytes;
	const uint8_t* src = image.data;
	std::vector<uint8_t> packed;
	if (stride != rowBytes) {
		packed.resize(rowBytes * image.height);
		for (int y = 0; y < image.height; y++)
			std::memcpy(packed.data() + y * rowBytes, image.data + y * stride, rowBytes);
		src = packed.data();
	}
	if (workWidth <= 0 || workWidth == image.width) {
		for (int p = 0; p < 3; p++) {
			planes[p] = Gray8(image.width, image.height);
			for (size_t i = 0; i < size_t(image.width) * image.height; i++)
				planes[p].data[i] = src[i * channels + index[p]];
		}
		return;
	}
	// cv2.resize(image, (width, round(h · width / w)), INTER_AREA), as data.load
	double f = double(workWidth) / image.width;
	int dstH = int(std::nearbyint(image.height * f));
	resizeArea(src, image.width, image.height, channels, workWidth, dstH, index, 3, planes);
}

namespace {

// sRGB → linear via a 256-entry table (inputs are 8-bit), D65 white, the CIE formulas with OpenCV's constants
const float* linearTable () {
	static float lin[256];
	static bool ready = false;
	if (!ready) {
		for (int i = 0; i < 256; i++) {
			double c = i / 255.0;
			lin[i] = float(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4));
		}
		ready = true;
	}
	return lin;
}

inline void labOf (const float* lin, int b, int g, int r, float* d) {
	auto f = [](double t) { return t > 0.008856 ? std::cbrt(t) : 7.787 * t + 16.0 / 116.0; };
	double B = lin[b], G = lin[g], R = lin[r];
	double X = (0.412453 * R + 0.357580 * G + 0.180423 * B) / 0.950456;
	double Y = 0.212671 * R + 0.715160 * G + 0.072169 * B;
	double Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.088754;
	double fy = f(Y);
	d[0] = float(Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y);
	d[1] = float(500 * (f(X) - fy));
	d[2] = float(200 * (fy - f(Z)));
}

}

LabImage toLab (const Gray8 planes[3]) {
	const float* lin = linearTable();
	LabImage out;
	out.width = planes[0].width;
	out.height = planes[0].height;
	out.data.resize(size_t(out.width) * out.height * 3);
	const size_t n = size_t(out.width) * out.height;
	for (size_t i = 0; i < n; i++)
		labOf(lin, planes[0].data[i], planes[1].data[i], planes[2].data[i], out.data.data() + i * 3);
	return out;
}

LabImage toLabWithin (const Gray8 planes[3], const std::vector<std::array<int, 4>>& boxes) {
	// only the pixels of the given boxes (x0, y0, x1, y1, exclusive) are converted; the rest stays 0
	const float* lin = linearTable();
	LabImage out;
	out.width = planes[0].width;
	out.height = planes[0].height;
	out.data.assign(size_t(out.width) * out.height * 3, 0.0f);
	std::vector<uint8_t> done(size_t(out.width) * out.height, 0);
	for (const auto& b : boxes)
		for (int y = std::max(b[1], 0); y < std::min(b[3], out.height); y++)
			for (int x = std::max(b[0], 0); x < std::min(b[2], out.width); x++) {
				size_t i = size_t(y) * out.width + x;
				if (done[i])
					continue;
				done[i] = 1;
				labOf(lin, planes[0].data[i], planes[1].data[i], planes[2].data[i], out.data.data() + i * 3);
			}
	return out;
}

Gray8 toGray (const Gray8 planes[3]) {
	// cv2.cvtColor(BGR2GRAY) on 8 bits: fixed point, 14 fraction bits
	Gray8 g(planes[0].width, planes[0].height);
	for (size_t i = 0; i < g.data.size(); i++)
		g.data[i] = uint8_t((planes[0].data[i] * 1868 + planes[1].data[i] * 9617 + planes[2].data[i] * 4899 + (1 << 13)) >> 14);
	return g;
}

namespace {

// np.median of floats: the middle value, or the mean of the two middle ones
float median (std::vector<float>& v) {
	size_t n = v.size(), h = n / 2;
	std::nth_element(v.begin(), v.begin() + h, v.end());
	float hi = v[h];
	if (n % 2)
		return hi;
	float lo = *std::max_element(v.begin(), v.begin() + h);
	return (lo + hi) / 2;
}

}

bool sampleQuad (const LabImage& lab, const std::array<double, 8>& quad, std::array<float, 3>& out, int& pixels) {
	const int w = lab.width, h = lab.height;
	float q[8];
	for (int i = 0; i < 8; i++)
		q[i] = float(quad[i]);
	float minX = std::min({q[0], q[2], q[4], q[6]}), maxX = std::max({q[0], q[2], q[4], q[6]});
	float minY = std::min({q[1], q[3], q[5], q[7]}), maxY = std::max({q[1], q[3], q[5], q[7]});
	int x0 = std::max(int(std::floor(minX)), 0), y0 = std::max(int(std::floor(minY)), 0);
	int x1 = std::min(int(std::ceil(maxX)) + 1, w), y1 = std::min(int(std::ceil(maxY)) + 1, h);
	pixels = 0;
	if (x1 - x0 < 1 || y1 - y0 < 1)
		return false;
	// vertices on the quarter-pixel grid of fillConvexPoly(shift=2); a pixel is in if its centre is in the quad
	double vx[4], vy[4];
	for (int i = 0; i < 4; i++) {
		vx[i] = std::nearbyint((q[2 * i] - x0) * 4.0) / 4.0;
		vy[i] = std::nearbyint((q[2 * i + 1] - y0) * 4.0) / 4.0;
	}
	double area = 0;
	for (int i = 0; i < 4; i++)
		area += vx[i] * vy[(i + 1) % 4] - vx[(i + 1) % 4] * vy[i];
	double sgn = area >= 0 ? 1 : -1;
	std::vector<float> c[3];
	for (int y = y0; y < y1; y++) {
		for (int x = x0; x < x1; x++) {
			double px = x - x0, py = y - y0;
			bool in = true;
			for (int i = 0; i < 4 && in; i++) {
				int j = (i + 1) % 4;
				in = sgn * ((vx[j] - vx[i]) * (py - vy[i]) - (vy[j] - vy[i]) * (px - vx[i])) >= 0;
			}
			if (in) {
				const float* p = lab.at(x, y);
				for (int k = 0; k < 3; k++)
					c[k].push_back(p[k]);
			}
		}
	}
	pixels = int(c[0].size());
	if (pixels < 3) {
		// colors.sample: the pixel at the quad's mean (truncated), if inside the image
		int cx = int(float(q[0] + q[2] + q[4] + q[6]) / 4), cy = int(float(q[1] + q[3] + q[5] + q[7]) / 4);
		pixels = 0;
		if (cx < 0 || cx >= w || cy < 0 || cy >= h)
			return false;
		const float* p = lab.at(cx, cy);
		out = {p[0], p[1], p[2]};
		return true;
	}
	for (int k = 0; k < 3; k++)
		out[k] = median(c[k]);
	return true;
}

std::array<double, 4> weigh (const float lab[3]) {
	double L = lab[0], a = lab[1], b = lab[2];
	double C = std::hypot(a, b), h = std::atan2(b, a), sat = std::min(C / C_SAT, 1.0);
	return {HUE_SCALE * sat * std::cos(h), HUE_SCALE * sat * std::sin(h), LOGC_SCALE * std::log1p(C), L_WEIGHT * L};
}

}
