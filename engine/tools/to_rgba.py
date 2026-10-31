"""Images to raw RGBA files for tools/bench.cpp (the engine has no image decoder).

	python engine/tools/to_rgba.py OUT_DIR IMAGE [IMAGE ...]

Prints per image the arguments bench takes: FILE.rgba WIDTH HEIGHT.
"""
import os, sys

import cv2


def main():
	if len(sys.argv) < 3:
		sys.exit(__doc__)
	out = sys.argv[1]
	os.makedirs(out, exist_ok=True)
	args = []
	for path in sys.argv[2:]:
		img = cv2.imread(path)
		if img is None:
			sys.exit(f"{path}: not an image")
		name = os.path.join(out, os.path.splitext(os.path.basename(path))[0] + ".rgba")
		cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).tofile(name)
		args += [name, str(img.shape[1]), str(img.shape[0])]
	print(" ".join(args))


if __name__ == "__main__":
	main()
