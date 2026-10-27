"""Rubik's cube state reconstruction from photos with rule-based image processing (no learning).

Experiment project: the target is a native mobile app, here the principle comes first. A group = three photos of one
cube from three corners (each pair sharing one face, all six faces covered); the result is the 54-facelet state.

Modules
	cube       facelet conventions (shared with klstudio's sample generator), whole-cube rotations, state comparison
	data       labelled samples in ~/data/rubik-renderer (renders, seedream background / hand edits)
	geometry   per view: where the 27 visible stickers are (line segments → cube axes → lattice fit); an oracle
	colors     sticker colour sampling and the 9-per-colour classification
	fusion     aligning the three views' cube frames (24 rotations each) by their shared faces
	pipeline   one group end to end
	evaluate   metrics over many groups:  python -m rubik_detector.evaluate --help
"""
