# crashbox-openradioss

OpenRadioss models and scripts for an AI-assisted auxetic crash box study (EG6011 capstone).

## Solver
Siemens retired the official OpenRadioss project in late 2026 (github.com/OpenRadioss is empty and
openradioss.org redirects to Simcenter Radioss). The code remains under AGPL-3.0.
`build-openradioss.yml` compiles it from a public mirror:

- source: upstream commit `157425b512ba` (28 May 2026), the last revision that matches the mirrored
  external libraries (extlib v59). The final upstream commit `bad66886` (25 Sep 2026) needs extlib v82,
  which is not mirrored.
- two small patches to the open-source input reader (shown in the build log and shipped as
  `openreader_patch.diff`): config version 2612 is mapped to 2026, and an unused rigid-part helper is stubbed.

## Workflows
- `build-openradioss.yml`: build starter + engine (Ubuntu 24.04), upload as artifact.
- `publish-solver-branch.yml`: copy a build to the `solver-bin` branch.
- `run-case.yml`: generate a deck, run it, post-process, commit outputs to the `results` branch.

## Scripts
- `scripts/gen_baseline.py`: baseline AA6061-T6 square tube deck (placeholder geometry and impact values).
- `scripts/thread.py`: reader for Radioss binary time-history files.
- `scripts/postprocess.py`: force-displacement, EA, SEA, PCF, MCF, CFE (rigid-wall impulse differentiated in time).
- `scripts/animate.py`: GIF of the deforming tube from nodal displacements written by the solver.
