# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

### Changed

- The `--visualize-follow` flag now has two choices `arms, kleisin`, tracking the SMC arms or kleisin respectively.
- The `average_plot.py` script now converts to `nm` and `s` units when creating plots by default.

### Fixed

- The `process_displacement.py` script no longer raises an error when no `obstacle.lammpstrj` is present.

## [0.1.1] - 2025-06-17

### Added

- LAMMPS script now prints the SMC state (`APO`, `ATP`, `ADP`) to the screen.

### Changed

- Removed work-around for hybrid bond_style when selecting rigid vs non-rigid hinge.

### Fixed

- Parameters `steps_APO`, `steps_ADP`, `steps_ATP` were used incorrectly, change their definition
  in `src/smc_lammps/generate/default_parameters.py` to match up with the use in lammps code.
- Arms now close properly in the `APO` state, due to decreased repulsion with upper site.

## [0.1.0] - 2025-06-02

First release for SMC_LAMMPS.
