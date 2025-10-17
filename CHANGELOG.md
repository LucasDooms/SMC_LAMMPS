# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- New `spaced_beads` parameters allow simulating arrays of obstacles on a DNA strand with
tunable size, spacing, and angle stiffness.
- New `lower_site_cycle_period`, `lower_site_toggle_delay` parameters allow simulating strand swapping with
a semi-permanent lower binding site.
- New `--quiet` flag reduces console output produced by `smc-lammps`.

### Changed

- Simulations run with the --continue flag will now produce output.lammpstrj.1, output.lammpstrj.2, ... files
instead of appending to the existing output.lammpstrj file.

### Fixed

- The `smc_perspective.py` script now correctly writes the atom types and ids in the lammpstrj format,
this fixes the colors in VMD.
- The `basepair_mass` is now correctly calculated (0.003 ag -> 0.001 ag)

## [0.1.2] - 2025-07-01

### Added

- The `average_plot.py` script now also plots a linear fit with the average velocity.
- Add new option `use_toroidal_hinge`, to allow simulating old non-toroidal hinge.

### Changed

- The `--visualize-follow` flag now has two choices `arms, kleisin`, tracking the SMC arms or kleisin respectively.
- The `average_plot.py` script now converts to `nm` and `s` units when creating plots by default.
- The `average_plot.py` script now removes `-1` values when finding the smallest common number of steps.

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
