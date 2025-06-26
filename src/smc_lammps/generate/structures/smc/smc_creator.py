# Copyright (c) 2024 Lucas Dooms

import math
from dataclasses import dataclass, fields
from typing import Any, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from smc_lammps.generate.generator import Nx3Array
from smc_lammps.generate.structures.structure_creator import (
    attach_chain,
    get_circle_segment_unit_radius,
    get_straight_segment,
)


@dataclass
class SMC_Pos:
    r_arm_dl: Nx3Array
    r_arm_ul: Nx3Array
    r_arm_ur: Nx3Array
    r_arm_dr: Nx3Array
    r_ATP: Nx3Array
    r_kleisin: Nx3Array
    r_upper_site: Nx3Array
    r_middle_site: Nx3Array
    r_lower_site: Nx3Array
    # None when using non-toroidal hinge!
    r_hinge: Nx3Array | None

    def iter(self) -> List[Any]:
        """Returns a list of all fields"""
        return [getattr(self, field.name) for field in fields(SMC_Pos)]

    @staticmethod
    def none_wrapper(func):
        """Handle None values in the first argument by returning None."""

        def new_func(*args, **kwargs):
            if args[0] is None:
                return None
            return func(*args, **kwargs)

        return new_func

    def apply(self, func) -> None:
        """Update the object inplace by applying a function to every field"""
        func = self.none_wrapper(func)
        for field in fields(self.__class__):
            setattr(self, field.name, func(getattr(self, field.name)))

    def map(self, func) -> List[Any]:
        """Apply a function to every field and return the resulting list"""
        return [func(x) for x in self.iter() if x is not None]


@dataclass
class SMC_Creator:
    """Computes the positions of all SMC beads"""

    SMC_spacing: float

    # Horizontal distance between top binding sites (units bead spacing)
    upper_site_h: float
    # Vertical distance of top binding sites from hinge (units of bead spacing)
    upper_site_v: float
    # Horizontal distance between middle binding sites (units bead spacing)
    middle_site_h: float
    # Vertical distance of middle binding sites from bridge (units of bead spacing)
    middle_site_v: float
    # Horizontal distance between bottom binding sites (units bead spacing)
    lower_site_h: float
    # Distance of bottom binding sites from kleisin (units of bead spacing)
    lower_site_v: float

    arm_length: float
    bridge_width: float
    use_toroidal_hinge: bool
    # Not used if use_toroidal_hinge = False
    hinge_radius: float
    hinge_opening: float

    kleisin_radius: float

    folding_angle_APO: float

    small_noise: float = 1e-5

    def get_arms(self) -> Tuple[Nx3Array, Nx3Array, Nx3Array, Nx3Array]:
        # Number of beads forming each arm segment (err on the high side)
        n_arm_segments = math.ceil(self.arm_length / (2 * self.SMC_spacing))

        # z and y lengths of each arm (2 aligned segments), for the initial triangular geometry
        z_arm = self.bridge_width / 2.0
        y_arm = math.sqrt(self.arm_length**2 - z_arm**2)

        # u: up, d: down
        # l: left, r: right
        direction_dl = [0, y_arm, z_arm]
        direction_ul = [0, y_arm, z_arm]
        direction_ur = [0, -y_arm, z_arm]
        direction_dr = [0, -y_arm, z_arm]

        factor = self.arm_length / 2.0 / n_arm_segments
        r_arm_dl = get_straight_segment(n_arm_segments, direction_dl) * factor
        r_arm_ul = get_straight_segment(n_arm_segments, direction_ul) * factor
        r_arm_ur = get_straight_segment(n_arm_segments, direction_ur) * factor
        r_arm_dr = get_straight_segment(n_arm_segments, direction_dr) * factor

        r_arm_dl, r_arm_ul, r_arm_ur, r_arm_dr = attach_chain(
            r_arm_dl, [[r_arm_ul, False], [r_arm_ur, False], [r_arm_dr, False]]
        )

        # move lower center to origin (where bridge will be placed)
        center = (r_arm_dl[0] + r_arm_dr[-1]) / 2.0

        r_arm_dl -= center
        r_arm_ul -= center
        r_arm_ur -= center
        r_arm_dr -= center

        return r_arm_dl, r_arm_ul, r_arm_ur, r_arm_dr

    def get_bridge(self) -> Nx3Array:
        # Number of beads forming the ATP ring (err on the high side)
        n_ATP = math.ceil(self.bridge_width / self.SMC_spacing)

        # We want an odd number (necessary for angle/dihedral interactions)
        if n_ATP % 2 == 0:
            n_ATP += 1

        # Positions
        r_ATP = get_straight_segment(n_ATP, [0, 0, 1])

        # use the bridgeWidth as the absolute truth (SMC spacing may be slightly off)
        r_ATP *= self.bridge_width / n_ATP

        # move center to origin
        r_ATP -= (r_ATP[0] + r_ATP[-1]) / 2.0

        return r_ATP

    def get_heads_kleisin(self) -> Nx3Array:
        # Circle-arc radius
        # radius = (self.HKradius**2 + (self.bridgeWidth / 2.0)**2) / (2.0 * self.HKradius)
        bridge_radius = self.bridge_width / 2.0
        radius = self.kleisin_radius
        if radius < bridge_radius:
            raise ValueError(
                f"The kleisin radius ({radius}) is too small (< {bridge_radius}) based on the bridgeWidth {self.bridge_width}"
            )

        # from the y-axis
        starting_angle = math.asin(bridge_radius / radius)
        # Opening angle of circular arc (away from the bridge = 2*pi - angle towards the bridge)
        phi0 = 2.0 * math.pi - 2.0 * starting_angle

        # Number of beads forming the heads/kleisin complex (err on the high side)
        n_kleisin = math.ceil(phi0 * radius / self.SMC_spacing)

        # We want an odd number (necessary for angle/dihedral interactions)
        if n_kleisin % 2 == 0:
            n_kleisin += 1

        ending_angle = 2.0 * math.pi - starting_angle
        # add pi/2, since circle_segment calculation starts from x-axis
        starting_angle += math.pi / 2.0
        ending_angle += math.pi / 2.0

        r_kleisin = get_circle_segment_unit_radius(
            n_kleisin,
            end_inclusive=True,
            theta_start=starting_angle,
            theta_end=ending_angle,
            normal_direction=[1, 0, 0],
        )

        r_kleisin *= radius
        # move the bridge-gap to the origin
        r_kleisin[:, 1] -= r_kleisin[0][1]

        return r_kleisin

    @staticmethod
    def shielded_site_template(
        n_inner_beads: int,
        n_outer_beads_per_inner_bead: int,
        inner_spacing: float,
        outer_spacing: float,
    ) -> Nx3Array:
        """create a line of beads surrounded by a protective shell/shield"""
        axis = np.array([1, 0, 0])
        # Inner/Attractive beads
        inner_beads = get_straight_segment(n_inner_beads, direction=axis) * inner_spacing
        # put center at the origin
        inner_beads -= (inner_beads[0] + inner_beads[-1]) / 2.0

        # Repulsive/Outer beads, forming a surrounding shell
        shells = []
        for inner_bead in inner_beads:
            shells.append(
                get_circle_segment_unit_radius(
                    n_outer_beads_per_inner_bead,
                    end_inclusive=True,
                    theta_start=0,
                    theta_end=np.pi,
                    normal_direction=axis,
                )
                * outer_spacing
            )
            # place center of shell at inner bead
            shells[-1] += inner_bead

        # Horizontal shield at two ends
        end_first = (inner_beads[0] - inner_spacing * axis).reshape(1, 3)
        end_last = (inner_beads[-1] + inner_spacing * axis).reshape(1, 3)

        return np.concatenate([inner_beads, *shells, end_first, end_last])

    @staticmethod
    def transpose_rotate_transpose(rotation, *arrays: Nx3Array) -> Tuple[Nx3Array]:
        return tuple(rotation.dot(arr.transpose()).transpose() for arr in arrays)

    def get_interaction_sites(
        self, lower_site_points_down: bool
    ) -> Tuple[Nx3Array, Nx3Array, Nx3Array]:
        # U = upper  interaction site
        # M = middle interaction site
        # D = lower  interaction site

        # UPPER SITE
        if self.use_toroidal_hinge:
            r_upper_site = get_straight_segment(3)
            r_upper_site -= r_upper_site[1]
        else:
            r_upper_site = self.shielded_site_template(3, 4, self.upper_site_h, 1)
            # Inert bead connecting site to arms at top
            r_upper_site = np.concatenate(
                [r_upper_site, np.array([0.0, self.upper_site_v, 0.0]).reshape(1, 3)]
            )

        rotate_around_x_axis = Rotation.from_rotvec(math.pi * np.array([1.0, 0.0, 0.0])).as_matrix()

        # MIDDLE SITE
        r_middle_site = self.shielded_site_template(1, 4, self.middle_site_h, 1)
        (r_middle_site,) = self.transpose_rotate_transpose(rotate_around_x_axis, r_middle_site)

        # take last bead and use it as an extra inner bead
        r_middle_site = np.concatenate([r_middle_site[:1], r_middle_site[-1:], r_middle_site[1:-1]])
        # move, so that this bead is at the origin
        r_middle_site -= r_middle_site[1]

        # LOWER SITE
        r_lower_site = self.shielded_site_template(3, 4, self.lower_site_h, 1)
        if not lower_site_points_down:
            (r_lower_site,) = self.transpose_rotate_transpose(rotate_around_x_axis, r_lower_site)

        return r_upper_site, r_middle_site, r_lower_site

    def get_hinge(self) -> Nx3Array:
        radius = self.hinge_radius

        spacing = self.SMC_spacing * 0.8

        n_ring = math.ceil(2 * np.pi * radius / spacing)
        # should be multiple of 2 but not of 4
        if n_ring % 2 == 1:
            n_ring -= 1
        if n_ring % 4 == 0:
            n_ring += 2

        r_hinge = get_circle_segment_unit_radius(
            n_ring, end_inclusive=False, normal_direction=(0, 1, 0)
        )

        # rotate slightly
        angle = np.linalg.norm(r_hinge[1] - r_hinge[0]) / 2.0
        rotation = Rotation.from_rotvec(angle * np.array([0.0, 1.0, 0.0])).as_matrix()

        r_hinge *= radius

        (r_hinge,) = self.transpose_rotate_transpose(rotation, r_hinge)

        # separate pieces
        half = len(r_hinge) // 2
        # overlap
        r_hinge[:half, 2] -= r_hinge[0, 2]
        r_hinge[half:, 2] -= r_hinge[-1, 2]

        r_hinge[:half, 2] -= self.hinge_opening / 2.0
        r_hinge[half:, 2] += self.hinge_opening / 2.0

        return r_hinge

    def get_smc(self, lower_site_points_down: bool, extra_rotation=None) -> SMC_Pos:
        r_arm_dl, r_arm_ul, r_arm_ur, r_arm_dr = self.get_arms()
        r_ATP = self.get_bridge()
        r_kleisin = self.get_heads_kleisin()
        r_upper_site, r_middle_site, r_lower_site = self.get_interaction_sites(
            lower_site_points_down
        )
        if self.use_toroidal_hinge:
            r_hinge = self.get_hinge()
        else:
            r_hinge = np.empty(shape=(0, 3), dtype=r_ATP.dtype)

        # Inert bead, used for breaking folding symmetry
        r_middle_site = np.concatenate([r_middle_site, np.array([1.0, -1.0, 0.0]).reshape(1, 3)])
        r_middle_site[:, 1] += self.middle_site_v
        if lower_site_points_down:
            r_lower_site[:, 1] -= self.lower_site_v
        else:
            r_lower_site[:, 1] += self.lower_site_v

        # scale properly
        r_upper_site *= self.SMC_spacing
        r_middle_site *= self.SMC_spacing
        r_lower_site *= self.SMC_spacing

        if self.use_toroidal_hinge:
            # place hinge at center of top
            r_hinge += r_arm_ur[0]
            # place bead slightly below
            r_upper_site += r_arm_ur[0]
            r_upper_site[:, 1] -= self.SMC_spacing
        else:
            r_upper_site += r_arm_ur[0] - r_upper_site[-1]

        if self.use_toroidal_hinge:
            # rotate upper arms away to attach to hinge properly
            left_attach_hinge = len(r_hinge) // 4
            rot = Rotation.align_vectors(
                r_arm_ul[-1] - r_arm_ul[0], r_hinge[left_attach_hinge] - r_arm_ul[0]
            )[0]
            r_arm_ur = (
                self.transpose_rotate_transpose(rot.as_matrix(), r_arm_ur - r_arm_ur[-1])[0]
                + r_arm_ur[-1]
            )
            r_arm_ul = (
                self.transpose_rotate_transpose(rot.inv().as_matrix(), r_arm_ul - r_arm_ul[0])[0]
                + r_arm_ul[0]
            )

        # move into the correct location
        r_middle_site += r_ATP[len(r_ATP) // 2]
        r_lower_site += r_kleisin[len(r_kleisin) // 2]

        ############################# Fold upper compartment ############################

        # Rotation matrix (clockwise about z axis)
        rotation = Rotation.from_rotvec(
            -math.radians(self.folding_angle_APO) * np.array([0.0, 0.0, 1.0])
        ).as_matrix()

        # Rotate upper segments only
        r_arm_dl, r_arm_ul, r_arm_ur, r_arm_dr, r_upper_site, r_middle_site, r_hinge = (
            self.transpose_rotate_transpose(
                rotation,
                r_arm_dl,
                r_arm_ul,
                r_arm_ur,
                r_arm_dr,
                r_upper_site,
                r_middle_site,
                r_hinge,
            )
        )

        self.generated_positions = SMC_Pos(
            r_arm_dl=r_arm_dl,
            r_arm_ul=r_arm_ul,
            r_arm_ur=r_arm_ur,
            r_arm_dr=r_arm_dr,
            r_ATP=r_ATP,
            r_kleisin=r_kleisin,
            r_upper_site=r_upper_site,
            r_middle_site=r_middle_site,
            r_lower_site=r_lower_site,
            r_hinge=r_hinge if self.use_toroidal_hinge else None,
        )

        # apply extra rotation to entire SMC
        if extra_rotation is not None:
            rotation = Rotation.from_rotvec(extra_rotation).as_matrix()
            self.generated_positions.apply(
                lambda pos: self.transpose_rotate_transpose(rotation, pos)
            )

        return self.generated_positions

    def get_mass_per_atom(self, total_mass: float) -> float:
        return total_mass / sum(self.generated_positions.map(len))
