from ..structure_creator import get_straight_segment, get_circle_segment, attach, attach_chain, get_interpolated
import numpy as np
import math


def get_closest(array, position) -> int:
    """returns the index of the array that is closest to the given position"""
    distances = np.linalg.norm(array - position, axis=1)
    return int(np.argmin(distances))


def get_dna_coordinates_straight(nDNA: int, DNAbondLength: float):
    rDNA = get_straight_segment(nDNA, [1, 0, 0])

    # Shift X-coordinate to get DNA end-point at X = 0

    rDNA[:,0] -= rDNA[0][0]

    # get correct bead spacings

    rDNA *= DNAbondLength

    # Rotate (flip the x-component)

    rDNA[:,0] *= -1

    return [rDNA]


def get_dna_coordinates_safety_belt(nDNA: int, DNAbondLength: float):
    rDNA = get_interpolated(
        DNAbondLength,
        10 * DNAbondLength * np.array(
            [
                [4, 0, 0], [2, 0, 0], # right, straight piece
                [0, 1.0, 0], [-0.5, 1.0, 0], [-3, 1.25, 0], # up, to the left
                [-3.25, 0.75, 0], # down, to the left
                [-0.5, -1, 0], [0, -1, 0], [0.7, -1, 0], [1.5, -1.5, 0], # down, to the right
                [1.0, -2, 0], [-2, -2, 0], # down, to the left
                [-3, -0.5, 0], [-4, 0, 0], # left, straight
            ],
            dtype=float
        )
    )
    distances = np.linalg.norm(rDNA - 10*DNAbondLength*np.array([0, -1.0, 0.0]), axis=1)
    belt_index = np.where(distances == np.min(distances))[0][0]

    remaining = nDNA - len(rDNA)
    nLeft = int(remaining * 0.05)
    nRight = remaining - nLeft
    left = get_straight_segment(nLeft + 1, [-1, 0, 0]) * DNAbondLength
    right = get_straight_segment(nRight + 1, [-1, 0, 0]) * DNAbondLength
    right, rDNA, left = attach_chain(
        right, [[rDNA, True], [left, True]]
    )

    belt_location = rDNA[belt_index]
    return [np.concatenate([right, rDNA, left])], belt_location, belt_index + len(right), len(right) - 40


def get_dna_coordinates_advanced_safety_belt(nDNA: int, DNAbondLength: float):
    smc_pos = np.array([3.3, -2.1, 0.0])
    rDNA = get_interpolated(
        DNAbondLength,
        10 * DNAbondLength * np.array(
            [
                [6, -2.0, 0], [5, -1.4, 0], [4, 0, 0], [2, 0, 0], # right, straight piece
                [0, 1.0, 0], [-3, 1.5, 0], [-4, 1.25, 0], # up, to the left
                [-5, 0.75, 0], [-6, 0.5, 0], [-7, 0, 0], # down, to the left
                [-5.5, -1, 0], [-4, -1.3, 0], [0.7, -1.4, 0], [2.8, -1.8, 0], [3.5, -2.2, 0.0], [3.5, -2.5, 0.0], # down, to the right
                [3, -3.1, 0], [2.5, -3.2, 0], [2, -3.2, 0], # down, to the left
            ],
            dtype=float
        )
    )
    distances = np.linalg.norm(rDNA - 10*DNAbondLength*smc_pos, axis=1)
    belt_index = np.where(distances == np.min(distances))[0][0]

    remaining = nDNA - len(rDNA)
    right = get_straight_segment(remaining + 1, [-1, 0, 0]) * DNAbondLength
    rDNA = attach(right, rDNA, delete_overlap=True)

    belt_location = rDNA[belt_index]
    bead_to_tether_id = len(right) + 31

    return [np.concatenate([right, rDNA])], belt_location, belt_index + len(right), bead_to_tether_id


def get_dna_coordinates_advanced_safety_belt_plus_loop(nDNA: int, DNAbondLength: float):
    smc_pos = np.array([3.3, -2.1, 0.0])
    tether_pos = np.array([3.75, 0.0, 0.0])
    rDNA = get_interpolated(
        DNAbondLength,
        10 * DNAbondLength * np.array(
            [
                [6, -2.0, 0], [5, -1.4, 0], [4.7, -0.5, 0], # right piece
                [4.6, -0.4, 2], [4.7, -0.2, 4], [4.5, -0.2, 4], [4.0, -0.15, 4],
                [0.0, -0.15, 4], [1.0, -0.15, 2], [4.0, -0.05, 2.4], [4.2, 0, 2], # loop
                [4, 0, 0], [2, 0, 0],
                [0, 1.0, 0], [-3, 1.5, 0], [-4, 1.25, 0], # up, to the left
                [-5, 0.75, 0], [-6, 0.5, 0], [-7, 0, 0], # down, to the left
                [-5.5, -1, 0], [-4, -1.3, 0], [0.7, -1.4, 0], [2.8, -1.8, 0], [3.5, -2.2, 0.0], [3.5, -2.5, 0.0], # down, to the right
                [3, -3.1, 0], [2.5, -3.2, 0], [2, -3.2, 0], # down, to the left
            ],
            dtype=float
        )
    )
    belt_index = get_closest(rDNA, 10*DNAbondLength*smc_pos)
    bead_to_tether_id = get_closest(rDNA, 10*DNAbondLength*tether_pos)

    remaining = nDNA - len(rDNA)
    right = get_straight_segment(remaining + 1, [-1, 0, 0]) * DNAbondLength
    rDNA = attach(right, rDNA, delete_overlap=True)

    belt_location = rDNA[belt_index]
    bead_to_tether_id += len(right)

    return [np.concatenate([right, rDNA])], belt_location, belt_index + len(right), bead_to_tether_id


def get_dna_coordinates(nDNA: int, DNAbondLength: float, diameter: float, nArcStraight: int):
    # form vertical + quarter circle + straight + semi circle + horizontal parts

    # Number of beads forming the arced DNA piece (err on the high side)
    nArcedDNA = math.ceil( 3 / 4 * math.pi * diameter / DNAbondLength ) # 3 / 4 = 1 / 2 + 1 / 4 = semi + quarter

    # We want an odd number (necessary for angle/dihedral interactions)
    if nArcedDNA % 2 == 0:
        nArcedDNA += 1


    # Upper DNA piece

    nUpperDNA = (nDNA - nArcedDNA - nArcStraight) // 2

    rUpperDNA = get_straight_segment(nUpperDNA, [0, -1, 0])

    # Arced DNA piece

    nArcSemi = int(nArcedDNA * 2/3)
    nArcQuart = nArcedDNA - nArcSemi

    # since there will be overlap: use one extra, then delete it later (after pieces are assembled)
    rArcQuart = get_circle_segment(nArcQuart + 1, end_inclusive=True, theta_start=0, theta_end=-np.pi/2.0)

    rArcStraight = get_straight_segment(nArcStraight, [-1, 0, 0])

    rArcSemi = get_circle_segment(nArcSemi + 1, end_inclusive=True, theta_start=np.pi/2.0, theta_end=np.pi*3.0/2.0)

    # Lower DNA piece

    nLowerDNA = nDNA - nUpperDNA - nArcedDNA - nArcStraight

    rLowerDNA = get_straight_segment(nLowerDNA, [1, 0, 0])

    # Total DNA

    rUpperDNA, rArcQuart, rArcStraight, rArcSemi, rLowerDNA = \
            attach_chain(rUpperDNA, [[rArcQuart, True], [rArcStraight, False, 1.0], [rArcSemi, True], [rLowerDNA, False, 1.0]])

    # alternative, without attach_chain method:
    # rArcQuart = attach(rUpperDNA, rArcQuart, delete_overlap=True)
    # rArcStraight = attach(rArcQuart, rArcStraight, delete_overlap=False, extra_distance=1.0)
    # rArcSemi = attach(rArcStraight, rArcSemi, delete_overlap=True)
    # rLowerDNA = attach(rArcSemi, rLowerDNA, delete_overlap=False, extra_distance=1.0)

    rDNA = np.concatenate([rUpperDNA, rArcQuart, rArcStraight, rArcSemi, rLowerDNA])

    # Shift X-coordinate to get DNA end-point at X = 0

    rDNA[:,0] -= rDNA[0][0]

    # get correct bead spacings

    rDNA *= DNAbondLength

    # Rotate (flip the x-component)

    rDNA[:,0] *= -1

    # the position in the center of the semi arc
    offset = nUpperDNA + nArcQuart + nArcStraight
    centerCoordinate = rDNA[offset] + (rDNA[offset + nArcSemi] - rDNA[offset]) / 2.0

    return [rDNA], centerCoordinate


def get_dna_coordinates_twist(nDNA: int, DNAbondLength: float, diameter: float):
    # form upper + semi circle + horizontal parts

    # Number of beads forming the arced DNA piece (err on the high side)
    nArcedDNA = math.ceil( 1 / 2 * math.pi * diameter / DNAbondLength ) # 1 / 2 = semi circle

    # We want an odd number (necessary for angle/dihedral interactions)
    if nArcedDNA % 2 == 0:
        nArcedDNA += 1


    # Upper DNA piece

    nUpperDNA = (nDNA - nArcedDNA) // 2

    rUpperDNA = get_straight_segment(nUpperDNA, [-1, 0, 0])

    # Arced DNA piece

    nArcSemi = nArcedDNA

    # since there will be overlap: use one extra, then delete it later (after pieces are assembled)
    rArcSemi = get_circle_segment(nArcSemi + 1, end_inclusive=True, theta_start=np.pi/2.0, theta_end=np.pi*3.0/2.0)

    # Lower DNA piece

    nLowerDNA = nDNA - nUpperDNA - nArcedDNA

    rLowerDNA = get_straight_segment(nLowerDNA, [1, 0, 0])

    # Total DNA

    rUpperDNA, rArcSemi, rLowerDNA = \
            attach_chain(rUpperDNA, [[rArcSemi, True], [rLowerDNA, False, 1.0]])

    rDNA = np.concatenate([rUpperDNA, rArcSemi, rLowerDNA])

    # Shift X-coordinate to get DNA end-point at X = 0

    rDNA[:,0] -= rDNA[0][0]

    # get correct bead spacings

    rDNA *= DNAbondLength

    # the position in the center of the arc
    offset = nUpperDNA
    centerCoordinate = rDNA[offset] + (rDNA[offset + nArcSemi] - rDNA[offset]) / 2.0

    return [rDNA], centerCoordinate


def get_dna_coordinates_doubled(nDNA: int, DNAbondLength: float, diameter: float):
    nOuterDNA = math.ceil(nDNA / 1.9)
    nInnerDNA = nDNA - nOuterDNA
    [rOuterDNA], outerCenter = get_dna_coordinates_twist(nOuterDNA, DNAbondLength, diameter)
    [rInnerDNA], innerCenter = get_dna_coordinates_twist(nInnerDNA, DNAbondLength, diameter / 1.5)
    # align pieces
    rInnerDNA += outerCenter - innerCenter
    # create seperation (x direction)
    rInnerDNA[:,0] += 5.0 * DNAbondLength
    return [rOuterDNA, rInnerDNA], outerCenter
