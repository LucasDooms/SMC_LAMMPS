from structure_creator import get_straight_segment, get_circle_segment, attach, attach_chain
import numpy as np
import math


def get_dna_coordinates(nDNA: int, DNAbondLength: float, diameter: int, nArcStraight: int):
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

    # attach_chain(rUpperDNA, [[rArcQuart, True], [rArcStraight, False, 1.0], [rArcSemi, True], [rLowerDNA, False, 1.0]])

    rArcQuart = attach(rUpperDNA, rArcQuart, delete_overlap=True)
    rArcStraight = attach(rArcQuart, rArcStraight, delete_overlap=False, extra_distance=1.0)
    rArcSemi = attach(rArcStraight, rArcSemi, delete_overlap=True)
    rLowerDNA = attach(rArcSemi, rLowerDNA, delete_overlap=False, extra_distance=1.0)

    rDNA = np.concatenate([rUpperDNA, rArcQuart, rArcStraight, rArcSemi, rLowerDNA])

    # Shift X-coordinate to get DNA end-point at X = 0

    rDNA[:,0] -= rDNA[0][0]

    # get correct bead spacings

    rDNA *= DNAbondLength

    # Rotate (flip the x-component)

    rDNA[:,0] *= -1

    return rDNA, nLowerDNA
