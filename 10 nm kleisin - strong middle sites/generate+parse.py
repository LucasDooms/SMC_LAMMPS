import math
import numpy as np
import subprocess
#import matplotlib.pyplot as plt


#################################################################################
#                             Parse parameters-file                              #
#################################################################################


def readParameters(filename='parameters'):

    parameterFile = open(filename, 'r')

    # Remove lines starting with '#'
    lines = [x.split(' ') for x in parameterFile.read().split('\n') if len(x)>0 and x[0] != '#']

    # Remove '=' and spaces
    lines = [[y.strip() for y in x if y!='=' and y!=''] for x in lines]

    parameters = {}

    for line in lines:
        if len(line) < 3:
            parameters[line[0]] = line[1]
        else:
            parameters[line[0]] = line[1:]
   
    parameterFile.close()

    return parameters


#################################################################################
#                               Input parameters                                #
#################################################################################


# Initial loop size (DNA beads)
loop = int(readParameters()['loop'])

# Diameter of initial loop (nm)
diameter = int(readParameters()['diameter'])


# Number of DNA beads
nDNA = int(readParameters()['N'])

# Number of base pairs per DNA bead
DNAdiscr = float(readParameters()['n'])

# Length of each coiled-coil arm (nm)
armLength = float(readParameters()['armLength'])

# Width of ATP bridge (nm)
bridgeWidth = float(readParameters()['bridgeWidth'])

# Radius of lower circular-arc compartment (nm)
HKradius = float(readParameters()['HKradius'])

# LJ energy of repulsion (kT units)
epsilon3 = float(readParameters()['epsilon3'])

# LJ energy of lower site (kT units)
epsilon6 = float(readParameters()['epsilon6'])

# LJ cutoff of lower site (nm)
cutoff6 = float(readParameters()['cutoff6'])

# SMC-DNA repulsion radius (nm)
intRadSMCvsDNA = float(readParameters()['intRadSMCvsDNA'])

# Bending stiffness of arm-bridge angle (kT units)
armsStiffness = float(readParameters()['armsStiffness'])

# Bending stiffness of elbows (kT units)
elbowsStiffness = float(readParameters()['elbowsStiffness'])

# Alignment stiffness of binding sites (kT units)
siteStiffness = float(readParameters()['siteStiffness'])

# Folding angle of lower compartment (degrees)
foldingAngleAPO = float(readParameters()['foldingAngleAPO'])

# Folding stiffness of lower compartment (kT units)
foldingStiffness = float(readParameters()['foldingStiffness'])

# Folding asymmetry stiffness of lower compartment (kT units)
asymmetryStiffness = float(readParameters()['asymmetryStiffness'])

# Name of generated data file
filename_data = str('datafile')

# Name of generated parameter file
filename_para = str('parameterfile')


#################################################################################
#                               Other parameters                                #
#################################################################################


# Simulation temperature (K)
T = float(readParameters()['T'])

# Boltzmann's constant (pN nm / K)
kB = float(readParameters()['kB'])


#################################### Masses #####################################


#######
# DNA #
#######

# Mass per base pair (ag)
bpMass = 2 * 3.1575 * 5.24e-4

# Effective bead mass (ag)
mDNA = DNAdiscr * bpMass


#######
# SMC #
#######

# Total mass of SMC protein (ag)
mSMCtotal = 0.25


#################################### Lengths ####################################


################
# Interactions #
################

# DNA-DNA repulsion radius (nm)
intRadDNAvsDNA = 3.5


#######
# DNA #
#######

# Bending stiffness (nm)
DNAstiff = 50.

# Base pair step (nm)
bpStep = 0.34

# Effective bond length = DNA bead size (nm)
DNAbondLength = DNAdiscr * bpStep

# Total length of DNA (nm)
DNAcontourLength = DNAbondLength * nDNA


#######
# SMC #
#######

# Desirable SMC spacing (radius of 1 SMC bead is R = intRadSMCvsDNA)
# Equal to R:   Minimum diameter = sqrt(3)    = 1.73 R
# Equal to R/2: Minimum diameter = sqrt(15)/2 = 1.94 R
SMCspacing = intRadSMCvsDNA/2


#################
# Binding sites #
#################

# Vertical distance of top binding sites from hinge (units of bead spacing)
siteUvDist = 4

# Horizontal distance between top binding sites (units bead spacing)
siteUhDist = 2


# Vertical distance of middle binding sites from bridge (units of bead spacing)
siteMvDist = 1

# Horizontal distance between middle binding sites (units bead spacing)
siteMhDist = 2


# Distance of bottom binding sites from kleisin (units of bead spacing)
siteDvDist = 0.5

# Horizontal distance between bottom binding sites (units bead spacing)
siteDhDist = 2


##################
# Simulation box #
##################


# Width of cubic simulation box (nm)
boxWidth = 2*DNAcontourLength


################################## Interactions #################################


###########
# DNA-DNA #
###########

sigmaDNAvsDNA   = intRadDNAvsDNA
epsilonDNAvsDNA = epsilon3
rcutDNAvsDNA    = sigmaDNAvsDNA * 2**(1/6)


###########
# SMC-DNA #
###########

sigmaSMCvsDNA   = intRadSMCvsDNA
epsilonSMCvsDNA = epsilon3
rcutSMCvsDNA    = sigmaSMCvsDNA * 2**(1/6)


#############
# Sites-DNA #
#############

# Sigma of LJ attraction (same as those of the repulsive SMC sites)
sigmaSiteDvsDNA = sigmaSMCvsDNA

# Cutoff distance of LJ attraction
rcutSiteDvsDNA = cutoff6

# Epsilon parameter of LJ attraction
epsilonSiteDvsDNA = epsilon6 * kB*T


#################################################################################
#                                 SMC complex                                   #
#################################################################################


#################################### Arms #######################################


# Number of beads forming each arm segment (err on the high side)
nArmSegm = math.ceil(armLength / 2 / SMCspacing)

# Create list of 3 zeros for later coordinates of each arm bead for each arm segment (U=up, D=down, L=left, R=right)
rArmDL = np.zeros((nArmSegm,3))
rArmUL = np.zeros((nArmSegm,3))
rArmUR = np.zeros((nArmSegm,3))
rArmDR = np.zeros((nArmSegm,3))

# z and y lengths of each arm (2 aligned segments), for the initial triangular geometry
zArm = bridgeWidth / 2
yArm = ( armLength**2 - zArm**2 )**0.5 

# y positions
rArmDL[:,1] = np.linspace(      0,  yArm/2, nArmSegm)
rArmUL[:,1] = np.linspace( yArm/2,    yArm, nArmSegm)
rArmUR[:,1] = np.linspace(   yArm,  yArm/2, nArmSegm)
rArmDR[:,1] = np.linspace( yArm/2,       0, nArmSegm)

# z positions
rArmDL[:,2] = np.linspace(  -zArm, -zArm/2, nArmSegm)
rArmUL[:,2] = np.linspace(-zArm/2,       0, nArmSegm)
rArmUR[:,2] = np.linspace(      0,  zArm/2, nArmSegm)
rArmDR[:,2] = np.linspace( zArm/2,    zArm, nArmSegm)


# A bit of randomness, to avoid exact overlap (pressure is messed up in LAMMPS)
SMALL = 1e-9

rArmDL += np.random.standard_normal(rArmDL.shape) * SMALL
rArmUL += np.random.standard_normal(rArmUL.shape) * SMALL
rArmUR += np.random.standard_normal(rArmUR.shape) * SMALL
rArmDR += np.random.standard_normal(rArmDR.shape) * SMALL


################################# ATP bridge ####################################


# Number of beads forming the ATP ring (err on the high side)
nATP = math.ceil( bridgeWidth / SMCspacing )

# We want an odd number (necessary for angle/dihedral interactions)
if nATP%2 == 0:
    nATP += 1


# Positions
rATP      = np.zeros((nATP,3))
rATP[:,2] = np.linspace(-bridgeWidth/2, bridgeWidth/2, nATP)


# A bit of randomness
rATP += np.random.standard_normal(rATP.shape) * SMALL


################################ Heads/Kleisin ##################################


# Circle-arc radius
radius = ( HKradius**2 + (bridgeWidth/2)**2 ) / ( 2 * HKradius )

# Opening angle of circular arc
phi0 = 2 * math.asin( bridgeWidth / 2 / radius )
if HKradius > bridgeWidth/2:
    phi0 = 2*math.pi - phi0


# Number of beads forming the heads/kleisin complex (err on the high side)
nHK = math.ceil( phi0 * radius / SMCspacing )

# We want an odd number (necessary for angle/dihedral interactions)
if nHK%2 == 0:
    nHK += 1


# Polar angle
phi = np.linspace(-(math.pi-phi0)/2, -(math.pi+phi0)/2, nHK) 
if HKradius > bridgeWidth/2:
    phi = np.linspace((phi0-math.pi)/2, -math.pi-(phi0-math.pi)/2, nHK)


# Positions
rHK      = np.zeros((nHK,3))
rHK[:,1] = radius * np.sin(phi) - HKradius + radius
rHK[:,2] = radius * np.cos(phi)

# A bit of randomness
rHK += np.random.standard_normal(rHK.shape) * SMALL


############################### Interaction sites ###############################


# U = upper  interaction site
# M = middle interaction site
# D = lower  interaction site

# Number of beads per site
nSiteU = 18
nSiteM =  8
nSiteD = 17

rSiteU = np.zeros((nSiteU,3))
rSiteM = np.zeros((nSiteM,3))
rSiteD = np.zeros((nSiteD,3))


# Polar angles of a 4-bead semicircle
phi = np.arange(4) * np.pi/3


# UPPER SITE


# Attractive beads
rSiteU[0] = rArmUL[-1] + SMCspacing * np.array([-siteUhDist, -siteUvDist, 0])
rSiteU[1] = rArmUL[-1] + SMCspacing * np.array([          0, -siteUvDist, 0])
rSiteU[2] = rArmUL[-1] + SMCspacing * np.array([+siteUhDist, -siteUvDist, 0])

# Repulsive beads, forming a surrounding shell
for index in range(len(phi)):
    rSiteU[3 +index] = rSiteU[0] + SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteU[7 +index] = rSiteU[1] + SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteU[11+index] = rSiteU[2] + SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])

# Horizontal shield at two ends
rSiteU[15] = rSiteU[0] + SMCspacing * np.array([-siteUhDist,0,0])
rSiteU[16] = rSiteU[2] + SMCspacing * np.array([ siteUhDist,0,0])

# Inert bead connecting site to arms at top
rSiteU[17] = rArmUL[-1]


# MIDDLE SITE


# Attractive beads
rSiteM[0] = rATP[nATP//2] + SMCspacing * np.array([-siteMhDist, siteMvDist, 0])
rSiteM[1] = rATP[nATP//2] + SMCspacing * np.array([          0, siteMvDist, 0])

# Inert bead, used for breaking folding symmetry
rSiteM[2] = rATP[nATP//2] + SMCspacing * np.array([ 1, 0, 0])

# Repulsive beads, forming a surrounding shell
for index in range(len(phi)):
    rSiteM[3+index] = rSiteM[0] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])

# Horizontal shield at one end
rSiteM[7] = rSiteM[0] + SMCspacing * np.array([-siteMhDist, 0, 0])


# LOWER SITE


# Attractive beads
rSiteD[0] = rHK[nHK//2] + SMCspacing * np.array([-siteDhDist,  siteDvDist, 0])
rSiteD[1] = rHK[nHK//2] + SMCspacing * np.array([          0,  siteDvDist, 0])
rSiteD[2] = rHK[nHK//2] + SMCspacing * np.array([+siteDhDist,  siteDvDist, 0])

# Repulsive beads, forming a surrounding shell
for index in range(len(phi)):
    rSiteD[3 +index] = rSiteD[0] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteD[7 +index] = rSiteD[1] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteD[11+index] = rSiteD[2] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])

# Horizontal shield at two ends
rSiteD[15] = rSiteD[0] + SMCspacing * np.array([-siteDhDist,0,0])
rSiteD[16] = rSiteD[2] + SMCspacing * np.array([ siteDhDist,0,0])


# Add randomness
rSiteU += np.random.standard_normal(rSiteU.shape) * SMALL
rSiteM += np.random.standard_normal(rSiteM.shape) * SMALL
rSiteD += np.random.standard_normal(rSiteD.shape) * SMALL


############################# Fold upper compartment ############################


c = math.cos(math.radians(foldingAngleAPO))
s = math.sin(math.radians(foldingAngleAPO))

# Rotation matrix (counter-clockwise about z axis)
rotMat = np.array([ [c,s,0] , [-s,c,0], [0,0,1] ])

# Rotations
rArmDL = np.einsum('ij,...j->...i', rotMat, rArmDL)
rArmUL = np.einsum('ij,...j->...i', rotMat, rArmUL)
rArmUR = np.einsum('ij,...j->...i', rotMat, rArmUR)
rArmDR = np.einsum('ij,...j->...i', rotMat, rArmDR)
rSiteU = np.einsum('ij,...j->...i', rotMat, rSiteU)
rSiteM = np.einsum('ij,...j->...i', rotMat, rSiteM)


#################################################################################
#                                     DNA                                       #
#################################################################################


# Number of beads forming the arced DNA piece (err on the high side)

nArcedDNA = math.ceil( math.pi * diameter / 2 / DNAbondLength )


# We want an odd number (necessary for angle/dihedral interactions)
if nArcedDNA%2 == 0:
    nArcedDNA += 1


# Upper DNA piece

nUpperDNA = int((nDNA-nArcedDNA)/2)

rUpperDNAtemp      = np.zeros((nUpperDNA,3))
rUpperDNAtemp[:,0] = np.arange(1, nUpperDNA+1) * DNAbondLength
rUpperDNAtemp[:,1] = diameter

rUpperDNA = rUpperDNAtemp[::-1]


# Arced DNA piece

angle = np.linspace(math.pi/2, 3*math.pi/2, nArcedDNA)

rArcedDNA      = np.zeros((nArcedDNA,3))
rArcedDNA[:,0] = (diameter/2)*np.cos(angle)
rArcedDNA[:,1] = (diameter/2)*np.sin(angle) + diameter/2


# Lower DNA piece

nLowerDNA = int((nDNA-nArcedDNA)/2)

rLowerDNA      = np.zeros((nLowerDNA,3))
rLowerDNA[:,0] = np.arange(1, nLowerDNA+1) * DNAbondLength


# Total DNA

rDNA = np.concatenate((rUpperDNA,rArcedDNA,rLowerDNA))


# Shift X-coordinate to get DNA end-points at X = 0

rDNA[:,0] -= DNAbondLength*(nDNA-nArcedDNA)/2



#################################################################################
#                                  Shift SMC                                    #
#################################################################################


# Makes sure that the SMC encircles DNA at the right position and at the start of the initial loop
shift = [-DNAbondLength*(nDNA-loop+nArcedDNA)/2,0,0] - rSiteD[1] - np.array([0,1,0]) * sigmaSiteDvsDNA * 2**(1/6)

rArmDL += shift.reshape(1,3)
rArmUL += shift.reshape(1,3)
rArmUR += shift.reshape(1,3)
rArmDR += shift.reshape(1,3)
rHK    += shift.reshape(1,3)
rATP   += shift.reshape(1,3)
rSiteU += shift.reshape(1,3)
rSiteM += shift.reshape(1,3)
rSiteD += shift.reshape(1,3)


#################################################################################
#                                Print to file                                  #
#################################################################################


datafile = open(filename_data, 'w')


#### Header ####


# Total number of atoms, bonds, angles and dihedrals
nAtoms     = nDNA + 4*nArmSegm + nHK + nATP + nSiteU + nSiteM + nSiteD
nBonds     = nDNA + 13
nAngles    = nDNA + 2
nImpropers = 5

# Total number of atom, bond, angle and improper types
nAtomTypes     = 7
nBondTypes     = 4
nAngleTypes    = 3
nImproperTypes = 3

datafile.write("# LAMMPS data file\n")
datafile.write("%s atoms\n"       %nAtoms)
datafile.write("%s bonds\n"       %nBonds)
datafile.write("%s angles\n"      %nAngles)
datafile.write("%s impropers\n\n" %nImpropers)

datafile.write("%s atom types\n"       %nAtomTypes)
datafile.write("%s bond types\n"       %nBondTypes)
datafile.write("%s angle types\n"      %nAngleTypes)
datafile.write("%s improper types\n\n" %nImproperTypes)

datafile.write("# System size\n")
datafile.write("%s %s xlo xhi\n"   %(-boxWidth/2, boxWidth/2))
datafile.write("%s %s ylo yhi\n"   %(-boxWidth/2, boxWidth/2))
datafile.write("%s %s zlo zhi\n\n" %(-boxWidth/2, boxWidth/2))


#### Body ####


# Divide total mass evenly among the segments
mSMC = mSMCtotal / ( 4*nArmSegm + nHK + nATP + nSiteU + nSiteM + nSiteD )

datafile.write("Masses\n\n")
datafile.write("1 %s\n"   %(mDNA)) # DNA
datafile.write("2 %s\n"   %(mSMC)) # Arms and kleisin
datafile.write("3 %s\n"   %(mSMC)) # ATP bridge
datafile.write("4 %s\n"   %(mSMC)) # Upper site
datafile.write("5 %s\n"   %(mSMC)) # Middle site
datafile.write("6 %s\n"   %(mSMC)) # Lower site
datafile.write("7 %s\n\n" %(mSMC)) # Reference site


# Relative bond fluctuations
bondFlDNA = 1e-2
bondFlSMC = 1e-2

# Maximum relative bond extension (units of rest length)
bondMax = 1.

# Spring constant obeying equilibrium relative bond fluctuations
kBondDNA    = 3*kB*T/(DNAbondLength*bondFlDNA)**2
kBondSMC    = 3*kB*T/(   SMCspacing*bondFlSMC)**2
kBondAlign1 =  10*kB*T / SMCspacing**2
kBondAlign2 = 200*kB*T / SMCspacing**2


indL = np.argmin(np.linalg.norm(rSiteU[-2]-rArmUL, axis=1))
indR = np.argmin(np.linalg.norm(rSiteU[-2]-rArmUR, axis=1))
bondMin1 = np.linalg.norm(rSiteU[-2]-rArmUL[indL])
bondMin2 = np.linalg.norm(rSiteU[-2]-rArmUL[-1])

# Maximum bond length
maxLengthDNA = DNAbondLength*bondMax
maxLengthSMC =    SMCspacing*bondMax

datafile.write("Bond Coeffs # hybrid\n\n")
datafile.write("1 fene/expand %s %s %s %s %s\n" %(kBondDNA, maxLengthDNA, 0, 0, DNAbondLength))
datafile.write("2 fene/expand %s %s %s %s %s\n" %(kBondSMC, maxLengthSMC, 0, 0, 0))
datafile.write("3 harmonic %s %s\n"             %(kBondAlign1, bondMin1))
datafile.write("4 harmonic %s %s\n\n"           %(kBondAlign2, bondMin2))


# DNA bending rigidity
kDNA = DNAstiff * kB*T / DNAbondLength

# Angular trap constants
# kElbows: Bending of elbows (kinkable arms, hence soft)
# kArms:   Arms opening angle wrt ATP bridge (should be stiff)
kElbows = elbowsStiffness*kB*T
kArms   =   armsStiffness*kB*T

datafile.write("Angle Coeffs # hybrid\n\n")
datafile.write("1 cosine %s\n"        %  kDNA )
datafile.write("2 harmonic %s %s\n"   % ( kElbows, 180 ) )
#datafile.write("3 harmonic %s %s\n\n" % ( kArms,  np.rad2deg( math.acos( bridgeWidth / 2 / armLength ) ) ) )
datafile.write("3 harmonic %s %s\n\n" % ( kArms,  np.rad2deg( math.acos( bridgeWidth / armLength ) ) ) )


# Fixes site orientation (prevents free rotation, should be stiff)
kAlignSite = siteStiffness*kB*T

# Folding stiffness of lower compartment (should be stiff)
kFolding = foldingStiffness*kB*T

# Makes folding asymmetric (should be stiff)
kAsymmetry = asymmetryStiffness*kB*T

# We impose zero improper angle
datafile.write("Improper Coeffs # harmonic\n\n")
datafile.write("1 %s %s\n"   %( kAlignSite, 0 ))
datafile.write("2 %s %s\n"   %( kFolding,   180 - foldingAngleAPO ))
datafile.write("3 %s %s\n\n" %( kAsymmetry,  math.fabs(90 - foldingAngleAPO) ))


# Pair coefficients
datafile.write("PairIJ Coeffs # hybrid\n\n")

datafile.write("1 1 lj/cut %s %s %s\n" %(epsilonDNAvsDNA,   sigmaDNAvsDNA,   rcutDNAvsDNA))
datafile.write("1 2 lj/cut %s %s %s\n" %(epsilonSMCvsDNA,   sigmaSMCvsDNA,   rcutSMCvsDNA))
datafile.write("1 3 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("1 4 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("1 5 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("1 6 lj/cut %s %s %s\n" %(epsilonSiteDvsDNA, sigmaSiteDvsDNA, rcutSiteDvsDNA))
datafile.write("1 7 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("2 2 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("2 3 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("2 4 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("2 5 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("2 6 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("2 7 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("3 3 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("3 4 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("3 5 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("3 6 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("3 7 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("4 4 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("4 5 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("4 6 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("4 7 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("5 5 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("5 6 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("5 7 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("6 6 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("6 7 lj/cut %s %s %s\n" %(0, 0, 0))
datafile.write("7 7 lj/cut %s %s %s\n" %(0, 0, 0))

# The indexing of the SMC follows a clockwise order, 
# starting from the bottom-left arm and ending at
# the right part of the ATP bridge.

datafile.write("\nAtoms # molecular\n\n")

# LAMMPS indices for each atom
IDdNA   = np.arange(nDNA)     + 1
IDarmDL = np.arange(nArmSegm) + 1 + IDdNA[-1]
IDarmUL = np.arange(nArmSegm) + 1 + IDarmDL[-1]
IDarmUR = np.arange(nArmSegm) + 1 + IDarmUL[-1]
IDarmDR = np.arange(nArmSegm) + 1 + IDarmUR[-1]
IDhK    = np.arange(nHK)      + 1 + IDarmDR[-1]
IDaTP   = np.arange(nATP)     + 1 + IDhK[-1]
IDsiteU = np.arange(nSiteU)   + 1 + IDaTP[-1]
IDsiteM = np.arange(nSiteM)   + 1 + IDsiteU[-1]
IDsiteD = np.arange(nSiteD)   + 1 + IDsiteM[-1]

# Molecule for each rigid body
molDNA   = 1
molArmDL = 2
molArmUL = 3
molArmUR = 4
molArmDR = 5
molHK    = 6
molATP   = 7
molSiteU = 8
molSiteM = molATP
molSiteD = molHK

# DNA
for index in range(nDNA):
    datafile.write("%s %s 1 %s %s %s\n" %(IDdNA[index], molDNA, rDNA[index,0], rDNA[index,1], rDNA[index,2]) )


# Coiled-coil arms
for index in range(nArmSegm):
    datafile.write("%s %s 2 %s %s %s\n" %(IDarmDL[index], molArmDL, rArmDL[index,0], rArmDL[index,1], rArmDL[index,2]) )
for index in range(nArmSegm):
    datafile.write("%s %s 2 %s %s %s\n" %(IDarmUL[index], molArmUL, rArmUL[index,0], rArmUL[index,1], rArmUL[index,2]) )
for index in range(nArmSegm):
    datafile.write("%s %s 2 %s %s %s\n" %(IDarmUR[index], molArmUR, rArmUR[index,0], rArmUR[index,1], rArmUR[index,2]) )
for index in range(nArmSegm):
    datafile.write("%s %s 2 %s %s %s\n" %(IDarmDR[index], molArmDR, rArmDR[index,0], rArmDR[index,1], rArmDR[index,2]) )

# Kleisin circular-arc part
for index in range(nHK):
    datafile.write("%s %s 2 %s %s %s\n" %(IDhK[index], molHK, rHK[index,0], rHK[index,1], rHK[index,2]) )    

# ATP 
for index in range(nATP):
    datafile.write("%s %s 3 %s %s %s\n" %(IDaTP[index], molATP, rATP[index,0], rATP[index,1], rATP[index,2]) )    

# Top sites
for index in range(nSiteU):
    if index < 3:
        datafile.write("%s %s 4 %s %s %s\n" %(IDsiteU[index], molSiteU, rSiteU[index,0], rSiteU[index,1], rSiteU[index,2]) )
    else:
        datafile.write("%s %s 2 %s %s %s\n" %(IDsiteU[index], molSiteU, rSiteU[index,0], rSiteU[index,1], rSiteU[index,2]) )

# Middle sites
for index in range(nSiteM):
    if index < 2:
        datafile.write("%s %s 5 %s %s %s\n" %(IDsiteM[index], molSiteM, rSiteM[index,0], rSiteM[index,1], rSiteM[index,2]) )
    elif index == 2:
        datafile.write("%s %s 7 %s %s %s\n" %(IDsiteM[index], molSiteM, rSiteM[index,0], rSiteM[index,1], rSiteM[index,2]) )
    else:
        datafile.write("%s %s 3 %s %s %s\n" %(IDsiteM[index], molSiteM, rSiteM[index,0], rSiteM[index,1], rSiteM[index,2]) )

# Bottom sites
for index in range(nSiteD):
    if index < 3:
        datafile.write("%s %s 6 %s %s %s\n" %(IDsiteD[index], molSiteD, rSiteD[index,0], rSiteD[index,1], rSiteD[index,2]) )
    else:
        datafile.write("%s %s 2 %s %s %s\n" %(IDsiteD[index], molSiteD, rSiteD[index,0], rSiteD[index,1], rSiteD[index,2]) )


datafile.write("\nBonds\n\n")

# DNA connectivity
for index in range(nDNA-1):
    datafile.write("%s 1 %s %s\n" %(index+1, IDdNA[index], IDdNA[index+1]) )

# Every joint is kept in place through bonds
datafile.write("%s 2 %s %s\n"   %(nDNA,      IDarmDL[-1], IDarmUL[ 0]))
datafile.write("%s 2 %s %s\n"   %(nDNA+1,    IDarmUL[-1], IDarmUR[ 0]))
datafile.write("%s 2 %s %s\n"   %(nDNA+2,    IDarmUR[-1], IDarmDR[ 0]))
datafile.write("%s 2 %s %s\n"   %(nDNA+3,    IDarmUL[-1], IDsiteU[-1]))
datafile.write("%s 2 %s %s\n"   %(nDNA+4,    IDarmDR[-1],   IDaTP[-1]))
datafile.write("%s 2 %s %s\n"   %(nDNA+5,      IDaTP[ 0], IDarmDL[ 0]))
datafile.write("%s 2 %s %s\n"   %(nDNA+6,      IDaTP[-1],    IDhK[ 0]))
datafile.write("%s 2 %s %s\n"   %(nDNA+7,       IDhK[-1],   IDaTP[ 0]))
datafile.write("%s 3 %s %s\n"   %(nDNA+8,  IDarmUL[indL], IDsiteU[-2]))
datafile.write("%s 3 %s %s\n"   %(nDNA+9,  IDarmUL[indL], IDsiteU[-3]))
datafile.write("%s 3 %s %s\n"   %(nDNA+10, IDarmUR[indR], IDsiteU[-2]))
datafile.write("%s 3 %s %s\n"   %(nDNA+11, IDarmUR[indR], IDsiteU[-3]))
datafile.write("%s 4 %s %s\n"   %(nDNA+12,   IDarmUL[-1], IDsiteU[-2]))
datafile.write("%s 4 %s %s\n\n" %(nDNA+13,   IDarmUL[-1], IDsiteU[-3]))


datafile.write("\nAngles\n\n")

# DNA stiffness
for index in range(nDNA-2):
    datafile.write("%s 1 %s %s %s\n" %(index+1, IDdNA[index], IDdNA[index+1], IDdNA[index+2]))

# Arm-arm angles
datafile.write("%s 2 %s %s %s\n" %(nDNA-1, IDarmDL[ 0], IDarmUL[ 0], IDarmUL[-1]))
datafile.write("%s 2 %s %s %s\n" %(nDNA,   IDarmUR[ 0], IDarmUR[-1], IDarmDR[-1]))
# Arm-ATP angles
datafile.write("%s 3 %s %s %s\n" %(nDNA+1, IDarmDL[-1], IDarmDL[ 0], IDaTP[-1]))
datafile.write("%s 3 %s %s %s\n" %(nDNA+2, IDarmDR[ 0], IDarmDR[-1], IDaTP[ 0]))


datafile.write("\nImpropers\n\n")

# Fix orientation of ATP/kleisin bridge
datafile.write("1 1 %s %s %s %s\n"   %(IDarmDL[-1], IDarmDL[ 0], IDaTP[-1], IDsiteM[1]))
datafile.write("2 1 %s %s %s %s\n"   %(IDarmDR[ 0], IDarmDR[-1], IDaTP[ 0], IDsiteM[1]))
# Folding angle
datafile.write("7 2 %s %s %s %s\n"   %(IDarmDL[-1], IDarmDL[ 0], IDaTP[-1], IDhK[nHK//2]))
datafile.write("8 2 %s %s %s %s\n"   %(IDarmDR[ 0], IDarmDR[-1], IDaTP[ 0], IDhK[nHK//2]))
# Folding asymmetry
datafile.write("9 3 %s %s %s %s\n\n" %(IDsiteM[2], IDarmDL[ 0], IDarmDR[-1], IDhK[nHK//2]))


datafile.close()


"""
plt.plot(   rHK[:,2],    rHK[:,1], '.')
plt.plot(  rATP[:,2],   rATP[:,1], '.')
plt.plot(rArmDL[:,2], rArmDL[:,1], '.')
plt.plot(rArmUL[:,2], rArmUL[:,1], '.')
plt.plot(rArmUR[:,2], rArmUR[:,1], '.')
plt.plot(rArmDR[:,2], rArmDR[:,1], '.')
#plt.plot(rSiteU[0,2], rSiteU[0,1], '.')
#plt.plot(rSiteD[0,2], rSiteD[0,1], '.')

plt.axis('scaled')
plt.show()
"""


#################################################################################
#                           Print to parameterfile                              #
#################################################################################


parameterfile = open(filename_para, 'w')

parameterfile.write("# LAMMPS parameter file\n\n")

for key in readParameters():
	parameterfile.write("variable %s equal %s\n\n"       %(key, readParameters()[key]))

parameterfile.close()
