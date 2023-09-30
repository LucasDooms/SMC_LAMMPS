import subprocess
import numpy as np
from parse import readParameters

runStart = 13

# Number of independent runs
nRuns = 12

# Read parameters
parameters = readParameters()

datafile = 'initial.dat'

subprocess.call(['python3', 'generate.py',
                 '%s' %parameters['N'],
                 '%s' %parameters['n'],
                 '%s' %parameters['armLength'],
                 '%s' %parameters['bridgeWidth'],
                 '%s' %parameters['HKradius'],
                 '%s' %parameters['epsilon3'],
                 '%s' %parameters['epsilon6'],
                 '%s' %parameters['cutoff6'],
                 '%s' %parameters['intRadSMCvsDNA'],
                 '%s' %parameters['armsStiffness'],
                 '%s' %parameters['elbowsStiffness'],
                 '%s' %parameters['siteStiffness'],
                 '%s' %parameters['foldingAngleAPO'],
                 '%s' %parameters['foldingStiffness'],
                 '%s' %parameters['asymmetryStiffness'],
                 datafile ])

for run in range(runStart,nRuns+runStart):

    outputfile = '%spN_run_%s.lammpstrj' %(str(parameters['forces']).replace('.','p'), run)

    subprocess.Popen(['~/Documents/lammps-29Oct20/src/lmp_serial', '-in', 'input',
                     '-var', 'N',                  '%s' %parameters['N'],
                     '-var', 'force',              '%s' %parameters['forces'],
                     '-var', 'armLength',          '%s' %parameters['armLength'],
                     '-var', 'bridgeWidth',        '%s' %parameters['bridgeWidth'],
                     '-var', 'epsilon3',           '%s' %parameters['epsilon3'],
                     '-var', 'epsilon4',           '%s' %parameters['epsilon4'],
                     '-var', 'epsilon5',           '%s' %parameters['epsilon5'],
                     '-var', 'epsilon6',           '%s' %parameters['epsilon6'],
                     '-var', 'cutoff4',            '%s' %parameters['cutoff4'],
                     '-var', 'cutoff5',            '%s' %parameters['cutoff5'],
                     '-var', 'cutoff6',            '%s' %parameters['cutoff6'],
                     '-var', 'sigma',              '%s' %parameters['intRadSMCvsDNA'],
                     '-var', 'armsStiffness',      '%s' %parameters['armsStiffness'],
                     '-var', 'armsAngleATP',       '%s' %parameters['armsAngleATP'],
                     '-var', 'foldingStiffness',   '%s' %parameters['foldingStiffness'],
                     '-var', 'asymmetryStiffness', '%s' %parameters['asymmetryStiffness'],
                     '-var', 'foldingAngleAPO',    '%s' %parameters['foldingAngleAPO'],
                     '-var', 'foldingAngleATP',    '%s' %parameters['foldingAngleATP'],
                     '-var', 'stepsATP',           '%s' %parameters['stepsATP'],
                     '-var', 'stepsADP',           '%s' %parameters['stepsADP'],
                     '-var', 'stepsAPO',           '%s' %parameters['stepsAPO'],
                     '-var', 'datafile',           datafile,
                     '-var', 'outputfile',         outputfile,
                     '-var', 'seed',               '%s' %np.random.randint(1,100000),
                     '-log',    'run_%s.log'    %run,
                     '-screen', 'run_%s.screen' %run])

    subprocess.call(['sleep', '2.0'])
