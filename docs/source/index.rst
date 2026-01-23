smc-lammps documentation
========================

`smc-lammps`_ is a python package that uses `LAMMPS`_ to perform coarse-grained Molecular Dynamics simulations of `SMC complexes`_.

For information about installation and usage, view the `README`_ on github.

.. _`smc-lammps`: https://github.com/LucasDooms/SMC_LAMMPS
.. _`README`: https://github.com/LucasDooms/SMC_LAMMPS#user-content-dna-loop-extrusion-by-smccs-in-lammps
.. _`LAMMPS`: https://docs.lammps.org
.. _`SMC complexes`: https://en.wikipedia.org/wiki/SMC_protein

Workflow
--------

The diagram below shows the basic program flow of `smc-lammps`_.

.. raw:: html

   <div style="position: relative;">
   <div style="position: absolute; top: 0; right: 0; font-size: 0.85em;">
   <span style="color: #5ba4b5;">■ smc-lammps</span>
   &nbsp;&nbsp;
   <span style="color: #e67e22;">■ LAMMPS</span>
   </div>
   </div>

.. graphviz::

   digraph workflow {
       bgcolor="transparent"
       rankdir=LR
       ranksep=0.8
       nodesep=0.5

       node [
           shape=box,
           style="rounded,filled",
           fontname="Helvetica",
           fontsize=11,
           margin="0.2,0.1"
       ]

       edge [
           fontname="Helvetica",
           fontsize=10
       ]

       // Python (smc-lammps) nodes - blue
       parameters [label="User Input\n(parameters.py)", fillcolor="#e8f4f8", color="#5ba4b5"]
       vmd [label="Open in VMD", fillcolor="#e8f4f8", color="#5ba4b5"]
       analyze [label="Analyze SMC/DNA\npositions", fillcolor="#e8f4f8", color="#5ba4b5"]

       // LAMMPS nodes - orange
       inputs [label="LAMMPS Input Files\n(positions, coefficients,\nSMC states)", fillcolor="#fff3e0", color="#e67e22"]
       inputlmp [label="LAMMPS code\n(input.lmp)", fillcolor="#fff3e0", color="#e67e22"]
       outputs [label="Output Files\n(output.lammpstrj,\napo/atp/adp snapshots)", fillcolor="#fff3e0", color="#e67e22"]

       // Python edges - blue
       parameters -> inputs [label="--generate", color="#5ba4b5", fontcolor="#3d7a8a"]
       outputs -> vmd [label="--visualize", color="#5ba4b5", fontcolor="#3d7a8a"]
       outputs -> analyze [label="--post-process", color="#5ba4b5", fontcolor="#3d7a8a"]

       // LAMMPS edges - orange
       inputs -> outputs [label="--run", color="#e67e22", fontcolor="#c45a00"]
       inputlmp -> outputs [color="#e67e22"]
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   smc_lammps
