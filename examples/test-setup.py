import mdtraj as md
import os
import ensembletrek.gromacs as gro 

pdb=md.load("input-structs/vp35.pdb")

setup_dir = os.path.join("vp35-sims", "setup")
os.makedirs(setup_dir)

start_struct, top, itps = gro.create_system(pdb, os.path.join(setup_dir, "system-creation"))

minimized = gro.minimize(start_struct, os.path.join(setup_dir, "minimization"), top, itps)

equilibrated = gro.equilibrate(minimized, os.path.join(setup_dir, "equilibration"), top, itps, nsteps=50000)

inputs_dir = os.path.join("vp35-sims", "sim-inputs")
os.makedirs(inputs_dir, exist_ok=True)
equilibrated.save(os.path.join(inputs_dir, "system-solvated.gro"))
equilibrated.save(os.path.join(inputs_dir, "system-solvated.pdb"))
gro.trjconv(os.path.join(inputs_dir, "system-solvated.pdb"), os.path.join(inputs_dir, "system.pdb"), xtc_grps='prot-masses')
top_fn = os.path.join(inputs_dir, "topol.top")
gro.save_topology(top, top_fn)
gro.save_itps(itps, inputs_dir)

