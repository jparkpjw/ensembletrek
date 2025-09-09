from enspara.cluster import KHybrid
from enspara.msm.transition_matrices import assigns_to_counts
from enspara.util import array as ra
from enspara.util.load import load_as_concatenated
from functools import partial
import glob
import mdtraj as md
import numpy as np
import os
import shutil
import ensembletrek.rankings as rankings
import ensembletrek.gromacs as gro
import ensembletrek.launcher as launcher
import submitit

def run():
    # num sims and where to put them
    n_gens = 4
    n_treks = 2
    head_dir = "vp35-sims"
    fast_dir = os.path.join(head_dir, "fast-rmsd")
    os.makedirs(fast_dir, exist_ok=True)

    # simulation input files
    inputs_dir = os.path.join(head_dir, "sim-inputs")
    solv_struct_fn = os.path.join(inputs_dir, "system-solvated.pdb")
    solv_struct = md.load(solv_struct_fn, standard_names=False)
    struct_fn = os.path.join(inputs_dir, "system.pdb")
    struct = md.load(struct_fn, standard_names=False)
    top = gro.load_topology(os.path.join(inputs_dir, "topol.top"))
    itps = gro.load_itps(inputs_dir)

    # save inputs locally so easy to track in future
    os.system("cp -r %s %s" % (inputs_dir, os.path.join(fast_dir, "setup")))

    # gromacs settings
    hmr = True
    nsteps = 5000
    maxwarn = 1
    n_cpus_gromacs = 4
    n_gpus_gromacs = 0
    if n_gpus_gromacs > 0:
        gpu_only = True
    else:
        gpu_only = False
    xtc_grps = 'system' # so can restart from any frame with resolvating
    solute_grps = 'Protein' # what to store in unsolvated structures/traj
    max_sim_time = 60
    n_sims_per_node = 1
    mdp = gro.MDP.npt(hmr=hmr, gpu_only=gpu_only, xtc_grps=xtc_grps)

    # msm/clustering settings
    n_cpus_cluster = 4
    cluster_radius = 0.003 # in nm
    max_cluster_time = 60
    lag_time = 1
    atm_inds_cluster = struct.topology.select('name CA')
    width = 1.0

    # analysis settings
    rmsd_inds = atm_inds_cluster
    analyze = partial(md.rmsd, reference=struct, atom_indices=rmsd_inds)

    # how much to pad numbers in file names
    padding = 3

    # setup directory structure
    traj_dir = os.path.join(fast_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    solv_traj_dir = os.path.join(fast_dir, "trajectories_solvated")
    os.makedirs(solv_traj_dir, exist_ok=True)
    sim_dir = os.path.join(fast_dir, "raw-treks")
    os.makedirs(sim_dir, exist_ok=True)
    msm_dir = os.path.join(fast_dir, "msms")
    os.makedirs(msm_dir, exist_ok=True)
    center_dir = os.path.join(fast_dir, "centers")
    os.makedirs(center_dir, exist_ok=True)
    pick_dir = os.path.join(fast_dir, "picks")
    os.makedirs(pick_dir, exist_ok=True)
    analy_dir = os.path.join(fast_dir, "analysis")
    os.makedirs(analy_dir, exist_ok=True)

    # tools for running simulations
    mdrun = partial(gro.npt_run, top=top, itps=itps, nsteps=nsteps, maxwarn=maxwarn, nt=n_cpus_gromacs, ntmpi=1, ntomp=n_cpus_gromacs, mdp=mdp, xtc_grps=xtc_grps, gpu_only=gpu_only)
    sim_executor = submitit.AutoExecutor(folder=os.path.join(sim_dir, "logs"))
    sim_executor.update_parameters(timeout_min=max_sim_time, tasks_per_node=n_sims_per_node, cpus_per_task=n_cpus_gromacs, gpus_per_node=n_gpus_gromacs*n_sims_per_node)

    # tools for clustering simulations
    cluster_executor = submitit.AutoExecutor(folder=os.path.join(msm_dir, "logs"))
    cluster_executor.update_parameters(timeout_min=max_cluster_time, cpus_per_task=n_cpus_cluster)


    # all sims in first gen will start from given initial structure
    start_structures = solv_struct.slice(range(solv_struct.n_frames), copy=True)
    for i in range(1, n_treks):
        start_structures = start_structures + solv_struct

    gen = 0

    while gen < n_gens:
        gen_dir = os.path.join(sim_dir, "gen" + str(gen))
        base_treks_dir = os.path.join(gen_dir, "trek")
        if not os.path.exists(gen_dir):
            print("Running simulations for gen", gen)
            os.makedirs(gen_dir, exist_ok=True)

            # run the simulations
            launcher.launch_sims(mdrun, start_structures, base_treks_dir, n_sims_per_node, sim_executor)

        else:
            print("Already ran simulations for gen", gen)

        # move trajectories
        last_fn = os.path.join(traj_dir, "g%s-w%s.xtc" % (str(gen).zfill(padding), str(n_treks-1).zfill(padding)))
        if not os.path.exists(last_fn):
            print("Moving simulations for gen", gen)
            for w in range(n_treks):
                orig_fn = os.path.join(base_treks_dir + str(w), "simulation-nojump.xtc")
                solv_traj_fn = os.path.join(solv_traj_dir, "g%s-w%s.xtc" % (str(gen).zfill(padding), str(w).zfill(padding)))
                shutil.move(orig_fn, solv_traj_fn)
                traj_fn = os.path.join(traj_dir, "g%s-w%s.xtc" % (str(gen).zfill(padding), str(w).zfill(padding)))
                gro.trjconv(solv_traj_fn, traj_fn, ref_fn=solv_struct_fn, xtc_grps=solute_grps)
        else:
            print("Already moved simulations for gen", gen)

        # get list of all xtc files
        fns_to_cluster = np.sort(np.array(glob.glob(os.path.join(traj_dir, "*.xtc"))))

        # cluster
        curr_msm_dir = os.path.join(msm_dir, "gen%d" % gen)
        centers_fn = os.path.join(center_dir, "gen%d.xtc" % gen)
        if not os.path.exists(curr_msm_dir):
            print("Clustering gen", gen)
            os.makedirs(curr_msm_dir)
            job = cluster_executor.submit(cluster, fns_to_cluster, n_cpus_cluster, struct_fn, atm_inds_cluster, cluster_radius, curr_msm_dir, centers_fn)
            job.result()
        else:
            print("Already clustered gen", gen)

        # load relevant clustering results
        centers = md.load(centers_fn, top=struct)
        print("  #centers", len(centers))

        # analyze all states
        analy_fn = os.path.join(analy_dir, "gen%d.txt" % gen)
        geometric_feature = None
        if not os.path.exists(analy_fn):
            print("Analyzing gen", gen)
            geometric_feature = analyze(centers)
            np.savetxt(analy_fn, geometric_feature)
        else:
            print("Already analyzed gen", gen)
            geometric_feature = np.loadtxt(analy_fn)

        # get counds per state
        assign_fn = os.path.join(curr_msm_dir, "assignments.h5")
        assigns = ra.load(assign_fn)
        c_per_state = np.bincount(assigns.ravel())

        # pick states to simulate
        pick_inds_fn = os.path.join(pick_dir, "gen%d.txt" % gen)
        centers_pick = None
        if not os.path.exists(pick_inds_fn):
            print("Ranking states for gen", gen)
            centers_pick = rankings.fast_spread(n_treks, geometric_feature, c_per_state, centers, width)
            print("  states picked:", centers_pick)
            np.savetxt(pick_inds_fn, centers_pick, fmt="%d")
        else:
            print("Already ranked states for gen", gen)
            centers_pick = np.loadtxt(pick_inds_fn, dtype=int)

        # setup start_structures for next generation
        pick_struct_fn = os.path.join(pick_dir, "gen%d.xtc" % gen)
        start_structures = None
        if not os.path.exists(pick_struct_fn):
            print("Getting states picked for gen", gen)
            solv_traj_fns = np.sort(np.array(glob.glob(os.path.join(solv_traj_dir, "*.xtc"))))
            center_inds_fn = os.path.join(curr_msm_dir, "center_inds.txt")
            center_inds = np.loadtxt(center_inds_fn, dtype=int)
            start_structures = None
            for trj_ind, snapshot in center_inds[centers_pick]:
                trj = md.load(solv_traj_fns[trj_ind], top=solv_struct)
                if start_structures is None:
                    start_structures = trj.slice(snapshot, copy=True)
                else:
                    start_structures = start_structures + trj[snapshot]
            pick_struct_fn = os.path.join(pick_dir, "gen%d.xtc" % gen)
            start_structures.save(pick_struct_fn)
        else:
            print("Already got states picked for gen", gen)
            start_structures = md.load(pick_struct_fn, top=solv_struct)

        gen += 1

def cluster(fns_to_cluster, n_cpus_cluster, struct_fn, atm_inds_cluster, cluster_radius, msm_dir, centers_fn, cluster_setup_cmd=None):

    if cluster_setup_cmd is not None:
        os.system(cluster_setup_cmd)

    # load trajectories
    trj_lengths, xyzs = load_as_concatenated(
                filenames=fns_to_cluster, processes=n_cpus_cluster,
                top=struct_fn)

    struct = md.load(struct_fn)
    trjs = md.Trajectory(xyzs, struct.topology)
    trjs_cluster_atms = trjs.atom_slice(atm_inds_cluster)

    # cluster simulations
    cluster_obj = KHybrid(metric=md.rmsd, cluster_radius=cluster_radius, kmedoids_updates=1)
    cluster_obj.fit(trjs_cluster_atms)
    center_indices, distances, assignments, centers = \
            cluster_obj.result_.partition(trj_lengths)

    # save clustering output
    assign_fn = os.path.join(msm_dir, "assignments.h5")
    ra.save(assign_fn, assignments)
    dist_fn = os.path.join(msm_dir, "distances.h5")
    ra.save(dist_fn, distances)
    center_inds_fn = os.path.join(msm_dir, "center_inds.txt")
    np.savetxt(center_inds_fn, np.array(center_indices, dtype=int), fmt="%d")

    centers = trjs[cluster_obj.center_indices_]
    centers.superpose(struct)
    centers.save_xtc(centers_fn)

if __name__ == '__main__':
    run()

