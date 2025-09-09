from functools import partial
from multiprocessing import Process
import math

def async_sims(sim_func, structures, dir_names):
    """
    Run a simulation function asynchronously for a list of structures and directory names.

    Args:
        sim_func: The simulation function to run.
        structures: An mdtraj.Trajectory object containing the structures to starteach simulation from.
        dir_names: A list of directory names to save the results for each simulation in.
    """
    n_tasks = len(structures)
    tasks = []

    for i in range(n_tasks):
        tasks.append(Process(target=sim_func, args=(structures[i], dir_names[i])))
        tasks[i].start()

    for task in tasks:
        task.join()

def launch_sims(sim_func, structures, base_dir_name, n_sims_per_node, executor):
    """
    Launch a barch of simulations.
    If n_sims_per_node > 1, n_sims_per_node simulations will be run asynchronously on each node.

    Args:
        sim_func: The simulation function to run.
        structures: An mdtraj.Trajectory object containing the structures to starteach simulation from.
        base_dir_name: Results from simulation X will be saved in base_dir_nameX
        n_sims_per_node: The number of simulations to run per node.
        executor: A submitit.Executor object to use to run the simulations.
    """
    n_sims = len(structures)

    jobs = []
    if n_sims_per_node == 1:
        for i in range(n_sims):
            jobs.append(executor.submit(sim_func, structures[i], f"{base_dir_name}{i}"))
        
    else:
        n_jobs = math.ceil(n_sims / n_sims_per_node)

        for i in range(n_jobs):
            start_idx = i * n_sims_per_node
            end_idx = min(n_sims, start_idx + n_sims_per_node)
            node_structures = structures[start_idx:end_idx]
            node_dir_names = [f"{base_dir_name}{i}" for i in range(start_idx, end_idx)]
            jobs.append(executor.submit(async_sims, sim_func, node_structures, node_dir_names))   

    for job in jobs:
        job.result()
