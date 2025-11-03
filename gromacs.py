from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Tuple, List
import glob
import mdtraj as md
import os
import re
import shutil
import subprocess
import tempfile

@dataclass
class MDP:
    """Class for handling GROMACS MDP files.
    
    This class provides functionality to read, write, and modify GROMACS MDP files.
    It handles parameter validation and provides a convenient interface for working
    with MDP parameters.
    """
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    _filepath: Optional[str] = None
    
    @classmethod
    def from_file(cls, filepath: str) -> 'MDP':
        """Create an MDP instance from a file.
        
        Args:
            filepath: Path to the MDP file
            
        Returns:
            MDP instance with parameters loaded from file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MDP file not found: {filepath}")
            
        mdp = cls()
        mdp._filepath = filepath
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith(';'):
                    continue
                    
                # Parse parameter and value
                if '=' in line:
                    param, value = line.split('=', 1)
                    param = param.strip()
                    value = value.strip()
                    
                    # Convert value to appropriate type
                    if value.lower() == 'yes':
                        value = True
                    elif value.lower() == 'no':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string if not a number
                            
                    mdp.parameters[param] = value
        
        mdp._fix_case()
             
        return mdp
    
    @classmethod
    def minimize(cls) -> 'MDP':
        """Create an MDP instance configured for energy minimization.
        
        Returns:
            MDP instance with energy minimization parameters set
        """
        mdp = cls()
        
        # Energy minimization parameters
        mdp.parameters.update({
            'constraints': 'none',    # No constraints
            'integrator': 'steep',    # Steepest descent minimization
            'nsteps': 5000,           # Maximum number of steps
            'emtol': 1000,            # Stop minimization when max force < 1000 kJ/mol/nm
            'emstep': 0.01,           # Initial step size
            'dt': 0.001,              # Time step
            
            # Neighbor searching
            'ns_type': 'grid',        # Grid-based neighbor searching
            'rlist': 1.2,             # Neighbor list cutoff
            'rcoulomb': 1.2,          # Coulomb cutoff
            'rvdw': 1.2,              # Van der Waals cutoff
            'vdwtype': 'cut-off',    # Van der Waals type
            
            # Temperature and pressure coupling
            'Tcoupl': 'no',           # No temperature coupling
            'Pcoupl': 'no',           # No pressure coupling
            
            # Velocity generation
            'gen_vel': 'no',          # No velocity generation
        })
        
        mdp._fix_case()
        
        return mdp
    
    @classmethod
    def equilibrate(cls) -> 'MDP':
        """Create an MDP instance configured for equilibration.
        
        Returns:
            MDP instance with equilibration parameters set
        """
        mdp = MDP.npt()
        
        # Equilibration parameters
        mdp.set('define', '-DPOSRES')       # Position restraints
        mdp.set('nsteps', 250000)
        mdp.set('pcoupl', 'berendsen')
        mdp.set('nstcomm', 0)
        mdp.set('comm-mode', 'none')
        mdp.set('refcoord-scaling', 'all')
        
        mdp._fix_case()
        
        return mdp
    
    @classmethod
    def npt(cls,
        xtc_grps: str = 'Prot-Masses',
        gpu_only: bool = False,
        hmr: bool = False,
    ) -> 'MDP':
        """Create an MDP instance configured for NPT production simulation.
        
        Args:
            xtc_grps: Groups for compressed output (.xtc files)
            gpu_only: Whether to only use GPU for simulation
            hmr: Whether to use HMR (hydrogen mass repartitioning)

        Returns:
            MDP instance with NPT production parameters set
        """
        mdp = cls()
        
        # Basic MD settings
        mdp.parameters.update({
            'integrator': 'md',              # Molecular dynamics
            'dt': 0.002,                     # Time step
            
            # Output control
            'nsteps': 25000000,              # 50 ns
            'nstxout': 0,                    # No coordinate output
            'nstvout': 0,                    # No velocity output
            'nstfout': 0,                    # No force output
            'nstxout-compressed': 5000,      # Compressed trajectory output
            'nstenergy': 5000,               # Energy output frequency
            'nstlog': 5000,                  # Log output frequency
            'nstcalcenergy': 100,            # Energy calculation frequency
            
            'compressed-x-grps': xtc_grps,   # Groups for compressed output

            # Van der Waals and electrostatics
            'cutoff-scheme': 'Verlet',       # Verlet cutoff scheme
            'nstlist': 50,                   # Neighbor list update frequency
            'rlist': 1.1,                    # Neighbor list cutoff
            'coulombtype': 'pme',            # Particle Mesh Ewald
            'rcoulomb': 1.1,                 # Coulomb cutoff
            'vdwtype': 'cut-off',            # Van der Waals type
            'vdw-modifier': 'None',          # No force switching
            'rvdw': 0.9,                     # Van der Waals cutoff
            'DispCorr': 'EnerPres',          # No dispersion correction
            'pbc': 'xyz',                    # Periodic boundary conditions

            # Temperature control
            'tcoupl': 'v-rescale',           # Velocity rescaling thermostat
            'tc-grps': 'system',# Temperature coupling groups
            'tau_t': '0.1',              # Temperature coupling time constants
            'ref_t': '300',              # Reference temperatures
            
            # Pressure control
            'pcoupl': 'Parrinello-Rahman',   # Parrinello-Rahman barostat
            'pcoupltype': 'isotropic',       # Isotropic pressure coupling
            'tau_p': 5.0,                    # Pressure coupling time constant
            'compressibility': 4.5e-5,       # Isothermal compressibility
            'ref_p': 1.0,                # Reference pressure (1 atm)
            
            # Bond parameters
            'constraint_algorithm': 'lincs', # LINCS algorithm
            'constraints': 'h-bonds',        # H-bonds constrained
            'lincs_iter': 1,                # LINCS iterations
            'lincs_order': 4,               # LINCS order

            # Initial conditions
            'gen_vel': 'yes',                # Generate velocities
            'gen_temp': 300,                 # Initial temperature
            
            # COM removal
            'nstcomm': 100,                  # COM removal frequency
            'comm-mode': 'linear',           # Linear COM removal
            'comm-grps': 'System',           # COM removal groups
        })

        if gpu_only:
            mdp.set('nstlist', 300)
        if hmr:
            mdp.update_dt_and_nst(0.004)
            mdp.set('mass-repartition-factor', 2)
            mdp.set('constraints', 'h-bonds') # had tried 'all' but that doesn't work on gpu only
            mdp.set('lincs_iter', 2)
            mdp.set('lincs_order', 6)
        
        mdp._fix_case()

        return mdp
    
    def _fix_case(self):
        """Force all parameter names to lowercase, except flags set with define which are uppercase."""
        temp = {}
        for k, v in self.parameters.items():
            new_k = str(k).lower()
            if new_k == 'define':
                temp[new_k] = str(v).upper()
            else:
                temp[new_k] = str(v).lower()

            
        self.parameters.clear()
        self.parameters.update(temp)

    def write(self, filepath: Optional[str] = None) -> None:
        """Write the MDP parameters to a file.
        
        Args:
            filepath: Path to write the MDP file to. If None, uses the original filepath.
            
        Raises:
            ValueError: If no filepath is provided and no original filepath exists
        """
        if filepath is None:
            filepath = self._filepath
            if filepath is None:
                raise ValueError("No filepath provided and no original filepath exists")
                
        with open(filepath, 'w') as f:
            for param, value in sorted(self.parameters.items()):
                if isinstance(value, bool):
                    value = 'yes' if value else 'no'
                f.write(f"{param} = {value}\n")
                
    def get(self, param: str, default: Any = None) -> Any:
        """Get a parameter value.
        
        Args:
            param: Parameter name
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value or default if not found
        """
        return self.parameters.get(param, default)
        
    def set(self, param: str, value: Any) -> None:
        """Set a parameter value.
        
        Args:
            param: Parameter name
            value: Parameter value
        """
        self.parameters[str(param).lower()] = str(value).lower()
        
    def update_dt_and_nst(self, new_dt: float) -> None:
        """Change the time step (dt) and update all the nst parameters to keep them at the same frequency in ps.
        
        Args:
            new_dt: New time step (ps)
        """
        old_dt = float(self.parameters['dt'])
        ratio = old_dt / new_dt
        self.parameters['dt'] = new_dt

        nst_params = ['nsteps', 'nstxout', 'nstvout', 'nstfout',
            'nstxout-compressed', 'nstenergy', 'nstlog', 'nstcalcenergy',
            'nstlist', 'nstcomm']
        
        for param in nst_params:
            if param in self.parameters.keys():
                self.parameters[param] = int(int(self.parameters[param]) * ratio)

    def validate(self) -> bool:
        """Validate the MDP parameters.
        WARNING: This method may be incomplete/inaccurate as 1) it was partly generated by AI and
        2) the Gromacs developers have ocassionally changed parameters over time.
        
        Returns:
            True if parameters are valid, False otherwise.
        """
        self._fix_case()
        
        pass_test = True

        valid_parameters = {
            "integrator": ["md", "md-vv", "md-vv-avek", "sd", "bd", "steep", "cg", "l-bfgs", "nm", "tpi", "tpic", "mimic"],
            "dt": ["<float>"],
            "nsteps": ["<int>"],
            "mass-repartition-factor": ["<int>"],
            "define": ["<string>"],
            "emtol": ["<float>"],
            "emstep": ["<float>"],
            "comm-mode": ["linear", "angular", "none", "linear-acceleration-correction"],
            "nstcomm": ["<int>"],
            "comm-grps": ["<group names>"],
            "refcoord-scaling": ["no", "all", "com"],
            "nstxout": ["<int>"],
            "nstvout": ["<int>"],
            "nstfout": ["<int>"],
            "nstlog": ["<int>"],
            "nstcalcenergy": ["<int>"],
            "nstenergy": ["<int>"],
            "nstxout-compressed": ["<int>"],
            "compressed-x-grps": ["<group names>"],
            "xtc-precision": ["<int>"],
            "nstlist": ["<int>"],
            "ns_type": ["grid", "simple"],
            "cutoff-scheme": ["verlet", "group"],
            "pbc": ["xyz", "no", "xy", "yz", "xz"],
            "rlist": ["<float>"],
            "coulombtype": ["cut-off", "reaction-field", "pme", "pme-switch", "ewald", "generalized-reaction-field"],
            "rcoulomb": ["<float>"],
            "epsilon_r": ["<float>"],
            "vdwtype": ["cut-off", "switch", "shift", "user"],
            "vdw-modifier": ["none", "potential-shift", "force-switch", "potential-switch"],
            "rvdw": ["<float>"],
            "dispcorr": ["no", "ener", "enerpres"],
            "tcoupl": ["no", "berendsen", "nose-hoover", "v-rescale", "andersen"],
            "tc-grps": ["<group names>"],
            "tau_t": ["<float>"],
            "ref_t": ["<float>"],
            "pcoupl": ["no", "berendsen", "parrinello-rahman", "martini", "c-rescale", "mttk"],
            "pcoupltype": ["isotropic", "anisotropic", "semiisotropic", "surface-tension"],
            "tau_p": ["<float>"],
            "ref_p": ["<float>"],
            "compressibility": ["<float>"],
            "gen_vel": ["yes", "no"],
            "gen_temp": ["<float>"],
            "gen_seed": ["<int>"],
            "constraints": ["none", "h-bonds", "all-bonds", "h-angles", "all-angles"],
            "constraint_algorithm": ["lincs", "shake"],
            "continuation": ["yes", "no"],
            "lincs_order": ["<int>"],
            "lincs_iter": ["<int>"],
            "fourierspacing": ["<float>"],
            "pme_order": ["<int>"],
            "ewald_rtol": ["<float>"],
            "free_energy": ["yes", "no"],
            "init_lambda_state": ["<int>"],
            "delta_lambda": ["<float>"],
            "implicit_solvent": ["no", "gbsa"],
            "gb_algorithm": ["still", "hct", "obc", "gbsa"],
            "nstgbradii": ["<int>"],
            "rgbradii": ["<float>"],
            "gb_epsilon_solvent": ["<float>"],
            "morse": ["yes", "no"]
        }

        for key, val in self.parameters.items():
            if key not in valid_parameters:
                pass_test = False
                print(f"Invalid parameter: {key}")
            else:
                allowed = valid_parameters[key]
                if any(a.startswith("<") and a.endswith(">") for a in allowed):
                    # assume value is valid if it matches the type placeholder
                    placeholder = allowed[0]
                    try:
                        if placeholder == "<float>":
                            float(val)
                        elif placeholder == "<int>":
                            int(val)
                        else:
                            str(val)  # accept anything for <group names>
                    except ValueError:
                        pass_test = False
                        print(f"Invalid value for {key}: {val}")
                else:
                    if str(val) not in allowed:
                        pass_test = False
                        print(f"Invalid value for {key}: {val}")
        if not pass_test:
            print("WARNING: mdp validation may be incomplete/inaccurate as 1) it was partly generated by AI and 2) the Gromacs developers have ocassionally changed parameters over time.")
            print("If needed, you can update the validation method or turn off validation.")
        return pass_test

        
    def __str__(self) -> str:
        """String representation of the MDP parameters."""
        return '\n'.join(f"{param} = {value}" for param, value in sorted(self.parameters.items()))


def load_topology(top_file: str) -> str:
    """Load a GROMACS topology file (.top or .itp).
    
    Args:
        top_file: Path to the topology file
    """
    with open(top_file, "r") as file:
        return file.read()

def save_topology(top: str, top_file: str) -> None:
    """Save a GROMACS topology to a file.
    
    Args:
        top: Topology string
        top_file: Path to the topology file
    """
    with open(top_file, "w") as file:
        file.write(top)

def load_itps(dir_name: str) -> List[Tuple[str, str]]:
    """Load all itp files in a directory.
    
    Args:
        dir_name: Path to the directory
    """
    itp_files = glob.glob(os.path.join(dir_name, "*.itp"))
    itps = []
    for itp_file in itp_files:
        itp = load_topology(itp_file)
        itps.append((os.path.basename(itp_file), itp))

    return itps

def save_itps(itps: List[Tuple[str, str]], dir_name: str) -> None:
    """Save a list of itp files to a directory.
    
    Args:
        itps: List of itp files (each a tuple of the file name and the file contents)
        dir_name: Path to the directory
    """
    os.makedirs(dir_name, exist_ok=True)
    for itp_file, itp in itps:
        save_topology(itp, os.path.join(dir_name, itp_file))

def create_system(
    structure: md.Trajectory, 
    output_dir: str = "system-creation",
    forcefield: str = 'amber03',
    water_model: str = 'tip3p',
    box_padding: float = 1.0,
    box_type: str = 'dodecahedron',
    add_ions: bool = True,
    neutralize: bool = True,
    ion_concentration: float = 0.1,
) -> Tuple[md.Trajectory, str, List[Tuple[str, str]]]:
    """Create a system for a GROMACS simulation from an MDTraj trajectory.
    
    Args:
        structure: MDTraj trajectory object
        output_dir: Directory to write simulation files
        forcefield: GROMACS forcefield to use
        water_model: Water model to use
        box_padding: Padding around protein in nm
        box_type: Type of simulation box ('cubic', 'triclinic', 'dodecahedron', 'octahedron')
        add_ions: Whether to add ions
        neutralize: Whether to neutralize the system
        ion_concentration: Ion concentration in mol/L
        
    Returns:
        A tuple of the final structure, topology, and a list of itp files (each a tuple of the file name and the file contents).
        The final structure is an MDTraj trajectory object.
        The topology is a string.
        The itp files are a list of tuples, each containing the file name and the file contents.
        
    Raises:
        RuntimeError: If GROMACS commands fail
        ValueError: If box_type is invalid
    """
    print("-" * 40)
    print("Setting up simulation...")

    # Validate box type
    valid_box_types = ['cubic', 'triclinic', 'dodecahedron', 'octahedron']
    if box_type not in valid_box_types:
        raise ValueError(f"box_type must be one of {valid_box_types}")
    
    # Create output directory if it doesn't exist
    orig_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Save trajectory as PDB
    pdb_file = 'system.pdb'
    structure.save_pdb(pdb_file)
 
    # Generate topology using pdb2gmx
    top_file = 'topol.top'
    gro_file = 'start.gro'
    
    cmd = ['gmx', 'pdb2gmx', '-f', pdb_file, '-o', gro_file, 
            '-water', water_model, '-ff', forcefield, '-p', top_file,
            '-ignh']
    
    try:
        with open("pdb2gmx.txt", "w") as f:
            subprocess.run(cmd, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        os.chdir(orig_dir)
        raise RuntimeError(f"pdb2gmx failed: {e.stderr}")
    
    # Define box and add solvent
    box_file = 'box.gro'
    cmd = ['gmx', 'editconf', '-f', gro_file, '-o', box_file,
            '-c', '-d', str(box_padding), '-bt', box_type]
    
    try:
        with open("editconf.txt", "w") as f:
            subprocess.run(cmd, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        os.chdir(orig_dir)
        raise RuntimeError(f"editconf failed: {e.stderr}")
    
    # Add water
    solv_file = 'solv.gro'
    cmd = ['gmx', 'solvate', '-cp', box_file, '-cs',
            '-o', solv_file, '-p', top_file]
    
    try:
        with open("solvate.txt", "w") as f:
            subprocess.run(cmd, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        os.chdir(orig_dir)
        raise RuntimeError(f"solvate failed: {e.stderr}")
    
    # Add ions if requested
    if neutralize or add_ions:
        # Create ions.mdp file
        ions_mdp = MDP.minimize()
        ions_mdp_file = 'ions.mdp'
        ions_mdp.write(ions_mdp_file)
        ions_tpr_file = 'ions.tpr'
        
        ions_file = 'ions.gro'
        cmd = ['gmx', 'grompp', '-f', ions_mdp_file, '-c', solv_file,
                '-p', top_file, '-o', ions_tpr_file]
        
        try:
            with open("grompp.txt", "w") as f:
                subprocess.run(cmd, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            os.chdir(orig_dir)
            raise RuntimeError(f"grompp for ions failed: {e.stderr}")
        
    
        # Add ions
        echo_process = subprocess.Popen(['echo', 'SOL'], stdout=subprocess.PIPE)
        cmd = ['gmx', 'genion', '-s', ions_tpr_file,
              '-o', ions_file, '-p', top_file, '-pname', 'NA', '-nname', 'CL',
              '-neutral' if neutralize else '', '-conc', str(ion_concentration)]
        
        try:
            with open("genion.txt", "w") as f:
                subprocess.run(cmd, stdin=echo_process.stdout, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
            echo_process.wait()
        except subprocess.CalledProcessError as e:
            os.chdir(orig_dir)
            raise RuntimeError(f"genion failed: {e.stderr}")
        
        final_gro_file = ions_file
    else:
        final_gro_file = solv_file
    
    shutil.copy2(final_gro_file, 'final_system.gro')

    # Convert to pdb so can preserve atom naming since mdtraj messes up gro files
    trjconv('final_system.gro', 'final_system.pdb')
    final_struct = md.load('final_system.pdb', standard_names=False)

    top = load_topology(top_file)

    itps = load_itps(".")

    os.chdir(orig_dir)
    
    return final_struct, top, itps

def print_warnings_and_notes(filename: str):
    with open(filename, 'r') as file:
        lines = file.readlines()

    in_block = False
    current_block = []

    for line in lines:
        stripped = line.strip()

        if not in_block:
            if stripped.startswith("NOTE") or stripped.startswith("WARNING"):
                in_block = True
                current_block.append("  " + line)
        else:
            if stripped == "":
                # End of current block
                print("".join(current_block))
                current_block = []
                in_block = False
            else:
                current_block.append("  " + line)

    # Print any remaining block at the end of file
    if in_block and current_block:
        print("".join(current_block))

def mdrun(
    structure: md.Trajectory,
    output_dir: str,
    top: str,
    itps: List[str],
    mdp: MDP,
    maxwarn: int = 0,
    restrain: bool = False,
    nsteps: int = None,
    xtc_grps: str = 'Prot-Masses',
    gpu_only: bool = False,
    validate: bool = True,
    add_cmd: str = None,
    solute_grps: str = None,
    **kwargs,
) -> md.Trajectory:
    """Run a simulation from a given starting structure.
    
    Args:
        structure: MDTraj trajectory object
        output_dir: Directory to write simulation files
        top: Topology string
        itps: List of itp files (each a string)
        mdp: MDP object
        maxwarn: Maximum number of warnings to ignore
        restrain: Whether to restrain the structure
        nsteps: Override the nsteps parameter in the mdp file
        xtc_grps: Groups for compressed trajectory output (xtc file)
        validate: Whether to validate the mdp file
        gpu_only: Whether to only use GPU for simulation
        add_cmd: Additional command to run before executing mdrun (e.g. module load module_name)
        **kwargs: Additional arguments to pass to mdrun (e.g. pin='on')

    Returns:
        MDTraj trajectory object containing final structure
    """

    if nsteps is not None:
        mdp.set('nsteps', nsteps)
    mdp.set('compressed-x-grps', xtc_grps)
    if validate:
        if not mdp.validate():
            raise ValueError("Invalid MDP parameters")
    
    # Create output directory if it doesn't exist
    orig_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Save structure as gro file
    input_gro_file = 'input-system.gro'
    structure.save(input_gro_file)

    # Write topology and itp files
    save_topology(top, 'topol.top')
    save_itps(itps, '.')

    # Write mdp file
    mdp.write('settings.mdp')

    tpr_file = 'simulation.tpr'

    # cmd = ['gmx', 'grompp', '-f', 'settings.mdp', '-c', input_gro_file,
    #        '-p', 'topol.top', '-o', 'simulation.tpr', '-maxwarn', str(maxwarn)]
    # if restrain:
    #     cmd += ['-r', input_gro_file]
    # print("  " + " ".join(cmd))

    # try:
    #     with open("grompp.txt", "w") as f:
    #         subprocess.run(cmd, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
    # except subprocess.CalledProcessError as e:
    #     os.chdir(orig_dir)
    #     raise RuntimeError(f"grompp failed: {e.stderr}")

    cmd = f' gmx grompp -f settings.mdp -c {input_gro_file} -p topol.top -o simulation.tpr -maxwarn {maxwarn}'
    if add_cmd:
        cmd = f'{add_cmd} && ' + cmd
    if restrain:
        cmd += f' -r {input_gro_file}'
    print("  " + cmd)

    try:
        with open("grompp.txt", "w") as f:
            subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        os.chdir(orig_dir)
        raise RuntimeError(f"grompp failed: {e.stderr}")

    print_warnings_and_notes("grompp.txt")
    
    xtc_file = 'simulation.xtc'
    xtc_intermediate_file = 'simulation-nojump.xtc'
    xtc_nojump_file = 'simulation-nojump-mol.xtc'
    xtc_nojump_file_solute = 'simulation-nojump-solute.xtc'
    xtc_gro_file = 'xtc-grps-nojump.gro'
    output_gro_file = 'output-system.gro'
    output_nojump_gro_file = 'output-system-nojump.gro'
    output_nojump_pdb_file = 'output-system-nojump.pdb'

    # cmd = ['gmx', 'mdrun', '-v', '-s', tpr_file, '-x', xtc_file, '-c', output_gro_file]
    # if kwargs:
    #     for k, v in kwargs.items():
    #         if v is not None:
    #             cmd += [f"-{k}", f"{v}"]
    #         else:
    #             cmd += [f"-{k}"]

    # if gpu_only:
    #     cmd += ['-nt', '1', '-ntmpi', '1', '-ntomp', '1', '-pin', 'off', '-nb', 'gpu', '-pme', 'gpu', '-bonded', 'gpu', '-update', 'gpu']
    # print("  " + " ".join(cmd))

    # try:
    #     with open("mdrun.txt", "w") as f:
    #         subprocess.run(cmd, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
    # except subprocess.CalledProcessError as e:
    #     os.chdir(orig_dir)
    #     raise RuntimeError(f"mdrun failed: {e.stderr}")
    # print_warnings_and_notes("mdrun.txt")

    cmd = f'gmx mdrun -v -s {tpr_file} -x {xtc_file} -c {output_gro_file}'
    if add_cmd:
        cmd = f'{add_cmd} && ' + cmd
    if kwargs:
        for k, v in kwargs.items():
            if v is not None:
                cmd += f" -{k} {v}"
            else:
                cmd += f" -{k}"

    if gpu_only:
        cmd += ' -nt 1 -ntmpi 1 -ntomp 1 -pin off -nb gpu -pme gpu -bonded gpu -update gpu'

    try:
        with open("mdrun.txt", "a") as f:
            subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        os.chdir(orig_dir)
        raise RuntimeError(f"mdrun failed: {e.stderr}")
    print_warnings_and_notes("mdrun.txt")

    # Remove pbc jumps
    trjconv(output_gro_file, output_nojump_gro_file, ref_fn='simulation.tpr', pbc='mol', add_cmd=add_cmd)
    if os.path.exists(xtc_file):
        trjconv(output_nojump_gro_file, xtc_gro_file, xtc_grps=xtc_grps, add_cmd=add_cmd)

        trjconv(xtc_file, xtc_intermediate_file, ref_fn='simulation.tpr', pbc='nojump', add_cmd=add_cmd)
        trjconv(xtc_intermediate_file, xtc_nojump_file, ref_fn='simulation.tpr', pbc='mol', add_cmd=add_cmd, ur='compact', center='')
        if solute_grps is not None:
            trjconv(xtc_nojump_file, xtc_nojump_file_solute, 
                ref_fn=xtc_gro_file, xtc_grps= solute_grps, add_cmd=add_cmd)

    # Convert to pdb so can preserve atom naming since mdtraj messes up gro files
    trjconv(output_nojump_gro_file, output_nojump_pdb_file, add_cmd=add_cmd)
    try:
        final_struct = md.load(output_nojump_pdb_file, standard_names=False)
    except:
        print(f'Could not load {output_nojump_pdb_file}.')
    os.chdir(orig_dir)

    return final_struct

def minimize(
    structure: md.Trajectory,
    output_dir: str,
    top: str,
    itps: List[str],
    mdp: MDP = MDP.minimize(),
    maxwarn: int = 0,
    nsteps: int = None,
    **kwargs,
) -> md.Trajectory:
    """Minimize a structure.
    
    Args:
        structure: MDTraj trajectory object
        output_dir: Directory to write simulation files
        top: Topology string
        itps: List of itp files (each a string)
        mdp: MDP object
        maxwarn: Maximum number of warnings to ignore
        nsteps: Override the nsteps parameter in the mdp file
        **kwargs: Additional arguments to pass to mdrun (e.g. pin='on')

    Returns:
        MDTraj trajectory object containing minimized structure
    """
    print("-" * 40)
    print("Performing minimization and printing warnings/notes...")
    output = mdrun(structure, output_dir, top, itps, mdp, maxwarn=maxwarn, nsteps=nsteps, **kwargs)
    fmax = extract_fmax(f"{output_dir}/mdrun.txt")
    if fmax < float(mdp.parameters['emtol']):
        print("  Success. Fmax reached", fmax, "and target was", mdp.parameters['emtol'])
    else:
        raise ValueError("  Failed. Fmax reached", fmax, "but target was", mdp.parameters['emtol'])
    
    return output

def equilibrate(
    structure: md.Trajectory,
    output_dir: str,
    top: str,
    itps: List[str],
    mdp: MDP = MDP.equilibrate(),
    maxwarn: int = 1,
    nsteps: int = None,
    **kwargs,
) -> md.Trajectory:
    """Equilibrate a structure.
    
    Args:
        structure: MDTraj trajectory object
        output_dir: Directory to write simulation files
        top: Topology string
        itps: List of itp files (each a string)
        mdp: MDP object
        maxwarn: Maximum number of warnings to ignore
        nsteps: Override the nsteps parameter in the mdp file
        **kwargs: Additional arguments to pass to mdrun (e.g. pin='on')

    Returns:
        MDTraj trajectory object containing equilibrated structure
    """
    print("-" * 40)
    print("Performing equilibration and printing warnings/notes...")
    return mdrun(structure, output_dir, top, itps, mdp, maxwarn=maxwarn, restrain=True, nsteps=nsteps, **kwargs)

def npt_run(
    structure: md.Trajectory,
    output_dir: str,
    top: str,
    itps: List[str],
    mdp: MDP = MDP.npt(),
    maxwarn: int = 0,
    nsteps: int = None,
    xtc_grps: str = 'Prot-Masses',
    solute_grps: str = None,
    **kwargs,
) -> md.Trajectory:
    """Run an NPT simulatoin from a structure.
    
    Args:
        structure: MDTraj trajectory object
        output_dir: Directory to write simulation files
        top: Topology string
        itps: List of itp files (each a string)
        mdp: MDP object
        maxwarn: Maximum number of warnings to ignore
        nsteps: Override the nsteps parameter in the mdp file
        xtc_grps: Groups for compressed trajectory output (xtc file)
        **kwargs: Additional arguments to pass to mdrun (e.g. pin='on')

    Returns:
        MDTraj trajectory object containing minimized structure
    """
    print("-" * 40)
    print(kwargs)
    print("Performing NPT simulation and printing warnings/notes...")
    return mdrun(structure, output_dir, top, itps, mdp, maxwarn=maxwarn, nsteps=nsteps, xtc_grps=xtc_grps, 
        solute_grps=solute_grps, **kwargs)

def trjconv(
    input_fn: str,
    output_fn: str,
    pbc: Optional[str] = None,
    ref_fn: Optional[str] = None,
    xtc_grps: Optional[str] = None,
    add_cmd: str = None,
    **kwargs,
) -> None:
    """Convert a file from one format to another.
    
    Args:
        input_fn: Input file name
        output_fn: Output file name
        pbc: how remove period boundary issues
        ref_fn: reference structure file for pbc removal
        xtc_grps: Groups for compressed trajectory output (xtc file)
        add_cmd: Additional command to run before executing mdrun (e.g. module load module_name)
        **kwargs: Additional commands to pass to Gromacs (e.g. -ur compact)
    """
    if ref_fn is None:
        ref_fn = input_fn

    cmd = f'gmx trjconv -f {input_fn} -s {ref_fn} -o {output_fn}'
    if pbc is not None:
        cmd += f' -pbc {pbc}'
    if add_cmd:
        cmd = f'{add_cmd} && ' + cmd
    if kwargs:
        cmd += ' ' + ' '.join(f"-{key} {value}" for key, value in kwargs.items())

    if xtc_grps is not None:
        if 'center' in kwargs:
            echo_process = subprocess.Popen(['echo', 'protein', xtc_grps], stdout=subprocess.PIPE)
        else:  
            echo_process = subprocess.Popen(['echo', xtc_grps], stdout=subprocess.PIPE)
    else:
        if 'center' in kwargs:
            echo_process = subprocess.Popen(['echo', 'protein', '0'], stdout=subprocess.PIPE)
        else:
            echo_process = subprocess.Popen(['echo', '0'], stdout=subprocess.PIPE)

    try:
        with open("trjconv.txt", "a") as f:
            #subprocess.run(cmd, stdin=echo_process.stdout, check=True, text=True, stdout=f, stderr=subprocess.STDOUT)
            subprocess.run(cmd, shell=True, executable="/bin/bash", stdin=echo_process.stdout, stdout=f, stderr=subprocess.STDOUT)
        echo_process.wait()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"trjconv failed: {e.stderr}")

def extract_fmax(filename: str) -> float:
    """
    Extracts final Fmax value from a GROMACS log file.
    
    Args:
        filename (str): The name of the GROMACS log file.

    Returns:
        float: The extracted Fmax value, or None if not found.
    """
    last_fmax_line = None

    with open(filename, 'r') as file:
        for line in file:
            if 'Fmax=' in line:
                last_fmax_line = line.strip()

    if last_fmax_line is not None:
        match = re.search(r"Fmax=\s*([0-9.+\-eE]+)", last_fmax_line)
        return float(match.group(1))
    else:   
        raise ValueError(f"No Fmax found in {filename}")
        
