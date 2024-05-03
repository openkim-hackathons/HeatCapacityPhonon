import os
import random
import subprocess
from typing import Iterable, List, Optional, Tuple, Sequence
import uuid
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.build import bulk
from ase.cell import Cell
from ase.utils import structure_comparator as sc
from kim_python_utils.ase import CrystalGenomeTest, KIMASEError

class HeatCapacityPhonon(CrystalGenomeTest):
    def reduce_and_avg(self, atoms, repeat):
        '''
        Function to reduce all atoms to unit cell position and return averaged unit cell

        @param atoms : repeated atoms object
        @param repeat : repeat tuple

        @return prim_cell : primitive cell
        '''

        # Scale cell of bulk.
        cell = atoms.get_cell()
        
        # Divide each unit vector by its number of repeats.
        # See https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element.
        cell = cell / np.array(repeat)[:, None]
        
        # Decrease size of cell in the atoms object.
        atoms.set_cell(cell)
        atoms.set_pbc((True, True, True))

        positions_in_prim_cell = np.zeros((len(atoms), 3))

        # Set averaging factor
        M = np.prod(repeat)

        # The scaled positions will automatically wrap back the repeated atoms on top of the original 
        # atoms in the primitive cell.
        scaled_positions = atoms.get_scaled_positions()

        for i in range(len(atoms)):
            for d in range(3):
                positions_in_prim_cell[i % n][d] += scaled_positions[i][d] / M

        atoms.set_scaled_positions(positions_in_prim_cell)

    def _calculate(self, structure_index: int, temperature: float, pressure: float, timestep: float, 
                   number_sampling_timesteps: int, repeat: Tuple[int, int, int] = (3, 3, 3), 
                   seed: Optional[int] = None) -> None:
        """
        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). 
            This indicates which is being used for the current calculation.

        temperature:
            Temperature in Kelvin at which the phonon contribution to the heat capacity 
            at constant volume is estimated. Must be strictly greater than zero.
        
        pressure:
            Pressure in bar of the NPT simulation for the initial equilibration of the 
            zero-temperature configuration. Must be strictly greater than zero.

        # TODO: Document arguments and add sensible default values.
        """
        # Check arguments.
        if not temperature > 0.0:
            raise RuntimeError("Temperature has to be larger than zero.")
        
        if not pressure > 0.0:
            raise RuntimeError("Pressure has to be larger than zero.")

        # TODO: Check all arguments.
        
        # TODO: Ask whether this is the correct way.
        # TODO: Filter out repeated species.
        # TODO: Just get the species as a test argument (they are definitely alphabetically ordered).
        # TODO: They changed the atoms object so this might actually be fixed.
        # Get species and masses of the atoms.
        atoms = self.atoms[structure_index]
        
        # TODO: Remove this hack at some point.
        species_of_each_atom = atoms.get_chemical_symbols()[:1]
        masses = atoms.get_masses()
        
        # Copy original atoms so that their information does not get lost when the new atoms are modified.
        atoms_new = atoms.copy()
        
        # UNCOMMENT THIS TO TEST A TRICLINIC STRUCTURE!
        # atoms_new = bulk('Ar', 'fcc', a=5.248)
  
        atoms_new = atoms_new.repeat(repeat)

        # Write lammps file.
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        atoms_new.write(structure_file, format="lammps-data")
        self._add_masses_to_structure_file(structure_file, masses)
        
        # LAMMPS for heat capacity
        if seed is None:
            # Get random 31-bit unsigned integer.
            seed = random.getrandbits(31)
            
        # TODO: Move damping factors to argument.
        pdamp = timestep * 100.0
        tdamp = timestep * 1000.0

        # Run NPT simulation.
        variables = {
            "modelname": self.model_name,
            "temperature": temperature,
            "temperature_seed": seed,
            "temperature_damping": tdamp,
            "pressure": pressure,
            "pressure_damping": pdamp,
            "timestep": timestep,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species_of_each_atom) 
        }
        # TODO: Possibly run MPI version of Lammps if available.
        command = (
            "lammps " 
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items()) 
            + " -in npt_equilibration.lammps")
        
        subprocess.run(command, check=True, shell=True)
        self._extract_and_plot() 
        
        # TODO: Philipp prevents that this is even happening.
        os.remove("average_position.dump.0")
        # TODO: Guanming changes this into a function call.
        subprocess.run("python compute_average_positions.py", check=True, shell=True)

        # Check symmetry - post-NPT
        atoms_new.set_positions(self._get_positions_from_lammps_dump("output/average_position_over_files.out"))
        atoms_new.set_cell(self._get_cell_from_lammps_dump("output/average_position_over_files.out"))

        # Reduce and average
        self.reduce_and_avg(atoms_new, repeat)

        # ASE Symmetry check
        comp = sc.SymmetryEquivalenceCheck()
        comp.compare(atoms, atoms_new)

        # AFLOW Symmetry check
        self._update_aflow_designation_from_atoms(structure_index, atoms_new)
        
        """
        # NVT simulation
        vol = np.loadtxt('volume.dat')

        lmp_nvt = 'modelname ${model_name} temperature ${temperature} temperature_seed ${seed} temperature_damping ${tdamp} volume ${vol} timestep ${timestep} number_control_timesteps ${number_control_timesteps}'
        script = 'nvt_equilibration.lammps'
        command = 'lammps -var %s -in %s'%(lmp_npt, script)
        subprocess.run(command, check=True, shell=True) 

        # Check symmetry - post-NVT
        try:
            self._update_aflow_designation_from_atoms(structure_index,atoms_new)
            raise RuntimeError("Symmetry of crystal changed during NVT equilibration!")

        except KIMASEError as e:
            print("We have successfully caught an exception with the following message:")
            print(e.msg)

        ####################################################
        # ACTUAL CALCULATION ENDS 
        ####################################################

        ####################################################
        # SOME USAGE EXAMPLES NOT NECESSARY FOR THE PRESENT TEST 
        ####################################################
        # This is unnecessary here because we are not changing the atoms object, but if we were and we needed to re-analyze, this is how you do it 
        self.atoms[structure_index].set_cell(original_cell,scale_atoms=True)
        self._update_aflow_designation_from_atoms(structure_index)

        # alternatively, you can update the `structure_index`-th AFLOW symmetry designation from a specified atoms object instead of from 
        # self.atoms[structure_index]. The function will also raise an error if the prototype label changes, so you can use it as a try-except to detect
        # symmetry changes
        atoms_new = atoms.copy()
        cell = atoms_new.get_cell()
        cell[1,2] += 0.5 # this is highly likely to change the symmetry
        atoms_new.set_cell(cell,scale_atoms=True)

        try:
            # this will intentionally raise an exception to demonstrate changing symmetry issue
            self._update_aflow_designation_from_atoms(structure_index,atoms_new)
            raise RuntimeError("We should not have gotten here")
        except KIMASEError as e:
            print("We have successfully caught an exception with the following message:")
            print(e.msg)
        ####################################################
        # USAGE EXAMPLES END
        ####################################################

        ####################################################
        # PROPERTY WRITING
        ####################################################

        # Import data
        cv = np.loadtxt('cv.dat')
        cp = np.loadtxt('cp.dat')

        # Assign property
        self._add_property_instance("heat_capacity")
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=False,write_temp=False) # last two default to False
        self._add_key_to_current_property_instance("constant_pressure_heat_capacity",constant_pressure_heat_capacity,"eV/Kelvin")
        self._add_key_to_current_property_instance("constant_volume_heat_capacity",constant_volume_heat_capacity,"eV/Kelvin")
        self._add_key_to_current_property_instance("volume", volume, "Angstroms^3")
        self._add_key_to_current_property_instance("pressure", pressure, "bars")
        ####################################################
        # PROPERTY WRITING END
        ####################################################
        """
    
    @staticmethod
    def _add_masses_to_structure_file(structure_file: str, masses: Iterable[float]) -> None:
        # TODO: This does not always work... (especially when cube is included in atoms)
        with open(structure_file, "a") as file:
            print(file=file)
            print("Masses", file=file)
            print(file=file)
            for i, mass in enumerate(masses):
                print(f"    {i+1} {mass}", file=file)
                # TODO: Remove this hack at some point.
                break
    
    @staticmethod
    def _extract_and_plot(property_name: str = "v_vol_metal") -> None:
        # extract data and save it as png file
        subprocess.run(f"python extract_table.py output/lammps_equilibration.log "
                       f"output/lammps_equilibration.csv {property_name}", check=True, shell=True)

    @staticmethod
    def _get_positions_from_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
        lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key = lambda x: x[0])     
        return [(line[2], line[3], line[4]) for line in lines]

    @staticmethod
    def _get_cell_from_lammps_dump(filename: str) -> npt.NDArray[float]:
        new_cell = np.loadtxt(filename, skiprows=5, max_rows=3)
        assert new_cell.shape == (3, 2) or new_cell.shape == (3, 3)

        # Save cell parameters
        xlo = new_cell[0,0]
        xhi = new_cell[0,1]
        ylo = new_cell[1,0]
        yhi = new_cell[1,1]
        zlo = new_cell[2,0]
        zhi = new_cell[2,1]

        # If not cubic add more cell params
        if new_cell.shape[-1] != 2:
            xy = new_cell[0,2]
            xz = new_cell[1,2]
            yz = new_cell[2,2]
        else:
            xy = 0.0
            xz = 0.0
            yz = 0.0

        # See https://docs.lammps.org/Howto_triclinic.html.
        cell = np.empty(shape=(3, 3))
        cell[0, :] = np.array([xhi - xlo, 0.0, 0.0])
        cell[1, :] = np.array([xy, yhi - ylo, 0.0])
        cell[2, :] = np.array([xz, yz, zhi - zlo])
        return cell

    @staticmethod
    def _set_atoms_from_lmp_file(atoms: Atoms, lmp_file: str) -> None:
        # HACKY!
        pos = np.loadtxt(lmp_file, skiprows=12, max_rows=len(atoms))
        atoms.set_positions([pos[i, 2:] for i in range(len(atoms))])
        with open(lmp_file, "r") as file:
            for index, line in enumerate(file):
                if index == 4:
                    ls = line.split()
                    assert len(ls) == 4
                    xlo = float(ls[0])
                    xhi = float(ls[1])
                elif index == 5:
                    ls = line.split()
                    assert len(ls) == 4
                    ylo = float(ls[0])
                    yhi = float(ls[1])
                elif index == 6:
                    ls = line.split()
                    assert len(ls) == 4
                    zlo = float(ls[0])
                    zhi = float(ls[1])
                elif index == 7:
                    ls = line.split()
                    assert len(ls) == 6
                    xy = float(ls[0])
                    xz = float(ls[1])
                    yz = float(ls[2])
        cell = np.empty(shape=(3, 3))
        cell[0, :] = np.array([xhi - xlo, 0.0, 0.0])
        cell[1, :] = np.array([xy, yhi - ylo, 0.0])
        cell[2, :] = np.array([xz, yz, zhi - zlo])
        atoms.set_cell(cell)


if __name__ == "__main__":
    atoms = bulk("AlCo", "cesiumchloride", a=2.8663, cubic=True)
    model_name = "EAM_Dynamo_VailheFarkas_1997_CoAl__MO_284963179498_005"

    atoms = bulk("Ar", "fcc", a=5.248, cubic=True)
    model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"

    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)

    test = HeatCapacityPhonon(model_name=model_name, atoms=atoms)
    test(temperature = 1.0, pressure = 1.0, timestep=0.001, number_sampling_timesteps=10, 
         repeat=(5, 5, 5))
