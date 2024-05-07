import os
import random
import subprocess
from typing import Iterable, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.build import bulk
from kim_test_utils.test_driver import CrystalGenomeTestDriver


class HeatCapacityPhonon(CrystalGenomeTestDriver):
    def _calculate(self, temperature: float, pressure: float, timestep: float, 
                   number_sampling_timesteps: int, repeat: Tuple[int, int, int] = (3, 3, 3), 
                   seed: Optional[int] = None, loose_triclinic_and_monoclinic=False, **kwargs) -> None:
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
        
        # Copy original atoms so that their information does not get lost when the new atoms are modified.
        atoms_new = self.atoms.copy()
        
        # UNCOMMENT THIS TO TEST A TRICLINIC STRUCTURE!
        # atoms_new = bulk('Ar', 'fcc', a=5.248)

        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_data
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))
        
        # Build supercell.
        atoms_new = atoms_new.repeat(repeat)

        # Write lammps file.
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        atoms_new.write(structure_file, format="lammps-data", masses=True)
        
        # Get random 31-bit unsigned integer.
        if seed is None:
            seed = random.getrandbits(31)
            
        # TODO: Move damping factors to argument.
        pdamp = timestep * 100.0
        tdamp = timestep * 1000.0

        # Run NPT simulation.
        variables = {
            "modelname": self.kim_model_name,
            "temperature": temperature,
            "temperature_seed": seed,
            "temperature_damping": tdamp,
            "pressure": pressure,
            "pressure_damping": pdamp,
            "timestep": timestep,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species) 
        }
        # TODO: Possibly run MPI version of Lammps if available.
        command = (
            "lammps " 
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items()) 
            + " -in npt_equilibration.lammps")
        subprocess.run(command, check=True, shell=True)

        # TODO: Remove subprocess call in this function.
        self._extract_and_plot(("v_vol_metal", "v_temp_metal"))
        
        # TODO: Guanming changes this into a function call and also removes the average_position.dump.* files.
        subprocess.run("python compute_average_positions.py", check=True, shell=True)

        # Check symmetry - post-NPT
        atoms_new.set_cell(self._get_cell_from_lammps_dump("output/average_position_over_files.out"))
        atoms_new.set_scaled_positions(self._get_positions_from_lammps_dump("output/average_position_over_files.out"))

        # Reduce and average
        self._reduce_and_avg(atoms_new, repeat)

        # AFLOW Symmetry check
        self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
            atoms_new, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)
        
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
    def _reduce_and_avg(atoms, repeat):
        '''
        Function to reduce all atoms to the original unit cell position.

        @param atoms : repeated atoms object
        @param repeat : repeat tuple
        '''
        cell = atoms.get_cell()
        
        # Divide each unit vector by its number of repeats.
        # See https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element.
        cell = cell / np.array(repeat)[:, None]

        # Decrease size of cell in the atoms object.
        atoms.set_cell(cell)
        atoms.set_pbc((True, True, True))

        # Set averaging factor
        M = np.prod(repeat)

        # Wrap back the repeated atoms on top of the reference atoms in the original unit cell.
        positions = atoms.get_positions(wrap=True)
        
        number_atoms = len(atoms)
        original_number_atoms = number_atoms // M
        assert number_atoms == original_number_atoms * M
        positions_in_prim_cell = np.zeros((original_number_atoms, 3))

        # Start from end of the atoms because we will remove all atoms except the reference ones.
        for i in reversed(range(number_atoms)):
            if i >= original_number_atoms:
                # Get the distance to the reference atom in the original unit cell with the 
                # minimum image convention.
                distance = atoms.get_distance(i % original_number_atoms, i,
                                              mic=True, vector=True)
                # Get the position that has the closest distance to the reference atom in the 
                # original unit cell.
                position_i = positions[i % original_number_atoms] + distance
                # Remove atom from atoms object.
                atoms.pop()
            else:
                # Atom was part of the original unit cell.
                position_i = positions[i]
            # Average.
            positions_in_prim_cell[i % original_number_atoms] += position_i / M

        atoms.set_positions(positions_in_prim_cell)
    
    @staticmethod
    def _extract_and_plot(property_names: Iterable[str]) -> None:
        # extract data and save it as png file
        subprocess.run(f"python extract_table.py output/lammps_equilibration.log "
                       f"output/lammps_equilibration.csv {' '.join(property_names)}", check=True, shell=True)

    @staticmethod
    def _get_positions_from_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
        lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key = lambda x: x[0])     
        return [(line[1], line[2], line[3]) for line in lines]

    @staticmethod
    def _get_cell_from_lammps_dump(filename: str) -> npt.NDArray[float]:
        new_cell = np.loadtxt(filename, skiprows=5, max_rows=3)
        assert new_cell.shape == (3, 2) or new_cell.shape == (3, 3)

        # See https://docs.lammps.org/Howto_triclinic.html.
        xlo_bound = new_cell[0,0]
        xhi_bound = new_cell[0,1]
        ylo_bound = new_cell[1,0]
        yhi_bound = new_cell[1,1]
        zlo_bound = new_cell[2,0]
        zhi_bound = new_cell[2,1]

        # If not cubic add more cell params
        if new_cell.shape[-1] != 2:
            xy = new_cell[0,2]
            xz = new_cell[1,2]
            yz = new_cell[2,2]
        else:
            xy = 0.0
            xz = 0.0
            yz = 0.0
        
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo = zlo_bound
        zhi = zhi_bound
        
        cell = np.empty(shape=(3, 3))
        cell[0, :] = np.array([xhi - xlo, 0.0, 0.0])
        cell[1, :] = np.array([xy, yhi - ylo, 0.0])
        cell[2, :] = np.array([xz, yz, zhi - zlo])
        return cell


if __name__ == "__main__":
    model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = HeatCapacityPhonon(model_name)
    test_driver(bulk("Ar", "fcc", a=5.248), temperature = 10.0, pressure = 1.0, timestep=0.001, 
                number_sampling_timesteps=10, repeat=(10, 10, 10), loose_triclinic_and_monoclinic=False)
