import os
import random
import subprocess
from typing import Iterable, Optional, Tuple, Sequence
import uuid
import numpy as np
from ase.build import bulk
from kim_python_utils.ase import CrystalGenomeTest, KIMASEError


class HeatCapacityPhonon(CrystalGenomeTest):
    def _calculate(self, structure_index: int, temperature: float, pressure: float, 
                   mass: Iterable[float], timestep: float, number_control_timesteps: int, 
                   number_sampling_timesteps: int, repeat: Tuple[int, int, int] = (3,3,3), 
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
        
        # Repeat atoms in given unit cell.
        atoms = self.atoms[structure_index]
        # TODO: Ask whether this is the correct way.
        species_of_each_atom = atoms.get_chemical_symbols()
        atoms = atoms.repeat(repeat)
        
        # Write lammps file.
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        atoms.write(structure_file, format="lammps-data")
        self._add_masses_to_structure_file(structure_file, mass)

        ####################################################
        # ACTUAL CALCULATION BEGINS 
        ####################################################
        # TODO: Possibly remove this when everything is finished.
        original_cell = atoms.get_cell() # do this instead of deepcopy
        natoms = len(atoms)
        
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
            "number_control_timesteps": number_control_timesteps,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species_of_each_atom) 
        }
        # TODO: Possibly run MPI version of Lammps if available.
        command = (
            "lammps " 
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items()) 
            + " -in npt_equilibration.lammps")
        subprocess.run(command, check=True, shell=True)

        exit()

        # Check symmetry - post-NPT
        # TODO: Fix loading txt according to created dump file.
        output = np.loadtxt('average_position.dump')
        new_species = []
        new_pos = []

        for line in output:
            new_species.append(output[i][0])
            new_pos.append([output[i,1], output[i,2], output[i,3]])

        new_atom = bulk()

        try:

            self._update_aflow_designation_from_atoms(structure_index,atoms_new)
            raise RuntimeError("Symmetry of crystal changed during NPT equilibration!")

        except KIMASEError as e:
            print("We have successfully caught an exception with the following message:")
            print(e.msg)

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
    
    @staticmethod
    def _add_masses_to_structure_file(structure_file: str, masses: Iterable[float]) -> None:
        with open(structure_file, "a") as file:
            print(file=file)
            print("Masses", file=file)
            print(file=file)
            for i, mass in enumerate(masses):
                print(f"    {i+1} {mass}", file=file)


if __name__ == "__main__":
    ####################################################
    # if called directly, do some debugging examples
    ####################################################


    # This queries for equilibrium structures in this prototype and builds atoms
    # test = BindingEnergyVsWignerSeitzRadius(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", stoichiometric_species=['Fe','P'], prototype_label='AB_oP8_62_c_c')
                    
    # Alternatively, for debugging, give it atoms object or a list of atoms objects
    atoms1 = bulk('NaCl','rocksalt',a=4.58)
    atoms2 = bulk('NaCl','cesiumchloride',a=4.58)
    model_name = "Sim_LAMMPS_EIM_Zhou_2010_BrClCsFIKLiNaRb__SM_259779394709_000"
    model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"

    atoms = bulk("Ar", "sc", a=3.6343565881252293)
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test = HeatCapacityPhonon(model_name=model_name, atoms=atoms)
    test(temperature = 298.0, pressure = 1.0, mass = atoms.get_masses(), 
         timestep=0.001, number_control_timesteps=10, number_sampling_timesteps=10,
         repeat=(5,5,5))

