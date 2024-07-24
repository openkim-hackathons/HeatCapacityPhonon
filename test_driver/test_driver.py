from math import ceil
import os
import random
import re
import subprocess
from typing import Iterable, List, Optional, Tuple
from ase import Atoms
from ase.build import bulk
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from kim_test_utils.test_driver import CrystalGenomeTestDriver
from .helper_functions import *


class HeatCapacityPhonon(CrystalGenomeTestDriver):
    def _calculate(self, temperature: float, pressure: float, temperature_offset_fraction: float,
                   timestep: float, number_sampling_timesteps: int, repeat: Tuple[int, int, int] = (3, 3, 3),
                   seed: Optional[int] = None, loose_triclinic_and_monoclinic=False, **kwargs) -> None:
        """
        Compute constant-pressure heat capacity from centered finite difference (see Section 3.2 in
        https://pubs.acs.org/doi/10.1021/jp909762j).

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
        #TDdirectory = os.path.dirname(os.path.realpath(__file__))
        #structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        structure_file = "output/zero_temperature_crystal.lmp"
        atoms_new.write(structure_file, format="lammps-data", masses=True)

        # Get random 31-bit unsigned integer.
        # TODO: Add seed to property.
        if seed is None:
            seed = random.getrandbits(31)

        # TODO: Move damping factors to argument.
        pdamp = timestep * 100.0
        tdamp = timestep * 1000.0

        # Run NPT simulation for equilibration.
        # TODO: If we notice that this takes too long, maybe use an initial temperature ramp.
        variables = {
            "modelname": self.kim_model_name,
            "temperature": temperature,
            "temperature_seed": seed,
            "temperature_damping": tdamp,
            "pressure": pressure,
            "pressure_damping": pdamp,
            "timestep": timestep,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species),
            "average_position_filename": "output/average_position_equilibration.dump.*",
            "average_cell_filename": "output/average_cell_equilibration.dump",
            "write_restart_filename": "output/final_configuration_equilibration.restart"
        }
        # TODO: Possibly run MPI version of Lammps if available.
        command = (
            "lammps "
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
            + " -log output/lammps_equilibration.log"
            + " -in npt_equilibration.lammps")
        subprocess.run(command, check=True, shell=True)

        # Analyse equilibration run.
        equilibration_time = self._extract_equilibration_step_from_logfile("output/lammps_equilibration.log")
        # Round to next multiple of 10000.
        equilibration_time = int(ceil(equilibration_time / 10000.0)) * 10000
        self._plot_property_from_lammps_log("output/lammps_equilibration.log", ("v_vol_metal", "v_temp_metal"))
        self._compute_average_positions_from_lammps_dump("output", "average_position_equilibration.dump",
                                                         "output/average_position_equilibration_over_dump.out",
                                                         skip_steps=equilibration_time)
        atoms_new.set_cell(self._get_cell(self._average_cell_over_steps("output/average_cell_equilibration.dump",
                                                                        skip_steps=equilibration_time)))
        atoms_new.set_scaled_positions(
            self._get_positions_from_lammps_dump("output/average_position_equilibration_over_dump.out"))
        reduced_atoms = self._reduce_and_avg(atoms_new, repeat)
        # AFLOW Symmetry check
        self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
            reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)

        # Run first NPT simulation at higher temperature.
        variables = {
            "modelname": self.kim_model_name,
            "temperature": (1 + temperature_offset_fraction) * temperature,
            "temperature_damping": tdamp,
            "pressure": pressure,
            "pressure_damping": pdamp,
            "timestep": timestep,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species),
            "average_position_filename": "output/average_position_high_temperature.dump.*",
            "average_cell_filename": "output/average_cell_high_temperature.dump",
            "read_restart_filename": "output/final_configuration_equilibration.restart"
        }
        # TODO: Possibly run MPI version of Lammps if available.
        command = (
            "lammps "
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
            + " -log output/lammps_high_temperature.log"
            + " -in npt_heat_capacity.lammps")
        subprocess.run(command, check=True, shell=True)

        # Analyse high-temperature NPT run.
        equilibration_time = self._extract_equilibration_step_from_logfile("output/lammps_high_temperature.log")
        # Round to next multiple of 10000.
        equilibration_time = int(ceil(equilibration_time / 10000.0)) * 10000
        self._plot_property_from_lammps_log("output/lammps_high_temperature.log",
                                            ("v_vol_metal", "v_temp_metal", "v_enthalpy_metal"))
        self._compute_average_positions_from_lammps_dump("output", "average_position_high_temperature.dump",
                                                         "output/average_position_high_temperature_over_dump.out",
                                                         skip_steps=equilibration_time)
        atoms_new.set_cell(self._get_cell(self._average_cell_over_steps("output/average_cell_high_temperature.dump",
                                                                        skip_steps=equilibration_time)))
        atoms_new.set_scaled_positions(
            self._get_positions_from_lammps_dump("output/average_position_high_temperature_over_dump.out"))
        reduced_atoms = self._reduce_and_avg(atoms_new, repeat)
        # AFLOW Symmetry check
        self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
            reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)
        
        # Run second NPT simulation at lower temperature.
        variables = {
            "modelname": self.kim_model_name,
            "temperature": (1 - temperature_offset_fraction) * temperature,
            "temperature_damping": tdamp,
            "pressure": [pressure] * 3 + [0] * 3,
            "pressure_damping": pdamp,
            "timestep": timestep,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species),
            "average_position_filename": "output/average_position_low_temperature.dump.*",
            "average_cell_filename": "output/average_cell_low_temperature.dump",
            "read_restart_filename": "output/final_configuration_equilibration.restart"
        }
        command = (
            "lammps "
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
            + " -log output/lammps_low_temperature.log"
            + " -in npt_heat_capacity.lammps")
        subprocess.run(command, check=True, shell=True)

        # Analyse low-temperature NPT run.
        equilibration_time = self._extract_equilibration_step_from_logfile("output/lammps_low_temperature.log")
        # Round to next multiple of 10000.
        equilibration_time = int(ceil(equilibration_time / 10000.0)) * 10000
        self._plot_property_from_lammps_log("output/lammps_low_temperature.log",
                                            ("v_vol_metal", "v_temp_metal", "v_enthalpy_metal"))
        self._compute_average_positions_from_lammps_dump("output", "average_position_low_temperature.dump",
                                                         "output/average_position_low_temperature_over_dump.out",
                                                         skip_steps=equilibration_time)
        atoms_new.set_cell(self._get_cell(self._average_cell_over_steps("output/average_cell_low_temperature.dump",
                                                                        skip_steps=equilibration_time)))
        atoms_new.set_scaled_positions(
            self._get_positions_from_lammps_dump("output/average_position_low_temperature_over_dump.out"))
        reduced_atoms = self._reduce_and_avg(atoms_new, repeat)
        # AFLOW Symmetry check
        self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
            reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)

        f1 = "output/lammps_high_temperature.log"
        f2 = "output/lammps_low_temperature.log"
        eps = temperature_offset_fraction * temperature
        c, c_err = self._compute_heat_capacity(f1, f2, eps, 2)

        # Print Result
        print('####################################')
        print('# NPT Phonon Heat Capacity Results #')
        print('####################################')
        print(f'C_p:\t{c}')
        print(f'C_p Error:\t{c_err}')

        # Uncertainty
        uncertainty_info = {
            "constant_pressure_heat_capacity_uncert_value": c_err
        }

        # I have to do this or KIM tries to save some coordinate file
        self.poscar = None

        # Write property
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-phonon-npt", write_stress=True, write_temp=True)  # last two default to False
        self._add_key_to_current_property_instance("constant_pressure_heat_capacity", c, "eV/Kelvin", uncertainty_info=uncertainty_info)
        self._add_key_to_current_property_instance("cell-cauchy-stress", variables['pressure'], "bars")


if __name__ == "__main__":
    model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = HeatCapacityPhonon(model_name)
    test_driver(bulk("Ar", "fcc", a=5.248), temperature=10.0, pressure=1.0, temperature_offset_fraction=0.01,
                timestep=0.001, number_sampling_timesteps=100, repeat=(7, 7, 7), loose_triclinic_and_monoclinic=False)
