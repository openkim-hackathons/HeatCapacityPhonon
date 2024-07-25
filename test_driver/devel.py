from concurrent.futures import as_completed, ProcessPoolExecutor
import copy
from math import ceil, sqrt
import os
import random
import re
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple
from ase import Atoms
from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data
import findiff
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize
from kim_tools.test_driver import CrystalGenomeTestDriver
from kim_tools import query_crystal_genome_structures


class HeatCapacity(CrystalGenomeTestDriver):
    def _calculate(self, temperature: float, pressure: float, temperature_step_fraction: float,
                   number_symmetric_temperature_steps: int, timestep: float, number_sampling_timesteps: int,
                   repeat: Tuple[int, int, int] = (3, 3, 3), loose_triclinic_and_monoclinic=False,
                   max_workers: Optional[int] = None, **kwargs) -> None:
        """
        Compute constant-pressure heat capacity from centered finite difference (see Section 3.2 in
        https://pubs.acs.org/doi/10.1021/jp909762j).

        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). 
            This indicates which is being used for the current calculation.

        temperature:
            Temperature in Kelvin at which the heat capacity at constant pressure is estimated. Must be strictly greater
            than zero.

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

        if not number_symmetric_temperature_steps > 0:
            raise RuntimeError("Number of symmetric temperature steps has to be bigger than zero.")

        if number_symmetric_temperature_steps * temperature_step_fraction >= 1.0:
            raise RuntimeError(
                "The given number of symmetric temperature steps and the given temperature-step fraction "
                "would yield zero or negative temperatures.")
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

        # Get temperatures that should be simulated.
        temperature_step = temperature_step_fraction * temperature
        temperatures = [temperature + i * temperature_step
                        for i in range(-number_symmetric_temperature_steps, number_symmetric_temperature_steps + 1)]
        assert len(temperatures) == 2 * number_symmetric_temperature_steps + 1
        assert all(t > 0.0 for t in temperatures)

        # Write lammps file.
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        atoms_new.write(structure_file, format="lammps-data", masses=True)

        # Handle cases where kim models expect different structure file formats.
        try:
            self._run_lammps(self.kim_model_name, 0, temperatures[0], pressure, timestep, number_sampling_timesteps,
                             species, test_file_read=True)
        except subprocess.CalledProcessError as e:
            filename = "output/lammps_temperature_0.log"
            log_file = os.path.join(TDdirectory, filename)
            wrong_format_error = self._check_lammps_log_for_wrong_structure_format(log_file)

            if wrong_format_error:
                # write the atom configuration file in the in the 'charge' format some models expect
                write_lammps_data(structure_file, atoms_new, atom_style="charge", masses=True)
                # try to read the file again, raise any exeptions that might happen
                self._run_lammps(self.kim_model_name, 0, temperatures[0], pressure, timestep, number_sampling_timesteps,
                                 species, test_file_read=True)

            else:
                raise e

        # Choose the correct accuracies file for kim-convergence based on whether the cell is orthogonal or not.
        if atoms_new.get_cell().orthorhombic:
            shutil.copyfile("accuracies_orthogonal.py", "accuracies.py")
        else:
            shutil.copyfile("accuracies_non_orthogonal.py", "accuracies.py")

        # Run Lammps simulations in parallel.
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, t in enumerate(temperatures):
                futures.append(executor.submit(
                    HeatCapacityPhonon._run_lammps, self.kim_model_name, i, t, pressure, timestep,
                    number_sampling_timesteps, species))

        # If one simulation fails, cancel all runs.
        for future in as_completed(futures):
            assert future.done()
            exception = future.exception()
            if exception is not None:
                for f in futures:
                    f.cancel()
                raise exception

        # Collect results and check that symmetry is unchanged after all simulations.
        log_filenames = []
        restart_filenames = []
        for future, t in zip(futures, temperatures):
            assert future.done()
            assert future.exception() is None
            log_filename, restart_filename, average_position_filename, average_cell_filename = future.result()
            log_filenames.append(log_filename)
            restart_filenames.append(restart_filename)
            restart_filenames.append(restart_filename)
            atoms_new.set_cell(self._get_cell_from_averaged_lammps_dump(average_cell_filename))
            atoms_new.set_scaled_positions(
                self._get_positions_from_averaged_lammps_dump(average_position_filename))
            reduced_atoms = self._reduce_and_avg(atoms_new, repeat)
            crystal_genome_designation = self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
                reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)

        c = self._compute_heat_capacity(temperatures, log_filenames, 2)

        alpha = self._compute_alpha(log_filenames, temperatures, crystal_genome_designation["prototype_label"])
        # Print result.
        print('####################################')
        print('# NPT Heat Capacity Results #')
        print('####################################')
        print(f'C_p:\t{c}')
        print('####################################')
        print('# NPT Linear Thermal Expansion Tensor Results #')
        print('####################################')
        print(f'alpha:\t{alpha}')

        # TODO: We should write some coordinate file.
        self.poscar = None

        # Write property
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-npt", write_stress=True, write_temp=True)  # last two default to False
        self._add_key_to_current_property_instance("constant_pressure_heat_capacity", c, "eV/Kelvin",
                                                   uncertainty_info=uncertainty_info)
        self._add_key_to_current_property_instance("cell-cauchy-stress", variables['pressure'], "bars")

        # Write property.
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-npt", write_stress=True, write_temp=True)  # last two default to False
        self._add_key_to_current_property_instance(
            "constant_pressure_heat_capacity", c["finite_difference_accuracy_2"][0], "eV/Kelvin",
            uncertainty_info={"constant_pressure_heat_capacity_uncert_value": c["finite_difference_accuracy_2"][1]})
        self._add_key_to_current_property_instance("pressure", [pressure] * 3 + [0.0] * 3, "bars")

        max_accuracy = len(temperatures) - 1

        alpha11 = alpha[0][0][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha11_err = alpha[0][0][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha12 = alpha[0][1][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha12_err = alpha[0][1][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha13 = alpha[0][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha13_err = alpha[0][2][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha22 = alpha[1][1][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha22_err = alpha[1][1][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha23 = alpha[1][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha23_err = alpha[1][2][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha33 = alpha[2][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha33_err = alpha[2][2][f"finite_difference_accuracy_{max_accuracy}"][1]

        # enforce tensor symmetries
        alpha21 = alpha12
        alpha31 = alpha13
        alpha32 = alpha23

        alpha21_err = alpha12_err
        alpha31_err = alpha13_err
        alpha32_err = alpha23_err

        alpha_final = np.asarray([[alpha11, alpha12, alpha13],
                                  [alpha21, alpha22, alpha23],
                                  [alpha31, alpha32, alpha33]])

        alpha_final_err = np.asarray([[alpha11_err, alpha12_err, alpha13_err],
                                      [alpha21_err, alpha22_err, alpha23_err],
                                      [alpha31_err, alpha32_err, alpha33_err]])

        self._add_property_instance_and_common_crystal_genome_keys("thermal-expansion-coefficient-npt",
                                                                   write_stress=True, write_temp=True)
        self._add_key_to_current_property_instance("alpha11", alpha11, "1/K")
        self._add_key_to_current_property_instance("alpha22", alpha22, "1/K")
        self._add_key_to_current_property_instance("alpha33", alpha33, "1/K")
        self._add_key_to_current_property_instance("alpha12", alpha12, "1/K")
        self._add_key_to_current_property_instance("alpha13", alpha13, "1/K")
        self._add_key_to_current_property_instance("alpha23", alpha23, "1/K")
        self._add_key_to_current_property_instance("thermal-expansion-coefficient", alpha_final, "1/K")

        self._add_key_to_current_property_instance("alpha11_err", alpha11_err, "1/K")
        self._add_key_to_current_property_instance("alpha22_err", alpha22_err, "1/K")
        self._add_key_to_current_property_instance("alpha33_err", alpha33_err, "1/K")
        self._add_key_to_current_property_instance("alpha12_err", alpha12_err, "1/K")
        self._add_key_to_current_property_instance("alpha13_err", alpha13_err, "1/K")
        self._add_key_to_current_property_instance("alpha23_err", alpha23_err, "1/K")
        self._add_key_to_current_property_instance("thermal-expansion-coefficient_err", alpha_final_err, "1/K")

        self._add_key_to_current_property_instance("temperature", temperature, "K")

        self.write_property_instances_to_file()

    @staticmethod
    def _run_lammps(modelname: str, temperature_index: int, temperature: float, pressure: float, timestep: float,
                    number_sampling_timesteps: int, species: List[str],
                    test_file_read=False) -> Tuple[str, str, str, str]:
        # Get random 31-bit unsigned integer.
        seed = random.getrandbits(31)

        pdamp = timestep * 100.0
        tdamp = timestep * 1000.0

        log_filename = f"output/lammps_temperature_{temperature_index}.log"
        restart_filename = f"output/final_configuration_temperature_{temperature_index}.restart"
        variables = {
            "modelname": modelname,
            "temperature": temperature,
            "temperature_seed": seed,
            "temperature_damping": tdamp,
            "pressure": pressure,
            "pressure_damping": pdamp,
            "timestep": timestep,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species),
            "average_position_filename": f"output/average_position_temperature_{temperature_index}.dump.*",
            "average_cell_filename": f"output/average_cell_temperature_{temperature_index}.dump",
            "write_restart_filename": restart_filename
        }
        if test_file_read:
            # do a minimal test to see if the model can read the structure file
            command = (
                    "lammps "
                    + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
                    + f" -log {log_filename}"
                    + " -in file_read_test.lammps")
            subprocess.run(command, check=True, shell=True)
        else:
            command = (
                    "lammps "
                    + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
                    + f" -log {log_filename}"
                    + " -in npt.lammps")

            subprocess.run(command, check=True, shell=True)

            HeatCapacityPhonon._plot_property_from_lammps_log(
                log_filename, ("v_vol_metal", "v_temp_metal", "v_enthalpy_metal"))

            equilibration_time = HeatCapacityPhonon._extract_equilibration_step_from_logfile(log_filename)
            # Round to next multiple of 10000.
            equilibration_time = int(ceil(equilibration_time / 10000.0)) * 10000

            full_average_position_file = f"output/average_position_temperature_{temperature_index}.dump.full"
            HeatCapacityPhonon._compute_average_positions_from_lammps_dump(
                "output", f"average_position_temperature_{temperature_index}.dump",
                full_average_position_file, equilibration_time)

            full_average_cell_file = f"output/average_cell_temperature_{temperature_index}.dump.full"
            HeatCapacityPhonon._compute_average_cell_from_lammps_dump(
                f"output/average_cell_temperature_{temperature_index}.dump", full_average_cell_file,
                equilibration_time)

            return log_filename, restart_filename, full_average_position_file, full_average_cell_file

    @staticmethod
    def _reduce_and_avg(atoms: Atoms, repeat: Tuple[int, int, int]) -> Atoms:
        '''
        Function to reduce all atoms to the original unit cell position.
        '''
        new_atoms = atoms.copy()

        cell = new_atoms.get_cell()

        # Divide each unit vector by its number of repeats.
        # See https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element.
        cell = cell / np.array(repeat)[:, None]

        # Decrease size of cell in the atoms object.
        new_atoms.set_cell(cell)
        new_atoms.set_pbc((True, True, True))

        # Set averaging factor
        M = np.prod(repeat)

        # Wrap back the repeated atoms on top of the reference atoms in the original unit cell.
        positions = new_atoms.get_positions(wrap=True)

        number_atoms = len(new_atoms)
        original_number_atoms = number_atoms // M
        assert number_atoms == original_number_atoms * M
        positions_in_prim_cell = np.zeros((original_number_atoms, 3))

        # Start from end of the atoms because we will remove all atoms except the reference ones.
        for i in reversed(range(number_atoms)):
            if i >= original_number_atoms:
                # Get the distance to the reference atom in the original unit cell with the
                # minimum image convention.
                distance = new_atoms.get_distance(i % original_number_atoms, i,
                                                  mic=True, vector=True)
                # Get the position that has the closest distance to the reference atom in the
                # original unit cell.
                position_i = positions[i % original_number_atoms] + distance
                # Remove atom from atoms object.
                new_atoms.pop()
            else:
                # Atom was part of the original unit cell.
                position_i = positions[i]
            # Average.
            positions_in_prim_cell[i % original_number_atoms] += position_i / M

        new_atoms.set_positions(positions_in_prim_cell)

        return new_atoms

    @staticmethod
    def _plot_property_from_lammps_log(in_file_path: str, property_names: Iterable[str]) -> None:
        '''
        The function to get the value of the property with time from ***.log 
        the extracted data are stored as ***.csv and ploted as property_name.png
        data_dir --- the directory contains lammps_equilibration.log 
        property_names --- the list of properties
        '''

        def get_table(in_file):
            if not os.path.isfile(in_file):
                raise FileNotFoundError(in_file + " not found")
            elif not ".log" in in_file:
                raise FileNotFoundError("The file is not a *.log file")
            is_first_header = True
            header_flags = ["Step", "v_pe_metal", "v_temp_metal", "v_press_metal"]
            eot_flags = ["Loop", "time", "on", "procs", "for", "steps"]
            table = []
            with open(in_file, "r") as f:
                line = f.readline()
                while line:  # Not EOF.
                    is_header = True
                    for _s in header_flags:
                        is_header = is_header and (_s in line)
                    if is_header:
                        if is_first_header:
                            table.append(line)
                            is_first_header = False
                        content = f.readline()
                        while content:
                            is_eot = True
                            for _s in eot_flags:
                                is_eot = is_eot and (_s in content)
                            if not is_eot:
                                table.append(content)
                            else:
                                break
                            content = f.readline()
                    line = f.readline()
            return table

        def write_table(table, out_file):
            with open(out_file, "w") as f:
                for l in table:
                    f.writelines(l)

        dir_name = os.path.dirname(in_file_path)
        in_file_name = os.path.basename(in_file_path)
        out_file_path = os.path.join(dir_name, in_file_name.replace(".log", ".csv"))

        table = get_table(in_file_path)
        write_table(table, out_file_path)
        df = np.loadtxt(out_file_path, skiprows=1)

        for property_name in property_names:
            with open(out_file_path) as file:
                first_line = file.readline().strip("\n")
            property_index = first_line.split().index(property_name)
            properties = df[:, property_index]
            step = df[:, 0]
            plt.plot(step, properties)
            plt.xlabel("step")
            plt.ylabel(property_name)
            img_file = os.path.join(dir_name, in_file_name.replace(".log", "_") + property_name + ".png")
            plt.savefig(img_file, bbox_inches="tight")
            plt.close()

    @staticmethod
    def _compute_average_positions_from_lammps_dump(data_dir: str, file_str: str, output_filename: str,
                                                    skip_steps: int) -> None:
        '''
        This function compute the average position over *.dump files which contains the file_str in data_dir and output it
        to data_dir/[file_str]_over_dump.out

        input:
        data_dir -- the directory contains all the data e.g average_position.dump.* files
        file_str -- the files whose names contain the file_str are considered
        output_filename -- the name of the output file
        skip_steps -- dump files with steps <= skip_steps are ignored
        '''

        def get_id_pos_dict(file_name):
            '''
            input: 
            file_name--the file_name that contains average postion data
            output:
            the dictionary contains id:position pairs e.g {1:array([x1,y1,z1]),2:array([x2,y2,z2])}
            for the averaged positions over files
            '''
            id_pos_dict = {}
            header4N = ["NUMBER OF ATOMS"]
            header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
            is_table_started = False
            is_natom_read = False
            with open(file_name, "r") as f:
                line = f.readline()
                count_content_line = 0
                N = 0
                while line:
                    if not is_natom_read:
                        is_natom_read = np.all([flag in line for flag in header4N])
                        if is_natom_read:
                            line = f.readline()
                            N = int(line)
                    if not is_table_started:
                        contain_flags = np.all([flag in line for flag in header4pos])
                        is_table_started = contain_flags
                    else:
                        count_content_line += 1
                        words = line.split()
                        id = int(words[0])
                        pos = np.array([float(words[1]), float(words[2]), float(words[3])])
                        id_pos_dict[id] = pos
                    if count_content_line > 0 and count_content_line >= N:
                        break
                    line = f.readline()
            if count_content_line < N:
                print("The file " + file_name +
                      " is not complete, the number of atoms is smaller than " + str(N))
            return id_pos_dict

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(data_dir + " does not exist")
        if not ".dump" in file_str:
            raise ValueError("file_str must be a string containing .dump")

        # Extract and store all the data.
        pos_list = []
        max_step, last_step_file = -1, ""
        for file_name in os.listdir(data_dir):
            if file_str in file_name:
                step = int(re.findall(r'\d+', file_name)[-1])
                if step <= skip_steps:
                    continue
                file_path = os.path.join(data_dir, file_name)
                id_pos_dict = get_id_pos_dict(file_path)
                id_pos = sorted(id_pos_dict.items())
                id_list = [pair[0] for pair in id_pos]
                pos_list.append([pair[1] for pair in id_pos])
                # Check if this is the last step.
                if step > max_step:
                    last_step_file, max_step = os.path.join(data_dir, file_name), step
        if max_step == -1 and last_step_file == "":
            raise RuntimeError("Found no files to average over.")
        pos_arr = np.array(pos_list)
        avg_pos = np.mean(pos_arr, axis=0)
        # Get the lines above the table from the file of the last step.
        with open(last_step_file, "r") as f:
            header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
            line = f.readline()
            description_str = ""
            is_table_started = False
            while line:
                description_str += line
                is_table_started = np.all([flag in line for flag in header4pos])
                if is_table_started:
                    break
                else:
                    line = f.readline()
        # Write the output to the file.
        with open(output_filename, "w") as f:
            f.write(description_str)
            for i in range(len(id_list)):
                f.write(str(id_list[i]))
                f.write("  ")
                for dim in range(3):
                    f.write('{:3.6}'.format(avg_pos[i, dim]))
                    f.write("  ")
                f.write("\n")

    @staticmethod
    def _compute_average_cell_from_lammps_dump(input_file: str, output_file: str, skip_steps: int) -> None:
        with open(input_file, "r") as f:
            f.readline()  # Skip the first line.
            header = f.readline()
            header = header.replace("#", "")
        property_names = header.split()
        data = np.loadtxt(input_file, skiprows=2)
        time_step_index = property_names.index("TimeStep")
        time_step_data = data[:, time_step_index]
        cutoff_index = np.argmax(time_step_data > skip_steps)
        assert time_step_data[cutoff_index] > skip_steps
        assert cutoff_index == 0 or time_step_data[cutoff_index - 1] <= skip_steps
        mean_data = data[cutoff_index:].mean(axis=0).tolist()
        with open(output_file, "w") as f:
            print("# Full time-averaged data for cell information", file=f)
            print(f"# {' '.join(name for name in property_names if name != 'TimeStep')}", file=f)
            print(" ".join(str(mean_data[i]) for i, name in enumerate(property_names) if name != "TimeStep"), file=f)

    @staticmethod
    def _get_positions_from_averaged_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
        lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key=lambda x: x[0])
        return [(line[1], line[2], line[3]) for line in lines]

    @staticmethod
    def _get_cell_from_averaged_lammps_dump(filename: str) -> npt.NDArray[np.float64]:
        cell_list = np.loadtxt(filename, comments='#')
        assert len(cell_list) == 6
        cell = np.empty(shape=(3, 3))
        cell[0, :] = np.array([cell_list[0], 0.0, 0.0])
        cell[1, :] = np.array([cell_list[3], cell_list[1], 0.0])
        cell[2, :] = np.array([cell_list[4], cell_list[5], cell_list[2]])
        return cell

    @staticmethod
    def _compute_heat_capacity(temperatures: List[float], log_filenames: List[str],
                               quantity_index: int) -> Dict[str, Tuple[float, float]]:
        enthalpy_means = []
        enthalpy_errs = []
        for log_filename in log_filenames:
            enthalpy_mean, enthalpy_conf = HeatCapacityPhonon._extract_mean_error_from_logfile(
                log_filename, quantity_index)
            enthalpy_means.append(enthalpy_mean)
            # Correct 95% confidence interval to standard error.
            enthalpy_errs.append(enthalpy_conf / 1.96)

        # Use finite differences to estimate derivative.
        temperature_step = temperatures[1] - temperatures[0]
        assert all(abs(temperatures[i + 1] - temperatures[i] - temperature_step)
                   < 1.0e-12 for i in range(len(temperatures) - 1))
        assert len(temperatures) >= 3
        max_accuracy = len(temperatures) - 1
        heat_capacity = {}
        for accuracy in range(2, max_accuracy + 1, 2):
            heat_capacity[
                f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                temperature_step, enthalpy_means, enthalpy_errs, accuracy)

        # Use linear fit to estimate derivative.
        heat_capacity["fit"] = HeatCapacityPhonon._get_slope_and_error(
            temperatures, enthalpy_means, enthalpy_errs)

        return heat_capacity

    ###############################################################################
    #
    # _compute_alpha
    #
    # compute the linear thermal expansion tensor from the simulation results
    #
    # Arguments:
    # cell_list: list of N list of pairs of list with 6 floats each,
    #            the first corresponding to cell sizes and angles,
    #            and the second their errors,
    #            where N is the number of temperatures sampled
    # temperatures: temperature list of length N
    # prototype_label: aflow crystal structure designation
    #
    ###############################################################################
    @staticmethod
    def _compute_alpha(log_filenames: List[str], temperatures: List[float], prototype_label: str):

        lx = []
        ly = []
        lz = []
        xy = []
        xz = []
        yz = []

        lx_errs = []
        ly_errs = []
        lz_errs = []
        xy_errs = []
        xz_errs = []
        yz_errs = []

        # pull out the space group from the prototype label
        # to use to determine the category of crystal symmetry
        space_group = int(prototype_label.split("_")[2])

        # must match the order of kim_convergence variables in npt.lammps
        convergence_indicies = {"lx": 3, "ly": 4, "lz": 5, "xy": 6, "xz": 7, "yz": 8}

        # get all of the box parameters and their uncertianties
        for filename in log_filenames:
            lxi, lx_err = HeatCapacityPhonon._extract_mean_error_from_logfile(filename, convergence_indicies["lx"])
            lyi, ly_err = HeatCapacityPhonon._extract_mean_error_from_logfile(filename, convergence_indicies["ly"])
            lzi, lz_err = HeatCapacityPhonon._extract_mean_error_from_logfile(filename, convergence_indicies["lz"])
            xyi, xy_err = HeatCapacityPhonon._extract_mean_error_from_logfile(filename, convergence_indicies["xy"])
            xzi, xz_err = HeatCapacityPhonon._extract_mean_error_from_logfile(filename, convergence_indicies["xz"])
            yzi, yz_err = HeatCapacityPhonon._extract_mean_error_from_logfile(filename, convergence_indicies["yz"])

            lx.append(lxi)
            ly.append(lyi)
            lz.append(lzi)
            xy.append(xyi)
            xz.append(xzi)
            yz.append(yzi)

            # Correct 95% confidence interval to standard error.
            lx_errs.append(lx_err / 1.96)
            ly_errs.append(ly_err / 1.96)
            lz_errs.append(lz_err / 1.96)
            xy_errs.append(xy_err / 1.96)
            xz_errs.append(xz_err / 1.96)
            yz_errs.append(yz_err / 1.96)

        lx = np.asarray(lx)
        ly = np.asarray(ly)
        lz = np.asarray(lz)
        xy = np.asarray(xy)
        xz = np.asarray(xz)
        yz = np.asarray(yz)

        lx_errs = np.asarray(lx_errs)
        ly_errs = np.asarray(ly_errs)
        lz_errs = np.asarray(lz_errs)
        xy_errs = np.asarray(xy_errs)
        xz_errs = np.asarray(xz_errs)
        yz_errs = np.asarray(yz_errs)

        # transform lammps cell parameters to lengths and angles
        # https://docs.lammps.org/Howto_triclinic.html#crystallographic-general-triclinic-representation-of-a-simulation-box
        a = lx
        b = np.sqrt(ly ** 2 + xy ** 2)
        c = np.sqrt(lz ** 2 + xz ** 2 + yz ** 2)
        alpha_angle = np.degrees(np.arccos((xy * xz + ly * yz) / (b * c)))
        beta_angle = np.degrees(np.arccos(xz / c))
        gamma_angle = np.degrees(np.arccos(xy / b))

        # uncertainty propagation for lattice parameter transformations
        a_errs = lx_errs
        b_errs = np.sqrt((ly / np.sqrt(ly ** 2 + xy ** 2)) ** 2 * ly_errs ** 2 + (
                    xz / np.sqrt(ly ** 2 + xy ** 2)) ** 2 * xz_errs ** 2)
        c_err_denom_squared = lz ** 2 + xz ** 2 + yz ** 2
        c_errs = np.sqrt(
            (lz ** 2 / c_err_denom_squared) * lz_errs ** 2 + (xz ** 2 / c_err_denom_squared) * xz_errs ** 2 +
            (yz ** 2 / c_err_denom_squared) * yz_errs ** 2)

        alpha_angle_denom = b * c * np.sqrt(1 - ((xy * xz + ly * lz) / (b * c) ** 2))

        alpha_angle_errs = np.sqrt(
            (xy / alpha_angle_denom) ** 2 * xy_errs ** 2 + (xz / alpha_angle_denom) ** 2 * xz_errs ** 2 +
            (ly / alpha_angle_denom) ** 2 * ly_errs ** 2 + (lz / alpha_angle_denom) ** 2 * lz_errs ** 2 +
            ((xy * xz + ly * lz) / b * alpha_angle_denom) ** 2 * b_errs ** 2 +
            ((xy * xz + ly * lz) / c * alpha_angle_denom) ** 2 * c_errs ** 2)

        beta_angle_denom = c * np.sqrt(1 - (xz / c) ** 2)
        beta_angle_errs = np.sqrt(beta_angle_denom ** 2 * xz_errs ** 2 + (xz / c * beta_angle_denom) ** 2 * c_errs ** 2)
        gamma_angle_denom = b * np.sqrt(1 - (xy / b) ** 2)
        gamma_angle_errs = np.sqrt(
            gamma_angle_denom ** 2 * xy_errs ** 2 + (xy / b * gamma_angle_denom) ** 2 * b_errs ** 2)

        # needed for all space groups
        # Use finite differences to estimate derivative.
        temperature_step = temperatures[1] - temperatures[0]
        assert all(abs(temperatures[i + 1] - temperatures[i] - temperature_step)
                   < 1.0e-12 for i in range(len(temperatures) - 1))
        assert len(temperatures) >= 3
        max_accuracy = len(temperatures) - 1
        aslope = {}
        for accuracy in range(2, max_accuracy + 1, 2):
            aslope[
                f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                temperature_step, a, a_errs, accuracy)

        # create entries of the same format
        # for zero-valued tensor components
        zero = {}
        for accuracy in range(2, max_accuracy + 1, 2):
            zero[f"finite_difference_accuracy_{accuracy}"] = (0.0, 0.0)

        # extract values of cell parameters at target temperature
        aval = a[int(len(a) / 2)]
        bval = b[int(len(b) / 2)]
        cval = c[int(len(c) / 2)]
        alphaval = alpha_angle[int(len(a) / 2)]
        betaval = beta_angle[int(len(a) / 2)]
        gammaval = gamma_angle[int(len(a) / 2)]

        aval_err = a_errs[int(len(a) / 2)]
        bval_err = b_errs[int(len(a) / 2)]
        cval_err = c_errs[int(len(a) / 2)]
        alphaval_err = alpha_angle_errs[int(len(a) / 2)]
        betaval_err = beta_angle_errs[int(len(a) / 2)]
        gammaval_err = gamma_angle_errs[int(len(a) / 2)]

        # if the space group is cubic, only compute alpha11
        if space_group >= 195:
            alpha11 = copy.deepcopy(zero)
            for accuracy in range(2, max_accuracy + 1, 2):
                alpha11[f"finite_difference_accuracy_{accuracy}"] = aslope[
                                                                        f"finite_difference_accuracy_{accuracy}"] / aval

                # uncertainty propagation for thermal expansion tensor calculation
                aslope_val = aslope[f"finite_difference_accuracy_{accuracy}"][0]
                aslope_err = aslope[f"finite_difference_accuracy_{accuracy}"][1]
                alpha11_err = np.sqrt((aslope_val / aval ** 2) ** 2 * aval_err ** 2 + (1 / aval) ** 2 * aslope_err ** 2)
                alpha11[f"finite_difference_accuracy_{accuracy}"][1] = alpha11_err

            alpha22 = alpha11
            alpha33 = alpha11

            alpha12 = copy.deepcopy(zero)
            alpha13 = copy.deepcopy(zero)
            alpha23 = copy.deepcopy(zero)

        # hexagona, trigonal, tetragonal space groups also compute alpha33
        elif space_group >= 75 and space_group <= 194:

            cslope = {}
            for accuracy in range(2, max_accuracy + 1, 2):
                cslope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, c, c_errs, accuracy)

            alpha11 = copy.deepcopy(zero)
            alpha33 = copy.deepcopy(zero)
            for accuracy in range(2, max_accuracy + 1, 2):
                alpha11[f"finite_difference_accuracy_{accuracy}"] = aslope[
                                                                        f"finite_difference_accuracy_{accuracy}"] / aval
                alpha33[f"finite_difference_accuracy_{accuracy}"] = cslope[
                                                                        f"finite_difference_accuracy_{accuracy}"] / cval

                # uncertainty propagation for thermal expansion tensor calculation
                aslope_val = aslope[f"finite_difference_accuracy_{accuracy}"][0]
                aslope_err = aslope[f"finite_difference_accuracy_{accuracy}"][1]
                alpha11_err = np.sqrt((aslope_val / aval ** 2) ** 2 * aval_err ** 2 + (1 / aval) ** 2 * aslope_err ** 2)
                alpha11[f"finite_difference_accuracy_{accuracy}"][1] = alpha11_err

                cslope_val = cslope[f"finite_difference_accuracy_{accuracy}"][0]
                cslope_err = cslope[f"finite_difference_accuracy_{accuracy}"][1]
                alpha33_err = np.sqrt((cslope_val / cval ** 2) ** 2 * cval_err ** 2 + (1 / cval) ** 2 * cslope_err ** 2)
                alpha33[f"finite_difference_accuracy_{accuracy}"][1] = alpha33_err

            alpha22 = alpha33

            alpha12 = copy.deepcopy(zero)
            alpha13 = copy.deepcopy(zero)
            alpha23 = copy.deepcopy(zero)

        # orthorhombic, also compute alpha22
        elif space_group >= 16 and space_group <= 74:
            bslope = {}
            cslope = {}
            for accuracy in range(2, max_accuracy + 1, 2):
                bslope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, b, b_errs, accuracy)
                cslope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, c, c_errs, accuracy)

            alpha11 = copy.deepcopy(zero)
            alpha22 = copy.deepcopy(zero)
            alpha33 = copy.deepcopy(zero)
            for accuracy in range(2, max_accuracy + 1, 2):
                alpha11[f"finite_difference_accuracy_{accuracy}"] = aslope[
                                                                        f"finite_difference_accuracy_{accuracy}"] / aval
                alpha22[f"finite_difference_accuracy_{accuracy}"] = bslope[
                                                                        f"finite_difference_accuracy_{accuracy}"] / bval
                alpha33[f"finite_difference_accuracy_{accuracy}"] = cslope[
                                                                        f"finite_difference_accuracy_{accuracy}"] / cval

                # uncertainty propagation for thermal expansion tensor calculation
                aslope_val = aslope[f"finite_difference_accuracy_{accuracy}"][0]
                aslope_err = aslope[f"finite_difference_accuracy_{accuracy}"][1]
                alpha11_err = np.sqrt((aslope_val / aval ** 2) ** 2 * aval_err ** 2 + (1 / aval) ** 2 * aslope_err ** 2)
                alpha11[f"finite_difference_accuracy_{accuracy}"][1] = alpha11_err

                bslope_val = bslope[f"finite_difference_accuracy_{accuracy}"][0]
                bslope_err = bslope[f"finite_difference_accuracy_{accuracy}"][1]
                alpha22_err = np.sqrt((bslope_val / bval ** 2) ** 2 * bval_err ** 2 + (1 / bval) ** 2 * bslope_err ** 2)
                alpha22[f"finite_difference_accuracy_{accuracy}"][1] = alpha22_err

                cslope_val = cslope[f"finite_difference_accuracy_{accuracy}"][0]
                cslope_err = cslope[f"finite_difference_accuracy_{accuracy}"][1]
                alpha33_err = np.sqrt((cslope_val / cval ** 2) ** 2 * cval_err ** 2 + (1 / cval) ** 2 * cslope_err ** 2)
                alpha33[f"finite_difference_accuracy_{accuracy}"][1] = alpha33_err

            alpha12 = copy.deepcopy(zero)
            alpha13 = copy.deepcopy(zero)
            alpha23 = copy.deepcopy(zero)

        # monoclinic or triclinic
        elif space_group <= 15:
            bslope = {}
            cslope = {}
            alpha_angle_slope = {}
            beta_angle_slope = {}
            gamma_angle_slope = {}
            gamma_star_prime = {}

            # calculating reciprocal lattice angle gamma_star
            gamma_star_array = np.arccos((np.cos(np.radians(alpha_angle)) * np.cos(np.radians(beta_angle)) -
                                          np.cos(np.radians(gamma_angle))) / (
                                                     np.sin(np.radians(alpha_angle)) * np.sin(np.radians(beta_angle))))

            # calculate error propagation for reciprocal lattice angle gamma_star
            gamma_star_errs_denom = 1 - (np.cos(np.radians(alpha_angle)) * np.cos(np.radians(beta_angle)) -
                                         (1 / (np.sin(np.radians(alpha_angle)))) * (
                                                     1 / np.cos(np.radians(beta_angle))) * np.cos(
                        np.radians(gamma_angle))) ** 2

            gamma_star_errs = np.sqrt(((1 / np.tan(np.radians(alpha_angle))) * (1 / np.sin(np.radians(alpha_angle))) * (
                        1 / np.cos(np.radians(beta_angle))) * np.cos(gamma_angle)
                                       - np.sin(np.radians(alpha_angle)) * np.cos(
                        np.radians(beta_angle))) ** 2 / gamma_star_errs_denom * alpha_angle_errs ** 2 +
                                      ((1 / np.sin(np.radians(alpha_angle))) * np.tan(np.radians(beta_angle)) * (
                                                  1 / np.cos(np.radians(beta_angle))) * np.cos(
                                          np.radians(gamma_angle)) +
                                       np.cos(np.radians(alpha_angle)) * np.cos(np.radians(
                                                  beta_angle))) ** 2 / gamma_star_errs_denom * beta_angle_errs ** 2 +
                                      ((1 / np.sin(np.radians(alpha_angle))) * (
                                                  1 / np.cos(np.radians(beta_angle))) * np.sin(np.radians(
                                          gamma_angle))) ** 2 / gamma_star_errs_denom * gamma_angle_errs ** 2)

            # pull out central value/error at target temperature
            gamma_star = gamma_star_array[int(len(gamma_star_array) / 2)]
            gamma_star_err = gamma_star_errs[int(len(gamma_star_array) / 2)]

            alpha11 = copy.deepcopy(zero)
            alpha22 = copy.deepcopy(zero)
            alpha33 = copy.deepcopy(zero)
            alpha12 = copy.deepcopy(zero)
            alpha13 = copy.deepcopy(zero)
            alpha23 = copy.deepcopy(zero)

            for accuracy in range(2, max_accuracy + 1, 2):
                # calculate temperature derivatives
                bslope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, b, b_errs, accuracy)
                cslope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, c, c_errs, accuracy)
                alpha_angle_slope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, alpha_angle, alpha_angle_errs, accuracy)
                beta_angle_slope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, beta_angle, beta_angle_errs, accuracy)
                gamma_angle_slope[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, gamma_angle, gamma_angle_errs, accuracy)
                # temperature derivative of gamma_star, only needed for triclinic structures
                gamma_star_prime[
                    f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                    temperature_step, gamma_star_array, gamma_star_errs, accuracy)

            for accuracy in range(2, max_accuracy + 1, 2):
                # parse out values and associated uncertianties
                aslope_val = aslope[f"finite_difference_accuracy_{accuracy}"][0]
                aslope_err = aslope[f"finite_difference_accuracy_{accuracy}"][1]
                bslope_val = bslope[f"finite_difference_accuracy_{accuracy}"][0]
                bslope_err = bslope[f"finite_difference_accuracy_{accuracy}"][1]
                cslope_val = cslope[f"finite_difference_accuracy_{accuracy}"][0]
                cslope_err = cslope[f"finite_difference_accuracy_{accuracy}"][1]
                alpha_angle_slope_val = alpha_angle_slope[f"finite_difference_accuracy_{accuracy}"][0]
                alpha_angle_slope_err = alpha_angle_slope[f"finite_difference_accuracy_{accuracy}"][1]
                beta_angle_slope_val = beta_angle_slope[f"finite_difference_accuracy_{accuracy}"][0]
                beta_angle_slope_err = beta_angle_slope[f"finite_difference_accuracy_{accuracy}"][1]
                gamma_angle_slope_val = gamma_angle_slope[f"finite_difference_accuracy_{accuracy}"][0]
                gamma_angle_slope_err = gamma_angle_slope[f"finite_difference_accuracy_{accuracy}"][1]
                gamma_star_slope_val = gamma_star_prime[f"finite_difference_accuracy_{accuracy}"][0]
                gamma_star_slope_err = gamma_star_prime[f"finite_difference_accuracy_{accuracy}"][1]
                # monoclinic
                if space_group > 2:

                    alpha11[f"finite_difference_accuracy_{accuracy}"] = (1 / aval) * aslope_val
                    alpha22[f"finite_difference_accuracy_{accuracy}"] = ((1 / bval) * bslope_val +
                                                                         gamma_angle_slope_val *
                                                                         (1 / np.tan(np.radians(gammaval))))

                    alpha33[f"finite_difference_accuracy_{accuracy}"] = (1 / cval) * cslope_val
                    alpha12[f"finite_difference_accuracy_{accuracy}"] = ((-1 / 2) * ((1 / aval) * aslope_val -
                                                                                     (1 / bval) * bslope_val) * (
                                                                                     1 / np.tan(np.radians(gammaval))) -
                                                                         (1 / 2) * gamma_angle_slope_val)

                    alpha13[f"finite_difference_accuracy_{accuracy}"] = (-1 / 2) * beta_angle_slope_val
                    alpha23[f"finite_difference_accuracy_{accuracy}"] = (
                                (1 / 2) * ((-1 / np.sin(np.radians(gammaval))) * alpha_angle_slope_val +
                                           beta_angle_slope_val * (1 / np.tan(np.radians(gammaval)))))

                    # uncertainty propagation for thermal expansion tensor calculation
                    alpha11_err = np.sqrt(
                        (aslope_val / aval ** 2) ** 2 * aval_err ** 2 + (1 / aval) ** 2 * aslope_err ** 2)
                    alpha11[f"finite_difference_accuracy_{accuracy}"][1] = alpha11_err

                    alpha22_err = np.sqrt(
                        (bslope_val / bval ** 2) ** 2 * bval_err ** 2 + (1 / bval) ** 2 * bslope_err ** 2 +
                        ((1 / np.tan(gammaval) * gamma_angle_slope_err) ** 2 -
                         (gamma_angle_slope_val * 1 / np.sin(gammaval) * gammaval_err) ** 2))
                    alpha22[f"finite_difference_accuracy_{accuracy}"][1] = alpha22_err

                    alpha33_err = np.sqrt(
                        (cslope_val / cval ** 2) ** 2 * cval_err ** 2 + (1 / cval) ** 2 * cslope_err ** 2)
                    alpha33[f"finite_difference_accuracy_{accuracy}"][1] = alpha33_err

                    alpha12_err = np.sqrt((aslope_val * 1 / np.tan(gammaval) / (2 * aval ** 2)) ** 2 * aval_err ** 2 +
                                          (bslope_val * 1 / np.tan(gammaval) / (2 * bval ** 2)) ** 2 * bval_err ** 2 +
                                          ((1 / 2) * (aslope_val / aval - bslope_val / bval) * (
                                                      1 / np.sin(gammaval)) ** 2) ** 2 * gammaval_err ** 2 +
                                          (1 / (2 * aval * np.tan(gammaval))) ** 2 * aslope_err ** 2 +
                                          (1 / (2 * bval * np.tan(gammaval))) ** 2 * bslope_err ** 2 +
                                          (1 / 4) * gamma_angle_slope_err ** 2)
                    alpha12[f"finite_difference_accuracy_{accuracy}"][1] = alpha12_err

                    alpha13_err = np.sqrt((1 / 4) * beta_angle_slope_err ** 2)
                    alpha13[f"finite_difference_accuracy_{accuracy}"][1] = alpha13_err

                    alpha23_err = np.sqrt((1 / (2 * np.sin(gammaval))) ** 2 * alpha_angle_slope_err ** 2 +
                                          (1 / (2 * np.tan(gammaval))) ** 2 * beta_angle_slope_err ** 2 +
                                          ((1 / 2) * (1 / np.sin(gammaval)) * (
                                                      alpha_angle_slope_val * (1 / np.tan(gammaval)) -
                                                      beta_angle_slope_val * (
                                                                  1 / np.sin(gammaval)))) ** 2 * gammaval_err ** 2)
                    alpha23[f"finite_difference_accuracy_{accuracy}"][1] = alpha23_err

                # triclinic
                elif space_group <= 2:

                    alpha11[f"finite_difference_accuracy_{accuracy}"] = ((1 / aval) * aslope_val +
                                                                         beta_angle_slope_val *
                                                                         (1 / np.tan(np.radians(betaval))))
                    alpha22[f"finite_difference_accuracy_{accuracy}"] = ((1 / bval) * bslope_val +
                                                                         alpha_angle_slope_val *
                                                                         (1 / np.tan(np.radians(
                                                                             alphaval))) + gamma_star_slope_val *
                                                                         (1 / np.tan(gamma_star)))
                    alpha33[f"finite_difference_accuracy_{accuracy}"] = ((1 / cval) * cslope_val)
                    alpha12[f"finite_difference_accuracy_{accuracy}"] = (
                                (1 / 2) * (1 / np.tan(gamma_star)) * ((1 / aval) * aslope_val -
                                                                      (1 / bval) * bslope_val -
                                                                      alpha_angle_slope_val * (1 / np.tan(alphaval)) +
                                                                      beta_angle_slope_val * (1 / np.tan(betaval))) + (
                                            1 / 2) *
                                gamma_star_slope_val)
                    alpha13[f"finite_difference_accuracy_{accuracy}"] = ((1 / 2) * ((1 / aval) * aslope_val -
                                                                                    (1 / cval) * cslope_val) * (
                                                                                     1 / np.tan(betaval)) -
                                                                         (1 / 2) * beta_angle_slope_val)
                    alpha23[f"finite_difference_accuracy_{accuracy}"] = ((1 / 2) * (((1 / aval) * aslope_val -
                                                                                     (1 / cval) * cslope_val) * (
                                                                                                1 / np.tan(
                                                                                            gamma_star)) *
                                                                                    (1 / np.tan(betaval)) + ((
                                                                                                                         1 / bval) * bslope_val -
                                                                                                             (
                                                                                                                         1 / cval) * cslope_val) *
                                                                                    (1 / (np.tan(alphaval) * np.sin(
                                                                                        gamma_star))) - ((1 / np.sin(
                                gamma_star)) *
                                                                                                         alpha_angle_slope_val +
                                                                                                         beta_angle_slope_val * (
                                                                                                                     1 / np.tan(
                                                                                                                 gamma_star)))))

                    # uncertainty propagation for thermal expansion tensor calculation
                    alpha11_err = np.sqrt((aslope_val / aval ** 2) ** 2 * aval_err ** 2 +
                                          (1 / aval) ** 2 * aslope_err + (beta_angle_slope * (
                                1 / np.sin(np.radians(betaval))) ** 2) ** 2 * betaval_err ** 2 +
                                          (1 / np.tan(np.radians(betaval))) ** 2 * beta_angle_slope_err ** 2)
                    alpha11[f"finite_difference_accuracy_{accuracy}"][1] = alpha11_err

                    alpha22_err = np.sqrt((bslope_val / bval ** 2) ** 2 * bval_err ** 2 +
                                          (1 / bval) ** 2 * bslope_err ** 2 +
                                          (alpha_angle_slope_val * (
                                                      1 / np.sin(np.radians(alphaval))) ** 2) ** 2 * alphaval_err ** 2 +
                                          (1 / np.tan(np.radians(alphaval))) ** 2 * alpha_angle_slope_err ** 2 +
                                          (gamma_star_prime * (1 / np.sin(
                                              np.radians(gamma_star))) ** 2) ** 2 * gamma_star_err ** 2 +
                                          (1 / np.tan(np.radians(gamma_star))) ** 2 * gamma_star_slope_err ** 2)
                    alpha22[f"finite_difference_accuracy_{accuracy}"][1] = alpha22_err

                    alpha33_err = np.sqrt((cslope_val / cval ** 2) ** 2 * cval_err ** 2 +
                                          (1 / cval) ** 2 * cslope_err ** 2)
                    alpha33[f"finite_difference_accuracy_{accuracy}"][1] = alpha33_err

                    alpha12_err = np.sqrt(
                        (aslope_val / (2 * aval ** 2 * np.tan(np.radians(gamma_star)))) ** 2 * aval_err ** 2 +
                        (1 / (2 * aval * np.tan(np.radians(gamma_star)))) ** 2 * aslope_err ** 2 +
                        (bslope_val / (2 * bval ** 2 * np.tan(np.radians(gamma_star)))) ** 2 * bval_err ** 2 +
                        (1 / (2 * bval * np.tan(np.radians(gamma_star)))) ** 2 * bslope_err ** 2 +
                        (alpha_angle_slope_val / (np.tan(np.radians(gamma_star)) * np.sin(
                            np.radians(alphaval)) ** 2)) ** 2 * alphaval_err ** 2 +
                        (1 / (2 * np.tan(np.radians(gamma_star)) * np.tan(
                            np.radians(alphaval))) ** 2 * alpha_angle_slope_err ** 2) +
                        (beta_angle_slope_val / (np.tan(np.radians(gamma_star)) * np.sin(
                            np.radians(betaval)) ** 2)) ** 2 * betaval_err ** 2 +
                        (1 / (2 * np.tan(np.radians(gamma_star)) * np.tan(
                            np.radians(betaval))) ** 2 * beta_angle_slope_err ** 2) +
                        ((1 / np.sin(np.radians(gamma_star))) ** 2 * ((aval / aslope_val) - (bval / bslope_val) - (
                                    alpha_angle_slope_val / np.tan(np.radians(alphaval))) +
                                                                      (beta_angle_slope_val / np.tan(np.radians(
                                                                          betaval))))) ** 2 * gamma_star_err ** 2 +
                        (1 / 2) ** 2 * gamma_star_slope_err ** 2)

                    alpha12[f"finite_difference_accuracy_{accuracy}"][1] = alpha12_err

                    alpha13_err = np.sqrt(
                        (aslope_val / (2 * aval ** 2 * np.tan(np.radians(betaval)))) ** 2 * aval_err ** 2 +
                        (1 / (2 * aval * np.tan(np.radians(betaval)))) ** 2 * aslope_err ** 2 +
                        (cslope_val / (2 * cval ** 2 * np.tan(np.radians(betaval)))) ** 2 * cval_err ** 2 +
                        (1 / (2 * cval * np.tan(np.radians(betaval)))) ** 2 * cslope_err ** 2 +
                        ((1 / (2 * np.sin(np.radians(betaval)))) * (
                                    (aslope_val / aval) - (cslope_val / cval))) ** 2 * betaval_err ** 2 +
                        (1 / 2) ** 2 * beta_angle_slope_err ** 2)
                    alpha13[f"finite_difference_accuracy_{accuracy}"][1] = alpha13_err

                    alpha23_prefactor1 = ((1 / (2 * np.tan(np.radians(gamma_star)) * np.tan(np.radians(betaval)))) + (
                                1 / (2 * np.sin(np.radians(gamma_star)) * np.tan(np.radians(alphaval))))) ** 2
                    alpha23_prefactor2 = ((1 / np.sin(np.radians(gamma_star))) ** 2 * (
                                (1 / (2 * np.tan(np.radians(betaval)))) * (
                                    (aslope_val / aval) - (cslope_val / cval)) + (beta_angle_slope_val / 2)) -
                                          (alpha_angle_slope_val + (1 / np.tan(np.radians(alphaval))) * (
                                                      (bslope_val / bval) - (cslope_val / cval))) / (
                                                      np.tan(np.radians(gamma_star)) * np.sin(
                                                  np.radians(gamma_star)))) ** 2
                    alpha23_err = np.sqrt(((aslope_val / aval ** 2) * (1 / (2 * np.tan(np.radians(gamma_star)) * np.tan(
                        np.radians(betaval))))) ** 2 * aval_err ** 2 +
                                          (1 / (2 * aval * np.tan(np.radians(gamma_star)) * np.tan(
                                              np.radians(betaval)))) ** 2 * aslope_err ** 2 +
                                          (bslope_val / (2 * np.sin(np.radians(gamma_star)) * np.tan(
                                              np.radians(alphaval)) * bval ** 2)) ** 2 * bval_err ** 2 +
                                          (1 / (2 * np.sin(np.radians(gamma_star)) * np.tan(
                                              np.radians(alphaval)) * bval)) ** 2 * bslope_err ** 2 +
                                          (cslope_val / cval ** 2) ** 2 * alpha23_prefactor1 * cval_err ** 2 + (
                                                      1 / cval) ** 2 * alpha23_prefactor1 * cslope_err ** 2 +
                                          ((bslope_val / bval) - (cslope_val / cval)) / (
                                                      2 * np.sin(np.radians(gamma_star)) * np.sin(
                                                  np.radians(alphaval)) ** 2) ** 2 * alphaval_err ** 2 +
                                          (1 / (2 * np.sin(np.radians(gamma_star)))) ** 2 * alpha_angle_slope_val ** 2 +
                                          ((aslope_val / aval) - (cslope_val / cval)) / (
                                                      2 * np.tan(np.radians(gamma_star)) * np.sin(
                                                  np.radians(betaval)) ** 2) ** 2 * betaval_err ** 2 +
                                          (1 / (2 * np.tan(np.radians(gamma_star)))) ** 2 * beta_angle_slope_err ** 2 +
                                          alpha23_prefactor2 * gamma_star_err ** 2)
                    alpha23[f"finite_difference_accuracy_{accuracy}"][1] = alpha23_err
        else:
            raise RuntimeError("invalid space group in prototype label")

        # enforce tensor symmetries
        alpha21 = alpha12
        alpha31 = alpha13
        alpha32 = alpha23

        alpha = np.array([[alpha11, alpha12, alpha13],
                          [alpha21, alpha22, alpha23],
                          [alpha31, alpha32, alpha33]])

        # thermal expansion coeff tensor
        return alpha

    @staticmethod
    def _check_lammps_log_for_wrong_structure_format(log_file):
        wrong_format_in_structure_file = False

        try:
            with open(log_file, "r") as logfile:
                data = logfile.read()
                data = data.split("\n")
                final_line = data[-2]

                if final_line == "Last command: read_data output/zero_temperature_crystal.lmp":
                    wrong_format_in_structure_file = True
        except FileNotFoundError:
            pass

        return wrong_format_in_structure_file

    @staticmethod
    def _get_slope_and_error(x_values: List[float], y_values: List[float], y_errs: List[float]):
        popt, pcov = scipy.optimize.curve_fit(lambda x, m, b: m * x + b, x_values, y_values,
                                              sigma=y_errs, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        return popt[0], perr[0]

    @staticmethod
    def _get_center_finite_difference_and_error(diff_x: float, y_values: List[float], y_errs: List[float],
                                                accuracy: int):
        assert len(y_values) == len(y_errs)
        assert len(y_values) > accuracy
        assert len(y_values) % 2 == 1
        center_index = len(y_values) // 2
        coefficients = findiff.coefficients(deriv=1, acc=accuracy)["center"]["coefficients"]
        offsets = findiff.coefficients(deriv=1, acc=accuracy)["center"]["offsets"]
        finite_difference = 0.0
        finite_difference_error_squared = 0.0
        for coefficient, offset in zip(coefficients, offsets):
            finite_difference += coefficient * y_values[center_index + offset]
            finite_difference_error_squared += (coefficient * y_errs[center_index + offset]) ** 2
        finite_difference /= diff_x
        finite_difference_error_squared /= (diff_x * diff_x)
        return finite_difference, sqrt(finite_difference_error_squared)

    @staticmethod
    def _extract_mean_error_from_logfile(filename: str, quantity: int) -> Tuple[float, float]:
        """
        Function to extract the average from a LAAMPS log file for a given quantity

        @param filename : name of file
        @param quantity : quantity to take from
        @return mean : reported mean value
        """

        # Get content.
        with open(filename, "r") as file:
            data = file.read()

        # Look for print pattern.
        exterior_pattern = r'print "\${run_var}"\s*\{(.*?)\}\s*variable run_var delete'
        mean_pattern = r'"mean"\s*([^ ]+)'
        error_pattern = r'"upper_confidence_limit"\s*([^ ]+)'
        match_init = re.search(exterior_pattern, data, re.DOTALL)
        mean_matches = re.findall(mean_pattern, match_init.group(), re.DOTALL)
        error_matches = re.findall(error_pattern, match_init.group(), re.DOTALL)
        if mean_matches is None:
            raise ValueError("Mean not found")
        if error_matches is None:
            raise ValueError("Error not found")

        # Get correct match.
        mean = float(mean_matches[quantity])
        error = float(error_matches[quantity])

        return mean, error

    @staticmethod
    def _extract_equilibration_step_from_logfile(filename: str) -> int:
        # Get file content.
        with open(filename, 'r') as file:
            data = file.read()

        # Look for pattern.
        exterior_pattern = r'print "\${run_var}"\s*\{(.*?)\}\s*variable run_var delete'
        mean_pattern = r'"equilibration_step"\s*([^ ]+)'
        match_init = re.search(exterior_pattern, data, re.DOTALL)
        equil_matches = re.findall(mean_pattern, match_init.group(), re.DOTALL)
        if equil_matches is None:
            raise ValueError("Equilibration step not found")

        # Return largest match.
        return max(int(equil) for equil in equil_matches)


if __name__ == "__main__":
    # model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"
    model_name = "Sim_LAMMPS_ReaxFF_BrugnoliMiyataniAkaji_SiCeNaClHO_2023__SM_282799919035_000"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = HeatCapacityPhonon(model_name)
    list_of_queried_structures = query_crystal_genome_structures(kim_model_name=model_name,
                                                                 stoichiometric_species=['Ce', 'O'],
                                                                 prototype_label='AB2_cF12_225_a_c')
    for queried_structure in list_of_queried_structures:
        test_driver(**queried_structure, temperature=293.15, pressure=1.0, temperature_step_fraction=0.01,
                    number_symmetric_temperature_steps=2, timestep=0.001, number_sampling_timesteps=100,
                    repeat=(3, 3, 3), loose_triclinic_and_monoclinic=True, max_workers=5)
