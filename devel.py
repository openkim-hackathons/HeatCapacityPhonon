from concurrent.futures import as_completed, ProcessPoolExecutor
from math import ceil, sqrt
import os
import random
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple
from ase import Atoms
from ase.build import bulk
import findiff
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize
from kim_test_utils.test_driver import CrystalGenomeTestDriver


class HeatCapacityPhonon(CrystalGenomeTestDriver):
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

        if not number_symmetric_temperature_steps > 0:
            raise RuntimeError("Number of symmetric temperature steps has to be bigger than zero.")

        if number_symmetric_temperature_steps * temperature_step_fraction >= 1.0:
            raise RuntimeError("The given number of symmetric temperature steps and the given temperature-step fraction "
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

        # Write lammps file.
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        atoms_new.write(structure_file, format="lammps-data", masses=True)

        # Get temperatures that should be simulated.
        temperature_step = temperature_step_fraction * temperature
        temperatures = [temperature + i * temperature_step
                        for i in range(-number_symmetric_temperature_steps, number_symmetric_temperature_steps + 1)]
        assert len(temperatures) == 2 * number_symmetric_temperature_steps + 1
        assert all(t > 0.0 for t in temperatures)

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
            self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
                reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)

        c = self._compute_heat_capacity(temperatures, log_filenames, 2)

        # Print result.
        print('####################################')
        print('# NPT Heat Capacity Results #')
        print('####################################')
        print(f'C_p:\t{c}')

        # I have to do this or KIM tries to save some coordinate file.
        self.poscar = None

        # Write property.
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-phonon-npt", write_stress=True, write_temp=True)  # last two default to False
        self._add_key_to_current_property_instance(
            "constant_pressure_heat_capacity", c["finite_difference_accuracy_2"][0], "eV/Kelvin")
        self._add_key_to_current_property_instance(
            "constant_pressure_heat_capacity_err", c["finite_difference_accuracy_2"][1], "eV/Kelvin")
        self._add_key_to_current_property_instance("pressure", pressure, "bars")

    @staticmethod
    def _run_lammps(modelname: str, temperature_index: int, temperature: float, pressure: float, timestep: float,
                    number_sampling_timesteps: int, species: List[str]) -> Tuple[str, str, str, str]:
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

        command = (
            "lammps "
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
            + f" -log {log_filename}"
            + " -in npt_equilibration.lammps")
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
            img_file = os.path.join(dir_name, in_file_name.replace(".log", "_")+property_name + ".png")
            plt.savefig(img_file, bbox_inches="tight")
            plt.close()

    @staticmethod
    def _compute_average_positions_from_lammps_dump(data_dir: str, file_str: str, output_filename: str, skip_steps: int) -> None:
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
        assert all(abs(temperatures[i+1] - temperatures[i] - temperature_step)
                   < 1.0e-12 for i in range(len(temperatures) - 1))
        assert len(temperatures) >= 3
        max_accuracy = len(temperatures) - 1
        heat_capacity = {}
        for accuracy in range(2, max_accuracy + 1, 2):
            heat_capacity[f"finite_difference_accuracy_{accuracy}"] = HeatCapacityPhonon._get_center_finite_difference_and_error(
                temperature_step, enthalpy_means, enthalpy_errs, accuracy)

        # Use linear fit to estimate derivative.
        heat_capacity["fit"] = HeatCapacityPhonon._get_slope_and_error(
            temperatures, enthalpy_means, enthalpy_errs)

        return heat_capacity

    @staticmethod
    def _get_slope_and_error(x_values: List[float], y_values: List[float], y_errs: List[float]):
        popt, pcov = scipy.optimize.curve_fit(lambda x, m, b: m * x + b, x_values, y_values,
                                              sigma=y_errs, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        return popt[0], perr[0]

    @staticmethod
    def _get_center_finite_difference_and_error(diff_x: float, y_values: List[float], y_errs: List[float], accuracy: int):
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
    model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = HeatCapacityPhonon(model_name)
    test_driver(bulk("Ar", "fcc", a=5.248), temperature=10.0, pressure=1.0, temperature_step_fraction=0.01,
                number_symmetric_temperature_steps=2, timestep=0.001, number_sampling_timesteps=100,
                repeat=(7, 7, 7), loose_triclinic_and_monoclinic=False, max_workers=5)
