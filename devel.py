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
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
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
        self._plot_property_from_lammps_log("output/lammps_equilibration.log", ("v_vol_metal", "v_temp_metal"))
        self._compute_average_positions_from_lammps_dump("output", "average_position_equilibration.dump",
                                                         "output/average_position_equilibration_over_dump.out")
        atoms_new.set_cell(self._get_cell_from_lammps_dump("output/average_position_equilibration_over_dump.out"))
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
        self._plot_property_from_lammps_log("output/lammps_high_temperature.log",
                                            ("v_vol_metal", "v_temp_metal", "v_enthalpy_metal"))
        self._compute_average_positions_from_lammps_dump("output", "average_position_high_temperature.dump",
                                                         "output/average_position_high_temperature_over_dump.out")
        atoms_new.set_cell(self._get_cell_from_lammps_dump("output/average_position_high_temperature_over_dump.out"))
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
            "pressure": pressure,
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
        self._plot_property_from_lammps_log("output/lammps_low_temperature.log",
                                            ("v_vol_metal", "v_temp_metal", "v_enthalpy_metal"))
        self._compute_average_positions_from_lammps_dump("output", "average_position_low_temperature.dump",
                                                         "output/average_position_low_temperature_over_dump.out")
        atoms_new.set_cell(self._get_cell_from_lammps_dump("output/average_position_low_temperature_over_dump.out"))
        atoms_new.set_scaled_positions(
            self._get_positions_from_lammps_dump("output/average_position_low_temperature_over_dump.out"))
        #*** here we try the new function _compute_average_positions_from_lammps_dump() and 
        #*** _average_cell_over_steps(), feel free to remove these two lines 
        self._compute_average_positions_from_lammps_dump("output", "average_position_low_temperature.dump",
                                                         "output/average_position_low_temperature_over_dump_skip_30000.out",
                                                         skip_steps = 30000)
        property_value_dict = self._average_cell_over_steps("./output/average_cell_equilibration.dump")
        
        # Reduce and average
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

        # I have to do this or KIM tries to save some coordinate file
        self.poscar = None

        # Write property
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-phonon-npt", write_stress=True, write_temp=True)  # last two default to False
        self._add_key_to_current_property_instance("constant_pressure_heat_capacity", c, "eV/Kelvin")
        self._add_key_to_current_property_instance("constant_pressure_heat_capacity_err", c_err, "eV/Kelvin")
        self._add_key_to_current_property_instance("pressure", variables['pressure'], "bars")

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
                raise FileNotFoundError(in_file + "not found")
            elif not ".log" in in_file:
                raise FileNotFoundError("The file is not a *.log file")
            is_first_header = True
            header_flags = ["Step", "v_pe_metal", "v_temp_metal", "v_press_metal"]
            eot_flags = ["Loop", "time", "on", "procs", "for", "steps"]
            table = []
            with open(in_file, "r") as f:
                line = f.readline()
                while line:  # not EOF
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
                        # pos = np.array([float(words[2]),float(words[3]),float(words[4])])
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

        # extract and store all the data
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
                # check if this is the last step
                if step > max_step:
                    last_step_file, max_step = os.path.join(data_dir, file_name), step
        if max_step == -1 and last_step_file == "":
            raise RuntimeError("Found no files to average over.")
        pos_arr = np.array(pos_list)
        avg_pos = np.mean(pos_arr, axis=0)
        # get the lines above the table from the file of the last step
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
        # Write the output to the file
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
    def _average_cell_over_steps(input_file: str, skip_steps: int) -> List[float]:
        '''
        average cell properties over time steps
        args:
        input_file: the input file e.g "./output/average_cell_low_temperature.dump"
        return:
        the dictionary contains the property_name and its averaged value
        e.g. {v_lx_metal:1.0,v_ly_metal:2.0 ...}
        '''
        with open(input_file, "r") as f:
            f.readline()  # skip the first line
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
        property_dict = {property_names[i]: mean_data[i] for i in range(len(mean_data)) if property_names[i] != "TimeStep"}
        return [property_dict["v_lx_metal"], property_dict["v_ly_metal"], property_dict["v_lz_metal"], 
                property_dict["v_xy_metal"], property_dict["v_xz_metal"], property_dict["v_xz_metal"]]

    @staticmethod
    def _get_positions_from_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
        lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key=lambda x: x[0])
        return [(line[1], line[2], line[3]) for line in lines]

    @staticmethod
    def _get_cell_from_lammps_dump(filename: str) -> npt.NDArray[np.float64]:
        new_cell = np.loadtxt(filename, skiprows=5, max_rows=3)
        assert new_cell.shape == (3, 2) or new_cell.shape == (3, 3)

        # See https://docs.lammps.org/Howto_triclinic.html.
        xlo_bound = new_cell[0, 0]
        xhi_bound = new_cell[0, 1]
        ylo_bound = new_cell[1, 0]
        yhi_bound = new_cell[1, 1]
        zlo_bound = new_cell[2, 0]
        zhi_bound = new_cell[2, 1]

        # If not cubic add more cell params
        if new_cell.shape[-1] != 2:
            xy = new_cell[0, 2]
            xz = new_cell[1, 2]
            yz = new_cell[2, 2]
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

    @staticmethod
    def _compute_heat_capacity(f1: str, f2: str, eps: float, quantity: int) -> Tuple[float, float]:
        """
        Function to compute heat capacity by finite difference from two simulations

        @param f1 : filename of first file
        @param f2 : filename of second file
        @param eps : epsilon
        @param quantity : number quantity in the log file
        @return c : heat capacity
        @return err : the error in the reported value
        """

        # Extract mean values
        H_plus_mean, H_plus_err = HeatCapacityPhonon._extract_mean_error_from_logfile(f1, quantity)
        H_minus_mean, H_minus_err = HeatCapacityPhonon._extract_mean_error_from_logfile(f2, quantity)

        # Compute heat capacity
        c = (H_plus_mean - H_minus_mean) / (2 * eps)

        # Error computation
        dx = H_plus_err
        dy = H_minus_err
        c_err = np.sqrt(((dx / (2 * eps)) ** 2) + (- (dy / (2 * eps)) ** 2))

        # Return
        return c, c_err

    @staticmethod
    def _extract_mean_error_from_logfile(filename: str, quantity: int) -> Tuple[float, float]:
        """
        Function to extract the average from a LAAMPS log file for a given quantity

        @param filename : name of file
        @param quantity : quantity to take from
        @return mean : reported mean value
        """

        # Get content
        with open(filename, "r") as file:
            data = file.read()

        # Look for print pattern
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

        # Get correct match
        mean = float(mean_matches[quantity])
        error = float(error_matches[quantity])

        return mean, error


if __name__ == "__main__":
    model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = HeatCapacityPhonon(model_name)
    test_driver(bulk("Ar", "fcc", a=5.248), temperature=10.0, pressure=1.0, temperature_offset_fraction=0.01,
                timestep=0.001, number_sampling_timesteps=100, repeat=(7, 7, 7), loose_triclinic_and_monoclinic=False)
