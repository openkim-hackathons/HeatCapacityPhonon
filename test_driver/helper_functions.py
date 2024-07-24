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
            property_dict["v_xy_metal"], property_dict["v_xz_metal"], property_dict["v_yz_metal"]]


def _get_positions_from_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
    lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key=lambda x: x[0])
    return [(line[1], line[2], line[3]) for line in lines]

def _get_cell(cell_list: List[float]) -> npt.NDArray[np.float64]:
    assert len(cell_list) == 6
    cell = np.empty(shape=(3, 3))
    cell[0, :] = np.array([cell_list[0], 0.0, 0.0])
    cell[1, :] = np.array([cell_list[3], cell_list[1], 0.0])
    cell[2, :] = np.array([cell_list[4], cell_list[5], cell_list[2]])
    return cell

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
    c_err = np.sqrt(((dx / (2 * eps)) ** 2) + ((dy / (2 * eps)) ** 2))
    # Return
    return c, c_err

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

def _extract_equilibration_step_from_logfile(filename: str) -> int:
    # Get file content
    with open(filename, 'r') as file:
        data = file.read()

    # Look for pattern
    exterior_pattern = r'print "\${run_var}"\s*\{(.*?)\}\s*variable run_var delete'
    mean_pattern = r'"equilibration_step"\s*([^ ]+)'
    match_init = re.search(exterior_pattern, data, re.DOTALL)
    equil_matches = re.findall(mean_pattern, match_init.group(), re.DOTALL)
    if equil_matches is None:
        raise ValueError("Equilibration step not found")

    # Return largest match
    return max(int(equil) for equil in equil_matches)