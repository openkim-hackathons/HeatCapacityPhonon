import numpy as np
import os
import sys
import re
'''
Notes: run this file calculate the averaged position over files(e.g average_position_step100.dump,average_position_step200.dump)
if run "python compute_average_position.py dir_path"
average_positionXXX.dupm files will be loaded from dir_path
and the output is writen to dir_path/average_position_over_files.ouput
'''

def get_id_pos_dict(file_name:str) -> dict:
    '''
    input: 
    file_name--the file_name that contains average postion data
    output:
    the dictionary contains id:position pairs e.g {1:array([x1,y1,z1]),2:array([x2,y2,z2])}
    for the averaged positions over files
    '''
    id_pos_dict = {}
    header4N = ["NUMBER OF ATOMS"]
    header4pos = ["id","f_avePos[1]","f_avePos[2]","f_avePos[3]"]
    is_table_started = False
    is_natom_read = False
    
    with open(file_name,"r") as f:
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
                #pos = np.array([float(words[2]),float(words[3]),float(words[4])])
                pos = np.array([float(words[1]),float(words[2]),float(words[3])])
                id_pos_dict[id] = pos 
            if count_content_line > 0 and count_content_line >= N:
                break
            line = f.readline()
    if count_content_line < N:
        print("The file " + file_name +
              " is not complete, the number of atoms is smaller than " + str(N))
    return id_pos_dict
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # searh for data from ./output by default
        data_dir = "./output"
    else:
        data_dir = sys.argv[1]

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir + " does not exist")
    
    # extract and store all the data
    pos_list = []
    max_step,last_step_file = -1, ""
    for file_name in os.listdir(data_dir):
        if ("average_position_equilibration" in file_name) and ("dump" in file_name):
            file_path = os.path.join(data_dir,file_name)
            id_pos_dict = get_id_pos_dict(file_path)
            id_pos = sorted(id_pos_dict.items())
            id_list = [pair[0] for pair in id_pos]
            pos_list.append([pair[1] for pair in id_pos])
            # check if this is the last step
            step = int(re.findall(r'\d+', file_name)[-1])
            if step > max_step:
                last_step_file,max_step = os.path.join(data_dir ,file_name),step 
    pos_arr = np.array(pos_list)
    avg_pos = np.mean(pos_arr,axis=0)
    # get the lines above the table from the file of the last step
    with open(last_step_file,"r") as f:
        header4pos = ["id","f_avePos[1]","f_avePos[2]","f_avePos[3]"]
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

    # write the output to the file
    output_file = os.path.join(data_dir,"average_position_over_files.out")
    with open(output_file,"w") as f:
        f.write(description_str)
        #header_str = "id  pos[1]  pos[2]  pos[3]"
        #f.write(header_str)
        for i in range(len(id_list)):
            f.write(str(id_list[i]))
            f.write("  ")
            for dim in range(3):
                f.write('{:3.6}'.format(avg_pos[i,dim]))
                f.write("  ")
            f.write("\n")

        
