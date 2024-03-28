import pandas as pd
import matplotlib.pyplot as plt
import sys 
import os

# usage: python extract_table.py <input_file> <output_file> <property_name>
# e.g 
#    python extract_table.py ./output/lammps_equilibration.log ./output/lammps_equilibration.csv v_vol_metal
#
# properties includes(the header of the table)
#     v_pe_metal,  v_temp_metal, v_press_metal,v_vol_metal    
#     v_xlo_metal, v_xhi_metal,  v_ylo_metal, v_yhi_metal 
#     v_zlo_metal, v_zhi_metal

     
# generate clean table from in_file_name 
# and save it to out_file_name
# plot steps VS property(e.g volume)
# save the figure to output/property_name.png


if len(sys.argv) < 2:
    # plot the volume using lammps_equilibration.log if there is no explicit inputs
    in_file_name = "./output/lammps_equilibration.log"
    out_file_name = "./output/lammps_equilibration.csv"
    property_name = "v_vol_metal"
else:
    in_file_name = sys.argv[1]
    out_file_name = sys.argv[2]
    property_name = sys.argv[3]


def get_table(in_file):
    if not os.path.isfile(in_file):
        print(in_file)
        raise FileNotFoundError
    is_first_header = True
    header_flags  = ["Step","v_pe_metal","v_temp_metal","v_press_metal"]
    eot_flags  = ["Loop","time","on","procs","for","steps"]
    table = []
    with open(in_file,"r") as f:
        line = f.readline()
        while line: # not EOF
            is_header = True
            for _s in header_flags:
                is_header = is_header and (_s in line)
            if is_header:
                if is_first_header:
                    table.append(line)
                    is_first_header = False
                while True:
                    content = f.readline()
                    is_eot = True
                    for _s in eot_flags:
                        is_eot = is_eot and (_s in content)
                    if not is_eot:
                        table.append(content)
                    else:
                        break
            line = f.readline()
    return table

def write_table(out_file):
    with open(out_file,"w") as f:
        for l in table:
            f.writelines(l)


if os.path.isfile(out_file_name):
    df = pd.read_csv(out_file_name,delim_whitespace=True)
else:
    table = get_table(in_file_name)
    write_table(out_file_name)
    df = pd.read_csv(out_file_name,delim_whitespace=True)
    
volumes = df[property_name]
step = df['Step']
plt.plot(step,volumes)
for i in range(max(step)//1000):
    plt.gca().axvline(i*1000, color="k", linestyle="dashed")
plt.xlabel("step")
plt.ylabel(property_name)
img_file =  "./output/" + property_name +".png"
plt.savefig(img_file,bbox_inches='tight')
plt.show()


