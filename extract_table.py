import pandas as pd
import matplotlib.pyplot as plt

in_file_name = "./output/lammps_equilibration.log"
out_file_name = "./output/lammps_equlibration.csv"
# generate the clean table and save it to out_file_name
# plot steps VS volume

is_first_header = True
header_flags  = ["Step","v_pe_metal","v_temp_metal","v_press_metal"]
eot_flags  = ["Loop","time","on","procs","for","steps"]

table = []

with open(in_file_name,"r") as f:
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
with open(out_file_name,"w") as f:
    for l in table:
        f.writelines(l)

df = pd.read_csv(out_file_name,delim_whitespace=True)
print(df.columns)
volumes = df['v_vol_metal']
step = df['Step']
plt.plot(step,volumes)
for i in range(22):
    plt.gca().axvline(i*1000, color="k", linestyle="dashed")
plt.show()



        
