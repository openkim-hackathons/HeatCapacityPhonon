kim init sed_model_string metal unit_conversion_mode

boundary f f f

variable boxextent equal 25.0
region box block -${boxextent} ${boxextent} &
                 -${boxextent} ${boxextent} &
                 -${boxextent} ${boxextent}

create_box 1 box

create_atoms 1  single 0.0 0.0 0.0

change_box all x scale ${_u_distance} &
               y scale ${_u_distance} &
               z scale ${_u_distance} &
               remap

variable mass_converted equal sed_mass_string*${_u_mass}
mass 1 ${mass_converted}

atom_modify sort 0 0

kim interactions sed_species_string

# Use nsq neighlist method instead of binning since this is a small system
variable neigh_skin equal 2.0*${_u_distance}
neighbor ${neigh_skin} nsq

# Variables used to rescale the positions and forces so that the quantities in the
# dumpfile are in the original metal units (angstrom and eV/angstrom) even if we're
# running with a Simulator Model that uses different units
variable  pe_metal  equal  "c_thermo_pe/v__u_energy"
thermo_style custom v_pe_metal

run 0

print "Isolated atom energy: ${pe_metal} eV"
