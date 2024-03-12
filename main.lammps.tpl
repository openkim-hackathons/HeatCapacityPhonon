kim init sed_model_string metal unit_conversion_mode

# Isolated atom energy for this species in eV (computed in a separate LAMMPS calculation)
variable isolated_atom_energy equal sed_isolated_atom_energy

# Define looping variables
variable loopcount loop sed_numberofattemptedspacings_string
variable latticeconst index sed_latticeconst_string

# Periodic boundary conditions along all three dimensions
boundary p p p

neigh_modify one 4000

# Create a sed_latticetype_string lattice using a single conventional (orthogonal) unit
# cell with a lattice constant from the 'latticeconst' variable defined on line 15 above
variable latticeconst_converted equal ${latticeconst}*${_u_distance}
lattice sed_latticetype_string ${latticeconst_converted}
region box block 0 1 0 1 0 1 units lattice
create_box 1 box
create_atoms 1 box
mass 1 1.0 # Mass inconsequential since we're not performing time integration

kim interactions sed_species_string

# Variables used to rescale the box parameters, positions and forces so that the
# quantities in the thermo output and dumpfile are in the original metal units
# (Angstroms and eV/Angstrom) even if we're running with a Simulator Model that uses
# different units
variable     pe_metal  equal       "c_thermo_pe/v__u_energy"
variable    xlo_metal  equal              xlo/${_u_distance}
variable    xhi_metal  equal              xhi/${_u_distance}
variable    ylo_metal  equal              ylo/${_u_distance}
variable    yhi_metal  equal              yhi/${_u_distance}
variable    zlo_metal  equal              zlo/${_u_distance}
variable    zhi_metal  equal              zhi/${_u_distance}
variable     xy_metal  equal               xy/${_u_distance}
variable     xz_metal  equal               xz/${_u_distance}
variable     yz_metal  equal               yz/${_u_distance}
variable  press_metal  equal  "c_thermo_press/v__u_pressure"
variable    pxx_metal  equal              pxx/${_u_pressure}
variable    pyy_metal  equal              pyy/${_u_pressure}
variable    pzz_metal  equal              pzz/${_u_pressure}
variable    pxy_metal  equal              pxy/${_u_pressure}
variable    pxz_metal  equal              pxz/${_u_pressure}
variable    pyz_metal  equal              pyz/${_u_pressure}
variable      x_metal   atom                x/${_u_distance}
variable      y_metal   atom                y/${_u_distance}
variable      z_metal   atom                z/${_u_distance}
variable     fx_metal   atom                  fx/${_u_force}
variable     fy_metal   atom                  fy/${_u_force}
variable     fz_metal   atom                  fz/${_u_force}

# Set what thermodynamic information to print to log
thermo_style custom step atoms v_xlo_metal v_xhi_metal v_ylo_metal v_yhi_metal v_zlo_metal v_zhi_metal &
                    v_pe_metal press v_press_metal v_pxx_metal v_pyy_metal v_pzz_metal &
                    v_pxy_metal v_pxz_metal v_pyz_metal
thermo 10 # Print every 10 steps

# Set what information to write to dump file
dump dumpid all custom 10 output/lammps.dump id type v_x_metal v_y_metal v_z_metal &
                                            v_fx_metal v_fy_metal v_fz_metal

dump_modify dumpid format line "%d %d %16.7f %16.7f %16.7f %16.7f %16.7f %16.7f"

# Compute the energy and forces for this lattice spacing
run 0

# Calculate cohesive energy
variable natoms    equal "count(all)"
variable ecohesive equal "v_pe_metal/v_natoms - v_isolated_atom_energy"

# Output cohesive energy and equilibrium lattice constant
print "Cohesive energy: ${ecohesive} eV/atom"

# Queue next loop
clear # Clear existing atoms, variables, and allocated memory
next latticeconst # Increment latticeconst to next value
next loopcount # Increment loopcount to next value
jump SELF # Reload this input script with the new variable values
