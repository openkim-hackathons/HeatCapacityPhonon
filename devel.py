from kim_python_utils.ase import CrystalGenomeTest,KIMASEError
from numpy import multiply
import numpy as np
from math import pi
from crystal_genome_util.aflow_util import get_stoich_reduced_list_from_prototype
from ase.build import bulk
import os
class HeatCapacityPhonon(CrystalGenomeTest):
    def _calculate(self, structure_index: int, temperature: float, pressure: float, mass:list, timestep: float, number_control_timesteps: int,repeat:tuple=(3,3,3)):
        """
        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). 
            This indicates which is being used for the current calculation.

        temperature:
            Temperature in Kelvin at which the phonon contribution to the heat capacity 
            at constant volume is estimated. Must be strictly greater than zero.
        
        pressure:
            Pressure in bar of the NPT simulation for the initial equilibration of the 
            zero-temperature configuration. Must be strictly greater than zero.
        """
        if not temperature > 0.0:
            raise RuntimeError("Temperature has to be larger than zero.")
        
        if not pressure > 0.0:
            raise RuntimeError("Pressure has to be larger than zero.")
        
        atoms = self.atoms[structure_index]
        
        """TODO: Guanming puts code here."""
        print(atoms)
        atoms = atoms.repeat(repeat)
        print(atoms)        
        proto = self.prototype_label
        
        # __file__ is the location of the current file
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory,"zero_temperature_crystal.lmp")
        print(structure_file)
        atoms.write(structure_file,format="lammps-data")
        # TODO: HOW DO WE GET THE .lmp file?
        
        # TODO: Should we call the lammps executable explicitly or is there some internal 
        # to do it (say via self.model)?
        self.add_masses_to_data_file(structure_file,mass)
        ####################################################
        # ACTUAL CALCULATION BEGINS 
        ####################################################
        original_cell = atoms.get_cell() # do this instead of deepcopy
        natoms = len(atoms)

        for i in range(N):
            a_frac = a_frac_step*i + a_min_frac
            print("evaluating a_frac = " + str(a_frac) + " ...")
            atoms.set_cell(multiply(original_cell,a_frac),scale_atoms = True)
        
        # LAMMPS for heat capacity
        seed = np.random.randint(0, 1000)
        pdamp = timestep * 100
        tdamp = timestep * 100

        # NPT simulation
        lmp_npt = 'modelname ${model_name} temperature ${temperature} temperature_seed ${seed} temperature_damping ${tdamp} pressure ${pressure} pressure_damping ${pdamp} timestep ${timestep} number_control_timesteps ${number_control_timesteps}'
        script = 'npt_equilibration.lammps'
        command = 'lammps -var %s -in %s'%(lmp_npt, script)
        subprocess.run(command, check=True, shell=True)

        # Check symmetry - post-NPT
        try:

            self._update_aflow_designation_from_atoms(structure_index,atoms_new)
            raise RuntimeError("We should not have gotten here")

        except KIMASEError as e:
            print("We have successfully caught an exception with the following message:")
            print(e.msg)

        # NVT simulation
        vol = np.loadtxt('volume.dat')

        lmp_nvt = 'modelname ${model_name} temperature ${temperature} temperature_seed ${seed} temperature_damping ${tdamp} volume ${vol} timestep ${timestep} number_control_timesteps ${number_control_timesteps}'
        script = 'nvt_equilibration.lammps'
        command = 'lammps -var %s -in %s'%(lmp_npt, script)
        subprocess.run(command, check=True, shell=True) 

        # Check symmetry - post-NVT
        try:

            self._update_aflow_designation_from_atoms(structure_index,atoms_new)
            raise RuntimeError("We should not have gotten here")

        except KIMASEError as e:
            print("We have successfully caught an exception with the following message:")
            print(e.msg)

        ####################################################
        # ACTUAL CALCULATION ENDS 
        ####################################################

        ####################################################
        # SOME USAGE EXAMPLES NOT NECESSARY FOR THE PRESENT TEST 
        ####################################################
        # This is unnecessary here because we are not changing the atoms object, but if we were and we needed to re-analyze, this is how you do it 
        self.atoms[structure_index].set_cell(original_cell,scale_atoms=True)
        self._update_aflow_designation_from_atoms(structure_index)

        # alternatively, you can update the `structure_index`-th AFLOW symmetry designation from a specified atoms object instead of from 
        # self.atoms[structure_index]. The function will also raise an error if the prototype label changes, so you can use it as a try-except to detect
        # symmetry changes
        atoms_new = atoms.copy()
        cell = atoms_new.get_cell()
        cell[1,2] += 0.5 # this is highly likely to change the symmetry
        atoms_new.set_cell(cell,scale_atoms=True)

        try:
            # this will intentionally raise an exception to demonstrate changing symmetry issue
            self._update_aflow_designation_from_atoms(structure_index,atoms_new)
            raise RuntimeError("We should not have gotten here")
        except KIMASEError as e:
            print("We have successfully caught an exception with the following message:")
            print(e.msg)
        ####################################################
        # USAGE EXAMPLES END
        ####################################################

        ####################################################
        # PROPERTY WRITING
        ####################################################
        
        # Import data
        cv = np.loadtxt('cv.dat')
        cp = np.loadtxt('cp.dat')

        # Assign property
        self._add_property_instance("heat_capacity")
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=False,write_temp=False) # last two default to False
        self._add_key_to_current_property_instance("constant_pressure_heat_capacity",constant_pressure_heat_capacity,"eV/Kelvin")
        self._add_key_to_current_property_instance("constant_volume_heat_capacity",constant_volume_heat_capacity,"eV/Kelvin")
        self._add_key_to_current_property_instance("volume", volume, "Angstroms^3")
        self._add_key_to_current_property_instance("pressure", pressure, "bars")
        ####################################################
        # PROPERTY WRITING END
        ####################################################
       # add masses to data file
    def add_masses_to_data_file(self,data_file,masses):
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory,data_file)
    
        with open(structure_file,"r") as infile:
            data = infile.readlines()
   
        mass_lines = []
        mass_lines.append("\n")
        mass_lines.append("Masses \n")
        mass_lines.append("\n")

        for i in range(1,len(masses)+1):
            mass_lines.append("    "+str(i))

        mass_lines.append("\n")
    
        all_lines = data + mass_lines
        with open(structure_file,"w") as outfile:
            outfile.writelines(all_lines)

if __name__ == "__main__":
    ####################################################
    # if called directly, do some debugging examples
    ####################################################


    # This queries for equilibrium structures in this prototype and builds atoms
    # test = BindingEnergyVsWignerSeitzRadius(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", stoichiometric_species=['Fe','P'], prototype_label='AB_oP8_62_c_c')
                    
    # Alternatively, for debugging, give it atoms object or a list of atoms objects
    atoms1 = bulk('NaCl','rocksalt',a=4.58)
    atoms2 = bulk('NaCl','cesiumchloride',a=4.58)
    test = HeatCapacityPhonon(model_name="Sim_LAMMPS_EIM_Zhou_2010_BrClCsFIKLiNaRb__SM_259779394709_000", atoms=atoms1)
    test(temperature = 298, pressure = 1.0, mass = [1.0,.10],timestep=0.001, number_control_timesteps=10,repeat=(3,3,3))

