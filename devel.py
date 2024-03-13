from kim_python_utils.ase import CrystalGenomeTest,KIMASEError
from numpy import multiply
import numpy as np
from math import pi
from crystal_genome_util.aflow_util import get_stoich_reduced_list_from_prototype
from ase.build import bulk

class HeatCapacityPhonon(CrystalGenomeTest):
    def _calculate(self, structure_index: int, temperature: float, pressure: float, timestep: float, number_control_timesteps: int):
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

        # TODO: HOW DO WE GET THE .lmp file?
        
        # TODO: Should we call the lammps executable explicitly or is there some internal 
        # to do it (say via self.model)?
        
        ####################################################
        # ACTUAL CALCULATION BEGINS 
        ####################################################
        average_wigner_seitz_radius = []
        binding_potential_energy_per_atom = []
        binding_potential_energy_per_formula = []
        a_frac_range = a_max_frac - a_min_frac
        a_frac_step = a_frac_range/(N-1)
        original_cell = atoms.get_cell() # do this instead of deepcopy
        natoms = len(atoms)

        for i in range(N):
            a_frac = a_frac_step*i + a_min_frac
            print("evaluating a_frac = " + str(a_frac) + " ...")
            atoms.set_cell(multiply(original_cell,a_frac),scale_atoms = True)
            average_wigner_seitz_radius.append(((atoms.get_volume()/natoms)*(3/4)/pi)**(1/3))            
            pe_per_atom = atoms.get_potential_energy()/natoms
            binding_potential_energy_per_atom.append(pe_per_atom)
            binding_potential_energy_per_formula.append(pe_per_atom*sum(get_stoich_reduced_list_from_prototype(self.prototype_label)))

        
        # LAMMPS for heat capacity
        seed = np.random.randint(0, 1000)
        pdamp = timestep * 100
        tdamp = timestep * 100

        # NPT simulation
        lmp_npt = 'lmp modelname ${model_name} temperature ${temperature} temperature_seed ${seed} temperature_damping ${tdamp} pressure ${pressure} pressure_damping ${pdamp} timestep ${timestep} number_control_timesteps ${number_control_timesteps}'
        script = 'npt_equilibration.lammps'
        

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
        self._add_property_instance("heat_capacity")
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=False,write_temp=False) # last two default to False
        self._add_key_to_current_property_instance("average-wigner-seitz-radius",average_wigner_seitz_radius,"angstrom")
        self._add_key_to_current_property_instance("constant_pressure_heat_capacity",constant_pressure_heat_capacity,"eV")
        self._add_key_to_current_property_instance("constant_volume_heat_capacity",constant_volume_heat_capacity,"eV")
        ####################################################
        # PROPERTY WRITING END
        ####################################################
        
if __name__ == "__main__":
    ####################################################
    # if called directly, do some debugging examples
    ####################################################


    # This queries for equilibrium structures in this prototype and builds atoms
    # test = BindingEnergyVsWignerSeitzRadius(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", stoichiometric_species=['Fe','P'], prototype_label='AB_oP8_62_c_c')
                    
    # Alternatively, for debugging, give it atoms object or a list of atoms objects
    atoms1 = bulk('NaCl','rocksalt',a=4.58)
    atoms2 = bulk('NaCl','cesiumchloride',a=4.58)

    # Hard-coded values for testing
    test = HeatCapacityPhonon(model_name="Sim_LAMMPS_EIM_Zhou_2010_BrClCsFIKLiNaRb__SM_259779394709_000", atoms=atoms1, temperature=298, pressure=0.1, timestep=0.001, number_control_timesteps=10)