from test_driver.test_driver import TestDriver
from ase.build import bulk

atoms = bulk("Al")
td=TestDriver("EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_006")
td(
        atoms,
        temperature_K=293.15,
        target_size=1000,
        rlc_min_samples=100,
        msd_timesteps=5000,
        lammps_command= "mpirun -np 2 --bind-to numa lmp",
        max_workers= 3,
)
td.write_property_instances_to_file()



"""
if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Pass arguments to the KIM test driver')
    parser.add_argument('-m', '--model', help='Pass model for test driver to run', required=True)
    parser.add_argument('-s', '--stoichiometry', help='Stoichiometry of structure to test', nargs='+',
                        required=True)
    parser.add_argument('-p', '--prototype', help='Prototype ASE label', required=True)
    args = parser.parse_args()

    # Get arguments
    model_name = args.model
    stoich = args.stoichiometry
    prot = args.prototype

    # Run test
    subprocess.run(f"kim-api-collections-management install user {model_name}", shell=True)
    test_driver = TestDriver(model_name)
    list_of_queried_structures = query_crystal_structures(kim_model_name=model_name,
                                                          stoichiometric_species=stoich,
                                                          prototype_label=prot)
    for i, queried_structure in enumerate(list_of_queried_structures):
        try:
            test_driver(queried_structure, temperature_K=293.15,
                        cell_cauchy_stress_eV_angstrom3=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        temperature_step_fraction=0.01, number_symmetric_temperature_steps=1, timestep_ps=0.001,
                        number_sampling_timesteps=100, repeat=(4, 4, 4), max_workers=3, msd_threshold=0.1,
                        lammps_command="lmp", random_seeds=(1, 2, 3))
        except Exception as e:
            print(f"Got exception {repr(e)}")
    test_driver.write_property_instances_to_file()
"""