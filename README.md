# adversarial_env_generator

CHAI internship project with Michael Dennis.

## List of files

#### mini_grid_env.py
Defines a more complex mini grid environment space. The environment space contains 48 possible mazes, all with a T-shape wall in the centre dividing the maze into two chambers.
(Also contains some testing code to check that a single agent trains in this environment as expected.)

#### multi_bit_env.py
Defines a multi-bit guessing environment

#### sequential_multi_bit_env.py
Defines a multi-bit guessing environment where the antagonist/protagonist guess at each bit sequentially.

#### separator_wrapper.py
Defines the central separator (builder/describer/antagonist/protagonist) environment as a MultiAgentEnv.

#### simple_mini_grid_env.py
Defines a mini grid environment space with two possible environments with the goal in the bottom left and bottom right respectively.
(Also contains some testing code to check that a single agent trains in this environment as expected.)

#### single_bit_env.py
Defines a single-bit guessing environment.

#### test.py
Contains the code necessary to configure, produce and log training runs.

#### test_policies.py
Defines the UniformBuilder and PerfectDescriber policies that can be used for debugging.

## Configuration Options

**double_antagonist**: If false, run with a single antagonist agent that receives a single-bit observation from the describer along with the environment. If true, run with two angtagonist agents where one of the two is chosen for each roll out depending on the bit given by the describer.

**test_policies**:  
- **uniform_builder:** If true, fixes the builder's policy to be a uniform sample from the environment class.
- **perfect_describer:** If true, fixes the describer's policy to be that given by the optimal_partition function of the environment class. (So, by editing optimal_partition, you can fix and customise the policy of the describer.)

**environment_class**: A string that determines which environment class will be used as the base for the algorithm. (The string is converted to a environment class using env_string_to_env_class.)

**start_from_checkpoint**:
- **checkpoint_address**: The folder that the checkpoint will be loaded from.
- **iteration_number**: The epoch corresponding to the checkpoint to be loaded.

**iterations**: The number of iterations to train for.

**sample_size**: The number of roll outs to be used when generating logs.
