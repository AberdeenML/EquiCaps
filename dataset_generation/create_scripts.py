import os

num_chunks = 12

base_script = """#!/bin/bash --login
#SBATCH --partition=PARTITION 
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-task=N_CPUs       # Number of cores
#SBATCH --mem=MEM                    # Memory pool for all cores
#SBATCH -o slurm.%j.out              # STDOUT
#SBATCH -e slurm.%j.err              # STDERR
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@90

blenderproc run main_3DIEBench_T_generation.py --models-path "path/to/models" \\
                              --output-dir "path/to/dataset/output_directory/" \\
                              --objects ./all_objects/all_objects_chunk_{chunk}.npy \\
                              --views-per-object 50 --image-size 256 --seed {seed}
"""

output_dir = "./scripts"
os.makedirs(output_dir, exist_ok=True)

chunk = 0
for chunk in range(num_chunks):
    script_content = base_script.format(chunk=chunk, seed=chunk)

    script_path = os.path.join(output_dir, f"run_chunk_{chunk}.sh")
    with open(script_path, "w") as script_file:
        script_file.write(script_content)
