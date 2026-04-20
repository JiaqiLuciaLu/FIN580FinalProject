#!/bin/bash
#SBATCH --job-name=myjob            # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G            # memory per cpu-core (4G is default)
#SBATCH --time=00:01:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2025.6
conda activate pytools-env

python myscript.py


################ multithread ####################
#!/bin/bash
#SBATCH --job-name=multithread      # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G            # memory per cpu-core (4G is default)
#SBATCH --time=00:15:00             # maximum time needed (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=<YourNetID>@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge
module load matlab/R2025a

matlab -nodisplay -nosplash -r for_loop