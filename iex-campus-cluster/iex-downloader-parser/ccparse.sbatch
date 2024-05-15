#!/bin/bash
##values are default unless otherwise noted
#SBATCH --time=02:00:10	
#SBATCH --account="24sp-ie421onl-eng"   ##custom for class
#SBATCH --ntasks=16
#SBATCH --nodes=8
#SBATCH --cpus-per-task=4
#SBATCH --output="LOG.log"
#SBATCH --job-name="iex_parser" 
#SBATCH --partition=secondary	       ## no access to primary
########## Run your executable ######################
## Load required modules (Step 3 has been done)
module load anaconda3/2024.02-Feb
source activate my.anaconda
conda install tqdm
conda install requests
conda install pytz
## Run script
bash download.sh
bash parse_all.sh