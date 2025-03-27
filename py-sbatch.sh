#!/bin/bash

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#



NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
NODE_NAME="gipdeep10"
JOB_NAME=$(basename $1 .py)
MAIL_USER="itamarp@campus.technion.ac.il"
MAIL_TYPE=END,FAIL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL


# Default scans value (in case --scans is not provided)
SCANS_VALUE=""

# Parse command-line arguments to find --scans value
for arg in "$@"; do
    if [[ $arg == --scans=* ]]; then
        SCANS_VALUE="${arg#--scans=}" # Extract value after "--scans="
    fi
done

# Append scans value to job name
JOB_NAME="${JOB_NAME}_scans${SCANS_VALUE}"


###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=gs2mesh

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	-w $NODE_NAME \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh

conda deactivate
conda activate $CONDA_ENV

# Run python with the args to the script
python $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

