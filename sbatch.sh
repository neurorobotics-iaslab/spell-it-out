#!/bin/bash
#SBATCH --job-name=spellitout_AMICA
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-46%5 # set it to 0-49 when you have all the subjects
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tommaso.cortecchia@studenti.unipd.it

WORK_DIR=/home/cortecchia/spellitout_ws
DATA_DIR=/nfsd/iaslab4/Users/cortecchia/ 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MNE_LOGGING_LEVEL=WARNING
export MNE_DATASETS_SPELLITOUT_DATA_PATH=$DATA_DIR/spellitout

source $(conda info --base)/etc/profile.d/conda.sh
conda activate eeg

# Generate arrays of valid parameter combinations
TASKS=()
SUBJECTS=()
for task in 'silent' 'overt' ; do
    for subject in {1..25}; do
        if [[ "$task" == "overt" && ( "$subject" == "16" || "$subject" == "24" || "$subject" == "25" ) ]]; then
            continue
        fi
        TASKS+=("$task")
        SUBJECTS+=("$subject")
    done
done

CURRENT_TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
CURRENT_SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================================="
echo "Starting Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Processing Task: $CURRENT_TASK | Subject: $CURRENT_SUBJECT"
echo "Assigned Node: $SLURMD_NODENAME"
echo "=========================================================="

cd $DATA_DIR


# Execute the script for this specific job's combination
python $WORK_DIR/scripts/compute_AMICA.py \
    --condition "$CURRENT_TASK" \
    --subjects "$CURRENT_SUBJECT" \
    --overwrite \
    --derivatives-path $DATA_DIR/derivatives/preprocessing

echo "Job completed successfully."
