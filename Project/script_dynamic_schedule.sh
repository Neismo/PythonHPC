#!/bin/sh 
### General options 

### -- specify queue -- 
#BSUB -q hpc

### -- set the job Name -- 
#BSUB -J simulate_100_floors_dynamic

### -- ask for number of cores (default: 1) --
#BSUB -n 16

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need X amount of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"

#BSUB -R "select[model == XeonGold6226R]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 01:30

### -- set the email address -- (uncomment and set your email address if needed)
##BSUB -u your_email_address

### -- send notification at start -- (uncomment if needed)
##BSUB -B 

### -- send notification at completion -- (uncomment if needed)
##BSUB -N 

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo simulate_100_floors_dynamic.out 
#BSUB -eo simulate_100_floors_dynamic.err 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

echo "Running on 1 core..."
python -u simulate_dynamic.py 100 1

echo "Running on 2 cores..."
python -u simulate_dynamic.py 100 2

echo "Running on 4 cores..."
python -u simulate_dynamic.py 100 4

echo "Running on 8 cores..."
python -u simulate_dynamic.py 100 8

echo "Running on 16 cores..."
python -u simulate_dynamic.py 100 16