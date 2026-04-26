#!/bin/sh 
### General options 

### -- specify queue -- 
#BSUB -q hpc

### -- set the job Name -- 
#BSUB -J simulate_20_floors_static

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need X amount of memory per core/slot -- 
#BSUB -R "rusage[mem=64MB]"

#BSUB -R "select[model == XeonGold6226R]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 00:30

### -- set the email address -- (uncomment and set your email address if needed)
##BSUB -u your_email_address

### -- send notification at start -- (uncomment if needed)
##BSUB -B 

### -- send notification at completion -- (uncomment if needed)
##BSUB -N 

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo simulate_20_floors_static.out 
#BSUB -eo simulate_20_floors_static.err 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python -u simulate.py 20 4