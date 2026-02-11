#!/bin/sh 
### General options 

### -- specify queue -- 
#BSUB -q hpc

### -- set the job Name -- 
#BSUB -J simplest

### -- ask for number of cores (default: 1) --
#BSUB -n 16

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need X amount of memory per core/slot -- 
#BSUB -R "rusage[mem=512MB]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 00:02

### -- set the email address -- (uncomment and set your email address if needed)
##BSUB -u your_email_address

### -- send notification at start -- (uncomment if needed)
##BSUB -B 

### -- send notification at completion -- (uncomment if needed)
##BSUB -N 

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o output_%J.out 
#BSUB -e output_%J.err 

/bin/sleep 60