# CPU
#BSUB -n 2
#BSUB -R "span[hosts=1]"

# GPU
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA40"

# Queue
#BSUB -q "gpu-compute"

#BSUB -N
#BSUB -J auto_icd
#BSUB -R "rusage[mem=25]"
#BSUB -e VS_err.%J
#BSUB -u k.mugyeommatthew
#BSUB -o VS.%J

singularity exec --nv ../li_et_al/dl_summer_glibcss.sif /bin/bash -c '
        # Activate Conda environment
        source /opt/conda/etc/profile.d/conda.sh
        conda activate base

        nvidia-smi

        # Run script
        python3 main.py
'
