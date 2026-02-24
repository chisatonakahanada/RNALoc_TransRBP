#!/bin/bash

timestamp=$(date +%m%d_%H%M%S)

# create log_dir
log_dir="log/run_${timestamp}"
mkdir -p "$log_dir"

# script
main_script="main.py"

# output
log_file="${log_dir}/output.log"

# run command
echo "Running with logs in ${log_dir}"
export CUDA_VISIBLE_DEVICES=1
nohup python $main_script --log_dir $log_dir > "$log_file" 2>&1 &
