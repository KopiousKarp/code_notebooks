#!/bin/bash
# Iterate over all subdirectories in the current directory
for dir in */; do
    # Remove trailing slash
    dir=${dir%/}
    
    echo "Processing directory: $dir"
    
    # Use a subshell to change directory and run the script
    (
        cd "$dir" || exit
        
        # Source the Python environment
        source /opt/sam2_env/bin/activate
        
        if [ -f "experiment.py" ]; then
            echo "Running experiment.py in $dir"
            python experiment.py
        else
            echo "experiment.py not found in $dir"
        fi
    )
done

