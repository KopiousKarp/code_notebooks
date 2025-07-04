#!/bin/bash
# Iterate over all subdirectories in the current directory
for dir in */; do
    # Remove trailing slash
    dir=${dir%/}
    
    echo "Processing directory: $dir"
    # Declare associative arrays to track experiment results
    declare -A roots
    declare -A annFiles
    roots["2021_exclude"]='/work/2022_annot/images /work/2023_annot/images /work/2024_annot/images'
    annFiles["2021_exclude"]='/work/2022_annot/2022_annotations.json /work/2023_annot/2023_annotations_corrected.json /work/2024_annot/2024_annotations.json'                       
    roots["2022_exclude"]='/work/2021_annot/images /work/2023_annot/images /work/2024_annot/images'
    annFiles["2022_exclude"]='/work/2021_annot/2021_annotations.json /work/2023_annot/2023_annotations_corrected.json /work/2024_annot/2024_annotations.json'
    roots["2023_exclude"]='/work/2021_annot/images /work/2022_annot/images /work/2024_annot/images'
    annFiles["2023_exclude"]='/work/2021_annot/2021_annotations.json /work/2022_annot/2022_annotations.json /work/2024_annot/2024_annotations.json'
    roots["2024_exclude"]='/work/2021_annot/images /work/2022_annot/images /work/2023_annot/images'
    annFiles["2024_exclude"]='/work/2021_annot/2021_annotations.json /work/2022_annot/2022_annotations.json /work/2023_annot/annotations_corrected.json'   
    roots["2021_solo"]='/work/2021_annot/images'
    annFiles["2021_solo"]='/work/2021_annot/2021_annotations.json'
    roots["2022_solo"]='/work/2022_annot/images'
    annFiles["2022_solo"]='/work/2022_annot/2022_annotations.json'
    roots["2023_solo"]='/work/2023_annot/images'
    annFiles["2023_solo"]='/work/2023_annot/annotations_corrected.json'
    roots["2024_solo"]='/work/2024_annot/images'
    annFiles["2024_solo"]='/work/2024_annot/2024_annotations.json'
    roots["Multi_year"]='/work/2021_annot/images /work/2022_annot/images /work/2023_annot/images /work/2024_annot/images'
    annFiles["Multi_year"]='/work/2021_annot/2021_annotations.json /work/2022_annot/2022_annotations.json /work/2023_annot/2023_annotations_corrected.json /work/2024_annot/2024_annotations.json'
    # Record start time
    start_time=$(date +%s)
    # Use a subshell to change directory and run the script
    (
        cd "$dir" || exit
        # Get the current directory name without the full path
        current_dir=$(basename "$(pwd)")
        echo "Checking accessibility of annotation files: ${annFiles[$current_dir]}"
        # For single file paths, this will list the file. For multiple, it will list all.
        # If a file is not found, ls will output an error for that specific file.
        ls -l ${annFiles[$current_dir]} 
        echo "Checking accessibility of root directories: ${roots[$current_dir]}"
        ls -ld ${roots[$current_dir]}
        # Source the Python environment
        source /opt/sam2_env/bin/activate
        
        if [ -f "Kfold_CV_experiment.py" ]; then
            echo "Running experiment .py in $dir"
            python Kfold_CV_experiment.py \
                --coco_roots ${roots[$current_dir]} \
                --coco_annFiles ${annFiles[$current_dir]} 
        else
            echo "experiment.py not found in $dir"
        fi
    )
done

