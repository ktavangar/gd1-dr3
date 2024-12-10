#!/bin/bash

# Activate the Python virtual environment
source stream-env/bin/activate

# Define combinations of args

# Define arrays of possible values for arg1, arg2, and arg3
arg1_values=("[40,40,40]" "[30,30,30]") #"[20,20,20]"
arg2_values=("[5,10,10,10]" "[5,15,15,15]" "[5,20,20,20]"  "[5,24,24,24]") # "[10,10,10,10]" "[10,15,15,15]" "[10,20,20,20]" "[10,24,24,24]")
arg3_values=("[4,1]" "[2,0.5]" "[1,0.25]") # "[2,1]" "[1,1]")

combinations=()
for arg1 in "${arg1_values[@]}"; do
  for arg2 in "${arg2_values[@]}"; do
    for arg3 in "${arg3_values[@]}"; do
      combinations+=("$arg1 $arg2 $arg3")
    done
  done
done

# Export function to run the script
run_script() {
  local args=($1)
  echo "Running command: python scripts/run_full_gd1_model.py '${args[0]}' '${args[1]}' '${args[2]}'"
  python scripts/run_full_gd1_model.py "${args[0]}" "${args[1]}" "${args[2]}"
}
export -f run_script

# Run in parallel using xargs
printf "%s\n" "${combinations[@]}" | xargs -n 3 -P 4 -I {} bash -c 'run_script "{}"'

deactivate
