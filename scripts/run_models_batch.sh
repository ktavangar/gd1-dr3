#!/bin/bash

# Activate the Python virtual environment
source stream-env/bin/activate

# Define combinations of args

# Define arrays of possible values for arg1, arg2, and arg3
arg1_values=("[30,30,30]" "[40,40,40]" "[20,20,20]")
arg2_values=("[5,15,15,15]" "[5,20,20,20]" "[5,10,10,10]" "[5,25,25,25]" "[10,10,10,10]" "[10,15,15,15]" "[10,20,20,20]" "[5,5,5,5]")
arg3_values=("[6,2]" "[4,1.5]" "[3,1]" "[1.5,0.5]" "[2,1]" "[1,1]")

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
  echo "Running command: python script.py '${args[0]}' '${args[1]}' '${args[2]}'"
  python scripts/run_full_gd1_model.py "${args[0]}" "${args[1]}" "${args[2]}"
}
export -f run_script

# Run in parallel using xargs
printf "%s\n" "${combinations[@]}" | xargs -n 3 -P 6 -I {} bash -c 'run_script "{}"'

deactivate
