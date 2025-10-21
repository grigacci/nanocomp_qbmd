#!/bin/bash

# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# Set the executable path relative to the script directory
EXE_PATH="${SCRIPT_DIR}/simu.exe"

# Check if enough arguments are provided
if [ $# -lt 7 ]; then
    echo "Usage: $0 <arg1> <arg2> <arg3> <arg4> <arg5> <arg6> <arg7> [base_output_dir]"
    echo "Example: $0 5 2.0d0 7.0d0 2.5d0 1 2.0d0 7.0d0"
    exit 1
fi

# Check if executable exists
if [ ! -f "$EXE_PATH" ]; then
    echo "Error: Executable not found at $EXE_PATH"
    exit 1
fi

# Store arguments
ARG1=$1
ARG2=$2
ARG3=$3
ARG4=$4
ARG5=$5
ARG6=$6
ARG7=$7

# Base output directory (default to temp_files relative to script location, or use 8th argument)
BASE_OUTPUT_DIR="${8:-${SCRIPT_DIR}/runs}"

# Convert to absolute path
BASE_OUTPUT_DIR=$(realpath -m "$BASE_OUTPUT_DIR")

# Generate folder name from arguments
# Format: 05x02.0_07.0__02.5__01x02.0_07.0
FOLDER_NAME=$(printf "%02dx%s_%s__%s__%02dx%s_%s" \
    "$ARG1" \
    "$(echo $ARG2 | sed 's/d0//')" \
    "$(echo $ARG3 | sed 's/d0//')" \
    "$(echo $ARG4 | sed 's/d0//')" \
    "$ARG5" \
    "$(echo $ARG6 | sed 's/d0//')" \
    "$(echo $ARG7 | sed 's/d0//')")

# Create full output path with trailing slash
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${FOLDER_NAME}/"

# Create the directory if it doesn't exist
echo "Creating directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if directory was created successfully
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Failed to create directory $OUTPUT_DIR"
    exit 1
fi

# Run the executable with arguments
echo "Running simulation..."
echo "Executable: $EXE_PATH"
echo "Command: $EXE_PATH $ARG1 $ARG2 $ARG3 $ARG4 $ARG5 $ARG6 $ARG7 \"$OUTPUT_DIR\""

"$EXE_PATH" "$ARG1" "$ARG2" "$ARG3" "$ARG4" "$ARG5" "$ARG6" "$ARG7" "$OUTPUT_DIR"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Simulation completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
else
    echo "Error: Simulation failed with exit code $?"
    exit 1
fi

