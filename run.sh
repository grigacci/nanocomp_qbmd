#!/bin/bash

# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# Set the executable path relative to the script directory
EXE_PATH="${SCRIPT_DIR}/simu.exe"

# Check if enough arguments are provided
if [ $# -lt 8 ]; then
    echo "Usage: $0 <arg1> <arg2> <arg3> <arg4> <arg5> <arg6> <arg7> <output_dir>"
    echo "Example: $0 5 2.0d0 7.0d0 2.5d0 1 2.0d0 7.0d0 /path/to/output"
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

# Use the output directory provided by Python (8th argument)
OUTPUT_DIR=$(realpath -m "$8")

# Ensure trailing slash
if [[ "${OUTPUT_DIR}" != */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/"
fi

# Create the directory if it doesn't exist (remove trailing slash for mkdir)
MKDIR_PATH="${OUTPUT_DIR%/}"
echo "Creating directory: $MKDIR_PATH"
mkdir -p "$MKDIR_PATH"

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
