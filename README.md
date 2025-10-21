# E-Mulate Simulator

Fortran-based superlattice eigenstates and photocurrent simulator.

## Setup

Make the script and executable files executable:

```
chmod +x run.sh
chmod +x simu.exe
```


## Usage

Run the simulation with the following command:

```
./run.sh 5 2.0d0 7.0d0 2.5d0 1 2.0d0 7.0d0
```


### Parameters

The script accepts 7 required arguments and 1 optional argument:
- Arguments 1-7: Simulation parameters
- Argument 8 (optional): Custom output directory path

The script automatically creates an output directory based on the input parameters and runs the simulation with absolute paths.

## Output

Simulation results are saved to `/temp_files/` with automatically generated folder names based on input parameters.

