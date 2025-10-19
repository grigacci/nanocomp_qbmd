# Superlattice Eigensates & Photocurrent Simulation 

This repository contains the compiled **Fortran** executable only for linux
`20240825_Superlattice_Eigensates_Photocurrent_inputs.exe` for simulating **quantum superlattices** 
and calculating **eigenstates and photocurrent** in QBMD.

---

## 📁 Repository Contents

- `fortran_files/20240825_Superlattice_Eigensates_Photocurrent_inputs.exe` – Main executable.  
- `temp_files/` folder – Example folder where simulation results can be saved.  

---

## 1️⃣ Running the Program

To run on **Linux**, open a terminal and execute:

```
chmod +x /complete_path/e-mulate/linux_executable/fortran_files/20240825_Superlattice_Eigensates_Photocurrent_inputs.exe
```

chmod is a Unix/Linux command used to change file permissions.
+x adds the "executable" permission to the file.


### Required Arguments

The program requires arguments in the following order:

| Order | Argument   | Description |
|-------|------------|------------|
| 1     | RW         | Number of wells on the left |
| 2     | RQWt       | Thickness of the left quantum wells (nm) |
| 3     | RQBt       | Thickness of the left barriers (nm) |
| 4     | MQWt       | Thickness of the main quantum well (nm) |
| 5     | LW         | Number of wells on the right |
| 6     | LQWt       | Thickness of the right quantum wells (nm) |
| 7     | LQBt       | Thickness of the right barriers (nm) |
| 8     | out_folder | Output folder where `.txt` files will be saved. Must be enclosed in `""` and end with `/`. |

**it is important to put the out_folder complete, not a relative path!**

**Important Comments:**

- The executable was compiled from **Fortran** code and expects parameters via the command line.  
- The output folder must exist, or the program will create it automatically if it has permissions.  

---

## 2️⃣ Value Format

- **Number of wells:** use simple integers (`1`, `2`, `3`, …).  
- **Thicknesses (nm):** use **double precision** in Fortran, with the suffix `d0`.

Examples:



- 2.0d0 → 2 nm
- 2.5d0 → 2.5 nm
- 10.1d0 → 10.1 nm


**Explanation:**

- The `d0` format ensures **numerical precision** in the simulations, avoiding rounding errors typical of single precision.  

---

## 3️⃣ Simulation Organization

It is recommended to create a **separate folder for each simulation**, named according to the parameters:

**05x02.0_07.0__02.5__01x02.0_07.0/**


### Folder name format:


```
05x02.0_07.0__02.5__01x02.0_07.0/
RWxRQWt_RQBt__MQWt__LWxLQWt_LQBt/
```

**Comments:**

- This organization allows each simulation to have its own folder and keeps results organized.  
- Useful for later analysis or comparison between simulations.  

---

## 4️⃣ Example Execution


/complete_path/e-mulate/linux_executable/fortran_files/20240825_Superlattice_Eigensates_Photocurrent_inputs.exe 5 2.0d0 7.0d0 2.5d0 1 2.0d0 7.0d0 "/complete_path/e-mulate/linux_executable/temp_files/05x02.0_07.0__02.5__01x02.0_07.0/"


**Comments:**

- During execution, the program will automatically create `.txt` files in the specified folder.  
- The files contain the **eigenstates and photocurrent data** resulting from the simulation.  

---

## 5️⃣ Important Notes

- The `d0` format is mandatory for numerical precision.  
- Quotes `""` and the `/` at the end of the folder path are required for the program to recognize the path correctly, even if it contains spaces or special characters.  
- Ensure the thickness values are consistent with the physical system being simulated.  

---

## 6️⃣ Recommendations

- Always create a separate folder for each simulation to keep files organized.  
- Use descriptive folder names based on the parameters.  
- Check that parameters are within the appropriate physical range before running the program.  

---


*This README is fully in Markdown, including instructions, explanations, examples, and comments about simulation organization.*