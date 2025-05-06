# ANSYS Neo-Hookean Simulation Pipeline

This Python-based pipeline automates the simulation, result extraction, and visualization of Neo-Hookean hyperelastic materials in ANSYS. It is modularly designed to support additional material models, analysis types (e.g. thermal, modal), and output processing in future use cases.

---

## 📦 Features

- Reads material parameters (`C1`, `D1`) from an Excel sheet
- Auto-generates `.dat` input files for ANSYS
- Batch-runs ANSYS simulations using `subprocess`
- Extracts results from `.rst` files using `ansys-mapdl-reader`
- Plots per-material deformation vs time
- Saves 3D views and geometry of deformed shapes
- Summarizes maximum deformations across all materials

---

## ⚙️ Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

**Minimum Python version:** 3.8  
**Key packages:**
- `ansys-mapdl-reader`
- `pyvista`
- `matplotlib`
- `pandas`
- `tqdm`

---

### 2. Prepare your material data

Create an Excel file (e.g. `NeoHookMaterials.xlsx`) with the following columns:

| Material Name       | C1 (MPa) | D1 (1/MPa) |
|---------------------|----------|------------|
| Natural Rubber       | 0.35     | 0.005      |
| Silicone Rubber      | 0.80     | 0.002      |
| ...                 | ...      | ...        |

---

### 3. Edit the ANSYS `.dat` template

Create a `ds.dat` file that contains a Neo-Hookean definition like:

```ansys
TB,HYPE,1,1,1,NEO
TBDATA,1,1.5,0.026
```

> ⚠️ These values (`1.5`, `0.026`) will be replaced by the script automatically based on the Excel input.

---

### 4. Run the script

```bash
python main.py
```

You can customize:
- `excel_path`: path to your Excel file
- `template_path`: path to your `.dat` template
- `ansys_exe_path`: executable name or path for your ANSYS install (e.g. `ansys190`)

---

## 🧪 Output Files

For each material:
- `<material>.dat`: input file for ANSYS
- `result.rpt`: output log from simulation
- `file.rst`: result file (binary)
- `<material>_deformation_plot.png`: deformation vs time
- `<material>_deformed.png`: 3D view of final deformation
- `<material>_deformed.vtp`: VTK geometry for 3D inspection

In the root output folder:
- `all_materials_max_deformation.png`: comparison of max deformation across materials

---

## 🛠️ Engineering Applications

This pipeline can be used in:
- Rubber-like material design and testing
- Soft robotics simulation and optimization
- Biomedical material modeling (e.g., PDMS, Ecoflex)
- Comparative studies of hyperelastic performance
- Parameter sweeps for material fitting

---

## 📚 References

### ANSYS Documentation
- [ANSYS MAPDL Commands Reference](https://ansyshelp.ansys.com/)
- [Writing Input Files (`.dat`)](https://ansyshelp.ansys.com/account/secured?returnurl=/Views/Secured/corp/v201/en/ans_elem/Hlp_E_Elements1.html)
- [Reading `.rst` Files with PyAnsys](https://docs.pyansys.com/reader/)

---

## 📂 Repository Structure

```
project/
├── NeoHookMaterials.xlsx       # Input materials
├── ds.dat                      # ANSYS template
├── generated_dat_files/        # Output per-material folders
│   └── <material>/
│       ├── *.dat, *.rst, *.rpt
│       ├── *_deformed.png/.vtp
│       └── *_deformation_plot.png
├── main.py                     # Main script
├── requirements.txt
└── README.md
```

---

## 🔄 Extending the Project

You can easily adapt this framework to:
- Use different material models (e.g. Mooney-Rivlin, Ogden)
- Switch analysis type (thermal, modal, static structural)
- Add support for strain/stress extraction, animation, or optimization

Feel free to fork and expand.

---

## 🧑‍💻 Author

Cioconea Adina Mariana
License: MIT
