import pandas as pd
import re
import subprocess
from pathlib import Path
import pyvista as pv
import numpy as np
from ansys.mapdl import reader as pyansys
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- INPUT FILES ---
excel_path = "NeoHookMaterials.xlsx"
template_path = "ds.dat"
ansys_exe_path = "ansys190"

# --- LOAD TEMPLATE ---
with open(template_path, "r") as f:
    template_content = f.read()

# === CLEANERS ===
def to_float(value):
    if isinstance(value, str):
        value = re.sub(r"[^\d.\-eE]", "", value.split("–")[0])
    try:
        return float(value)
    except:
        return None

def clean_filename(name):
    safe_name = re.sub(r'[\\/*?:"<>|()\-]', "_", name)
    safe_name = re.sub(r'\s+', '_', safe_name)
    return safe_name.lower().strip("_")

# === SMART PARAMETER REPLACER ===
def replace_neohook_parameters(text, c1, d1):
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().upper().startswith("TBDATA") and ",1" in line:
            parts = line.split(",")
            if len(parts) >= 4:
                parts[2] = str(c1)
                parts[3] = str(d1)
                new_line = ",".join(parts)
                new_lines.append(new_line)
                continue
        new_lines.append(line)
    return "\n".join(new_lines)

# === LOAD DATA ===
df = pd.read_excel(excel_path)
materials = []
for _, row in df.iterrows():
    materials.append({
        "name": row["Material Name"],
        "c1": to_float(row["C1 (MPa)"]),
        "d1": to_float(row["D1 (1/MPa)"])
    })

output_base_dir = Path("generated_dat_files")
output_base_dir.mkdir(parents=True, exist_ok=True)
generated_folders = []

print("\n📦 Generating .dat files with Neo-Hookean parameters...")
for mat in tqdm(materials, desc="Generating .dat files"):
    print(f"\n  🔧 Replacing values for: {mat['name']}")
    new_content = replace_neohook_parameters(template_content, mat["c1"], mat["d1"])
    folder_name = clean_filename(mat["name"])
    folder_path = output_base_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    output_path = folder_path / f"{folder_name}.dat"
    with open(output_path, "w") as f:
        f.write(new_content)

    generated_folders.append(folder_path)

print(f"\n✅ Done generating {len(generated_folders)} .dat files.\n")

# === RUN ANSYS ===
def run_ansys(ansys_exe_path, folder):
    dat_file = next(folder.glob("*.dat"))
    output_file = folder / "result.rpt"
    command = [ansys_exe_path, "-b", "-i", str(dat_file.name), "-o", str(output_file.name)]
    print(f"\n  🚀 Running ANSYS for: {folder.name}")
    process = subprocess.run(command, cwd=str(folder.resolve()))
    if process.returncode == 0:
        print(f"  ✅ Finished ANSYS for: {folder.name}")
    else:
        print(f"\n  ❌ Error running ANSYS for: {folder.name}")

# === VISUALIZE RESULTS ===
all_max_deformations = {}

def visualize_deformation_and_stress(folder_material):
    rst_file = folder_material / "file.rst"
    if not rst_file.exists():
        print(f"  ⚠️ No .rst file in {folder_material.name}")
        return

    try:
        result = pyansys.read_binary(str(rst_file))
        print(rst_file)
        times = result.time_values
        max_def_list = []
        avg_def_list = []

        for i, t in enumerate(times):
            nnum, displacement = result.nodal_displacement(i)
            if displacement.shape[1] > 3:
                displacement = displacement[:, :3]
            elif displacement.shape[1] < 3:
                print(f"\n  ❌ Not enough displacement components at timestep {i} in {folder_material.name}: {displacement.shape}")
                return

            total_deformation = np.linalg.norm(displacement, axis=1)
            max_def_list.append(total_deformation.max())
            avg_def_list.append(total_deformation.mean())

        # Save summary
        all_max_deformations[folder_material.name] = max_def_list

        # Plot max and average deformation
        plt.figure(figsize=(10, 6))
        plt.plot(times, max_def_list, label="Max Deformation", marker='o')
        plt.plot(times, avg_def_list, label="Avg Deformation", marker='x')
        plt.title(f"Deformation vs Time - {folder_material.name}")
        plt.xlabel("Time")
        plt.ylabel("Total Deformation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(folder_material / f"{folder_material.name}_deformation_plot.png")
        plt.close()

        # Final timestep visualization
        nnum, displacement = result.nodal_displacement(-1)
        if displacement.shape[1] > 3:
            displacement = displacement[:, :3]
        elif displacement.shape[1] < 3:
            print(f"\n  ❌ Invalid displacement shape for {folder_material.name}: {displacement.shape}")
            return

        total_deformation = np.linalg.norm(displacement, axis=1)
        nodes = result.mesh.nodes

        if nodes.shape != displacement.shape:
            print(f"\n  ⚠️ Shape mismatch in {folder_material.name}: nodes {nodes.shape} vs disp {displacement.shape}")
            return

        deformed_nodes = nodes + displacement
        grid = pv.PolyData(deformed_nodes)
        grid.point_data["Total Deformation"] = total_deformation

        # Save deformed PNG (default camera aligned to X)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, scalars="Total Deformation", cmap="viridis", show_edges=False)
        plotter.add_scalar_bar(title="Total Deformation")
        plotter.set_background("white")
        plotter.view_vector((0,0, 1))
        plotter.show(screenshot=str(folder_material / f"{folder_material.name}_deformed.png"))

        # Save VTK 3D file
        grid.save(folder_material / f"{folder_material.name}_deformed.vtp")
        print(f"\n  ✅ Saved plots and 3D file for: {folder_material.name}")

    except Exception as e:
        print(f"\n  ❌ Visualization failed for {folder_material.name}: {e}")

# === EXECUTION ===
for folder in tqdm(generated_folders, desc="Running ANSYS"):
    run_ansys(ansys_exe_path, folder)

for folder in tqdm(generated_folders, desc="Visualizing results"):
    print(f"\n📊 Visualizing: {folder.name}")
    visualize_deformation_and_stress(folder)


# === SUMMARY PLOT ===
plt.figure(figsize=(12, 6))
for name, max_vals in all_max_deformations.items():
    plt.plot(range(len(max_vals)), max_vals, label=name, marker='o')

plt.title("Max Total Deformation Over Time for All Materials")
plt.xlabel("Time Step")
plt.ylabel("Max Total Deformation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_base_dir / "all_materials_max_deformation.png")
plt.close()
print("\n📈 Saved summary graph of max deformation for all materials.")
