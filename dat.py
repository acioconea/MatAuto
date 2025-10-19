from time import sleep

import pandas as pd
import re
import subprocess
from pathlib import Path
import pyvista as pv
import numpy as np
from ansys.mapdl import reader as pyansys
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

def clean_pyvista():
    import pyvista as pv
    pv.plotting.plotting._ALL_PLOTTERS.clear()
    pv.global_theme.restore_defaults()


def clean_ansys_reader(result=None):
    """Forcefully cleans up PyAnsys reader and memory."""
    if result is not None:
        del result
    pv.global_theme.restore_defaults()  # reset PyVista config (optional)
    gc.collect()
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


def visualize_deformation(folder_material):
    rst_files = list(folder_material.glob("*.rst"))
    if not rst_files:
        print(f"❌ No .rst file found in {folder_material}")
        return
    rst_file = rst_files[0]  # or loop if needed

    try:
        result = pyansys.read_binary(str(rst_file))
        times=[]
        times = result.time_values
        max_def_list = []
        avg_def_list = []

        # Loop over time steps to extract deformation
        for i in range(len(times)):
            _, displacement = result.nodal_displacement(i)

            if displacement.shape[1] >= 3:
                displacement = displacement[:, :3]
            else:
                print(
                    f"  ❌ Not enough displacement components at timestep {i} in {folder_material.name}: {displacement.shape}")
                return

            total_deformation = np.linalg.norm(displacement, axis=1)
            max_def_list.append(total_deformation.max())
            avg_def_list.append(total_deformation.mean())

        # Save summary deformation to global dict if needed
        if 'all_max_deformations' in globals():
            all_max_deformations[folder_material.name] = max_def_list

        # === Plot deformation vs time ===
        print(max_def_list,rst_file)

        plt.figure(figsize=(10, 6))
        plt.plot(times, all_max_deformations[folder_material.name], label="Max Deformation", marker='o')
        plt.plot(times, avg_def_list, label="Avg Deformation", marker='x')
        plt.title(f"Deformation vs Time - {folder_material.name}")
        plt.xlabel("Time")
        plt.ylabel("Total Deformation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(folder_material / f"{folder_material.name}_deformation_plot.png")
        plt.figure()  # create new figure
        plt.close('all')  # ensure all figures are dropped
        sleep(1)

        # === Final time step: 3D visualization ===
        _, displacement = result.nodal_displacement(-1)
        displacement = displacement[:, :3]  # Enforce 3D

        nodes = result.mesh.nodes
        if nodes.shape != displacement.shape:
            print(f"  ⚠️ Shape mismatch in {folder_material.name}: nodes {nodes.shape} vs disp {displacement.shape}")
            return

        total_deformation = np.linalg.norm(displacement, axis=1)
        deformed_nodes = nodes + displacement

        grid = pv.PolyData(deformed_nodes)
        grid.point_data["Total Deformation"] = total_deformation

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, scalars="Total Deformation", cmap="viridis", show_edges=False)
        plotter.add_scalar_bar(title="Total Deformation")
        plotter.set_background("white")
        plotter.view_vector((0, 0, 1))
        plotter.show(screenshot=str(folder_material / f"{folder_material.name}_deformed.png"))
        plotter.close()

        grid.save(folder_material / f"{folder_material.name}_deformed.vtp")
        print(f"  ✅ Saved plots and 3D view for: {folder_material.name}")

    except Exception as e:
        print(f"  ❌ Visualization failed for {folder_material.name}: {e}")

def save_deformation_plot(folder):
    import matplotlib.pyplot as plt
    import numpy as np
    from ansys.mapdl.reader import read_binary

    # Prepare output folder
    plot_folder = folder / "plots"
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Find the .rst file
    rst_files = list(folder.glob("*.rst"))
    if not rst_files:
        print(f"❌ No .rst file found in {folder}")
        return

    rst_file = rst_files[0]
    try:

        result = read_binary(str(rst_file))
        times = result.time_values
        max_def_list = []
        avg_def_list = []

        for i in range(len(times)):
            _, displacement = result.nodal_displacement(i)
            if displacement.shape[1] >= 3:
                displacement = displacement[:, :3]
            else:
                print(f"❌ Invalid displacement shape at timestep {i} in {folder.name}")
                return
            total_deformation = np.linalg.norm(displacement, axis=1)
            max_def_list.append(total_deformation.max())
            avg_def_list.append(total_deformation.mean())

        # Plot and save to plots folder
        plt.figure(figsize=(10, 6))
        plt.plot(times, max_def_list, label="Max Deformation", marker='o')
        plt.plot(times, avg_def_list, label="Avg Deformation", marker='x')
        plt.title(f"Deformation vs Time - {folder.name}")
        plt.xlabel("Time")
        plt.ylabel("Total Deformation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = plot_folder / f"{folder.name}_deformation_plot.png"
        plt.savefig(save_path)
        plt.close()

        print(f"📈 Saved deformation plot to {save_path}")

        # Optionally return values
        return max_def_list
        clean_ansys_reader(result)
    except Exception as e:
        print(f"❌ Failed to save deformation plot for {folder.name}: {e}")
        return None


sleep(3)

# === EXECUTION ===
# for folder in tqdm(generated_folders, desc="Running ANSYS"):
#     run_ansys(ansys_exe_path, folder)
#
# for folder in generated_folders:
#     print(f"\n📊 Visualizing: {folder.name}")
#     save_deformation_plot(folder)
#     sleep(0.1)

def get_max_deformation_at_final_step(folder):
    import numpy as np
    from ansys.mapdl.reader import read_binary

    rst_files = list(folder.glob("*.rst"))
    if not rst_files:
        print(f"❌ No .rst file found in {folder}")
        return None

    rst_file = rst_files[0]
    try:
        result = read_binary(str(rst_file))
        _, displacement = result.nodal_displacement(-1)  # Last time step

        if displacement.shape[1] >= 3:
            displacement = displacement[:, :3]
        else:
            print(f"❌ Displacement shape invalid for {folder.name}")
            return None

        total_deformation = np.linalg.norm(displacement, axis=1)
        max_deformation = total_deformation.max()

        print(f"📏 Max deformation at final step for {folder.name}: {max_deformation:.6f}")
        return max_deformation

    except Exception as e:
        print(f"❌ Error processing {folder.name}: {e}")
        return None
final_deformations = {}

for folder in generated_folders:
    max_def = get_max_deformation_at_final_step(folder)
    if max_def is not None:
        final_deformations[folder.name] = max_def

import pandas as pd

df = pd.DataFrame(list(final_deformations.items()), columns=["Material", "MaxDeformation_FinalStep"])
df.to_csv("final_max_deformations.csv", index=False)
print("✅ Saved max deformations to final_max_deformations.csv")



# === SUMMARY PLOT ===
plt.figure(figsize=(12, 6))
for name, max_vals in all_max_deformations.items():
    plt.plot(range(len(max_vals)), max_vals, label=name, marker='o')
print(all_max_deformations)
plt.title("Max Total Deformation Over Time for All Materials")
plt.xlabel("Time Step")
plt.ylabel("Max Total Deformation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_base_dir / "all_materials_max_deformation.png")
plt.close()
print("\n📈 Saved summary graph of max deformation for all materials.")
