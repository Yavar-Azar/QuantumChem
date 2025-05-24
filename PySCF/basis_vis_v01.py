import numpy as np
import pyvista as pv
from pyscf import gto
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import cpk_colors
from itertools import combinations


import argparse

# --- Optional command-line argument parser ---
parser = argparse.ArgumentParser(description="Visualize atomic basis functions (AOs) from a molecule.xyz file")
parser.add_argument("--xyz", type=str, default="molecule.xyz", help="Path to XYZ file (default: molecule.xyz)")
args = parser.parse_args()

xyz_filename = args.xyz
# --- Constants ---
BOHR = 1.8897259886
cpk_color_dict = {Z: tuple(rgb) for Z, rgb in enumerate(cpk_colors)}

# --- CONFIGURATION ---

# --- Step 1: Read XYZ file ---
with open(xyz_filename) as f:
    lines = f.readlines()[2:]
atoms = [line.strip().split() for line in lines]
atom_str = '; '.join(f"{el} {x} {y} {z}" for el, x, y, z in atoms)
symbols = [el for el, *_ in atoms]

# --- Step 2: Build PySCF molecule ---
mol = gto.M(atom=atom_str, basis='6-31G*')
mol.build()

# --- Use positions from PySCF (already in Bohr) ---
positions = mol.atom_coords()

# --- Get AO labels ---
ao_labels = mol.ao_labels(fmt=True)

# --- Step 3: Create adaptive 3D grid around molecule ---
margin = 3.0  # Bohr
positions_array = np.array(positions)
min_corner = positions_array.min(axis=0) - margin
max_corner = positions_array.max(axis=0) + margin

nx = ny = nz = 100
x = np.linspace(min_corner[0], max_corner[0], nx)
y = np.linspace(min_corner[1], max_corner[1], ny)
z = np.linspace(min_corner[2], max_corner[2], nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# --- Step 4: Evaluate basis functions (AOs) ---
ao_values = mol.eval_gto("GTOval", coords)

# --- AO Visualizer ---
class AOVisualizer:
    def __init__(self, ao_values, grid, plotter):
        self.ao_values = ao_values
        self.grid = grid
        self.plotter = plotter
        self.ao_index = 7
        self.iso_level = 0.05
        self.meshes = []
        self.update()

    def update(self):
        for m in self.meshes:
            self.plotter.remove_actor(m)
        self.meshes.clear()

        values = self.ao_values[:, self.ao_index].reshape((nx, ny, nz))
        self.grid.cell_data["basis"] = values.flatten(order="F")
        self.grid = self.grid.cell_data_to_point_data()

        vmin, vmax = values.min(), values.max()
        level = self.iso_level * max(abs(vmin), abs(vmax))

        contours_pos = self.grid.contour([+level], scalars="basis")
        if contours_pos.n_points > 0:
            self.meshes.append(self.plotter.add_mesh(contours_pos, color="red", opacity=0.6))

        contours_neg = self.grid.contour([-level], scalars="basis")
        if contours_neg.n_points > 0:
            self.meshes.append(self.plotter.add_mesh(contours_neg, color="blue", opacity=0.6))

        self.plotter.add_title(f"AO #{self.ao_index}: {ao_labels[self.ao_index]}", font_size=10)

    def set_iso_level(self, value):
        self.iso_level = value
        self.update()

    def set_ao_index(self, value):
        self.ao_index = int(value)
        self.update()

# --- Step 6: Setup PyVista volume grid ---
grid = pv.ImageData()
grid.dimensions = np.array([nx, ny, nz]) + 1
grid.origin = tuple(min_corner)
grid.spacing = tuple((max_corner - min_corner) / np.array([nx, ny, nz]))

# --- Step 7: Plotting ---
plotter = pv.Plotter()
visualizer = AOVisualizer(ao_values, grid, plotter)

# --- Step 8: Add atoms ---
for symbol, pos in zip(symbols, positions):
    Z = atomic_numbers[symbol]
    radius = covalent_radii[Z] * 0.65   # convert to Bohr
    color = cpk_color_dict.get(Z, (1.0, 1.0, 1.0))
    sphere = pv.Sphere(radius=radius, center=pos, theta_resolution=64, phi_resolution=64)
    plotter.add_mesh(sphere, color=color, specular=0.6)

# --- Step 9: Add bonds ---
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2)) / BOHR  # in Å for comparison

for i, j in combinations(range(len(positions)), 2):
    Z1, Z2 = atomic_numbers[symbols[i]], atomic_numbers[symbols[j]]
    r1, r2 = covalent_radii[Z1], covalent_radii[Z2]
    max_bond = r1 + r2 + 0.3  # Å
    if distance(positions[i], positions[j]) <= max_bond:
        p1, p2 = positions[i], positions[j]
        center = (np.array(p1) + np.array(p2)) / 2
        direction = np.array(p2) - np.array(p1)
        height = np.linalg.norm(direction)
        direction /= height
        cylinder = pv.Cylinder(center=center, direction=direction, radius=0.12, height=height, resolution=64)
        plotter.add_mesh(cylinder, color="gray", specular=0.3)

# --- Step 10: Add sliders ---
plotter.add_slider_widget(
    callback=lambda value: visualizer.set_ao_index(int(value)),
    rng=[0, ao_values.shape[1] - 1],
    value=8,
    title="Basis Function Index",
    pointa=(0.025, 0.1),
    pointb=(0.31, 0.1),
    style="modern",
    fmt="%.0f"
)

plotter.add_slider_widget(
    callback=lambda value: visualizer.set_iso_level(value),
    rng=[0.01, 0.8],
    value=0.05,
    title="Isosurface Level",
    pointa=(0.35, 0.1),
    pointb=(0.64, 0.1),
    style="modern",
)

# --- Final view ---
plotter.add_axes()
plotter.show()
