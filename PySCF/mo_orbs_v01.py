import numpy as np
import pyvista as pv
from pyscf import gto, dft
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import cpk_colors
from itertools import combinations
import argparse

# --- Optional CLI Argument Parser ---
parser = argparse.ArgumentParser(description="Molecular Orbital Visualization")
parser.add_argument("--xyz", type=str, default="molecule.xyz", help="Path to XYZ file (default: molecule.xyz)")
args = parser.parse_args()
xyz_filename = args.xyz

# --- Constants ---
BOHR = 1.8897259886
cpk_color_dict = {Z: tuple(rgb) for Z, rgb in enumerate(cpk_colors)}

# --- Read XYZ file ---
with open(xyz_filename) as f:
    lines = f.readlines()[2:]
atoms = [line.strip().split() for line in lines]
symbols = [line[0] for line in atoms]
atom_str = '; '.join(f"{el} {x} {y} {z}" for el, x, y, z in atoms)

# --- Build PySCF molecule ---
mol = gto.M(atom=atom_str, basis='631+g*', cart=True)
mol.verbose = 4
mol.output = 'pyscf_output.log'
mol.build()

positions = mol.atom_coords()  # In Bohr

# --- Run SCF Calculation ---
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.max_cycle = 100
mf.conv_tol = 1e-9
mf.kernel()
# --- Log SCF status ---
if not mf.converged:
    print("⚠️ SCF did NOT converge!")
else:
    print("✅ SCF converged.")

print(f"Total Energy: {mf.e_tot:.12f} Hartree")

try:
    print(f"SCF Steps: {mf.scf_summary['num_cycle']}")
except (AttributeError, KeyError):
    print("SCF Steps: [Unavailable]")

print("Orbital Energies and Occupations:")
for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
    print(f"MO #{i:3d} | Energy: {e: .6f} | Occupation: {occ:.1f}")

# --- Get MO data ---
mo_coeff = mf.mo_coeff
mo_occ = mf.mo_occ
mo_energies = mf.mo_energy

# --- Create adaptive grid box around molecule ---
margin = 3.0  # Bohr
min_corner = positions.min(axis=0) - margin
max_corner = positions.max(axis=0) + margin

nx = ny = nz = 100
x = np.linspace(min_corner[0], max_corner[0], nx)
y = np.linspace(min_corner[1], max_corner[1], ny)
z = np.linspace(min_corner[2], max_corner[2], nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# --- Evaluate molecular orbitals on the grid ---
ao_values = mol.eval_gto("GTOval", coords)
mo_values = np.dot(ao_values, mo_coeff)

# --- Setup PyVista grid object ---
grid = pv.ImageData()
grid.dimensions = np.array([nx, ny, nz]) + 1
grid.origin = tuple(min_corner)
grid.spacing = tuple((max_corner - min_corner) / np.array([nx, ny, nz]))

# --- MO Visualizer Class ---
class MOVisualizer:
    def __init__(self, mo_values, grid, plotter):
        self.mo_values = mo_values
        self.grid = grid
        self.plotter = plotter
        self.mo_index = 0
        self.iso_level = 0.05
        self.meshes = []
        self.title = None
        self.update()

    def update(self):
        for m in self.meshes:
            self.plotter.remove_actor(m)
        self.meshes.clear()
        if self.title:
            self.plotter.remove_actor(self.title)

        values = self.mo_values[:, self.mo_index].reshape((nx, ny, nz))
        self.grid.cell_data["mo"] = values.flatten(order="F")
        grid_point = self.grid.cell_data_to_point_data()

        vmin, vmax = values.min(), values.max()
        level = self.iso_level * max(abs(vmin), abs(vmax))

        contours_pos = grid_point.contour([+level], scalars="mo")
        if contours_pos.n_points > 0:
            self.meshes.append(self.plotter.add_mesh(contours_pos, color="red", opacity=0.6))

        contours_neg = grid_point.contour([-level], scalars="mo")
        if contours_neg.n_points > 0:
            self.meshes.append(self.plotter.add_mesh(contours_neg, color="blue", opacity=0.6))

        self.title = self.plotter.add_text(
            f"MO #{self.mo_index} | Energy: {mo_energies[self.mo_index]:.3f} | Occ: {mo_occ[self.mo_index]:.1f}",
            position='upper_edge', font_size=10
        )

    def set_iso_level(self, value):
        self.iso_level = value
        self.update()

    def set_mo_index(self, value):
        self.mo_index = int(value)
        self.update()

# --- Initialize plotter ---
plotter = pv.Plotter()
visualizer = MOVisualizer(mo_values, grid, plotter)

# --- Add atoms (Bohr) ---
for symbol, pos in zip(symbols, positions):
    Z = atomic_numbers[symbol]
    radius = covalent_radii[Z] * 0.65  # scaled to Bohr
    color = cpk_color_dict.get(Z, (1.0, 1.0, 1.0))
    sphere = pv.Sphere(radius=radius, center=pos, theta_resolution=64, phi_resolution=64)
    plotter.add_mesh(sphere, color=color, specular=0.6)

# --- Add bonds ---
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2)) / BOHR  # in Å

for i, j in combinations(range(len(positions)), 2):
    Z1 = atomic_numbers[symbols[i]]
    Z2 = atomic_numbers[symbols[j]]
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

# --- Add sliders ---
plotter.add_slider_widget(
    callback=lambda value: visualizer.set_mo_index(round(value)),
    rng=[0, mo_values.shape[1] - 1],
    value=0,
    title="MO Index",
    pointa=(0.025, 0.1),
    pointb=(0.61, 0.1),
    style="modern",
    fmt="%.0f",
)


plotter.add_slider_widget(
    callback=lambda value: visualizer.set_iso_level(value),
    rng=[0.01, 0.8],
    value=0.05,
    title="Iso-surface Level",
    pointa=(0.62, 0.1),
    pointb=(0.94, 0.1),
    style="modern",
)

plotter.add_axes()
plotter.show()
