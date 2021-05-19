"""
Example setup and run script for a 3d example with five vertical fractures.
"""

import logging

import numpy as np
import porepy as pp

import utils
from fracture_propagation_model import THMPropagationModel

logger = logging.getLogger(__name__)


class CDM(THMPropagationModel, pp.THM):
    """
    This class provides the parameter specification differing from examples 1 and 2.
    """

    def _set_fields(self, params):
        super()._set_fields(params)
        self.length_scale = params["length_scale"]

        self.initial_aperture = 2e-3 / self.length_scale
        self.gravity_on = True
        self.T_gradient = 15e-2 * self.length_scale
        # Define the domain
        size = self.params["size"]
        self.box = {
            "xmin": 0,
            "xmax": size,
            "ymin": 0,
            "ymax": size,
            "zmin": 0,
            "zmax": size,
        }
        self.export_fields.append("fluxes_exp")

    def _fractures(self):
        """
        Define the two fractures.
        The first fracture is the one where injection takes place.
        """
        s = self.params["size"]
        z_1 = s
        n = 5 + 1
        x1 = 1 * s / n
        x2 = 2 * s / n
        x3 = 3 * s / n
        x4 = 4 * s / n
        x5 = 5 * s / n
        y_1 = 3 / 4 * s
        y_2 = 1 / 4 * s
        y_1 = 0.75 * s
        y_2 = 0.25 * s
        z_2 = 0.5 * s

        f_1 = np.array([[x1, x1, x1, x1], [y_1, y_1, y_2, y_2], [z_1, z_2, z_2, z_1]])
        f_2 = np.array([[x2, x2, x2, x2], [y_1, y_1, y_2, y_2], [z_1, z_2, z_2, z_1]])
        f_3 = np.array([[x3, x3, x3, x3], [y_1, y_1, y_2, y_2], [z_1, z_2, z_2, z_1]])
        f_4 = np.array([[x4, x4, x4, x4], [y_1, y_1, y_2, y_2], [z_1, z_2, z_2, z_1]])
        f_5 = np.array([[x5, x5, x5, x5], [y_1, y_1, y_2, y_2], [z_1, z_2, z_2, z_1]])

        self.fracs = [f_1, f_2, f_3, f_4, f_5][: n - 1]

    def _faces_to_fix(self, g):
        """
        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.
        """
        all_bf, *_ = self._domain_boundary_sides(g)
        point = np.array(
            [
                [(self.box["xmin"] + self.box["xmax"]) / 2],
                [(self.box["ymin"] + self.box["ymax"]) / 2],
                [self.box["zmin"]],
            ]
        )

        distances = pp.distances.point_pointset(point, g.face_centers[:, all_bf])
        indexes = np.argsort(distances)
        faces = all_bf[indexes[:3]]
        return faces

    def _bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        """
        We set Neumann values imitating an anisotropic background stress regime on all
        but three faces, which are fixed to ensure a unique solution.
        """
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        faces = self._faces_to_fix(g)
        bc = pp.BoundaryConditionVectorial(g, faces, "dir")
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def _bc_values_mechanics(self, g) -> np.ndarray:
        """Anisotropic mechanical BC values based on lithostatic traction."""
        bc_values = np.zeros((g.dim, g.num_faces))

        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        A = g.face_areas

        # Gravity acceleration
        gravity = (
            pp.GRAVITY_ACCELERATION
            * self.rock.DENSITY
            * self._depth(g.face_centers)
            / self.scalar_scale
        )

        # Anisotropy ratios
        we, sn, bt = 0.6, 1.2, 1

        bc_values[0, west] = (we * gravity[west]) * A[west]
        bc_values[0, east] = -(we * gravity[east]) * A[east]
        bc_values[1, south] = (sn * gravity[south]) * A[south]
        bc_values[1, north] = -(sn * gravity[north]) * A[north]
        bc_values[2, bottom] = (bt * gravity[bottom]) * A[bottom]
        bc_values[2, top] = -(bt * gravity[top]) * A[top]
        faces = self._faces_to_fix(g)
        bc_values[:, faces] = 0

        return bc_values.ravel("F")

    def _p_and_T_dir_faces(self, g):
        """
        We prescribe Dirichlet value at the fractures.
        No-flow for the matrix.
        """
        all_bf, *_ = self._domain_boundary_sides(g)
        return all_bf

    def _bc_values_scalar(self, g) -> np.ndarray:
        """
        Hydrostatic pressure BC values.
        """
        # Retrieve the boundaries where values are assigned
        bf = self._p_and_T_dir_faces(g)
        bc_values = np.zeros(g.num_faces)
        depth = self._depth(g.face_centers[:, bf])
        bc_values[bf] = self._hydrostatic_pressure(g, depth) / self.scalar_scale
        return bc_values

    def _bc_values_temperature(self, g) -> np.ndarray:
        """
        Cooling at the top of the fracture
        """
        # Retrieve the boundaries where values are assigned
        bf = self._p_and_T_dir_faces(g)
        bc_values = np.zeros(g.num_faces)
        # if g.dim == self._Nd:
        bc_values[bf] = (
            self.box["zmax"] - g.face_centers[2, bf]
        ) * self.T_gradient + self.T_0_Kelvin
        return bc_values

    def _set_rock_and_fluid(self):
        super()._set_rock_and_fluid()
        self.rock.PERMEABILITY = 1e-16

    def _initial_temperature(self, g) -> np.ndarray:
        return (
            self.box["zmax"] - g.cell_centers[2]
        ) * self.T_gradient + self.T_0_Kelvin

    def _depth(self, coords) -> np.ndarray:
        """
        Unscaled depth. We center the domain at 1 km below the surface.
        """
        return 3.0 * pp.KILO * pp.METER - self.length_scale * coords[2]

    def _hydrostatic_pressure(self, g, depth: np.ndarray):
        """

        Parameters
        ----------
        g : grid.
        depth : array
            Unscaled depth.

        Returns
        -------
        p : array
            Pressure.

        """
        rho_0 = 1e3 * (pp.KILOGRAM / pp.METER ** 3)
        p = rho_0 * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE
        return p

    def _set_time_parameters(self):
        """
        Specify time parameters.
        """
        # For the initialization run, we use the following
        # start time
        self.time = 0
        # and time step
        self.time_step = 2.5 * pp.YEAR

        # We use
        self.end_time = 70 * pp.YEAR
        self.max_time_step = self.end_time / 2
        self.phase_limits = [self.time, 20 * pp.YEAR, self.end_time]
        self.phase_time_steps = [self.time_step, 2 * pp.YEAR, self.time_step]
        self.time_step_factor = 1.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    folder_name = "exIV_revision"
    params = {
        "folder_name": folder_name,
        "nl_convergence_tol": 5e-6,
        "max_iterations": 60,
        "file_name": "natural_convection",
        "mesh_args": {},
        "length_scale": 100,
        "size": 400 / 100,
        "max_memory": 7e7,
        "nx": 36,  # multiple of (n_fracs+1)=6 along x axis
        "ny": 40,  # ny should be multiple of four, since fractures extend from .25 to .75
        "nz": 40,  # nz should be even, since fractures extend from 1/2 to 1
        "prepare_umfpack": True,
    }

    m = CDM(params)
    m._compute_initial_displacement()
    pp.run_time_dependent_model(m, params)
    m._export_pvd()
    data = {
        "fracture_sizes": m.fracture_sizes,
        "time_steps": m.export_times,
    }
    utils.write_pickle(data, folder_name + "/fracture_sizes")
