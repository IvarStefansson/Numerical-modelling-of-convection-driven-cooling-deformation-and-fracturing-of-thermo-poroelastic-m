"""
Run script for 2d example with two fractures. Dynamics driven by Dirichlet
values at the fracture endpoints, which are different from the matrix BC values.
Flow and cooling from left to right, leftmost fracture grows.

-----------------------
|                     |
|                     |
|                     |
|                     |
|----             ----|
|                     |
|                     |
|                     |
-----------------------
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from porepy.models.thm_model import THM

from fracture_propagation_model import THMPropagationModel
from utils import read_pickle, write_pickle

logger = logging.getLogger(__name__)


class Example2Model(THMPropagationModel, THM):
    """
    This class provides the parameter specification of the example, including grid/geometry,
    BCs, rock and fluid parameters and time parameters. Also provides some common modelling
    functions, such as the aperture computation from the displacement jumps, and data storage
    and export functions.
    """

    def _fractures(self):
        self.fracs = [
            np.array([[0.0, 0.5], [0.25, 0.5]]).T,
            np.array([[0.75, 0.5], [1, 0.5]]).T,
        ]

    def _depth(self, coords):
        return np.zeros(coords.shape[1])

    def _bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        """
        Dirichlet values at top and bottom.
        """
        all_bf, east, west, north, south, _, _ = self._domain_boundary_sides(g)
        dir_faces = south + north + g.tags["fracture_faces"]
        bc = pp.BoundaryConditionVectorial(g, dir_faces, "dir")
        return bc

    def _bc_values_mechanics(self, g) -> np.ndarray:
        """Dirichlet displacement on the top, fixed on bottom and 0 Neumann
        on left and right.
        """
        # Retrieve the boundaries where values are assigned
        bc_values = np.zeros((g.dim, g.num_faces))
        return bc_values.ravel("F")

    def _p_and_T_dir_faces(self, g):
        """
        We prescribe Dirichlet value at the fractures.
        No-flow for the matrix.
        """
        if g.dim == self._Nd:
            return np.empty(0, dtype=int)
        else:
            all_bf, east, west, north, south, _, _ = self._domain_boundary_sides(g)
            return (east + west).nonzero()[0]

    def _bc_values_scalar(self, g) -> np.ndarray:
        """
        See bc_type_scalar
        """
        # Retrieve the boundaries where values are assigned
        dir_faces = self._p_and_T_dir_faces(g)
        bc_values = np.zeros(g.num_faces)
        bc_values[dir_faces] = (
            5e4 / self.scalar_scale * (1 - g.face_centers[0, dir_faces])
        )
        return bc_values

    def _bc_values_temperature(self, g) -> np.ndarray:
        """Cooling on the left from the onset of phase III."""
        bc_values = np.zeros(g.num_faces)
        dir_faces = self._p_and_T_dir_faces(g)
        bc_values[dir_faces] = self.T_0_Kelvin - 50 * (1 - g.face_centers[0, dir_faces])

        return bc_values

    def _set_rock_and_fluid(self):
        """
        Set rock and fluid properties to those of granite and water.
        We ignore all temperature dependencies of the parameters.
        """
        super()._set_rock_and_fluid()

    def _hydrostatic_pressure(self, g, depth):
        """Set explicitly to zero to avoid the atmospheric pressure returned
        by the exIII/exIV function for depth=0.
        """
        return np.zeros_like(depth)

    def _set_time_parameters(self):
        """
        Specify time parameters.
        """
        # For the initialization run, we use the following
        # start time
        self.time = 0
        # and time step
        self.time_step = self.params.get("time_step")

        # We use
        self.end_time = 4 * pp.HOUR
        self.max_time_step = self.time_step
        self.phase_limits = np.array([self.end_time])
        self.phase_time_steps = np.array([self.time_step])

    def _set_fields(self, params):
        """
        Set various fields to be used in the model.
        """
        super()._set_fields(params)

        # Initial aperture, a_0
        self.initial_aperture = 1e-3 / self.length_scale

        self.gravity_on = False  # Mechanics not implemented for True
        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # dt in seconds
    reference = False
    time_steps = np.array([600, 300, 150, 75])

    # Number of cells in each dimension
    n_cells = np.array([32, 64, 128])
    if reference:
        n_cells = np.array([512])
        time_steps = np.array([25])
    fracture_sizes = {}
    export_times = {}
    mesh_size = 0.02
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": 0.5 * mesh_size,
        "mesh_size_bound": 3.6 * mesh_size,
    }

    folder_name = "exII_revision"
    if reference:
        folder_name += "_ref"
    params = {
        "nl_convergence_tol": 1e-8,
        "max_iterations": 50,
        "file_name": "tensile_stable_propagation",
        "mesh_args": mesh_args,
        "folder_name": folder_name,
        "nx": 10,
        "prepare_umfpack": False,
    }
    if reference:
        params["file_name"] = "tensile_stable_propagation_reference"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i, dt in enumerate(time_steps):
        params["time_step"] = dt
        for j, nx in enumerate(n_cells):
            logger.info("\nSolving for dt {} and nx {}.".format(dt, nx))
            params["nx"] = nx
            m = Example2Model(params)
            pp.run_time_dependent_model(m, params)
            fracture_sizes[(dt, nx)] = m.fracture_sizes
            export_times[(dt, nx)] = m.export_times

    m._export_pvd()
    plot = False
    if reference:
        data = read_pickle("exII/fracture_sizes")
        fracture_sizes.update(data["fracture_sizes"])
        time_steps = np.union1d(data["time_steps"], time_steps)
        export_times = data["export_times"].update(export_times)
        n_cells = np.union1d(data["n_cells"], n_cells)

    data = {
        "fracture_sizes": fracture_sizes,
        "time_steps": time_steps,
        "n_cells": n_cells,
        "export_times": export_times,
    }
    write_pickle(data, folder_name + "/fracture_sizes")

    if plot:
        fig, ax = plt.subplots()
        for i, dt in enumerate(time_steps):
            for j, nx in enumerate(n_cells):
                data = fracture_sizes[(dt, nx)]
                length = data[:, 2] - data[:, 1]
                ax.plot(data[:, 0], length, label="dt {} nx {}".format(dt, nx))
        ax.legend()
        plt.show()
