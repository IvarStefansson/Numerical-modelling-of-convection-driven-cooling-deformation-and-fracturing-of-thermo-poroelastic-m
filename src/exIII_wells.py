"""
Example setup and run script for a 3d example with two fractures.

"""

import numpy as np
import porepy as pp
import logging
from typing import Tuple
from fracture_propagation_model import THMPropagationModel
import utils


logger = logging.getLogger(__name__)


class Example3Model(THMPropagationModel, pp.THM):
    """
    This class provides the parameter specification differing from examples 1 and 2.
    """

    def set_fields(self, params):
        super().set_fields(params)
        self.length_scale = params["length_scale"]

        self.initial_aperture = 3.0e-4 / self.length_scale
        self.production_well_key = "production_well"

        self.export_fields.append("well")
        self.gravity_on = True
        size = 1e3 / self.length_scale
        self.box = {
            "xmin": 0,
            "xmax": size,
            "ymin": 0,
            "ymax": size,
            "zmin": 0,
            "zmax": size,
        }

    def _fractures(self):
        """
        Define the two fractures.
        The first fracture is the one where injection takes place.
        """

        s = self.box["xmax"]
        z_3 = s / 2
        z_2 = 2 / 3 * s
        z_1 = 1 / 3 * s

        y_1 = 3 / 12 * s
        y_2 = 5 / 12 * s
        y_3 = 7 / 12 * s
        y_4 = 9 / 12 * s

        x_1 = 1 / 3 * s
        x_2 = 2 / 3 * s
        x_3 = 0.5 * s

        f_1 = np.array(
            [[x_1, x_1, x_2, x_2], [y_1, y_2, y_2, y_1], [z_3, z_3, z_3, z_3]]
        )
        f_2 = np.array(
            [[x_3, x_3, x_3, x_3], [y_3, y_4, y_4, y_3], [z_2, z_2, z_1, z_1]]
        )

        self.fracs = [f_1, f_2]

    def create_grid(self):
        self._fractures()

        x = self.box["xmax"] - self.box["xmin"]
        y = self.box["ymax"] - self.box["ymin"]
        nx = self.params.get("nx")
        ny = self.params.get("ny")
        ncells = [nx, ny]
        dims = [x, y]
        if "zmax" in self.box:
            nz = self.params.get("nz")
            ncells.append(nz)
            dims.append(self.box["zmax"] - self.box["zmin"])
        gb = pp.meshing.cart_grid(self.fracs, ncells, physdims=dims)

        s = self.box["xmax"]

        # The following ugly code refines the grid around the two fractures
        y_old = np.array([0, 1 / 3]) * s
        y_new = np.array([0, 0.4]) * s
        y_values = np.array([0, 7 / ny, 13 / ny, 1.0]) * s
        z_values = np.array([0, 0.44, 0.56, 1.0]) * s
        z = [0.44, 0.56]
        x = [0.40, 0.6]

        y0 = 10 / 26
        y1 = 16 / 26
        x_0 = 1 / 3 - 3 / 36
        x_1 = 2 / 3 + 3 / 36
        old = np.array([[x_0, x_1], [1 / 4, 3 / 4], [1 / 3, 2 / 3]]) * s
        new = np.array([[x[0], x[1]], [y0, y1], [z[0], z[1]]]) * s
        utils.adjust_nodes(gb, old, new)

        # Ensure one layer of small cells around fractures
        k = 0.8
        dx = k * x[0] / (nx / 3)  # (x[1]-x[0])
        dy = k * 0.25 / (ny / 4)
        dz = k * z[0] / (nz / 3)  # (z[1]-z[0])

        old = np.array([[0, x_0], [0, 1 / 4], [0, 1 / 3 + dz]]) * s
        new = np.array([[0, x[0]], [0, y0 + dy], [0, z[0]]]) * s
        utils.adjust_nodes(gb, old, new)

        old = np.array([[x_1, 1], [3 / 4, 1], [2 / 3, 1]]) * s
        new = np.array([[x[1], 1], [y1 - dy, 1], [z[1] - dz, 1]]) * s
        utils.adjust_nodes(gb, old, new)

        # Adjustment finished
        gb.compute_geometry()
        pp.fracs.meshing.create_mortar_grids(gb)
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()

        # Tag the wells
        self._tag_well_cells()
        self.n_frac = len(gb.grids_of_dimension(self.Nd - 1))

    def _faces_to_fix(self, g):
        """
        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.
        """
        all_bf, *_ = self.domain_boundary_sides(g)
        point = np.array(
            [
                [(self.box["xmin"] + self.box["xmax"]) / 2],
                [(self.box["ymin"] + self.box["ymax"]) / 2],
                [self.box["zmax"]],
            ]
        )
        distances = pp.distances.point_pointset(point, g.face_centers[:, all_bf])
        indexes = np.argsort(distances)
        faces = all_bf[indexes[:4]]
        return faces

    def _tag_well_cells(self):
        """
        Tag well cells with unitary values, positive for injection cells and negative
        for production cells.
        """
        for g, d in self.gb:
            tags = np.zeros(g.num_cells)
            if g.dim < self.Nd:
                s = self.box["xmax"]
                ny = 25
                # Avoid specifying a point on a face (having non-unique nearest
                # cell centre neighbour) by adding eps
                eps = 0.01
                ny = 26  # cf grid construction
                p1 = np.array([[(s + eps) / 2], [11.7 * s / (ny)], [s / 2]])
                p2 = np.array([[s / 2], [(14.2) * s / ny], [(s + eps) / 2]])

                if d["node_number"] == 1:
                    distances = pp.distances.point_pointset(p1, g.cell_centers)
                    indexes = np.argsort(distances)
                    tags[indexes[0]] = 1  # injection well
                elif d["node_number"] == 2:
                    distances = pp.distances.point_pointset(p2, g.cell_centers)
                    indexes = np.argsort(distances)

                    tags[indexes[0]] = -1

            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})

    def source_flow_rates(self) -> Tuple[int, int]:
        """
        The rate is given in l/s = m^3/s e-3. Length scaling also needed to convert from
        the scaled length to m.
        The values returned depend on the simulation phase.
        """
        tol = 1e-2
        injection, production = 0, 0
        if self.time > self.phase_limits[0] + tol:
            injection, production = 5, 5

        w = pp.MILLI * (pp.METER / self.length_scale) ** self.Nd
        return injection * w, production * w

    def source_scalar(self, g) -> np.ndarray:
        """
        Source term for the scalar equation.
        For slightly compressible flow in the present formulation, this has units of m^3.

        Sources are handled by ScalarSource discretizations.
        The implicit scheme yields multiplication of the rhs by dt, but
        this is not incorporated in ScalarSource, hence we do it here.
        """
        rhs = super().source_scalar(g)
        injection, production = self.source_flow_rates()
        wells = injection * self.time_step * g.tags["well_cells"].clip(min=0)
        wells += production * self.time_step * g.tags["well_cells"].clip(max=0)

        return rhs + wells

    def source_temperature(self, g) -> np.ndarray:
        """
        Sources are handled by ScalarSource discretizations.
        The implicit scheme yields multiplication of the rhs by dt, but
        this is not incorporated in ScalarSource, hence we do it here.

        The sink (production well) is discretized using the MassMatrix
        discretization to ensure the extracted energy matches the current
        production well temperature.
        """
        injection, production = self.source_flow_rates()

        # Injection well
        t_in = -30
        weight = (
            self.fluid_density(g, dT=t_in * self.temperature_scale)
            * self.fluid.specific_heat_capacity(self.background_temp_C)
            * self.time_step
            / self.T_0_Kelvin
        )
        rhs = t_in * weight * injection * g.tags["well_cells"].clip(min=0)

        # Production well, discretized by MassMatrix on the lhs
        weight = (
            self.fluid_density(g)
            * self.fluid.specific_heat_capacity(self.background_temp_C)
            * self.time_step
            / self.T_0_Kelvin
        )

        lhs = -(weight * production * g.tags["well_cells"].clip(max=0) / g.cell_volumes)
        # HACK: Set this directly into d to avoid additional return
        d = self.gb.node_props(g)
        pp.initialize_data(g, d, self.production_well_key, {"mass_weight": lhs})
        return rhs

    def bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        """
        We set Neumann values imitating an anisotropic background stress regime on all
        but three faces, which are fixed to ensure a unique solution.
        """
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        faces = self._faces_to_fix(g)
        bc = pp.BoundaryConditionVectorial(g, faces, "dir")
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def bc_values_mechanics(self, g) -> np.ndarray:
        """
        Lithostatic mechanical BC values.
        """
        bc_values = np.zeros((g.dim, g.num_faces))

        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
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
        if g.dim < self.Nd:
            return np.empty(0, dtype=int)
        else:
            all_bf, *_ = self.domain_boundary_sides(g)
            return all_bf

    def bc_values_scalar(self, g) -> np.ndarray:
        """
        Hydrostatic pressure BC values.
        """
        # Retrieve the boundaries where values are assigned
        bf = self._p_and_T_dir_faces(g)
        bc_values = np.zeros(g.num_faces)
        if g.dim == self.Nd:
            depth = self._depth(g.face_centers[:, bf])
            bc_values[bf] = self.hydrostatic_pressure(g, depth) / self.scalar_scale
        return bc_values

    def bc_values_temperature(self, g) -> np.ndarray:
        """
        Hydrostatic pressure BC values.
        """
        # Retrieve the boundaries where values are assigned
        bc_values = np.zeros(g.num_faces)
        return bc_values

    def hydrostatic_pressure(self, g, depth: np.ndarray):
        """
        Iterate to get a "density adjusted" hydrostatic pressure.
        TODO: Discuss if this makes sense.

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
        self.time = 0 * pp.YEAR
        # and time step
        self.time_step = 0.1 * pp.YEAR

        # We use
        self.end_time = 2.5 * pp.YEAR
        self.max_time_step = self.end_time / 2
        self.phase_limits = [self.time, 8 * pp.YEAR, self.end_time]
        self.phase_time_steps = [self.time_step, 1 * pp.YEAR, 1]
        # self.phase_time_steps = [self.time_step, 5 * pp.HOUR, self.end_time / 15, 1]
        self.time_step_factor = 1.0

    def _depth(self, coords) -> np.ndarray:
        """
        Unscaled depth. We center the domain at 1 km below the surface.
        """
        return 2 * pp.KILO * pp.METER - self.length_scale * coords[2] * self.gravity_on

    def assign_discretizations(self) -> None:

        """
        Mass matrix for temperature production well.
        """
        # Call parent class for disrcetizations.
        super().assign_discretizations()

        discr = pp.MassMatrix(self.production_well_key)
        for g, d in self.gb:
            if g.dim == self.Nd - 1:
                d[pp.DISCRETIZATION][self.temperature_variable].update(
                    {"production_well": discr}
                )

    def export_step(self):
        """
        Save pressure and temperature values for the two well cells.

        Stored as self.pressures and self.temperatures, used to produce plots.
        """
        super().export_step()
        pressures = np.zeros(len(self.gb.grids_of_dimension(self.Nd - 1)))
        temperatures = np.zeros(len(self.gb.grids_of_dimension(self.Nd - 1)))

        for i, g in enumerate(self.gb.grids_of_dimension(self.Nd - 1)):
            d = self.gb.node_props(g)
            T = d[pp.STATE][self.temperature_variable] * self.temperature_scale
            p = d[pp.STATE][self.scalar_variable] * self.scalar_scale
            ind = np.nonzero(g.tags["well_cells"])
            pressures[i] = p[ind]
            temperatures[i] = T[ind]
        if hasattr(self, "pressures"):
            self.pressures = np.vstack((self.pressures, pressures))
            self.temperatures = np.vstack((self.temperatures, temperatures))
        else:
            self.pressures = pressures
            self.temperatures = temperatures

    def _map_variables(self, solution: np.ndarray):
        """
        In addition to super's mapping an initialization of all primary variables
        and face values (darcy_fluxes and stored boundary conditions), map
        quantities to be exported.


        """
        new_solution = super()._map_variables(solution)
        # EK: Rudimentary treatment of variables that must be updated for the code
        # to run. Likely, a better approach is needed
        for g, d in self.gb:
            cell_map = d["cell_index_map"]

            if "well_cells" in g.tags:
                g.tags["well_cells"] = (cell_map * g.tags["well_cells"].T).T
                d[pp.STATE]["well"] = (cell_map * d[pp.STATE]["well"].T).T

        return new_solution


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    folder_name = "exIII"
    params = {
        "folder_name": folder_name,
        "nl_convergence_tol": 2e-6,
        "max_iterations": 60,
        "file_name": "forced_convection",
        "length_scale": 10,
        "max_memory": 7e7,
        "nx": 36,  # multiple of 6
        "ny": 48,  # multiple of 12
        "nz": 36,  # multiple of 6
        "prepare_umfpack": True,
    }

    m = Example3Model(params)
    m.compute_initial_displacement()

    pp.run_time_dependent_model(m, params)
    m.export_pvd()
    data = {
        "fracture_sizes": m.fracture_sizes,
        "time_steps": m.export_times,
        "pressures": m.pressures,
        "temperatures": m.temperatures,
    }
    utils.write_pickle(data, folder_name + "/fracture_sizes")
