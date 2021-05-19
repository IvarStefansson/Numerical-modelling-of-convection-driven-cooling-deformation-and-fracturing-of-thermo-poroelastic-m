"""
Problem description: Single horizontal fracture in an infinite domain.
Pressure p0 acting on the fracture walls.

-----------------------
|                     |
|                     |
|                     |
|                     |
|       ------        |
|                     |
|                     |
|                     |
-----------------------


Analytical solution due to Sneddon, 1946:
    The distribution of stress in the neighbourhood of a crack in an inelastic solid

The analytical solution is used to assign Dirichlet boundary condition using the
boundary element method, see
Keilegavlen et al, 2020:
    PorePy: An Open-Source Software for Simulation of Multi-physics Processes
    in Fractured Porous Media
"""

import logging
import os

import numpy as np
import porepy as pp
import scipy.sparse.linalg as spla

from fracture_propagation_model import TensilePropagation
from utils import write_pickle

logger = logging.getLogger(__name__)

# First four functions copied from run scripts for Keilegavlen et al.


def get_bem_centers(a, h, n, theta, center):
    """
    Compute coordinates of the centers of the bem segments

    Parameter
    ---------
    a: half fracture length
    h: bem segment length
    n: number of bem segments
    theta: orientation of the fracture
    center: center of the fracture
    """
    bem_centers = np.zeros((3, n))
    x_0 = center[0] - (a - 0.5 * h)
    y_0 = center[1]
    for i in range(0, n):
        bem_centers[0, i] = x_0 + i * h
        bem_centers[1, i] = y_0

    return bem_centers


def transform(xc, x, alpha):
    """
    Coordinate transofrmation for the BEM method

    Parameter
    ---------
    xc: coordinates of BEM segment centre
    x: coordinates of boundary faces
    alpha: fracture orientation
    """
    x_bar = np.zeros_like(x)
    x_bar[0, :] = (x[0, :] - xc[0]) * np.cos(alpha) + (x[1, :] - xc[1]) * np.sin(alpha)
    x_bar[1, :] = -(x[0, :] - xc[0]) * np.sin(alpha) + (x[1, :] - xc[1]) * np.cos(alpha)
    return x_bar


def get_bc_val(g, bound_faces, xf, h, poi, alpha, du):
    """
    Compute analytical displacement using the BEM method.

    Parameter
    ---------
    g: grid bucket
    bound_faces: boundary faces
    xf: coordinates of boundary faces
    h: bem segment length
    poi: Poisson ratio
    alpha: fracture orientation
    du: Sneddon's analytical relative normal displacement
    """
    f2 = np.zeros(bound_faces.size)
    f3 = np.zeros(bound_faces.size)
    f4 = np.zeros(bound_faces.size)
    f5 = np.zeros(bound_faces.size)

    u = np.zeros((g.dim, g.num_faces))

    m = 1 / (4 * np.pi * (1 - poi))

    f2[:] = m * (
        np.log(np.sqrt((xf[0, :] - h) ** 2 + xf[1] ** 2))
        - np.log(np.sqrt((xf[0, :] + h) ** 2 + xf[1] ** 2))
    )

    f3[:] = -m * (
        np.arctan2(xf[1, :], (xf[0, :] - h)) - np.arctan2(xf[1, :], (xf[0, :] + h))
    )

    f4[:] = m * (
        xf[1, :] / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
        - xf[1, :] / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2)
    )

    f5[:] = m * (
        (xf[0, :] - h) / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
        - (xf[0, :] + h) / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2)
    )

    u[0, bound_faces] = du * (
        -(1 - 2 * poi) * np.cos(alpha) * f2[:]
        - 2 * (1 - poi) * np.sin(alpha) * f3[:]
        - xf[1, :] * (np.cos(alpha) * f4[:] + np.sin(alpha) * f5[:])
    )
    u[1, bound_faces] = du * (
        -(1 - 2 * poi) * np.sin(alpha) * f2[:]
        + 2 * (1 - poi) * np.cos(alpha) * f3[:]
        - xf[1, :] * (np.sin(alpha) * f4[:] - np.cos(alpha) * f5[:])
    )

    return u


def assign_bem(g, h, bound_faces, theta, bem_centers, u_a, poi):

    """
    Compute analytical displacement using the BEM method for the pressurized crack
    problem in question.

    Parameter
    ---------
    g: grid bucket
    h: bem segment length
    bound_faces: boundary faces
    theta: fracture orientation
    bem_centers: bem segments centers
    u_a: Sneddon's analytical relative normal displacement
    poi: Poisson ratio
    """

    bc_val = np.zeros((g.dim, g.num_faces))

    alpha = np.pi / 2 - theta

    bound_face_centers = g.face_centers[:, bound_faces]

    for i in range(0, u_a.size):

        new_bound_face_centers = transform(bem_centers[:, i], bound_face_centers, alpha)

        u_bound = get_bc_val(
            g, bound_faces, new_bound_face_centers, h, poi, alpha, u_a[i]
        )

        bc_val -= u_bound

    return bc_val


def fracture_2d(length, height, a, beta):
    """
    Return list of one fracture, possibly inclined.
    """

    y_0 = height / 2 - a * np.cos(beta)
    x_0 = length / 2 - a * np.sin(beta)
    y_1 = height / 2 + a * np.cos(beta)
    x_1 = length / 2 + a * np.sin(beta)
    f = np.array([[x_0, x_1], [y_0, y_1]])
    e = np.array([[0, 1]]).T
    return f, e


def L2_norm(val, area=None):
    if area is None:
        area = np.ones(val.size) / val.size
    return np.sqrt(np.sum(np.multiply(area, np.square(val))))


def L2_error(v_ref, v_approx, area):
    enum = L2_norm(v_approx - v_ref, area)
    denom = L2_norm(v_ref, area)
    return enum / denom


class SneddonSIFTest(TensilePropagation, pp.ContactMechanics):
    def __init__(self, params):
        TensilePropagation.__init__(self, params)
        pp.ContactMechanics.__init__(self, params)
        self.mesh_args = params.get("mesh_args")
        self.params = params
        self.initial_aperture: float = 0.0
        self.scalar_parameter_key = "flow"
        self.scalar_variable = "pressure"

        xdim, ydim = 2, 2
        xdim, ydim = 50, 50

        self.box = {
            "xmin": 0,
            "ymin": 0,
            "xmax": xdim,
            "ymax": ydim,
        }
        self.domain_center = np.array([self.box["xmax"] / 2, self.box["ymax"] / 2, 0])
        # Fracture radius or half length:
        self.a = 5

    def _fractures(self) -> None:
        """
        Store the fractures and, for simplex grids, the frac_edges attribute.

        Returns
        -------
        None.

        """
        c = self.domain_center
        self.fracs = [np.array([[c[0] - self.a, c[1] + self.a], [c[1], c[1]]])]
        self.frac_edges = np.array([[0], [1]])

    def create_grid(self):
        """
        Method that creates the GridBucket of a 3D domain with the two fractures
        defined by self.fractures().
        The grid bucket is represents the mixed-dimensional grid.
        """
        """
        Single-fracture tetrahedral gb.
        
        """
        self._fractures()
        if self.params.get("simplex"):
            self.network = pp.FractureNetwork2d(
                self.fracs[0], self.frac_edges, domain=self.box
            )
            gb = self.network.mesh(self.mesh_args)
        else:
            nc = self.params["n_cells"]
            dims = [self.box["xmax"], self.box["ymax"]]
            gb = pp.meshing.cart_grid(self.fracs, nc, physdims=dims)

        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()
        self.n_frac = len(gb.grids_of_dimension(self.Nd - 1))

    def _bc_type(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        """
        We set Neumann values imitating an anisotropic background stress regime on all
        but three faces, which are fixed to ensure a unique solution.
        """
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "dir")
        return bc

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """
        Assign displacement values, computed using the boundary element method,
        see Keilegavlen et al.

        Parameters
        ----------
        g : pp.Grid

        Returns
        -------
        bc_values : np.ndarray

        """
        bc_values = np.zeros((g.dim, g.num_faces))

        # Retrieve the boundaries where values are assigned
        all_bf, *_ = self._domain_boundary_sides(g)
        nu = self.params["poisson"]

        theta = self.params["inclination"]
        c = self.domain_center
        n = 1000
        h = 2 * self.a / n

        bem_centers = get_bem_centers(self.a, h, n, theta, c)
        eta = pp.geometry.distances.point_pointset(bem_centers, c)
        u_a = self.analytical_apertures(eta)

        bc_values = assign_bem(g, h / 2, all_bf, theta, bem_centers, u_a, nu)

        return bc_values

    def _set_parameters(self):
        poisson = self.params["poisson"]
        young = 1
        shear = young / (2 * (1 + poisson))
        self.params["shear"] = shear
        self.p0 = 1e-4

        for g, d in self.gb:
            if g.dim == self.Nd:
                bc, bc_val = self._bc_type(g), self._bc_values(g)

                # Rock parameters
                lam = 2 * shear * poisson / (1 - 2 * poisson) * np.ones(g.num_cells)
                mu = shear * np.ones(g.num_cells)
                C = pp.FourthOrderTensor(mu, lam)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val.ravel("f"),
                        "source": 0,
                        "fourth_order_tensor": C,
                        "shear_modulus": shear,
                        "poisson_ratio": poisson,
                    },
                )

            elif g.dim == self.Nd - 1:
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "friction_coefficient": 1,
                        "contact_mechanics_numerical_parameter": 1e2,
                        "dilation_angle": 0,
                    },
                )

            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "mass_weight": np.ones(g.num_cells),
                    "source": g.cell_volumes * self.params["boundary_traction"],
                },
            )

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            # Parameters for the surface diffusion. Not used as of now.
            pp.initialize_data(
                mg,
                d,
                self.mechanics_parameter_key,
                {},
            )

    def analytical_apertures(self, eta=None):
        """
        Analytical aperture solution.
        """
        gb, params = self.gb, self.params
        dim_h = gb.dim_max()
        g_l = gb.grids_of_dimension(dim_h - 1)[0]

        mu = params["shear"]
        nu = params["poisson"]
        if eta is None:
            eta = pp.geometry.distances.point_pointset(
                g_l.cell_centers, self.domain_center
            )
        cons = (1 - nu) / mu * self.p0 * self.a * 2
        return cons * np.sqrt(1 - np.power(eta / self.a, 2))

    def analytical_SIFs(self):

        K = np.zeros(2)
        const = self.p0 * np.sqrt(self.a * np.pi)
        K[0] = const
        return K

    def after_newton_convergence(self, solution, errors, iteration_counter):
        super().after_newton_convergence(solution, errors, iteration_counter)
        self._export_vtu()

    def _export_vtu(self) -> None:
        self.exporter = pp.Exporter(
            self.gb,
            self.params["file_name"],
            folder_name=self.params["folder_name"] + "_vtu",
        )
        for g, d in self.gb:
            if g.dim == self.Nd:
                pad_zeros = np.zeros((3 - g.dim, g.num_cells))
                u = d[pp.STATE][self.displacement_variable].reshape(
                    (self.Nd, -1), order="F"
                )
                u_exp = np.vstack((u, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["u_global"] = u_exp
                d[pp.STATE]["traction_exp"] = np.zeros(d[pp.STATE]["u_exp"].shape)
            elif g.dim == (self.Nd - 1):
                pad_zeros = np.zeros((2 - g.dim, g.num_cells))
                g_h = self.gb.node_neighbors(g)[0]
                data_edge = self.gb.edge_props((g, g_h))

                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, d["tangential_normal_projection"], from_iterate=False
                )
                u_exp = np.vstack((u_mortar_local, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp

                traction = d[pp.STATE][self.contact_traction_variable].reshape(
                    (self.Nd, -1), order="F"
                )
                d[pp.STATE]["traction_exp"] = np.vstack((traction, pad_zeros))
        export_fields = ["u_exp", "traction_exp"]
        self.exporter.write_vtu(export_fields)

    def _initial_condition(self) -> None:
        """
        Initialize fracture pressure.

        Returns
        -------
        None
        """
        super()._initial_condition()
        for g, d in self.gb:
            if g.dim == self.Nd - 1:
                pp.set_state(
                    d,
                    {
                        self.scalar_variable: self.params["boundary_traction"]
                        * np.ones(g.num_cells)
                    },
                )

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket:
        """
        super()._assign_variables()
        for g, d in self.gb:
            if g.dim == self.Nd - 1:
                d[pp.PRIMARY_VARIABLES].update(
                    {
                        self.scalar_variable: {"cells": 1},
                    }
                )

    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether or not to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        super()._assign_discretizations()

        # Shorthand
        key_s, key_m = self.scalar_parameter_key, self.mechanics_parameter_key
        var_s, var_d = self.scalar_variable, self.displacement_variable

        # Define discretization
        # For the Nd domain we solve linear elasticity with mpsa.
        mpsa = pp.Mpsa(key_m)
        mass_disc_s = pp.MassMatrix(key_s)
        source_disc_s = pp.ScalarSource(key_s)
        # Coupling discretizations

        # Assign node discretizations
        for g, d in self.gb:
            if g.dim == self.Nd:
                mpsa = d[pp.DISCRETIZATION][var_d]["mpsa"]

            elif g.dim == self.Nd - 1:
                d[pp.DISCRETIZATION].update(
                    {
                        var_s: {
                            "mass": mass_disc_s,
                            "source": source_disc_s,
                        },
                    }
                )

        fracture_scalar_to_force_balance = pp.FractureScalarToForceBalance(
            mpsa, mass_disc_s
        )

        for e, d in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)

            if g_h.dim == self.Nd:
                d[pp.COUPLING_DISCRETIZATION].update(
                    {
                        "fracture_scalar_to_force_balance": {
                            g_h: (var_d, "mpsa"),
                            g_l: (var_s, "mass"),
                            e: (
                                self.mortar_displacement_variable,
                                fracture_scalar_to_force_balance,
                            ),
                        }
                    }
                )

    def assemble_and_solve_linear_system(self, tol):

        A, b = self.assembler.assemble_matrix_rhs()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min"
            + f" {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )
        if self.params.get("prepare_umfpack", False):
            A.indices = A.indices.astype(np.int64)
            A.indptr = A.indptr.astype(np.int64)
        return spla.spsolve(A, b)

    def check_convergence(self, solution, prev_solution, init_solution, nl_params=None):
        g_max = self._nd_grid()

        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = np.any(np.isnan(solution))
            converged = not diverged
            error = np.nan if diverged else 0
            return error, converged, diverged

        mech_dof = self.dof_manager.dof_ind(g_max, self.displacement_variable)

        # Also find indices for the contact variables
        contact_dof = np.array([], dtype=np.int)
        jump_dof = np.array([], dtype=np.int)
        for e, _ in self.gb.edges():
            if e[0].dim == self.Nd:
                contact_dof = np.hstack(
                    (
                        contact_dof,
                        self.dof_manager.dof_ind(e[1], self.contact_traction_variable),
                    )
                )
                jump_dof = np.hstack(
                    (
                        contact_dof,
                        self.dof_manager.dof_ind(e, self.mortar_displacement_variable),
                    )
                )

        # Pick out the solution from current, previous iterates, as well as the
        # initial guess.
        u_mech_now = solution[mech_dof]
        u_mech_prev = prev_solution[mech_dof]
        u_mech_init = init_solution[mech_dof]

        contact_now = solution[contact_dof]
        contact_prev = prev_solution[contact_dof]
        contact_init = init_solution[contact_dof]

        jump_now = solution[jump_dof]
        jump_prev = prev_solution[jump_dof]
        jump_init = init_solution[jump_dof]

        # Calculate errors
        difference_in_iterates_mech = np.sum((u_mech_now - u_mech_prev) ** 2)
        difference_from_init_mech = np.sum((u_mech_now - u_mech_init) ** 2)

        contact_norm = np.sum(contact_now ** 2)
        difference_in_iterates_contact = np.sum((contact_now - contact_prev) ** 2)
        difference_from_init_contact = np.sum((contact_now - contact_init) ** 2)
        jump_norm = np.sum(jump_now ** 2)
        difference_in_iterates_jump = np.sum((jump_now - jump_prev) ** 2)
        difference_from_init_jump = np.sum((jump_now - jump_init) ** 2)

        tol_convergence = nl_params["nl_convergence_tol"]
        # Not sure how to use the divergence criterion
        # tol_divergence = nl_params["nl_divergence_tol"]

        converged_mech = False
        diverged = False

        # Check absolute convergence criterion
        if difference_in_iterates_mech < tol_convergence:
            converged_mech = True
            error_mech = difference_in_iterates_mech

        else:
            # Check relative convergence criterion
            if (
                difference_in_iterates_mech
                < tol_convergence * difference_from_init_mech
            ):
                converged_mech = True
            error_mech = difference_in_iterates_mech / difference_from_init_mech
        # 1e3 for scale difference between T and u
        # The if is intended to avoid division through zero
        if contact_norm < 1e-10 and difference_in_iterates_contact < 1e-10:
            # converged = True
            error_contact = difference_in_iterates_contact
            converged_contact = True
        else:
            error_contact = (
                difference_in_iterates_contact / difference_from_init_contact
            )
            converged_contact = error_contact < tol_convergence * 10

        # The if is intended to avoid division through zero
        if jump_norm < 1e-10 and difference_in_iterates_jump < 1e-10:
            # converged = True
            error_jump = difference_in_iterates_jump
            converged_jump = True
        else:
            error_jump = difference_in_iterates_jump / difference_from_init_jump
            converged_jump = error_jump < tol_convergence * 10
        converged = converged_jump and converged_contact and converged_mech

        logger.info(
            "Errors: displacement jump {:.2e}, contact force {:.2e}, matrix displacement {:.2e}, ".format(
                error_jump, error_contact, error_mech
            )
        )

        return error_mech, converged, diverged

    def compute_sifs_and_errors(self) -> np.ndarray:
        """
        Compute SIFs and the errors relative to the analytical solution for
        apertures, SIF I and SIF II.

        Returns
        -------
        errors : TYPE
            DESCRIPTION.

        """
        gb = self.gb
        errors = np.zeros(3)
        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)
            d_l, d_h = gb.node_props(g_l), gb.node_props(g_h)

            # Compute analytical and numerical SIFs
            K_an = m.analytical_SIFs()
            m._displacement_correlation(g_l, d_h, d_l, d)
            K_num = d_l[pp.PARAMETERS][m.mechanics_parameter_key]["SIFs"]
            ind = K_num[0].nonzero()[0]
            K_num = K_num[:, ind]

            # Compute analytical and numerical apertures
            projection = d_l["tangential_normal_projection"]
            a_num = m.reconstruct_local_displacement_jump(d, projection)[1]
            a_an = m.analytical_apertures()

            # Compute relative errors for all g_l.num_cells apertures and
            # mode one and two SIFs at the tips.
            errors[0] = L2_error(a_an, a_num, None)
            den = L2_norm(K_an[0])  # Fine since L2_norm(K_an[1]) = 0
            errors[1] = L2_norm(K_num[0] - K_an[0]) / den
            errors[2] = L2_norm(K_num[1] - K_an[1]) / den
            # print("\nErrors", errors)
        return errors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Axes for the error array are poisson ratios, mesh sizes and the three
    # errors computed (for apertures, SIF_I, SIF_II).
    all_errors = np.empty((3, 4, 3))
    poisson_ratios = np.linspace(0.1, 0.4, all_errors.shape[0])

    simplex = 1 == 1
    if simplex:
        file_name = "SIFs_simplex"
        mesh_sizes = np.array([2, 1, 0.5])  # , 0.25, 0.125])
        mesh_sizes = mesh_sizes[: all_errors.shape[1]]
    else:
        beta = np.pi / 2
        file_name = "SIFs_cartesian"
    folder_name = "exI"
    mesh_size = 10
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_bound": 5 * mesh_size,
    }

    nx = 50
    params = {
        "nl_convergence_tol": 1e-8,
        "max_iterations": 20,
        "file_name": file_name,
        "mesh_args": mesh_args,
        "folder_name": folder_name,
        "inclination": np.pi / 2,
        "poisson": 0.3,
        "boundary_traction": 1e-4,
        "simplex": simplex,
        "n_cells": [nx, nx, 4],
        "prepare_umfpack": False,
    }

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i, nu in enumerate(poisson_ratios):
        params["poisson"] = nu
        for j, h in enumerate(mesh_sizes):
            params["mesh_args"] = {
                "mesh_size_frac": h,
                "mesh_size_bound": 3 * h,
            }
            m = SneddonSIFTest(params)
            # Also compute mode II SIFs:
            m._is_tensile = False
            pp.run_stationary_model(m, params)
            all_errors[i, j] = m.compute_sifs_and_errors()
    data = {
        "all_errors": all_errors,
        "poisson_ratios": poisson_ratios,
        "mesh_sizes": mesh_sizes,
    }
    write_pickle(data, folder_name + "/all_errors")
