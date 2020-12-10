"""
Model class to be used together with an existing/"physical" model to yield a full propagation
model.

Will also be combined with case specific parameters.
"""
import scipy.sparse as sps
import time
import numpy as np
import porepy as pp
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


class TensilePropagation(pp.ConformingFracturePropagation):
    """
    One more round of cleaning remains for this and related classes!

    EK: On my reading, the only active function in this class is _candidate_faces(),
        which is a simplification of the corresponding method in the superclass.
        If correct, I suggest we try to integrate the present function as an option
        in the superclass, and drop this extra class.

    """

    def _sorted_propagation_faces(self, g_l: pp.Grid, d_l: Dict) -> np.ndarray:
        parameters_l = d_l[pp.PARAMETERS][self.mechanics_parameter_key]
        faces = parameters_l["propagate_faces"].nonzero()[0]
        faces = faces[g_l.tags["tip_faces"][faces]]
        K_equivalent = d_l[pp.PARAMETERS][self.mechanics_parameter_key][
            "SIFs_equivalent"
        ]
        ind = np.argsort(K_equivalent[faces])
        faces = np.atleast_1d(faces[ind][::-1])
        return faces

    def _pick_propagation_face(
        self,
        g_h: pp.Grid,
        g_l: pp.Grid,
        data_h: Dict,
        data_l: Dict,
        data_edge: Dict,
        face_l,
        neighbor_threshold: int = 0,
        force_neighbors: bool = False,
    ) -> None:
        """
        Pick out which matrix face to split for a fracture faces tagged as propagating
        using the precomputed propagation angle.

        Workflow:
            Check that the face_l is permissible
            Identify the corresponding edges_h (= nodes if self.Nd==2)
            The edges' faces_h are candidates for propagation
            Pick the candidate based on the propagation angle

        Parameters
        ----------
        g_h : pp.Grid
            Higer-dimensional grid.
        g_l : pp.Grid
            Lower-dimensional grid.
        data_h : Dict
            Data dictionary corresponding to g_h.
        data_l : Dict
            Data dictionary corresponding to g_l.
        data_edge : Dict
            Data dictionary corresponding to the edge formed by g_h and g_l.

        Returns
        -------
        None
            DESCRIPTION.
        Stores the matrix "propagation_face_map" identifying pairs of
        lower- and higherdimensional faces. During grid updates, the former will receive
        a new neighbour cell and the latter will be split.
        """
        nd = self.Nd
        # EK: I am almost sure this method is not used, and can be deleted.
        # Leave a breakpoint here, and take action if ever hit it.
        # NOTE: If we hit it, the signature of this method is likely wrong (at least it
        # is different from the corresponding method in the parent class), so we should
        # revise the implementation.
        print("The method was used after all. Remove breakpoint, do QC")
        breakpoint()
        face_l: np.ndarray = face_l[g_l.tags["tip_faces"][face_l]]
        if face_l.size == 0:
            face_faces = sps.csr_matrix((g_l.num_faces, g_h.num_faces))
            data_edge["propagation_face_map"]: sps.spmatrix = face_faces
            return

        fracture_faces_h = g_h.tags["fracture_faces"].nonzero()[0]

        tip_faces_l = g_l.tags["tip_faces"].nonzero()[0]
        tip_edges_h = tip_faces_l_to_edges_h(g_l, tip_faces_l, g_h)
        tip_edges_h.sort(axis=0)
        fracture_edges_h = np.empty((g_l.dim, 0), dtype=int)
        for frac_face_h in g_h.tags["fracture_faces"].nonzero()[0]:
            for frac_e_h in np.sort(edges_of_face(g_h, frac_face_h), axis=0).T:
                frac_e_h = frac_e_h.reshape((g_l.dim, 1))
                is_found = np.isin(fracture_edges_h, frac_e_h)
                is_found = np.any(np.all(is_found))
                if not is_found or fracture_edges_h.size == 0:
                    fracture_edges_h = np.hstack((fracture_edges_h, frac_e_h))
        edge_h = tip_faces_l_to_edges_h(g_l, face_l, g_h)
        fracture_nodes_h = np.unique(
            g_h.face_nodes[:, g_h.tags["fracture_faces"]].nonzero()[0]
        )

        faces_h_to_split = np.empty(0, dtype=int)
        faces_l_to_split = np.empty(0, dtype=int)

        candidate_faces_h, faces_l_loc = self._candidate_faces(
            g_h,
            edge_h,
            g_l,
            face_l,
            tip_edges_h,
            fracture_edges_h,
            fracture_faces_h,
            neighbor_threshold,
            force_neighbors,
        )
        if force_neighbors:
            face_h = candidate_faces_h
        else:
            faces_l_loc = np.empty(0, dtype=int)

            ## Pick the right candidate:
            # Direction of h-dim face centers from the tip
            tip_coords = np.reshape(g_l.face_centers[:nd, face_l], (nd, 1))
            face_center_vecs = g_h.face_centers[:nd, candidate_faces_h] - tip_coords
            face_center_vecs = face_center_vecs / np.linalg.norm(
                face_center_vecs, axis=0
            )
            # Propagation vector, with sign assuring a positive orientation
            # of the basis
            propagation_vector = self._propagation_vector(g_l, data_l, face_l)
            # Pick the candidate closest to the propagation point,
            # i.e. smallest angle between propagation vector and face center vector
            distances = pp.geometry.distances.point_pointset(
                propagation_vector, face_center_vecs
            )
            ind = np.argsort(distances)
            # There might be no candidate faces left after imposition of restriction
            # of permissible candidates
            if candidate_faces_h.size > 0:
                face_h = candidate_faces_h[ind[0]]

                edges_of_new_face = edges_of_face(g_h, face_h)
                edges_of_new_face.sort(axis=0)
                faces_l_loc = np.empty(0, dtype=int)
                for edge in edges_of_new_face.T:  # sort!
                    # Remove from tip edges if it was a tip, add if not
                    ind = np.all(np.isin(tip_edges_h, edge), axis=0)
                    if np.any(ind):
                        tip_edges_h = tip_edges_h[:, ~ind]
                        face_l_loc = tip_edge_h_to_face_l(g_l, g_h, edge)
                        if (
                            face_l_loc.size > 0
                        ):  # the else is a tip_edge_h arisen in this propagation step, and does not correspond to a tip to be opened
                            faces_l_loc = np.hstack((faces_l_loc, face_l_loc))

                    else:
                        tip_edges_h = np.hstack(
                            (tip_edges_h, edge.reshape((g_l.dim, 1)))
                        )
                        fracture_edges_h = np.hstack(
                            (fracture_edges_h, edge.reshape((g_l.dim, 1)))
                        )
        n_neigh = faces_l_loc.size
        if n_neigh > neighbor_threshold:
            faces_h_to_split = np.hstack((faces_h_to_split, np.tile(face_h, n_neigh)))
            faces_l_to_split = np.hstack((faces_l_to_split, faces_l_loc))
            fracture_faces_h = np.hstack((fracture_faces_h, face_h))

        face_faces = sps.csr_matrix(
            (np.ones(faces_l_to_split.shape), (faces_l_to_split, faces_h_to_split)),
            shape=(g_l.num_faces, g_h.num_faces),
        )
        data_edge["propagation_face_map"] = face_faces

    def _candidate_faces(
        self, g_h: pp.Grid, edge_h, g_l: pp.Grid, face_l: np.ndarray
    ) -> np.ndarray:
        """For a given edge (understood to be a fracture tip) in g_h, find the
        candidate faces that may be ready for a split.

        IMPLEMENTATION NOTE: This method is different from the identically named method
        in the parent class ConformingFracturePropagation in that fewer checks are done
        on the candidate faces. The present method is assumed to be used in a tensile
        fracturing regime, where the propagating fracture stays planar, and where the
        grid contains faces that fit this propagating geometry. In comparison, the method
        in the parent class aims at non-planar fractures, and thus needs to do much more
        checks to try to keep a reasonable fracture geometry also after propagation.

        """

        def faces_of_edge(g: pp.Grid, e: np.ndarray) -> np.ndarray:
            """
            Obtain indices of all faces sharing an edge.


            Parameters
            ----------
            g : pp.Grid
            e : np.ndarray
                The edge.

            Returns
            -------
            faces : np.ndarray
                Faces.
            """
            if g.dim == 1:
                faces = e
            elif g.dim == 2:
                faces = g.face_nodes[e].nonzero()[1]
            elif g.dim == 3:
                f_0 = g.face_nodes[e[0]].nonzero()[1]
                f_1 = g.face_nodes[e[1]].nonzero()[1]
                faces = np.intersect1d(f_0, f_1)
            else:
                raise ValueError("Grid dimension should be 1, 2 or 3")
            return faces

        # Find all the edge's neighboring faces
        candidate_faces = faces_of_edge(g_h, edge_h)

        # Exclude faces that are on a fracture
        are_fracture = g_h.tags["fracture_faces"][candidate_faces]
        candidate_faces = candidate_faces[np.logical_not(are_fracture)]
        return candidate_faces


class THMPropagationModel(TensilePropagation):
    def __init__(self, params):
        super().__init__(params)
        pp.THM.__init__(self, params)
        # Set additional case specific fields
        self.set_fields(params)

    ## THM + propagation specific methods
    def _initialize_new_variable_values(
        self, g: pp.Grid, d: Dict[str, Any], var: str, dofs: Dict[str, int]
    ) -> np.ndarray:
        """
        Overwrite the corresponding method in superclasses: The pressure variable is
        initialized to the atmospheric pressure. Apart from this, all other variables
        are initialized to zero.

        Parameters
        ----------
        g : pp.Grid
            Grid.
        d : Dict
            Data dictionary.
        var : str
            Name of variable.
        dofs : int
            Number of DOFs per cell (or face/node).

        Returns
        -------
        vals : np.ndarray
            Values for the new DOFs.

        """
        cell_dof = dofs.get("cells")
        n_new = d["cell_index_map"].shape[0] - d["cell_index_map"].shape[1]
        if var == self.scalar_variable:  # type: ignore
            vals = (
                np.ones(n_new * cell_dof) * pp.ATMOSPHERIC_PRESSURE / self.scalar_scale  # type: ignore
            )
        else:
            vals = np.zeros(n_new * cell_dof)
        return vals

    def _map_variables(self, solution: np.ndarray) -> np.ndarray:
        """
        In addition to super's mapping an initialization of all primary variables,
        map the face values (darcy_fluxes and stored boundary conditions) and
        quantities to be exported.

        Parameters
        ----------
        solution : np.ndarray
            Solution vector from before propagation.

        Returns
        -------
        new_solution : np.ndarray
            Mapped solution vector with initialized new DOFs.

        """
        # Map solution, and initialize for newly defined dofs
        new_solution = super()._map_variables(solution)
        self._map_face_values()
        return new_solution

    def _map_face_values(self) -> None:
        """
        Maps the following face values:
            old_bc_values, used by DivU
            darcy_fluxes, used by Upwind

        Returns
        -------
        None.

        """
        # g_h Darcy fluxes are first copied to both the split faces, then mapped
        # to the mortar grid and finally removed from d_h.
        # In d_l, we initialize zero fluxes on the new faces, since there was
        # no flux across fracture tips previous to propagation.

        t_key = self.temperature_parameter_key
        keys = (
            self.mechanics_parameter_key,
            self.mechanics_temperature_parameter_key,
        )
        gb = self.gb

        for g, d in gb:
            face_map: sps.spmatrix = d["face_index_map"]
            mapping = sps.kron(face_map, sps.eye(self.Nd))
            # Map darcy fluxes
            d[pp.PARAMETERS][t_key]["darcy_flux"] = (
                face_map * d[pp.PARAMETERS][t_key]["darcy_flux"]
            )
            if g.dim == self.Nd:
                # Duplicate darcy_fluxes for new faces ("other" side of new fracture)
                new_faces = d["new_faces"]
                old_faces = d["split_faces"]
                d[pp.PARAMETERS][t_key]["darcy_flux"][new_faces] = -d[pp.PARAMETERS][
                    t_key
                ]["darcy_flux"][old_faces]
                # Map bc values
                for key in keys:
                    old_vals = d[pp.PARAMETERS][key]["bc_values"]
                    new_vals = mapping * old_vals
                    new_ind = pp.fvutils.expand_indices_nd(d["new_faces"], self.Nd)
                    if new_ind.size > 0:
                        old_ind = pp.fvutils.expand_indices_nd(
                            d["split_faces"], self.Nd
                        )
                        new_vals[new_ind] = old_vals[old_ind]
                    d[pp.STATE][key]["bc_values"] = new_vals

        for e, d in gb.edges():
            cell_map: sps.spmatrix = d["cell_index_map"]
            mg: pp.MortarGrid = d["mortar_grid"]
            d[pp.PARAMETERS][t_key]["darcy_flux"] = (
                cell_map * d[pp.PARAMETERS][t_key]["darcy_flux"]
            )
            g_l, g_h = gb.nodes_of_edge(e)
            d_h = gb.node_props(g_h)
            new_ind = self._new_dof_inds(cell_map)
            fluxes_h: np.ndarray = d_h[pp.PARAMETERS][t_key]["darcy_flux"]
            new_mortar_fluxes = mg.primary_to_mortar_int() * fluxes_h
            d[pp.PARAMETERS][t_key]["darcy_flux"] += new_mortar_fluxes

        g = self._nd_grid()
        d = gb.node_props(g)
        d[pp.PARAMETERS][t_key]["darcy_flux"][g.tags["fracture_faces"]] = 0

    def before_newton_loop(self):
        self.convergence_status = False
        self._iteration = 0

    def update_discretizations(self):
        # For the moment, do a full rediscretization. A more targeted approach
        # should be possible.
        self._minimal_update_discretization()

    def before_newton_iteration(self) -> None:
        """Rediscretize non-linear terms.

        QUESTION: Should the parent be updated?
        """
        # First update parameters, then discretize all terms except those treated
        # by mpfa and mpsa in the highest dimension.
        # NOTE: We may end up unnecessarily rediscretizing a few terms, but the cost
        # of this is insignificant.
        self._iteration += 1

        ## First update parameters.
        # The Darcy fluxes were updated right after the previous Newton iteration
        # or in self.prepare_for_simulation(), thus no need to update these here.

        # Update apertures and specific volumes (e.g. compute from displacement jumps).
        # Store as iterate information.
        self.update_all_apertures(to_iterate=True)

        # Update parameters.
        # Depending on the implementation of set_parameters, this can for instance
        # update permeability as a function of aperture. Similarly, various other
        # quantities can be updated.
        self.set_parameters()

        ###
        # With updated parameters (including Darcy fluxes), we can now discretize
        # non-linear terms.

        # Discretize everything except terms relating to poro-elasticity and
        # diffusion (that is, discretize everything not handled by mpfa or mpsa).
        # NOTE: Accumulation terms in self.Nd could also have been excluded.
        term_list = [
            "!mpsa",
            "!stabilization",
            "!div_u",
            "!grad_p",
            "!diffusion",
        ]
        filt = pp.assembler_filters.ListFilter(term_list=term_list)
        # NOTE: No grid filter here, in pratice, all terms on lower-dimensional grids
        # (apart from diffusion) are discretized here, so is everything on the mortars
        self.assembler.discretize(filt=filt)

        # Discretize diffusion terms on lower-dimensional grids.
        for dim in range(self.Nd):
            grid_list = self.gb.grids_of_dimension(dim)
            if len(grid_list) == 0:
                continue
            filt = pp.assembler_filters.ListFilter(
                grid_list=grid_list,
                term_list=["diffusion"],
            )
            self.assembler.discretize(filt=filt)

    def after_propagation_loop(self):
        """
        TODO: Purge.

        Returns
        -------
        None.

        """
        ValueError("should not call this")

    def after_newton_iteration(self, solution: np.ndarray) -> None:

        super().after_newton_iteration(solution)

        # Update Darcy fluxes based on the newly converged pressure solution.
        # NOTE: For consistency between the discretization and solution, this is
        # done before updates to permeability or geometry (by fracture propagation).
        self.compute_fluxes()

    def after_newton_convergence(self, solution, errors, iteration_counter):
        """Propagate fractures if relevant. Update variables and parameters
        according to the newly calculated solution.
        """
        gb = self.gb

        # We export the converged solution *before* propagation:
        self.update_all_apertures(to_iterate=True)
        self.export_step()
        # NOTE: Darcy fluxes were updated in self.after_newton_iteration().
        # The fluxes are mapped to the new geometry (and fluxes are assigned for
        # newly formed faces) by the below call to self._map_variables().

        # Propagate fractures:
        #   i) Identify which faces to open in g_h
        #   ii) Split faces in g_h
        #   iii) Update g_l and the mortar grid. Update projections.
        self.evaluate_propagation()

        if self.propagated_fracture:
            # Update parameters and discretization

            for g, d in gb:
                if g.dim < self.Nd - 1:
                    # Should be really careful in this situation. Fingers crossed.
                    continue

                # Transfer information on new faces and cells from the format used
                # by self.evaluate_propagation to the format needed for update of
                # discretizations (see Discretization.update_discretization()).
                # TODO: This needs more documentation.
                new_faces = d.get("new_faces", np.array([], dtype=np.int))
                split_faces = d.get("split_faces", np.array([], dtype=np.int))
                modified_faces = np.hstack((new_faces, split_faces))
                update_info = {
                    "map_cells": d["cell_index_map"],
                    "map_faces": d["face_index_map"],
                    "modified_cells": d.get("new_cells", np.array([], dtype=np.int)),
                    "modified_faces": d.get("new_faces", modified_faces),
                }
                # d["update_discretization"] = update_info

            # Map variables after fracture propagation. Also initialize variables
            # for newly formed cells, faces and nodes.
            # Also map darcy fluxes and time-dependent boundary values (advection
            # and the div_u term in poro-elasticity).
            new_solution = self._map_variables(solution)

            # Update apertures: Both state (time step) and iterate.
            self.update_all_apertures(to_iterate=False)
            self.update_all_apertures(to_iterate=True)

            # Set new parameters.
            self.set_parameters()
            # For now, update discretizations will do a full rediscretization
            # TODO: Replace this with a targeted rediscretization.
            # We may want to use some of the code below (after return), but not all of
            # it.
            self._minimal_update_discretization()
        else:
            # No updates to the solution
            new_solution = solution

        # Finally, use super's method to do updates not directly related to
        # fracture propgation
        super().after_newton_convergence(new_solution, errors, iteration_counter)

        self.adjust_time_step()

        # Done!
        return

    def _minimal_update_discretization(self):
        # NOTE: Below here is an attempt at local updates of the discretization
        # matrices. For now, these are replaced by a full discretization at the
        # begining of each time step.

        # EK: Discretization is a pain, because of the flux term.
        # The advective term needs an updated (expanded faces) flux term,
        # to compute this, we first need to expand discretization of the
        # pressure diffusion terms.
        # It should be possible to do something smarter here, perhaps compute
        # fluxes before splitting, then transfer numbers and populate with other
        # values. Or something else.
        gb = self.gb

        t_0 = time.time()

        g_max = gb.grids_of_dimension(gb.dim_max())[0]
        grid_list = gb.grids_of_dimension(gb.dim_max() - 1).tolist()
        grid_list.append(g_max)

        data = gb.node_props(g_max)[pp.DISCRETIZATION_MATRICES]

        flow = {}
        for key in data["flow"]:
            flow[key] = data["flow"][key].copy()

        mech = {}
        for key in data["mechanics"]:
            mech[key] = data["mechanics"][key].copy()

        self.discretize_biot(update_after_geometry_change=False)

        for e, _ in gb.edges_of_node(g_max):
            grid_list.append((e[0], e[1], e))

        filt = pp.assembler_filters.ListFilter(
            variable_list=[self.scalar_variable, self.mortar_scalar_variable],
            term_list=[self.scalar_coupling_term],
            grid_list=grid_list,
        )
        self.assembler.discretize(filt=filt)

        grid_list = gb.grids_of_dimension(gb.dim_max() - 1).tolist()
        filt = pp.assembler_filters.ListFilter(
            term_list=["diffusion", "mass", "source"],
            variable_list=[self.scalar_variable],
            grid_list=grid_list,
        )
        # self.assembler.update_discretization(filt=filt)
        self.assembler.discretize(filt=filt)

        # Now that both variables and discretizations for the flux term have been
        # updated, we can compute the fluxes on the new grid.

        # self.compute_fluxes()

        # Update biot. Should be cheap.
        self.copy_biot_discretizations()
        # No need to update source term

        # Then the temperature discretizations. These are updated, to avoid full mpfa
        # in g_max
        temperature_terms = ["source", "diffusion", "mass", self.advection_term]
        filt = pp.assembler_filters.ListFilter(
            grid_list=[self._nd_grid()],
            variable_list=[self.temperature_variable],
            term_list=temperature_terms,
        )
        # self.assembler.update_discretization(filt=filt)
        self.assembler.discretize(filt=filt)

        # Pressure-temperature coupling terms
        coupling_terms = [self.s2t_coupling_term, self.t2s_coupling_term]
        filt = pp.assembler_filters.ListFilter(
            grid_list=[self._nd_grid()],
            variable_list=[self.temperature_variable, self.scalar_variable],
            term_list=coupling_terms,
        )
        self.assembler.discretize(filt=filt)

        # Build a list of all edges, and all couplings
        edge_list = []
        for e, _ in self.gb.edges():
            edge_list.append(e)
            edge_list.append((e[0], e[1], e))
        if len(edge_list) > 0:
            filt = pp.assembler_filters.ListFilter(grid_list=edge_list)
            self.assembler.discretize(filt=filt)

        # Finally, discretize terms on the lower-dimensional grids. This can be done
        # in the traditional way, as there is no Biot discretization here.
        for dim in range(0, self.Nd):
            grid_list = self.gb.grids_of_dimension(dim)
            if len(grid_list) > 0:
                filt = pp.assembler_filters.ListFilter(grid_list=grid_list)
                self.assembler.discretize(filt=filt)

        logger.info("Rediscretized in {} s.".format(time.time() - t_0))

    ## Methods specific to this project, but common to (some of) the examples
    def set_fields(self, params):
        """
        Set various fields to be used in the model.
        """
        # We operate on the temperature difference T-T_0, with T in Kelvin
        self.T_0_Kelvin = 500
        self.background_temp_C = pp.KELKIN_to_CELSIUS(self.T_0_Kelvin)
        # Scaling coefficients
        self.scalar_scale = 1e7
        self.temperature_scale = 1e0

        self.file_name = self.params["file_name"]
        self.folder_name = self.params["folder_name"]

        self.export_fields = [
            "u_exp",
            "p_exp",
            "T_exp",
            "traction_exp",
            "aperture_exp",
            "fluxes_exp",
            "cell_centers",
        ]

    # Geometry
    def create_grid(self) -> None:
        """
        Method that creates the GridBucket of a 2d or 3d domain.

        The geometry is defined through the method self._fractures() and the
        domain sizes stored in the dictionary self.box.
        This method sets self.gb and self.Nd.
        """

        # Define fractures
        self._fractures()

        x = self.box["xmax"] - self.box["xmin"]
        y = self.box["ymax"] - self.box["ymin"]
        nx = self.params.get("nx", 10)
        ny = self.params.get("ny", nx)
        ncells = [nx, ny]
        dims = [x, y]
        if "zmax" in self.box:
            ncells.append(self.params.get("nz", nx))
            dims.append(self.box["zmax"] - self.box["zmin"])
        gb = pp.meshing.cart_grid(self.fracs, ncells, physdims=dims)

        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()

        # Tag the wells
        self._tag_well_cells()
        self.n_frac = len(gb.grids_of_dimension(self.Nd - 1))

    # Numerics
    def assign_discretizations(self) -> None:
        """
        For long time steps, scaling the diffusive interface fluxes in the non-default
        way turns out to actually be beneficial for the condition number.
        """
        # Call parent class for disrcetizations.
        super().assign_discretizations()

        for e, d in self.gb.edges():
            d[pp.COUPLING_DISCRETIZATION][self.temperature_coupling_term][e][
                1
            ].kinv_scaling = False
            d[pp.COUPLING_DISCRETIZATION][self.scalar_coupling_term][e][
                1
            ].kinv_scaling = True

    def assemble_and_solve_linear_system(self, tol):
        if getattr(self, "report_A", True):
            A, b = self.assembler.assemble_matrix_rhs(add_matrices=False)
            for key in A.keys():
                logger.debug("{:.2e} {}".format(np.max(np.abs(A[key])), key))

        A, b = self.assembler.assemble_matrix_rhs()
        prepare_umfpack = self.params.get("prepare_umfpack", False)

        if prepare_umfpack:
            A.indices = A.indices.astype(np.int64)
            A.indptr = A.indptr.astype(np.int64)
        logger.debug("Max element in A {0:.2e}".format(np.max(np.abs(A))))
        logger.info(
            "Max {0:.2e} and min {1:.2e} A sum.".format(
                np.max(np.sum(np.abs(A), axis=1)), np.min(np.sum(np.abs(A), axis=1))
            )
        )
        t_0 = time.time()
        x = sps.linalg.spsolve(A, b)
        logger.info("Solved in {} s.".format(time.time() - t_0))
        return x

    def check_convergence(self, solution, prev_solution, init_solution, nl_params=None):
        g_max = self._nd_grid()
        uh_dof = self.assembler.dof_ind(g_max, self.displacement_variable)
        p_dof = np.array([], dtype=np.int)
        T_dof = np.array([], dtype=np.int)
        contact_dof = np.array([], dtype=np.int)
        for g, _ in self.gb:
            p_dof = np.hstack((p_dof, self.assembler.dof_ind(g, self.scalar_variable)))
            T_dof = np.hstack(
                (T_dof, self.assembler.dof_ind(g, self.temperature_variable))
            )
            if g.dim == self.Nd - 1:
                contact_dof = np.hstack(
                    (
                        contact_dof,
                        self.assembler.dof_ind(g, self.contact_traction_variable),
                    )
                )

        # Also find indices for the contact variables
        uj_dof = np.array([], dtype=np.int)
        for e, _ in self.gb.edges():
            if e[0].dim == self.Nd:
                uj_dof = np.hstack(
                    (
                        uj_dof,
                        self.assembler.dof_ind(e, self.mortar_displacement_variable),
                    )
                )

        # Pick out the solution from current, previous iterates, as well as the
        # initial guess.
        def differences(dofs):
            sol_now = solution[dofs]
            sol_prev = prev_solution[dofs]
            sol_init = init_solution[dofs]
            diff_iterates = np.sqrt(np.sum((sol_now - sol_prev) ** 2)) / sol_now.size
            diff_init = np.sqrt(np.sum((sol_now - sol_init) ** 2)) / sol_now.size
            norm = np.sqrt(np.sum(sol_now ** 2)) / sol_now.size
            return diff_iterates, diff_init, norm

        iterate_diff_T, init_diff_T, norm_T = differences(T_dof)
        iterate_diff_p, init_diff_p, norm_p = differences(p_dof)
        iterate_diff_uh, init_diff_uh, norm_uh = differences(uh_dof)
        iterate_diff_uj, init_diff_uj, norm_uj = differences(uj_dof)

        tol_convergence = nl_params["nl_convergence_tol"]
        # Not sure how to use the divergence criterion
        # tol_divergence = nl_params["nl_divergence_tol"]

        diverged = False

        # Check absolute convergence criterion
        def convergence(val, ref, atol, rtol=None):
            if rtol is None:
                rtol = atol
            if val < atol:
                return True, val
            error = val / ref
            return error < rtol, error

        scaled_convergence = 100 * tol_convergence
        converged_uh, error_uh = convergence(iterate_diff_uh, norm_uh, tol_convergence)
        converged_T, error_T = convergence(iterate_diff_T, norm_T, scaled_convergence)
        converged_p, error_p = convergence(iterate_diff_p, norm_p, tol_convergence)
        converged_uj, error_uj = convergence(iterate_diff_uj, norm_uj, tol_convergence)
        converged = (
            converged_uj
            # and converged_contact
            and converged_uh
            and converged_T
            and converged_p
        )

        logger.info(
            "Errors: displacement jump {:.2e}, matrix displacement {:.2e}, temperature {:.2e} and pressure {:.2e}".format(
                error_uj, error_uh, error_T, error_p
            )
        )
        logger.info(
            "Difference: displacement jump {:.2e}, matrix displacement {:.2e}, temperature {:.2e} and pressure {:.2e}".format(
                iterate_diff_uj, iterate_diff_uh, iterate_diff_T, iterate_diff_p
            )
        )
        return error_uh, converged, diverged

    def adjust_time_step(self):
        """
        Adjust the time step so that smaller time steps are used when the driving forces
        are changed. Also make sure to exactly reach the start and end time for
        each phase.
        """
        # Default is to just increase the time step somewhat
        self.time_step = getattr(self, "time_step_factor", 1.0) * self.time_step

        # We also want to make sure that we reach the end of each simulation phase
        for dt, lim in zip(self.phase_time_steps, self.phase_limits):
            diff = self.time - lim
            if diff < 0 and -diff <= self.time_step:
                self.time_step = -diff

            if np.isclose(self.time, lim):
                self.time_step = dt
        # And that the time step doesn't grow too large after the equilibration phase
        if self.time > 0:
            self.time_step = min(self.time_step, self.max_time_step)

    def compute_fluxes(self):
        """
        Compute fluxes.

        For 3d, the fluxes are damped after the fourth iteration.
        """
        use_smoothing = self.Nd == 3
        gb = self.gb
        for g, d in gb:
            pa = d[pp.PARAMETERS][self.temperature_parameter_key]
            if self._iteration > 1:
                pa["darcy_flux_1"] = pa["darcy_flux"].copy()
        for e, d in gb.edges():
            pa = d[pp.PARAMETERS][self.temperature_parameter_key]

            if self._iteration > 1:
                pa["darcy_flux_1"] = pa["darcy_flux"].copy()

        super().compute_fluxes()
        if not use_smoothing or self._iteration < 5:
            return
        a, b = 1, 1
        node_update, edge_update = 0, 0
        for g, d in gb:

            pa = d[pp.PARAMETERS][self.temperature_parameter_key]

            v1 = pa["darcy_flux_1"]
            v2 = pa["darcy_flux"]
            v_new = (a * v2 + b * v1) / (a + b)
            pa["darcy_flux"] = v_new
            node_update += np.sqrt(
                np.sum(np.power(v2 - v_new, 2)) / np.sum(np.power(v2, 2))
            )
        for e, d in gb.edges():
            pa = d[pp.PARAMETERS][self.temperature_parameter_key]

            v1 = pa["darcy_flux_1"]
            v2 = pa["darcy_flux"]
            v_new = (a * v2 + b * v1) / (a + b)
            pa["darcy_flux"] = v_new
            edge_update += np.sqrt(
                np.sum(np.power(v2 - v_new, 2)) / np.sum(np.power(v2, 2))
            )
        logger.info(
            "Smoothed fluxes by {:.2e} and edge {:.2e} at time {:.2e}".format(
                node_update, edge_update, self.time
            )
        )

    # Initialization etc.
    def initial_condition(self) -> None:
        """Initial values for the Darcy fluxes, p, T and u."""
        for g, d in self.gb:
            d[pp.PARAMETERS] = pp.Parameters()
            d[pp.PARAMETERS].update_dictionaries(
                [
                    self.mechanics_parameter_key,
                    self.mechanics_temperature_parameter_key,
                    self.scalar_parameter_key,
                    self.temperature_parameter_key,
                ]
            )
        self.update_all_apertures(to_iterate=False)
        self.update_all_apertures()
        super().initial_condition()

        for g, d in self.gb:
            u0 = self.initial_displacement(g)
            d[pp.PARAMETERS][self.temperature_parameter_key].update(
                {"darcy_flux": np.zeros(g.num_faces)}
            )
            p0 = self.initial_scalar(g)
            T0 = self.initial_temperature(g)
            state = {
                self.scalar_variable: p0,
                self.temperature_variable: T0,
            }
            iterate = {
                self.scalar_variable: p0,
                self.temperature_variable: T0,
                self.displacement_variable: u0,
            }

            pp.set_state(d, state)
            pp.set_iterate(d, iterate)
        for e, d in self.gb.edges():
            update = {self.mortar_displacement_variable: self.initial_displacement(e)}
            pp.set_state(d, update)
            pp.set_iterate(d, update)

    def initial_scalar(self, g) -> np.ndarray:
        """Hydrostatic pressure depending on _depth, which is set to 0 in exII."""
        depth = self._depth(g.cell_centers)
        return self.hydrostatic_pressure(g, depth) / self.scalar_scale

    def initial_temperature(self, g) -> np.ndarray:
        """Initial temperature is 0, but set to f(z) in exIV."""
        return np.zeros(g.num_cells)

    def initial_displacement(self, g):
        if isinstance(g, tuple):
            d = self.gb.edge_props(g)
            nc = d["mortar_grid"].num_cells
        else:
            d = self.gb.node_props(g)
            nc = g.num_cells
        return d[pp.STATE].get("initial_displacement", np.zeros((self.Nd * nc)))

    def compute_initial_displacement(self):
        """Is run prior to a time-stepping scheme. Use this to initialize
        displacement consistent with the given BCs, initial pressure and initial
        temperature.

        A modified version of the full equation system is solved. P and T are
        fixed by only considering the implicit mass matrix. The coupling
        contributions grad p and grad T are retained in the momentum balance.
        """
        self.prepare_simulation()
        var_d = self.displacement_variable

        # We need the source term for mechanics. Ensure no contribution for
        # p and T.
        for g, d in self.gb:
            d[pp.PARAMETERS][self.temperature_parameter_key]["source"] = np.zeros(
                g.num_cells
            )
            d[pp.PARAMETERS][self.scalar_parameter_key]["source"] = np.zeros(
                g.num_cells
            )

        # Terms to include. We have to retain the coupling terms to avoid a
        # singular matrix
        terms = [
            "mpsa",
            self.friction_coupling_term,
            "grad_p",
            "mass",
            "fracture_scalar_to_force_balance",
            self.advection_coupling_term,
            self.temperature_coupling_term,
            self.scalar_coupling_term,
            "empty",
            "source",
            # "matrix_temperature_to_force_balance",
            # "matrix_scalar_to_force_balance",
        ]
        filt = pp.assembler_filters.ListFilter(term_list=terms)
        A, b = self.assembler.assemble_matrix_rhs(filt=filt)

        if self.params.get("prepare_umfpack", False):
            A.indices = A.indices.astype(np.int64)
            A.indptr = A.indptr.astype(np.int64)
        x = sps.linalg.spsolve(A, b)
        self.assembler.distribute_variable(x)
        # Store the initial displacement (see method initial_displacement)
        g = self._nd_grid()
        d = self.gb.node_props(g)
        d[pp.STATE]["initial_displacement"] = d[pp.STATE][var_d].copy()
        for e, d in self.gb.edges():
            if e[0].dim == self.Nd:
                d[pp.STATE]["initial_displacement"] = d[pp.STATE][
                    self.mortar_displacement_variable
                ].copy()

    def prepare_simulation(self):
        """
        Copy of THM method which avoids overwriting self.gb and rediscretizing
        if the method is called a second time (after self.compute_initial_displacement).
        """
        first = not hasattr(self, "gb") or self.gb is None
        if first:
            self.create_grid()
            self.update_all_apertures(to_iterate=False)
            self.update_all_apertures()
            self._set_time_parameters()
            self.set_rock_and_fluid()
        self.initial_condition()
        self.set_parameters()
        if first:
            self.assign_variables()
            self.assign_discretizations()
            self.discretize()
        # Initialize Darcy fluxes
        self.compute_fluxes()

        self.initialize_linear_solver()
        self.export_step()

    def _tag_well_cells(self):
        """
        Tag well cells with unitary values, positive for injection cells and negative
        for production cells.
        """
        pass

    # Apertures and specific volumes
    def aperture(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the aperture of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["aperture"]
        else:
            return self.gb.node_props(g)[pp.STATE]["aperture"]

    def specific_volumes(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the specific volume of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["specific_volume"]
        else:
            return self.gb.node_props(g)[pp.STATE]["specific_volume"]

    def update_all_apertures(self, to_iterate=True):
        """
        To better control the aperture computation, it is done for the entire gb by a
        single function call. This also allows us to ensure the fracture apertures
        are updated before the intersection apertures are inherited.
        The aperture of a fracture is
            initial aperture + || u_n ||
        """
        gb = self.gb
        for g, d in gb:

            apertures = np.ones(g.num_cells)
            if g.dim == (self.Nd - 1):
                # Initial aperture

                apertures *= self.initial_aperture
                # Reconstruct the displacement solution on the fracture
                g_h = gb.node_neighbors(g)[0]
                data_edge = gb.edge_props((g, g_h))
                if pp.STATE in data_edge:
                    u_mortar_local = self.reconstruct_local_displacement_jump(
                        data_edge,
                        d["tangential_normal_projection"],
                        from_iterate=to_iterate,
                    )
                    # Magnitudes of normal components
                    # Absolute value to avoid negative volumes for non-converged
                    # solution (if from_iterate is True above)
                    apertures += np.absolute(u_mortar_local[-1])

            if to_iterate:
                pp.set_iterate(
                    d,
                    {"aperture": apertures.copy(), "specific_volume": apertures.copy()},
                )
            else:
                state = {
                    "aperture": apertures.copy(),
                    "specific_volume": apertures.copy(),
                }
                pp.set_state(d, state)

        for g, d in gb:
            parent_apertures = []
            num_parent = []
            if g.dim < (self.Nd - 1):
                for edges in gb.edges_of_node(g):
                    e = edges[0]
                    g_h = e[0]

                    if g_h == g:
                        g_h = e[1]

                    if g_h.dim == (self.Nd - 1):
                        d_h = gb.node_props(g_h)
                        if to_iterate:
                            a_h = d_h[pp.STATE][pp.ITERATE]["aperture"]
                        else:
                            a_h = d_h[pp.STATE]["aperture"]
                        a_h_face = np.abs(g_h.cell_faces) * a_h
                        mg = gb.edge_props(e)["mortar_grid"]
                        # Assumes g_h is primary
                        a_l = (
                            mg.mortar_to_secondary_avg()
                            * mg.primary_to_mortar_avg()
                            * a_h_face
                        )
                        parent_apertures.append(a_l)
                        num_parent.append(
                            np.sum(mg.mortar_to_secondary_int().A, axis=1)
                        )
                    else:
                        raise ValueError("Intersection points not implemented in 3d")
                parent_apertures = np.array(parent_apertures)
                num_parents = np.sum(np.array(num_parent), axis=0)

                apertures = np.sum(parent_apertures, axis=0) / num_parents

                specific_volumes = np.power(
                    apertures, self.Nd - g.dim
                )  # Could also be np.product(parent_apertures, axis=0)
                if to_iterate:
                    pp.set_iterate(
                        d,
                        {
                            "aperture": apertures.copy(),
                            "specific_volume": specific_volumes.copy(),
                        },
                    )
                else:
                    state = {
                        "aperture": apertures.copy(),
                        "specific_volume": specific_volumes.copy(),
                    }
                    pp.set_state(d, state)

        return apertures

    # Parameter assignment
    def set_mechanics_parameters(self):
        """Mechanical parameters.
        Note that we divide the momentum balance equation by self.scalar_scale.
        A homogeneous initial temperature is assumed.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                rock = self.rock
                lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                bc = self.bc_type_mechanics(g)
                bc_values = self.bc_values_mechanics(g)
                sources = self.source_mechanics(g)

                # In the momentum balance, the coefficient hits the scalar, and should
                # not be scaled. Same goes for the energy balance, where we divide all
                # terms by T_0, hence the term originally beta K T d(div u) / dt becomes
                # beta K d(div u) / dt = coupling_coefficient d(div u) / dt.
                coupling_coefficient = self.biot_alpha(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                        "source": sources,
                        "fourth_order_tensor": C,
                        "biot_alpha": coupling_coefficient,
                        "time_step": self.time_step,
                        "shear_modulus": self.rock.MU,
                        "poisson_ratio": self.rock.POISSON_RATIO,
                    },
                )

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_temperature_parameter_key,
                    {
                        "biot_alpha": self.biot_beta(g),
                        "bc_values": bc_values,
                    },
                )
            elif g.dim == self.Nd - 1:
                K_crit = self.rock.SIF_crit * np.ones((self.Nd, g.num_faces))
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "friction_coefficient": self.rock.FRICTION_COEFFICIENT,
                        "contact_mechanics_numerical_parameter": 1e1,
                        "dilation_angle": np.radians(3),
                        "time": self.time,
                        "SIFs_critical": K_crit,
                    },
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            # Parameters for the surface diffusion. Not used as of now.
            pp.initialize_data(
                mg,
                d,
                self.mechanics_parameter_key,
                {"mu": self.rock.MU, "lambda": self.rock.LAMBDA},
            )

    def set_scalar_parameters(self):

        for g, d in self.gb:
            specific_volumes = self.specific_volumes(g)

            # Define boundary conditions for flow
            bc = self.bc_type_scalar(g)
            # Set boundary condition values
            bc_values = self.bc_values_scalar(g)

            biot_coefficient = self.biot_alpha(g)
            compressibility = self.fluid.COMPRESSIBILITY

            mass_weight = compressibility * self.porosity(g)
            if g.dim == self.Nd:
                mass_weight += (
                    biot_coefficient - self.porosity(g)
                ) / self.rock.BULK_MODULUS

            mass_weight *= self.scalar_scale * specific_volumes

            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "biot_alpha": biot_coefficient,
                    "time_step": self.time_step,
                    "ambient_dimension": self.Nd,
                    "source": self.source_scalar(g),
                },
            )

            t2s_coupling = (
                self.scalar_temperature_coupling_coefficient(g)
                * specific_volumes
                * self.temperature_scale
            )
            pp.initialize_data(
                g,
                d,
                self.t2s_parameter_key,
                {"mass_weight": t2s_coupling, "time_step": self.time_step},
            )
        self.set_vector_source()

        self.set_permeability_from_aperture()

    def set_temperature_parameters(self):
        """temperature parameters.
        The entire equation is divided by the initial temperature in Kelvin.
        """

        for g, d in self.gb:
            T0 = self.T_0_Kelvin
            div_T_scale = self.temperature_scale / self.length_scale ** 2 / T0
            kappa_f = self.fluid.thermal_conductivity() * div_T_scale
            kappa_s = self.rock.thermal_conductivity() * div_T_scale

            heat_capacity_s = (
                self.rock.specific_heat_capacity(self.background_temp_C)
                * self.rock.DENSITY
            )
            heat_capacity_f = self.fluid_density(g) * self.fluid.specific_heat_capacity(
                self.background_temp_C
            )
            # Aperture and cross sectional area
            specific_volumes = self.specific_volumes(g)
            # Define boundary conditions for flow
            bc = self.bc_type_temperature(g)
            # Set boundary condition values
            bc_values = self.bc_values_temperature(g)
            # and source values
            biot_coefficient = self.biot_beta(g)

            mass_weight = (
                self._effective(g, heat_capacity_f, heat_capacity_s)
                * specific_volumes
                * self.temperature_scale
                / T0
            )

            thermal_conductivity = pp.SecondOrderTensor(
                self._effective(g, kappa_f, kappa_s) * specific_volumes
            )
            # darcy_fluxes are length scaled already
            advection_weight = heat_capacity_f * self.temperature_scale / T0

            pp.initialize_data(
                g,
                d,
                self.temperature_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "second_order_tensor": thermal_conductivity,
                    "advection_weight": advection_weight,
                    "biot_alpha": biot_coefficient,
                    "time_step": self.time_step,
                    "source": self.source_temperature(g),
                    "ambient_dimension": self.Nd,
                },
            )

            s2t_coupling = (
                self.scalar_temperature_coupling_coefficient(g)
                * specific_volumes
                * self.scalar_scale
            )
            pp.initialize_data(
                g,
                d,
                self.s2t_parameter_key,
                {"mass_weight": s2t_coupling, "time_step": self.time_step},
            )

        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            # T0 = self.T_0_Kelvin + self._T(mg)
            div_T_scale = (
                self.temperature_scale / self.length_scale ** 2 / self.T_0_Kelvin
            )
            kappa_f = self.fluid.thermal_conductivity() * div_T_scale
            a_l = self.aperture(g_l)
            V_h = self.specific_volumes(g_h)
            a_mortar = mg.secondary_to_mortar_avg() * a_l
            kappa_n = 2 / a_mortar * kappa_f
            tr = np.abs(g_h.cell_faces)
            V_j = mg.primary_to_mortar_int() * tr * V_h
            kappa_n = kappa_n * V_j
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.temperature_parameter_key,
                {"normal_diffusivity": kappa_n},
            )

    # BCs. Assumes _p_and_T_dir_faces
    def bc_type_scalar(self, g) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(g, self._p_and_T_dir_faces(g), "dir")

    def bc_type_temperature(self, g) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(g, self._p_and_T_dir_faces(g), "dir")

    # Common parameters
    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to those of granite and water.
        We ignore all temperature dependencies of the parameters.
        """
        self.rock = Granite()
        self.fluid = Water()

    def porosity(self, g) -> float:
        if g.dim == self.Nd:
            return 0.05
        else:
            return 1.0

    def _effective(self, g, param_f, param_s) -> float:
        """Compute effective thermal parameter as porosity weighted sum."""
        phi = self.porosity(g)
        return phi * param_f + (1 - phi) * param_s

    def biot_alpha(self, g) -> np.ndarray:
        if g.dim == self.Nd:
            return 0.8
        else:
            return 1.0

    def biot_beta(self, g):
        """
        For TM, the coefficient is the product of the bulk modulus (=inverse of
        the compressibility) and the volumetric thermal expansion coefficient.
        """
        if g.dim == self.Nd:
            # Factor 3 for volumetric/linear, since the pp.Granite
            # thermal expansion expansion coefficient is the linear one at 20 degrees C.
            return self.rock.BULK_MODULUS * 3 * self.rock.THERMAL_EXPANSION
        else:
            # Solution debendent coefficient computed from previous iterate,
            # see Eq. (xx)
            iterate = self.gb.node_props(g)[pp.STATE][pp.ITERATE]
            T_k = iterate[self.temperature_variable] * self.temperature_scale
            T0K = self.T_0_Kelvin

            return T_k / T0K * self.fluid_density(g)

    def scalar_temperature_coupling_coefficient(self, g) -> float:
        """
        The temperature-pressure coupling coefficient is porosity times thermal
        expansion. The pressure and
        scalar scale must be accounted for wherever this coefficient is used.
        """
        b_f = self.fluid.thermal_expansion(self.background_temp_C)
        if g.dim < self.Nd:
            coeff = -b_f
        else:
            b_s = self.rock.THERMAL_EXPANSION
            phi = self.porosity(g)
            coeff = -(phi * b_f + (self.biot_alpha(g) - phi) * b_s)
            # coeff = -self._effective(g, b_f, b_s)
        return coeff

    def fluid_density(self, g, dp=None, dT=None) -> np.ndarray:
        """Density computed from current pressure and temperature solution, both
        taken from the previous iterate.

        \rho = \rho_0 * exp[ compressibility * (p - p_0)
                            + thermal_expansion * (T-T_0) ],

        with    \rho_0 = 1000
                p_0 = 1 atm
                T_0 = 20 degrees C

        Clipping of the solution to aid convergence. Should not affect the
        converged solution given the chosen bounds.
        """
        iterate = self.gb.node_props(g)[pp.STATE][pp.ITERATE]
        if dp is None:
            p_k = iterate[self.scalar_variable] * self.scalar_scale
            dp = np.clip(p_k, a_min=-1e10, a_max=1e10)
            # Use hydrostatic pressure as reference
            dp = dp - pp.ATMOSPHERIC_PRESSURE
        if dT is None:
            T_k = iterate[self.temperature_variable] * self.temperature_scale
            dT = np.clip(T_k, a_min=-self.T_0_Kelvin, a_max=self.T_0_Kelvin)
            # Use 20 degrees C as reference
            dT = dT - (20 - self.background_temp_C)

        rho_0 = 1e3 * (pp.KILOGRAM / pp.METER ** 3) * np.ones(g.num_cells)
        rho = rho_0 * np.exp(
            dp * self.fluid.COMPRESSIBILITY - dT * self.fluid.thermal_expansion(dT)
        )
        return rho

    def set_permeability_from_aperture(self):
        """
        Cubic law in fractures, rock permeability in the matrix.
        """
        # Viscosity has units of Pa s, and is consequently divided by the scalar scale.
        viscosity = self.fluid.dynamic_viscosity() / self.scalar_scale
        gb = self.gb
        key = self.scalar_parameter_key
        for g, d in gb:
            if g.dim < self.Nd:
                # Use cubic law in fractures. First compute the unscaled
                # permeability
                apertures = self.aperture(g, from_iterate=True)
                apertures_unscaled = apertures * self.length_scale
                k = np.power(apertures_unscaled, 2) / 12 / viscosity
                d[pp.PARAMETERS][key]["perm_nu"] = k
                # Multiply with the cross-sectional area, which equals the apertures
                # for 2d fractures in 3d
                specific_volumes = self.specific_volumes(g, True)

                k = k * specific_volumes

                # Divide by fluid viscosity and scale back
                kxx = k / self.length_scale ** 2
            else:
                # Use the rock permeability in the matrix
                kxx = (
                    self.rock.PERMEABILITY
                    / viscosity
                    * np.ones(g.num_cells)
                    / self.length_scale ** 2
                )
            K = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][key]["second_order_tensor"] = K

        # Normal permeability inherited from the neighboring fracture g_l
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            g_l, g_h = gb.nodes_of_edge(e)
            data_l = gb.node_props(g_l)
            a = self.aperture(g_l, True)
            V = self.specific_volumes(g_l, True)
            V_h = self.specific_volumes(g_h, True)
            # We assume isotropic permeability in the fracture, i.e. the normal
            # permeability equals the tangential one
            k_s = data_l[pp.PARAMETERS][key]["second_order_tensor"].values[0, 0]
            # Division through half the aperture represents taking the (normal) gradient
            kn = mg.secondary_to_mortar_int() * np.divide(k_s, a * V / 2)
            tr = np.abs(g_h.cell_faces)
            V_j = mg.primary_to_mortar_int() * tr * V_h
            kn = kn * V_j
            pp.initialize_data(mg, d, key, {"normal_diffusivity": kn})

    def source_scalar(self, g) -> np.ndarray:
        """
        Source term for the scalar equation.
        In addition to regular source terms, we add a contribution compensating
        for the added volume in the conservation equation.

        For slightly compressible flow in the present formulation, this has units of m^3.

        Sources are handled by ScalarSource discretizations.
        The implicit scheme yields multiplication of the rhs by dt, but
        this is not incorporated in ScalarSource, hence we do it here.
        """
        rhs = np.zeros(g.num_cells)
        if g.dim < self.Nd:
            d = self.gb.node_props(g)
            new_cells = d.get("new_cells", np.array([], dtype=np.int))
            added_volume = self.initial_aperture * g.cell_volumes[new_cells]
            rhs[new_cells] -= added_volume

        return rhs

    def source_mechanics(self, g) -> np.ndarray:
        """
        Gravity term.
        """
        values = np.zeros((self.Nd, g.num_cells))
        if self.gravity_on:
            values[self.Nd - 1] = (
                pp.GRAVITY_ACCELERATION
                * self.rock.DENSITY
                * g.cell_volumes
                * self.length_scale
                / self.scalar_scale
                * self.gravity_on
            )
        return values.ravel("F")

    def set_vector_source(self):
        if not getattr(self, "gravity_on"):
            return
        for g, d in self.gb:
            grho = (
                pp.GRAVITY_ACCELERATION
                * self.fluid_density(g)
                / self.scalar_scale
                * self.length_scale
            )
            gr = np.zeros((self.Nd, g.num_cells))
            gr[self.Nd - 1, :] = -grho
            d[pp.PARAMETERS][self.scalar_parameter_key]["vector_source"] = gr.ravel("F")
        for e, data_edge in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            params_l = self.gb.node_props(g1)[pp.PARAMETERS][self.scalar_parameter_key]
            mg = data_edge["mortar_grid"]
            grho = (
                mg.secondary_to_mortar_avg()
                * params_l["vector_source"][self.Nd - 1 :: self.Nd]
            )
            a = mg.secondary_to_mortar_avg() * self.aperture(g1)
            gravity = np.zeros((self.Nd, mg.num_cells))
            gravity[self.Nd - 1, :] = grho * a / 2

            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"vector_source": gravity.ravel("F")},
            )

    # Solution storing and export
    def _set_exporter(self):
        self.exporter = pp.Exporter(
            self.gb,
            self.file_name,
            folder_name=self.viz_folder_name + "_vtu",
            fixed_grid=False,
        )
        self.export_times = []

    def export_step(self):
        """
        Export the current solution to vtu. The method sets the desired values in d[pp.STATE].
        For some fields, it provides zeros in the dimensions where the variable is not defined,
        or pads the vector values with zeros so that they have three components, as required
        by ParaView.
        We use suffix _exp on all exported variables, to separate from scaled versions also
        stored in d[pp.STATE].
        """
        if "exporter" not in self.__dict__:
            self._set_exporter()

        for g, d in self.gb:
            iterate = d[pp.STATE][pp.ITERATE]
            d[pp.STATE]["cell_centers"] = g.cell_centers.copy()
            ## First export Darcy fluxes:
            dis = d[pp.PARAMETERS][self.temperature_parameter_key]["darcy_flux"]
            if g.dim == self.Nd:
                for e in self.gb.edges_of_node(g):
                    d_e = self.gb.edge_props(e[0])
                    mg = d_e["mortar_grid"]
                    dis_e = d_e[pp.PARAMETERS][self.temperature_parameter_key][
                        "darcy_flux"
                    ]
                    faces_on_fracture_surface = (
                        mg.primary_to_mortar_int().tocsr().indices
                    )
                    sign = g.signs_and_cells_of_boundary_faces(
                        faces_on_fracture_surface
                    )[0]
                    dis = dis + mg.mortar_to_primary_int() * (sign * dis_e)
            fluxes = g.face_normals * dis / g.face_areas
            scalar_div = g.cell_faces

            # Vector extension, convert to coo-format to avoid odd errors when one
            # grid dimension is 1 (this may return a bsr matrix)
            # The order of arguments to sps.kron is important.
            block_div = sps.kron(scalar_div, sps.eye(3)).tocsc()

            proj = np.abs(block_div.transpose().tocsr())

            cell_flux = proj * (fluxes.ravel("F"))
            d[pp.STATE]["fluxes_exp"] = cell_flux.reshape((3, g.num_cells), order="F")

            ## Then handle u and contact traction, which are dimension dependent
            if g.dim == self.Nd:
                pad_zeros = np.zeros((3 - g.dim, g.num_cells))
                u = iterate[self.displacement_variable].reshape(
                    (self.Nd, -1), order="F"
                )
                u_exp = np.vstack((u * self.length_scale, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["traction_exp"] = np.zeros(d[pp.STATE]["u_exp"].shape)
            elif g.dim == (self.Nd - 1):
                pad_zeros = np.zeros((2 - g.dim, g.num_cells))
                g_h = self.gb.node_neighbors(g)[0]
                data_edge = self.gb.edge_props((g, g_h))

                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, d["tangential_normal_projection"], from_iterate=True
                )
                u_exp = np.vstack((u_mortar_local * self.length_scale, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp
                traction = (
                    iterate[self.contact_traction_variable].reshape(
                        (self.Nd, -1), order="F"
                    )
                    / g.cell_volumes
                )

                d[pp.STATE]["traction_exp"] = (
                    np.vstack((traction, pad_zeros)) * self.scalar_scale
                )

            ## Apertures, p and T
            d[pp.STATE]["aperture_exp"] = self.aperture(g) * self.length_scale

            d[pp.STATE]["p_exp"] = iterate[self.scalar_variable] * self.scalar_scale

            d[pp.STATE]["T_exp"] = (
                iterate[self.temperature_variable] * self.temperature_scale
            )

        self.exporter.write_vtk(self.export_fields, time_step=self.time, grid=self.gb)
        self.export_times.append(self.time)

        new_sizes = np.zeros(len(self.gb.grids_of_dimension(self.Nd - 1)))
        for i, g in enumerate(self.gb.grids_of_dimension(self.Nd - 1)):
            new_sizes[i] = np.sum(g.cell_volumes) * self.length_scale ** 2
        if hasattr(self, "fracture_sizes"):
            self.fracture_sizes = np.vstack((self.fracture_sizes, new_sizes))
        else:
            self.fracture_sizes = new_sizes

    def export_pvd(self):
        """
        At the end of the simulation, after the final vtu file has been exported, the
        pvd file for the whole simulation is written by calling this method.
        """
        self.exporter.write_pvd(np.array(self.export_times))

    def _update_iterate(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        Calls ContactMechanicsBiot version, and additionally updates the iterate solutions
        in d[pp.STATE][pp.ITERATE] are updated for the scalar variable, to be used
        for flux computations by compute_darcy_fluxes.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        super()._update_iterate(solution_vector)
        # HACK: This is one big hack to get the export working.
        # Ugly, but doesn't affect solution
        assembler = self.assembler
        variable_names = []
        for pair in assembler.block_dof.keys():
            variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in assembler.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                if isinstance(g, tuple):
                    continue
                else:
                    data = self.gb.node_props(g)

                    # g is a node (not edge)

                    # Save displacement for export. The export hacks are getting ugly!
                    if name == self.displacement_variable:
                        u = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.node_props(g)
                        data[pp.STATE][pp.ITERATE][
                            self.displacement_variable
                        ] = u.copy()


class Water:
    """
    Fluid phase.
    """

    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref
        self.VISCOSITY = 1 * pp.MILLI * pp.PASCAL * pp.SECOND
        self.COMPRESSIBILITY = 4e-10 / pp.PASCAL
        self.BULK_MODULUS = 1 / self.COMPRESSIBILITY

    def thermal_expansion(self, delta_theta):
        """ Units: m^3 / m^3 K, i.e. volumetric """
        return 4e-4

    def thermal_conductivity(self, theta=None):  # theta in CELSIUS
        """ Units: W / m K """
        if theta is None:
            theta = self.theta_ref
        return 0.6

    def specific_heat_capacity(self, theta=None):  # theta in CELSIUS
        """ Units: J / kg K """
        return 4200

    def dynamic_viscosity(self, theta=None):  # theta in CELSIUS
        """Units: Pa s"""
        return 0.001

    def hydrostatic_pressure(self, depth, theta=None):
        rho = 1e3 * (pp.KILOGRAM / pp.METER ** 3)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE


class Granite(pp.Granite):
    """
    Solid phase.
    """

    def __init__(self, theta_ref=None):
        super().__init__(theta_ref)
        self.BULK_MODULUS = pp.params.rock.bulk_from_lame(self.LAMBDA, self.MU)

        self.PERMEABILITY = 1e-14

        self.SIF_crit = 5e5  # Obs changed for ex 1 from 1e5

        # Increases with T https://link.springer.com/article/10.1007/s00603-020-02303-z
        self.THERMAL_EXPANSION = 5e-5

        self.FRICTION_COEFFICIENT = 0.8

    def thermal_conductivity(self, theta=None):
        return 2.0  # Ranges approx 1.7 to 4 according to Wikipedia


# EK: My guess is we can delete functions below.


def tip_faces_l_to_edges_h(g_l, faces_l, g_h):
    # Find the edges
    nodes_l, _, _ = sps.find(g_l.face_nodes[:, faces_l])
    # Obtain the global index of all nodes
    global_nodes = g_l.global_point_ind[nodes_l]

    # Prepare for checking intersection. ind_l is used to reconstruct non-unique
    # nodes later.
    global_nodes, ind_l = np.unique(global_nodes, return_inverse=True)
    # Find g_h indices of unique global nodes
    nodes_l, nodes_h, inds = np.intersect1d(
        g_h.global_point_ind, global_nodes, assume_unique=False, return_indices=True
    )
    # Reconstruct non-unique and reshape to edges (first dim is 2 if nd=3)
    edges_h = np.reshape(nodes_h[ind_l], (g_l.dim, faces_l.size), order="f")
    return edges_h


def tip_edge_h_to_face_l(g_l: pp.Grid, g_h: pp.Grid, edge_h: np.ndarray) -> np.ndarray:
    """
    Assumes all edges_h actually correspond to some face in g_l.

    Parameters
    ----------
    g_l : pp.Grid
        DESCRIPTION.
    g_h : pp.Grid
        DESCRIPTION.
    edges_h : np.ndarray
        DESCRIPTION.

    Returns
    -------
    faces_l : np.ndarray
        DESCRIPTION.

    """
    # Obtain the global index of all nodes
    global_nodes = g_h.global_point_ind[edge_h]

    # Find g_l indices of unique global nodes
    _, nodes_l, _ = np.intersect1d(
        g_l.global_point_ind, global_nodes, assume_unique=False, return_indices=True
    )
    if nodes_l.size == edge_h.size:

        face_l = faces_of_nodes(g_l, nodes_l)
        return face_l
    else:
        return np.empty(0, dtype=int)


def edges_of_face(g, face):
    local_nodes = g.face_nodes[:, face].nonzero()[0]
    pts = g.nodes[:, local_nodes]

    # Faces are defined by one node in 1d and two in 2d. This requires
    # dimension dependent treatment:
    if g.dim == 3:
        # Sort nodes clockwise (!)
        # ASSUMPTION: This assumes that the new cell is star-shaped with respect to the
        # local cell center. This should be okay.
        map_to_sorted = pp.utils.sort_points.sort_point_plane(
            pts, g.face_centers[:, face]
        )
        local_nodes = local_nodes[map_to_sorted]
        edges = np.vstack((local_nodes, np.hstack((local_nodes[1:], local_nodes[0]))))
    else:
        edges = np.atleast_2d(local_nodes)
    return edges


def faces_of_nodes(g: pp.Grid, e: np.ndarray) -> np.ndarray:
    """
    Obtain indices of all faces sharing one or two nodes.


    Parameters
    ----------
    g : pp.Grid
    e : np.ndarray
        The edge.

    Returns
    -------
    faces : np.ndarray
        Faces.
    """
    # if g.dim == 1:
    #     faces = e
    if e.size < 2:
        assert g.dim < 3
        faces = g.face_nodes[e[0]].nonzero()[1]
    elif e.size == 2:
        f_0 = g.face_nodes[e[0]].nonzero()[1]
        f_1 = g.face_nodes[e[1]].nonzero()[1]
        faces = np.intersect1d(f_0, f_1)
    else:
        raise NotImplementedError
    return faces


def fracture_edges(g_h):
    fracture_edges = np.empty((g_h.dim - 1, 0), dtype=int)
    for frac_face in g_h.tags["fracture_faces"].nonzero()[0]:
        for frac_e in np.sort(edges_of_face(g_h, frac_face), axis=0).T:
            frac_e = frac_e.reshape((g_h.dim - 1, 1))
            is_found = np.isin(fracture_edges, frac_e)
            is_found = np.any(np.all(is_found, axis=0))
            if not is_found or fracture_edges.size == 0:
                fracture_edges = np.hstack((fracture_edges, frac_e))
    return fracture_edges
