// C++ headers

// Deal.II headers

// Project headers
#include<EddElasticityLin.h>

namespace StructuralOptimization {

  template<int dim>
  EddElasticityLin<dim>::EddElasticityLin(Parameter &par_, Mesh<dim> &mesh_)
      :
      EddBVP_<dim>(par_, mesh_),
      mpi_communicator(par_.mpi_communicator),
      n_mpi_processes(par_.n_mpi_processes),
      this_mpi_process(par_.this_mpi_process),
      pcout(par_.pcout),
      compute_timer(par_.compute_timer),
      _fe_void(FE_Nothing<dim>(), dim),
      _fe_solid(FE_Q<dim>(par_.data.poly_degree), dim),
      _fe_shape(FE_Q<dim - 1, dim>(par_.data.poly_degree), dim),
      _mat(par_.data),
      bvp_output(par_.data, par_.mpi_communicator, par_.compute_timer, "edd-LE", *mesh_.base) {
// From this point remember in all collections the first item is void and the 2nd item is solid.
    _fe_collection.push_back(_fe_void);
    _fe_collection.push_back(_fe_solid);

    if (par_.data.problem_type != "edd")
      bvp_output.SwitchOffOutput();

  }

  template<int dim>
  EddElasticityLin<dim>::~EddElasticityLin() {
    this->_dof_handler_base.clear();
    this->_dof_handler_shape.clear();
    this->_smoothing_dof_handler.clear();
  }

  template<int dim>
  void EddElasticityLin<dim>::_SetupSystem() {

    TimerOutput::Scope t(compute_timer, "EddElasticityLin<dim>::_SetupSystem");

    // Setting active fe index
    /** Loop over all the cells in hp_dof_handler_hold_all and check if cell
    * is outside or inside using the _cell_is_outside(cell) or _cell_is_inside(cell)
    * function and set the flag cell->set_active_fe_index(0) or cell->set_active_fe_index(1)*/
    for (const auto &cell : this->_dof_handler_base.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;
     // if(cell->material_id() == e_outside_cell)
       // continue;
      if (EddTools::CellIsOutside<dim>(cell))
        cell->set_active_fe_index(0);
      else if (EddTools::CellIsInside<dim>(cell))
        cell->set_active_fe_index(1);
      else
        std::cerr << __FILE__ << ":" << __LINE__ << "| Cell is neither inside nor outside" << std::endl;
    }

    this->_dof_handler_base.distribute_dofs(_fe_collection);
    this->_dof_handler_shape.distribute_dofs(_fe_shape);

    // DoFRenumbering::subdomain_wise(this->_dof_handler_base);

// Now we get the locally owned dofs, that is the dofs that our local
// to this processor. These dofs corresponding entries in the
// matrix and vectors that we will write to.
    this->_locally_owned_dofs = this->_dof_handler_base.locally_owned_dofs();

// In additon to the locally owned dofs, we also need the the locally
// relevant dofs.  These are the dofs that have read access to and we
// need in order to do computations on our processor, but, that
// we do not have the ability to write to.
    DoFTools::extract_locally_relevant_dofs(this->_dof_handler_base, this->_locally_relevant_dofs);

    this->_solution.reinit(this->_locally_owned_dofs, this->_locally_relevant_dofs, mpi_communicator);
    this->_system_rhs.reinit(this->_locally_owned_dofs, mpi_communicator);

// Setup constraints
    this->_constraints.clear();
    this->shape_constraints.clear();

    this->_constraints.reinit(this->_locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(this->_dof_handler_base, this->_constraints);
    DoFTools::make_hanging_node_constraints(this->_dof_handler_shape, this->shape_constraints);

// From deal.ii test- trillinos_step-27.cc
#ifdef DEBUG
// We did not think about hp constraints on ghost cells yet.
// Thus, we are content with verifying their consistency for now.
    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(this->_dof_handler_base, locally_active_dofs);
    AssertThrow(this->_constraints.is_consistent_in_parallel(
        this->_dof_handler_base.locally_owned_dofs_per_processor(),
        locally_active_dofs,
        mpi_communicator,
/*verbose=*/true),
                ExcMessage(
                    "AffineConstraints object contains inconsistencies!"));
#endif

    this->_constraints.close();

    DynamicSparsityPattern dsp(this->_locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(this->_dof_handler_base, dsp, this->_constraints, false);
    SparsityTools::distribute_sparsity_pattern(
        dsp,
        this->_locally_owned_dofs,
        mpi_communicator,
        this->_locally_relevant_dofs);
    this->_system_matrix.reinit(this->_locally_owned_dofs, this->_locally_owned_dofs, dsp, mpi_communicator);

    unsigned int dof_index;
    //Point<dim> vertex;
    double comp = 0;

    for (const auto &shape_cell : this->_dof_handler_shape.active_cell_iterators()) {
      for (unsigned int id = 0; id < this->data.non_design_id.size(); ++id) {

        if (shape_cell->material_id() == (unsigned int) this->data.non_design_id[id]) {
          comp = this->data.non_design_comp[id];

          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
            dof_index = shape_cell->vertex_dof_index(v, comp);
            this->shape_constraints.add_line(dof_index);
            this->shape_constraints.set_inhomogeneity(dof_index, 0);
          }
        }
      }
    }

    this->shape_constraints.close();

    this->SetupSmoothingSystem();
    _SetupQPointHistory();

  } // end of _SetupSystem() function


  template<int dim>
  void EddElasticityLin<dim>::_SetupQPointHistory() {
    std::vector<QPointHistory> tmp;
    tmp.swap(quadrature_point_history);

    QGauss<dim> solid_quadrature(_fe_solid.degree + 1);
    quadrature_point_history.resize(this->_tria_base.n_active_cells() * solid_quadrature.size());

    unsigned int history_index = 0;
    for (auto &cell : this->_tria_base.active_cell_iterators()) {
      if (!cell->is_locally_owned()) {
        history_index += solid_quadrature.size();
        continue;
      }
      cell->set_user_pointer(&quadrature_point_history[history_index]);
      history_index += solid_quadrature.size();
    }
  }


  template<int dim>
  void EddElasticityLin<dim>::_UpdateQPointHistory() {
    const QGauss<dim> void_quadrature(_fe_solid.degree + 1);
    const QGauss<dim> solid_quadrature(_fe_solid.degree + 1);

    hp::QCollection<dim> q_collection;
    q_collection.push_back(void_quadrature);
    q_collection.push_back(solid_quadrature);

    std::vector<Tensor<2, dim>> local_displacement_gradients(solid_quadrature.size());

    hp::FEValues<dim> fe_values_hp(_fe_collection, q_collection,
                                   update_values | update_quadrature_points | update_JxW_values | update_gradients);

    const FEValuesExtractors::Vector u_extractor(0);

    SymmetricTensor<2, dim> e, s;

    bool should_get_density_data = (this->data.problem_type == "couple");

    for (auto &cell : this->_dof_handler_base.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;
      if(cell->material_id() == e_outside_cell)
        continue;
      auto *local_quadrature_points_history = reinterpret_cast<QPointHistory *>(cell->user_pointer());

      fe_values_hp.reinit(cell);
      const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();
      fe_values[u_extractor].get_function_gradients(this->_solution, local_displacement_gradients);

      for (unsigned int q = 0; q < solid_quadrature.size(); ++q) {
        auto grad_disp = local_displacement_gradients[q];
        e = symmetrize(grad_disp);

        if(should_get_density_data && (cell->material_id() == e_inside_cell) ) {
          this->penalty = this->data.initial_penalty;
          s = _mat.GetCauchyStress(e, this->pseudo_densities[cell->id()], this->penalty);
        }
        else
          s = _mat.GetCauchyStress(e);

        local_quadrature_points_history[q].strain = e;
        local_quadrature_points_history[q].stress = s;
        local_quadrature_points_history[q].strain_energy = e*s;
      }
    }
  }

  template<>
  void EddElasticityLin<2>::_DirichletPenaltyMatrix() {
    types::material_id dirichlet_id;
    Point<2> v1, v2, segment_p1, segment_p2;
    FullMatrix<double> cell_penalty_matrix;
    Vector<double> cell_penalty_rhs;
    std::vector<unsigned int> local_dof_indices;

    SymmetricTensor<4, 2> elasticity_tensor = _mat.GetElasticityTensor();
    const double elasticity_norm = elasticity_tensor.norm();

    typename DoFHandler<2>::active_cell_iterator base_cell = _dof_handler_base.begin_active();

    // For open cluster elimination
    this->shape_to_base_cell_assignment.clear();

    // Loop over dirichlet boundary condition
    for (unsigned int id = 0; id < this->data.dbc_id.size(); ++id) {
      dirichlet_id = this->data.dbc_id[id];

      for (auto shape_cell : this->_dof_handler_shape.active_cell_iterators()) {
        if (shape_cell->material_id() == dirichlet_id) {
          v1 = shape_cell->vertex(0);
          v2 = shape_cell->vertex(1);

          // loop over base cells that are intersected by corresponding dirichlet id
          for (unsigned int c = 0; c < this->boundary_cell_assignment[dirichlet_id].size(); ++c) {
            base_cell = this->boundary_cell_assignment[dirichlet_id][c];
            if (!base_cell->is_locally_owned())
              continue;

            // check if this particular cell is intersected with the shape cell.
            if (EddTools::BaseCellIsCutByShapeCell(base_cell, v1, v2)) {

              // For open cluster elimination
              this->shape_to_base_cell_assignment[shape_cell->id()].push_back(base_cell->id());

              // Find the segment of line v1-v2 is insterseted by base_cell, the segment is made of point segment_p1-segment-p2
              EddTools::DetermineCellIntersectionSegment(base_cell, v1, v2, segment_p1, segment_p2);

              // Integrate_cell_penalty_matrix
              unsigned int dofs_per_cell = base_cell->get_fe().dofs_per_cell;

              Point<2> tmp_unit_q_point, tmp_q_point;
              std::vector<Point<2> > unit_q_points;

              double N_i, N_j, frac, d_ext;
              double segment_jacobian, segment_gauss_weight, segment_penalty;
              unsigned int dirichlet_comp, component_index_i, component_index_j;
              double dirichlet_value = 0.0;

              const unsigned int n_q_points = 2;
              const double frac_1 = 0.2113249;
              const double frac_2 = 0.7886751;

              cell_penalty_matrix.reinit(dofs_per_cell, dofs_per_cell);
              cell_penalty_rhs.reinit(dofs_per_cell);
              unit_q_points.clear();

              // determine location of quadrature points within unit cell
              for (unsigned int q = 0; q < n_q_points; ++q) {
                frac = frac_1;
                if (q == 1)
                  frac = frac_2;

                tmp_q_point = (1 - frac) * segment_p1;
                tmp_q_point += frac * segment_p2;

                if (!base_cell->point_inside(tmp_q_point))
                  CERR_DEBUG << "integration point not within cell domain.." << std::endl;

                for (unsigned int d = 0; d < 2; ++d) {
                  d_ext = 2 * std::abs((base_cell->center()[d] - base_cell->vertex(0)[d]));
                  tmp_unit_q_point[d] = (tmp_q_point[d] - base_cell->vertex(0)[d]) / d_ext;
                }

                unit_q_points.push_back(tmp_unit_q_point);
              }

              segment_jacobian = segment_p1.distance(segment_p2);
              segment_gauss_weight = .5;
              segment_penalty = data.penalty;

              dirichlet_comp = this->data.dbc_comp[id];
              dirichlet_value = this->data.dbc_value[id];

// integrate quadrature point contributions
              for (unsigned int q = 0; q < n_q_points; ++q) {
                tmp_unit_q_point = unit_q_points[q];

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  component_index_i = _fe_solid.system_to_component_index(i).first;
                  N_i = _fe_solid.shape_value(i, tmp_unit_q_point);
                  if (component_index_i == dirichlet_comp) {
                    cell_penalty_rhs(i) +=
                        N_i * dirichlet_value * segment_penalty * elasticity_norm * segment_gauss_weight *
                        segment_jacobian;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                      component_index_j = _fe_solid.system_to_component_index(j).first;
                      N_j = _fe_solid.shape_value(j, tmp_unit_q_point);
                      if (component_index_j == dirichlet_comp) {
                        cell_penalty_matrix(i, j) +=
                            N_i * N_j * segment_penalty * elasticity_norm * segment_gauss_weight *
                            segment_jacobian;
                      }
                    }
                  }
                }
              }

// add to system matrix..
              local_dof_indices.resize(base_cell->get_fe().dofs_per_cell);
              base_cell->get_dof_indices(local_dof_indices);

              this->_constraints.distribute_local_to_global(cell_penalty_matrix,
                                                            cell_penalty_rhs,
                                                            local_dof_indices,
                                                            this->_system_matrix,
                                                            this->_system_rhs);

            }
          } // end of loop over base cell map
        } // end of if cell is dirichlet cell
      } // end of loop over shape cells
    } // end of loop over dirichlet id

  }


  template<>
  void EddElasticityLin<3>::_DirichletPenaltyMatrix() {

    types::material_id dirichlet_id;
    Point<3> v1, v2, v3;
    Point<3> dummy_p1, dummy_p2;
    FullMatrix<double> cell_penalty_matrix;
    Vector<double> cell_penalty_rhs;
    std::vector<unsigned int> local_dof_indices;
    std::vector<Point<3> > tmp_intersection_vertices;
    std::vector<std::vector<Point<3> > > subtriangulation;

    SymmetricTensor<4, 3> elasticity_tensor = _mat.GetElasticityTensor();
    const double elasticity_norm = elasticity_tensor.norm();

    typename DoFHandler<3>::active_cell_iterator base_cell = _dof_handler_base.begin_active();

// Loop over dirichlet boundary condition
    for (unsigned int id = 0; id < this->data.dbc_id.size(); ++id) {
      dirichlet_id = this->data.dbc_id[id];

      for (auto shape_cell : this->_dof_handler_shape.active_cell_iterators()) {
        if (shape_cell->material_id() == dirichlet_id) {

          for (unsigned int tria = 0; tria < 2; ++tria) {
            v1 = shape_cell->vertex(0);

            if (tria == 0) {
              v2 = shape_cell->vertex(1);
              v3 = shape_cell->vertex(3);
            } else if (tria > 0) {
              v2 = shape_cell->vertex(3);
              v3 = shape_cell->vertex(2);
            }

// loop over base cells that are intersected by corresponding dirichlet id
            for (unsigned int c = 0; c < this->boundary_cell_assignment[dirichlet_id].size(); ++c) {
              base_cell = this->boundary_cell_assignment[dirichlet_id][c];
              if (!base_cell->is_locally_owned())
                continue;

              if (EddTools::BaseCellIsCutByShapeTria(base_cell, v1, v2, v3, tmp_intersection_vertices))
                if (tmp_intersection_vertices.size() > 2) {
                  EddTools::TriangulateIntersectionVertices(tmp_intersection_vertices, subtriangulation);

// Integrate_cell_penalty_matrix
                  unsigned int dofs_per_cell = base_cell->get_fe().dofs_per_cell;

                  cell_penalty_matrix.reinit(dofs_per_cell, dofs_per_cell);
                  cell_penalty_rhs.reinit(dofs_per_cell);

                  std::vector<Point<3> > tmp_tria;
                  std::vector<Point<3> > unit_q_points;
                  Tensor<1, 3> edge_a, edge_b, edge_c;
                  Point<3> q_point, unit_q_point;
                  unsigned int component_index_i, component_index_j;
                  double len_a, len_b, len_c, s;
                  double tmp_area, d_ext, N_i, N_j;

                  const unsigned int n_q_points = 3;
                  double gauss_weight = 1.0 / 3.0;
//double unit_tria_area = 0.5;
                  double segment_penalty = data.penalty;
                  unsigned int dirichlet_comp = data.dbc_comp[id];
                  double dirichlet_value = data.dbc_value[id];

// loop over subtriangulation
                  for (unsigned int tria = 0; tria < subtriangulation.size(); ++tria) {
// get current tria
                    tmp_tria = subtriangulation[tria];

                    edge_a = tmp_tria[1] - tmp_tria[0];
                    edge_b = tmp_tria[2] - tmp_tria[0];
                    edge_c = tmp_tria[2] - tmp_tria[1];

                    len_a = edge_a.norm();
                    len_b = edge_b.norm();
                    len_c = edge_c.norm();

// evaluate area of current tria
                    s = 0.5 * (len_a + len_b + len_c);
                    tmp_area = std::pow(s * (s - len_a) * (s - len_b) * (s - len_c), 0.5);

// determine quadrature point locations within unit cell
                    unit_q_points.clear();
                    for (unsigned int q = 0; q < n_q_points; ++q) {
                      if (q == 0)
                        q_point = 0.5 * (tmp_tria[0] + tmp_tria[1]);
                      else if (q == 1)
                        q_point = 0.5 * (tmp_tria[0] + tmp_tria[2]);
                      else if (q == 2)
                        q_point = 0.5 * (tmp_tria[1] + tmp_tria[2]);

                      for (unsigned int d = 0; d < 3; ++d) {
                        d_ext = 2 *
                                std::abs((base_cell->center()[d] - base_cell->vertex(0)[d]));
                        unit_q_point[d] = (q_point[d] - base_cell->vertex(0)[d]) / d_ext;
                      }

                      unit_q_points.push_back(unit_q_point);
                    }

// loop over quadrature points of current tria and integrate penalty matrix contribution
                    for (unsigned int q = 0; q < n_q_points; ++q) {
                      unit_q_point = unit_q_points[q];

                      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        component_index_i = _fe_solid.system_to_component_index(i).first;
                        N_i = _fe_solid.shape_value(i, unit_q_point);

                        if (component_index_i == dirichlet_comp) {
                          cell_penalty_rhs(i) +=
                              N_i * dirichlet_value * segment_penalty * elasticity_norm * gauss_weight *
                              tmp_area;

                          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                            component_index_j = _fe_solid.system_to_component_index(
                                j).first;
                            N_j = _fe_solid.shape_value(j, unit_q_point);
                            if (component_index_j == dirichlet_comp) {
                              cell_penalty_matrix(i, j) +=
                                  N_i * N_j * segment_penalty * elasticity_norm * gauss_weight *
                                  tmp_area;
                            }
                          }
                        }
                      }
                    }
                  }

                  if (subtriangulation.size() < 1)
                    std::cout << "subtriangulation.size(): " << subtriangulation.size()
                              << std::endl;

// add to system matrix..
                  local_dof_indices.resize(base_cell->get_fe().dofs_per_cell);
                  base_cell->get_dof_indices(local_dof_indices);

                  this->_constraints.distribute_local_to_global(cell_penalty_matrix,
                                                                cell_penalty_rhs,
                                                                local_dof_indices,
                                                                this->_system_matrix,
                                                                this->_system_rhs);
                }
            }
          }
        }
      } // end of loop over shape cells
    } // end of loop over dirichlet id
  } // end of _DirichletPenaltyMatrix function


  template<int dim>
  void EddElasticityLin<dim>::_AssembleSystem() {
    TimerOutput::Scope t(this->compute_timer, "EddElasticityLin<dim>::_AssembleSystem");

    bool should_get_density_data = (this->data.problem_type == "couple");
// Reset the system matrix and rhs vector
    this->_system_matrix = 0.0;
    this->_system_rhs = 0.0;

// 2 different quadrature rules for inside and boundary cells,
// For the inside cells, std quadrature formula,
// For the boundary cells, we consider higher number of quadrature points
    QGauss<dim> quadrature_formula_inside(_fe_solid.degree + 1);
    QGauss<dim> quadrature_formula_void(1);
    QGauss<dim> quadrature_formula_boundary(this->data.cut_cell_quadrature);

// We have 2 QCollection, for inside and boundary cells,
// There are 2 FESystems in our problem, void and solid.
// Remember the statement in constructor, in collection the first item is void, the next is solid.
    hp::QCollection<dim> q_collection_inside, q_collection_boundary;

    q_collection_inside.push_back(quadrature_formula_void); // void cells
    q_collection_inside.push_back(quadrature_formula_inside); // solid cells

    q_collection_boundary.push_back(quadrature_formula_void); // void cells
    q_collection_boundary.push_back(quadrature_formula_boundary); // solid cells

// FEValues for inside
    hp::FEValues<dim> fe_values_hp_inside(this->_fe_collection, q_collection_inside,
                                          update_values | update_gradients | update_quadrature_points |
                                          update_JxW_values);

// FEValues for boundary
    hp::FEValues<dim> fe_values_hp_boundary(this->_fe_collection, q_collection_boundary,
                                            update_values | update_gradients | update_quadrature_points |
                                            update_JxW_values);

    const FEValuesExtractors::Vector u_extractor(0);

// Weak material contribution. 1 for inside q_points, 10e-3 or similar for outside q_points.
    double q_point_contribution;
    Point<dim> q_point; // This is used to determine if the q_point is inside or outside shape.

    unsigned int dofs_per_cell = _fe_solid.dofs_per_cell;

// Variable to collect local cell contribution.
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> unit_cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> dummy_cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    Tensor<4, dim> elasticity_tensor = _mat.GetElasticityTensor();

    if(this->data.problem_type == "couple") {
      for (const auto &cell : this->_dof_handler_base.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;

        if(cell->material_id() == e_outside_cell)
          this->pseudo_densities[cell->id()] = 0.0;
        // Just for visualization - display boundary cells as black, but at the same time exclude
        // them from the feasible range of [1e-3, 1.0] by assigning 1.0+1e.3 value.
        if(cell->material_id() == e_boundary_cell)
          this->pseudo_densities[cell->id()] = 1.0 + this->data.min_density;
      }
    }


    for (const auto &cell : this->_dof_handler_base.active_cell_iterators()) {
// if cell belong to some other domain, do nothing
      if (!cell->is_locally_owned())
        continue;

      if (cell->material_id() == e_outside_cell)
        continue;

      cell_matrix = 0.0;
      dummy_cell_rhs = 0.0;

      fe_values_hp_inside.reinit(cell);
      fe_values_hp_boundary.reinit(cell);

// Check if the cell is not boundary.
      if (cell->material_id() != e_boundary_cell) {
        const FEValues<dim> &fe_values_inside = fe_values_hp_inside.get_present_fe_values();

        for (unsigned int q = 0; q < fe_values_inside.n_quadrature_points; ++q) {
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            SymmetricTensor<2, dim> Grad_N_i = symmetrize(fe_values_inside[u_extractor].gradient(i, q));
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              SymmetricTensor<2, dim> Grad_N_j = symmetrize(fe_values_inside[u_extractor].gradient(j, q));
              cell_matrix(i, j) +=
                  contract3(Grad_N_i, elasticity_tensor, Grad_N_j) * fe_values_inside.JxW(q);

            }
          }
        } // end of loop over quadrature points

        // TODO: bubble on
        // Check if we deal with pseudo densities
        if(should_get_density_data) {
          this->cell_stiffness_matrices[cell->id()] = cell_matrix;
          this->penalty = this->data.initial_penalty;
          double density = std::pow(this->pseudo_densities[cell->id()], this->penalty);
          cell_matrix *= density;
        }

      }// end of cell->material_id() != 1

// base cells intersected by shape: gauss-point oversampling
      if (cell->material_id() == e_boundary_cell) {

        const FEValues<dim> &fe_values_boundary = fe_values_hp_boundary.get_present_fe_values();
        for (unsigned int q = 0; q < fe_values_boundary.n_quadrature_points; ++q) {
          q_point = fe_values_boundary.quadrature_point(q);
          q_point_contribution = 1;

          if (!GeometryAlgorithms::PointInShape(q_point, this->_tria_shape, this->min_face_length_base))
            q_point_contribution = this->data.weak_material;

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            SymmetricTensor<2, dim> Grad_N_i = symmetrize(fe_values_boundary[u_extractor].gradient(i, q));
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              SymmetricTensor<2, dim> Grad_N_j = symmetrize(fe_values_boundary[u_extractor].gradient(j, q));
              cell_matrix(i, j) +=
                  contract3(Grad_N_i, elasticity_tensor, Grad_N_j) * fe_values_boundary.JxW(q) *
                  q_point_contribution;

            }
          }
        } // end of loop over quadrature points
      }// end of cell->material_id() == 1

      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      this->_constraints.distribute_local_to_global(cell_matrix,
                                                    dummy_cell_rhs,
                                                    local_dof_indices,
                                                    this->_system_matrix,
                                                    this->_system_rhs);

      // Whenever a base cell is left outside the solid, we reassign its density to 1.0.
      // This is because in a specific case when this particular base cell is reintroduced into the solid domain
      // it would lay, in a most realistic scenario and because of shape step sizes being small enough,
      // in a vicinity of boundary elements, which are always considered
      // as "fully" solid. Therefore, there is no sudden change in density values.
      //if(!this->pseudo_densities.empty())
          //this->pseudo_densities[cell->id()] = 1.0;
    }// end of loop over cells

// Add the PenaltyMatrix to enforce Dirichlet boundary condition.
    _DirichletPenaltyMatrix();

    this->_system_matrix.compress(VectorOperation::add);
    this->_system_rhs.compress(VectorOperation::add);

  } // end of _AssembleSystem function


  template<>
  void EddElasticityLin<2>::_IntegrateCellRhs(typename DoFHandler<2>::active_cell_iterator &cell,
                                              Point<2> &segment_p1, Point<2> &segment_p2,
                                              std::vector<std::vector<Point<2>>> &subtriangulation,
                                              unsigned int &id,
                                              Vector<double> &rhs) {
    (void) subtriangulation;

    unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;

    rhs.reinit(dofs_per_cell);

    Point<2> tmp_unit_q_point, tmp_q_point;
    std::vector<Point<2> > unit_q_points;

    double N_i, frac, d_ext;
    double segment_jacobian, segment_gauss_weight, neumann_magnitude;
    unsigned int neumann_comp, component_index;

    const unsigned int n_q_points = 2;
    const double frac_1 = 0.2113249;
    const double frac_2 = 0.7886751;

    unit_q_points.clear();

// determine location of quadrature points within unit cell
    for (unsigned int q = 0; q < n_q_points; ++q) {
      frac = frac_1;
      if (q == 1)
        frac = frac_2;

      tmp_q_point = (1 - frac) * segment_p1;
      tmp_q_point += frac * segment_p2;

      if (!cell->point_inside(tmp_q_point))
        std::cout << "integration point not within cell domain.." << std::endl;

// Todo this is only valid for base cells aligned with principal coordinate system
      for (unsigned int d = 0; d < 2; ++d) {
        d_ext = 2 * std::abs((cell->center()[d] - cell->vertex(0)[d]));
        tmp_unit_q_point[d] = (tmp_q_point[d] - cell->vertex(0)[d]) / d_ext;
      }

      unit_q_points.push_back(tmp_unit_q_point);
    }

    segment_jacobian = segment_p1.distance(segment_p2);
    segment_gauss_weight = .5;

    neumann_magnitude = data.nbc_value[id];
    neumann_comp = data.nbc_comp[id];

// integrate quadrature point contributions
    for (unsigned int q = 0; q < n_q_points; ++q) {
      tmp_unit_q_point = unit_q_points[q];

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        component_index = _fe_solid.system_to_component_index(i).first;
        if (component_index == neumann_comp) {
          N_i = _fe_solid.shape_value(i, tmp_unit_q_point);
          rhs(i) += N_i * neumann_magnitude * segment_gauss_weight * segment_jacobian;
        }
      }
    }
  }

  template<>
  void EddElasticityLin<3>::_IntegrateCellRhs(typename DoFHandler<3>::active_cell_iterator &cell,
                                              Point<3> &segment_p1, Point<3> &segment_p2,
                                              std::vector<std::vector<Point<3>>> &subtriangulation,
                                              unsigned int &id,
                                              Vector<double> &rhs) {
// dim=3 --> ignore segment_p1, segment_p2 input
    (void) segment_p1, (void) segment_p2;

    unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;

    rhs.reinit(dofs_per_cell);

    std::vector<Point<3> > tmp_tria;
    std::vector<Point<3> > unit_q_points;
    Tensor<1, 3> edge_a, edge_b, edge_c;
    Point<3> q_point, unit_q_point;
    unsigned int component_index;
    double len_a, len_b, len_c, s;
    double tmp_area, d_ext, N_i;

    const unsigned int n_q_points = 3;
    double gauss_weight = 1.0 / 3.0;
    double unit_tria_area = 0.5;
    double neumann_magnitude = data.nbc_value[id];
    std::vector<double> vec_neumann_magnitude;
    double tmp_value = 0.0;
    unsigned int neumann_comp = data.nbc_comp[id];

// loop over subtriangulation
    for (unsigned int tria = 0; tria < subtriangulation.size(); ++tria) {
// get current tria
      tmp_tria = subtriangulation[tria];

      edge_a = tmp_tria[1] - tmp_tria[0];
      edge_b = tmp_tria[2] - tmp_tria[0];
      edge_c = tmp_tria[2] - tmp_tria[1];

      len_a = edge_a.norm();
      len_b = edge_b.norm();
      len_c = edge_c.norm();

// evaluate area of current tria
      s = 0.5 * (len_a + len_b + len_c);
      tmp_area = std::pow(s * (s - len_a) * (s - len_b) * (s - len_c), 0.5);

// determine quadrature point locations within unit cell
      unit_q_points.clear();
      vec_neumann_magnitude.clear();
      for (unsigned int q = 0; q < n_q_points; ++q) {
        if (q == 0)
          q_point = 0.5 * (tmp_tria[0] + tmp_tria[1]);
        else if (q == 1)
          q_point = 0.5 * (tmp_tria[0] + tmp_tria[2]);
        else if (q == 2)
          q_point = 0.5 * (tmp_tria[1] + tmp_tria[2]);

        for (unsigned int d = 0; d < 3; ++d) {
          d_ext = 2 * std::abs((cell->center()[d] - cell->vertex(0)[d]));
          unit_q_point[d] = (q_point[d] - cell->vertex(0)[d]) / d_ext;
        }

        tmp_value *= neumann_magnitude;
        vec_neumann_magnitude.push_back(tmp_value);

        unit_q_points.push_back(unit_q_point);
      }

      neumann_magnitude = data.nbc_value[id];
      neumann_comp = data.nbc_comp[id];

// loop over quadrature points of current tria and integrate rhs contribution
      for (unsigned int q = 0; q < n_q_points; ++q) {
        unit_q_point = unit_q_points[q];

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          component_index = _fe_solid.system_to_component_index(i).first;
          if (component_index == neumann_comp) {
            N_i = _fe_solid.shape_value(i, unit_q_point);
            rhs(i) += N_i * neumann_magnitude * gauss_weight * unit_tria_area * tmp_area;
          }
        }
      }
    }

  }

  template<>
  void EddElasticityLin<2>::_AssembleRhs() {
    TimerOutput::Scope t(this->compute_timer,
                         "EddElasticityLin<dim>::_AssembleRhs");

    Vector<double> cell_rhs;
    std::vector<unsigned int> local_dof_indices;
    Point<2> v1, v2, segment_p1, segment_p2;
    std::vector<std::vector<Point<2>>> dummy_subtriangulation;

    typename DoFHandler<2>::active_cell_iterator base_cell = _dof_handler_base.begin_active();

    // computing load area.
    // create a list of neumann_id with no repetation of nemann id.
    std::vector<int> unique_neumann_id = this->data.nbc_id;
    sort(unique_neumann_id.begin(), unique_neumann_id.end());
    unique_neumann_id.erase(std::unique(unique_neumann_id.begin(), unique_neumann_id.end()),
                            unique_neumann_id.end());

    // create a map with neumann id vs load area
    std::map<int, double> nemann_id_area;
    for (auto neumann_id: unique_neumann_id)
      nemann_id_area[neumann_id] = 0.0;

    for (const auto &shape_cell : this->_dof_handler_shape.active_cell_iterators()) {
      for (auto neumann_id: unique_neumann_id) {
        if (shape_cell->material_id() == (unsigned int) neumann_id) {
          nemann_id_area[neumann_id] += shape_cell->measure();
        }
      }
    }

// loop over neumann boundary
    for (unsigned int id = 0; id < data.nbc_id.size(); ++id) {

      unsigned int neumann_id = data.nbc_id[id];

      for (const auto &shape_cell : this->_dof_handler_shape.active_cell_iterators()) {
        if (shape_cell->material_id() == neumann_id) {
          v1 = shape_cell->vertex(0);
          v2 = shape_cell->vertex(1);

          for (unsigned int c = 0; c < this->boundary_cell_assignment[neumann_id].size(); ++c) {
            base_cell = this->boundary_cell_assignment[neumann_id][c];

            if (!base_cell->is_locally_owned())
              continue;

            if (EddTools::BaseCellIsCutByShapeCell(base_cell, v1, v2)) {
// determine segment_p1 and segment_p2
              EddTools::DetermineCellIntersectionSegment(base_cell, v1, v2, segment_p1, segment_p2);

// integrate cell contribution from segment_p1 to segment_p2..
              _IntegrateCellRhs(base_cell, segment_p1, segment_p2, dummy_subtriangulation, id, cell_rhs);

              cell_rhs /= nemann_id_area[neumann_id];

// add cell contribution to system rhs
              local_dof_indices.resize(base_cell->get_fe().dofs_per_cell);
              base_cell->get_dof_indices(local_dof_indices);

              this->_constraints.distribute_local_to_global(cell_rhs, local_dof_indices, this->_system_rhs);
            }
          }
        }
      }
    }

    this->_system_rhs.compress(VectorOperation::add);

  }

  template<>
  void EddElasticityLin<3>::_AssembleRhs() {
    TimerOutput::Scope t(this->compute_timer,
                         "EddElasticityLin<dim>::_AssembleRhs");

    Vector<double> cell_rhs;
    std::vector<unsigned int> local_dof_indices;
    std::vector<Point<3> > tmp_intersection_vertices;
    std::vector<std::vector<Point<3>>> subtriangulation;
    Point<3> v1, v2, v3;
    Point<3> dummy_p1, dummy_p2;

    typename DoFHandler<3>::active_cell_iterator base_cell = this->_dof_handler_base.begin_active();

    // computing load area.
    // create a list of neumann_id with no repetation of nemann id.
    Tensor<1, 3> edge_a, edge_b, edge_c;
    double len_a, len_b, len_c, s;
    std::vector<int> unique_neumann_id = this->data.nbc_id;
    sort(unique_neumann_id.begin(), unique_neumann_id.end());
    unique_neumann_id.erase(std::unique(unique_neumann_id.begin(), unique_neumann_id.end()),
                            unique_neumann_id.end());

    // create a map with neumann id vs load area
    std::map<int, double> nemann_id_area;
    for (auto neumann_id: unique_neumann_id)
      nemann_id_area[neumann_id] = 0.0;

    for (const auto &shape_cell : this->_dof_handler_shape.active_cell_iterators()) {
      for (auto neumann_id: unique_neumann_id) {
        if (shape_cell->material_id() == (unsigned int) neumann_id) {
          for (unsigned int tria = 0; tria < 2; ++tria) {
            v1 = shape_cell->vertex(0);

            if (tria == 0) {
              v2 = shape_cell->vertex(1);
              v3 = shape_cell->vertex(3);
            } else if (tria > 0) {
              v2 = shape_cell->vertex(3);
              v3 = shape_cell->vertex(2);
            }

            edge_a = v2 - v1;
            edge_b = v3 - v1;
            edge_c = v3 - v2;

            len_a = edge_a.norm();
            len_b = edge_b.norm();
            len_c = edge_c.norm();

            s = 0.5 * (len_a + len_b + len_c);
            nemann_id_area[neumann_id] += std::pow(s * (s - len_a) * (s - len_b) * (s - len_c), 0.5);
          }
        }
      }
    }

// loop over neumann boundary
    for (unsigned int id = 0; id < data.nbc_id.size(); ++id) {
      unsigned int neumann_id = data.nbc_id[id];

      for (const auto &shape_cell : this->_dof_handler_shape.active_cell_iterators()) {
        if (shape_cell->material_id() == neumann_id) {
          for (unsigned int tria = 0; tria < 2; ++tria) {
            v1 = shape_cell->vertex(0);

            if (tria == 0) {
              v2 = shape_cell->vertex(1);
              v3 = shape_cell->vertex(3);
            } else if (tria > 0) {
              v2 = shape_cell->vertex(3);
              v3 = shape_cell->vertex(2);
            }

            for (unsigned int c = 0; c < this->boundary_cell_assignment[neumann_id].size(); ++c) {
              base_cell = this->boundary_cell_assignment[neumann_id][c];

              if (!base_cell->is_locally_owned())
                continue;

              if (EddTools::BaseCellIsCutByShapeTria(base_cell, v1, v2, v3, tmp_intersection_vertices))
                if (tmp_intersection_vertices.size() > 2) {
                  EddTools::TriangulateIntersectionVertices(tmp_intersection_vertices, subtriangulation);
                  _IntegrateCellRhs(base_cell, dummy_p1, dummy_p2, subtriangulation, id, cell_rhs);
                  cell_rhs /= nemann_id_area[neumann_id];
                  if (subtriangulation.size() < 1)
                    std::cout << "subtriangulation.size(): " << subtriangulation.size()
                              << std::endl;

// add cell contribution to system rhs
                  local_dof_indices.resize(base_cell->get_fe().dofs_per_cell);
                  base_cell->get_dof_indices(local_dof_indices);
                  this->_constraints.distribute_local_to_global(cell_rhs, local_dof_indices,
                                                                this->_system_rhs);
                }
            }
          }
        }
      }
    }

    this->_system_rhs.compress(VectorOperation::add);

  }

  template<int dim>
  void EddElasticityLin<dim>::_SolveSystem() {
    TimerOutput::Scope t(this->compute_timer, "EddElasticityLin<dim>::_SolveSystem");

    LA::MPI::Vector completely_distributed_solution(this->_system_rhs);
    completely_distributed_solution = 0.0;
    this->_constraints.set_zero(completely_distributed_solution);

    this->SolveLinearSystem(this->_system_matrix,
                            completely_distributed_solution,
                            this->_system_rhs);

    this->_constraints.distribute(completely_distributed_solution);

    this->_solution = completely_distributed_solution;

    std::vector<std::string> sol_names(dim, "sol"), rhs_names(dim, "rhs");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        sol_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector),
        rhs_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

    bvp_output.template PushDataName<LA::MPI::Vector>(this->_system_rhs, rhs_names,
                                                                    rhs_data_component_interpretation,
                                                                    &this->_dof_handler_base);
    bvp_output.template PushDataName<LA::MPI::Vector>(this->_solution, sol_names,
                                                                    sol_data_component_interpretation,
                                                                    &this->_dof_handler_base);

  }


  template<int dim>
  void EddElasticityLin<dim>::_Postprocess() {

    TimerOutput::Scope t(this->compute_timer, "EddElasticityLin<dim>::_Postprocess");

    _UpdateQPointHistory();
    this->AssembleSmoothingMatrix();

    if (dim == 2) {
      for (unsigned int i = e_sym_grad_u_11; i <= e_sym_grad_u_22; ++i) {
        _AssembleSmoothingRhs(i);
        this->SolveSmoothingSystem();
        _UpdatePostprocessingData(i);

      }
      for (unsigned int i = e_cauchy_s_11; i <= e_cauchy_s_22; ++i) {
        _AssembleSmoothingRhs(i);
        this->SolveSmoothingSystem();
        _UpdatePostprocessingData(i);
      }
    }

    if (dim == 3) {
      for (unsigned int i = e_sym_grad_u_11; i <= e_sym_grad_u_33; ++i) {
        _AssembleSmoothingRhs(i);
        this->SolveSmoothingSystem();
        _UpdatePostprocessingData(i);
      }
      for (unsigned int i = e_cauchy_s_11; i <= e_cauchy_s_33; ++i) {
        _AssembleSmoothingRhs(i);
        this->SolveSmoothingSystem();
        _UpdatePostprocessingData(i);
      }
    }

    unsigned int enum_index = e_von_mises_s;
    _AssembleSmoothingRhs(enum_index);
    this->SolveSmoothingSystem();
    _UpdatePostprocessingData(enum_index);

    enum_index = e_strain_energy;
    _AssembleSmoothingRhs(enum_index);
    this->SolveSmoothingSystem();
    _UpdatePostprocessingData(enum_index);

  }

  template<int dim>
  void EddElasticityLin<dim>::_AssembleSmoothingRhs(unsigned int assembly_flag) {

    this->_smoothing_rhs = 0;

    const QGauss<dim> void_quadrature(this->_smoothing_fe_void.degree + 1);
    const QGauss<dim> solid_quadrature(this->_smoothing_fe_solid.degree + 1);

    hp::QCollection<dim> q_collection;
    q_collection.push_back(void_quadrature);
    q_collection.push_back(solid_quadrature);

    hp::FEValues<dim> fe_values_hp(this->_smoothing_fe_collection, q_collection,
                                   update_values | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs;
    std::vector<unsigned int> local_dof_indices;
    double q_value;

    for (auto &cell : this->_smoothing_dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;
      if (cell->material_id() == e_outside_cell)
        continue;

      fe_values_hp.reinit(cell);
      cell_rhs.reinit(cell->get_fe().dofs_per_cell);

      auto *local_quadrature_points_history = reinterpret_cast<QPointHistory *>(cell->user_pointer());

      if (EddTools::CellIsInside<dim>(cell)) {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
          q_value = 0;

          if (assembly_flag == e_sym_grad_u_11)
            q_value = local_quadrature_points_history[q].strain[0][0];
          else if (assembly_flag == e_sym_grad_u_12)
            q_value = local_quadrature_points_history[q].strain[0][1];
          else if (assembly_flag == e_sym_grad_u_22)
            q_value = local_quadrature_points_history[q].strain[1][1];

          else if (assembly_flag == e_sym_grad_u_13)
            q_value = local_quadrature_points_history[q].strain[0][2];
          else if (assembly_flag == e_sym_grad_u_23)
            q_value = local_quadrature_points_history[q].strain[1][2];
          else if (assembly_flag == e_sym_grad_u_33)
            q_value = local_quadrature_points_history[q].strain[2][2];

          else if (assembly_flag == e_cauchy_s_11)
            q_value = local_quadrature_points_history[q].stress[0][0];
          else if (assembly_flag == e_cauchy_s_12)
            q_value = local_quadrature_points_history[q].stress[0][1];
          else if (assembly_flag == e_cauchy_s_22)
            q_value = local_quadrature_points_history[q].stress[1][1];

          else if (assembly_flag == e_cauchy_s_13)
            q_value = local_quadrature_points_history[q].stress[0][2];
          else if (assembly_flag == e_cauchy_s_23)
            q_value = local_quadrature_points_history[q].stress[1][2];
          else if (assembly_flag == e_cauchy_s_33)
            q_value = local_quadrature_points_history[q].stress[2][2];

          else if (assembly_flag == e_strain_energy)
            q_value = local_quadrature_points_history[q].strain_energy;

          else if (assembly_flag == e_von_mises_s)
            q_value = _mat.GetVonMisesStress(local_quadrature_points_history[q].strain);

          else if (assembly_flag == e_compliance_topo)
            q_value = _mat.GetComplianceSens(local_quadrature_points_history[q].strain);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            cell_rhs(i) += q_value * fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
        }
      }
      // add to system smoothing rhs
      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      this->_smoothing_constraints.distribute_local_to_global(cell_rhs, local_dof_indices, this->_smoothing_rhs);
    }
    this->_smoothing_rhs.compress(VectorOperation::add);
  }

  template<int dim>
  void EddElasticityLin<dim>::_UpdatePostprocessingData(unsigned int assembly_flag) {

    Vector<double> reduced_smoothing_solution(this->_smoothing_solution);

    if (assembly_flag == e_sym_grad_u_11) {
      postprocessing_data.base_strain11.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_strain11 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain11");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_12) {
      postprocessing_data.base_strain12.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_strain12 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain12");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_22) {
      postprocessing_data.base_strain22.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_strain22 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain22");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_13) {
      postprocessing_data.base_strain13.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_strain13 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain13");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_23) {
      postprocessing_data.base_strain23.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_strain23 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain23");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_33) {
      postprocessing_data.base_strain33.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_strain33 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain33");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_cauchy_s_11) {
      postprocessing_data.base_stress11.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_stress11 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "stress11");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_cauchy_s_12) {
      postprocessing_data.base_stress12.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_stress12 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "stress12");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_cauchy_s_22) {
      postprocessing_data.base_stress22.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_stress22 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "stress22");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_cauchy_s_13) {
      postprocessing_data.base_stress13.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_stress13 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "stress13");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_cauchy_s_23) {
      postprocessing_data.base_stress23.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_stress23 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "stress23");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_cauchy_s_33) {
      postprocessing_data.base_stress33.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_stress33 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "stress33");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_strain_energy) {
      postprocessing_data.base_strain_energy.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_strain_energy = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain_energy");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_von_mises_s) {
      postprocessing_data.base_vm_stress.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_vm_stress = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "vm_stress");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_compliance_topo) {
      postprocessing_data.base_comp_topo.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.base_comp_topo = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "comp_topo");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    }

  }


  template<int dim>
  void EddElasticityLin<dim>::Run() {

    _SetupSystem();
    // Decide whether pseudo densities were given or not
    _AssembleSystem();
    _AssembleRhs();
    _SolveSystem();
    _Postprocess();
    _OutputResults(cycle);
    ++cycle;

  } // end of run

  template<int dim>
  Vector<double> EddElasticityLin<dim>::GetPostprocessingData(unsigned int &data_flag) {
    Vector<double> postprocessing_vector;
    if (data_flag == e_sym_grad_u_11) postprocessing_vector = postprocessing_data.base_strain11;
    else if (data_flag == e_sym_grad_u_12) postprocessing_vector = postprocessing_data.base_strain12;
    else if (data_flag == e_sym_grad_u_22) postprocessing_vector = postprocessing_data.base_strain22;
    else if (data_flag == e_sym_grad_u_13) postprocessing_vector = postprocessing_data.base_strain13;
    else if (data_flag == e_sym_grad_u_23) postprocessing_vector = postprocessing_data.base_strain23;
    else if (data_flag == e_sym_grad_u_33) postprocessing_vector = postprocessing_data.base_strain33;
    else if (data_flag == e_cauchy_s_11) postprocessing_vector = postprocessing_data.base_stress11;
    else if (data_flag == e_cauchy_s_12) postprocessing_vector = postprocessing_data.base_stress12;
    else if (data_flag == e_cauchy_s_22) postprocessing_vector = postprocessing_data.base_stress22;
    else if (data_flag == e_cauchy_s_13) postprocessing_vector = postprocessing_data.base_stress13;
    else if (data_flag == e_cauchy_s_23) postprocessing_vector = postprocessing_data.base_stress23;
    else if (data_flag == e_cauchy_s_33) postprocessing_vector = postprocessing_data.base_stress33;
    else if (data_flag == e_strain_energy) postprocessing_vector = postprocessing_data.base_strain_energy;
    else if (data_flag == e_von_mises_s) postprocessing_vector = postprocessing_data.base_vm_stress;
    else if (data_flag == e_compliance_topo) postprocessing_vector = postprocessing_data.base_comp_topo;
    else throw std::runtime_error( "Incorrect postprocessing data request.");
    return postprocessing_vector;
  }

  template<int dim>
  hp::FECollection<dim> EddElasticityLin<dim>::GetBaseFE() {
    return _fe_collection;
  }

  template<int dim>
  void EddElasticityLin<dim>::_OutputResults(const unsigned int cycle) {

    // pushing data to output vector
    Vector<float> mat_id(this->_tria_base.n_active_cells());
    unsigned int counter = 0;
    for (const auto &cell : this->_tria_base.active_cell_iterators()) {
      if (!cell->is_locally_owned()) {
        ++counter;
        continue;
      }
      mat_id[counter] = cell->material_id();
      ++counter;
    }

    std::vector<std::string> data_names(1, "mat_id");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(DataComponentInterpretation::component_is_scalar);
    bvp_output.template PushDataName<Vector<float>>(mat_id, data_names, data_component_interpretation,
                                                    &this->_dof_handler_base);

    bvp_output.WriteDataOutput(cycle);
  }

} // end of StructuralOptimization namespace

template
class StructuralOptimization::EddElasticityLin<2>;

template
class StructuralOptimization::EddElasticityLin<3>;
