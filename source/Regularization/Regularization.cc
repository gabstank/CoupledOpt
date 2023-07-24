#include <Regularization.h>

namespace StructuralOptimization {


  template<int dim>
  Regularization<dim>::Regularization(Parameter &par_)
      : _data(par_.data),
        mpi_communicator(par_.mpi_communicator),
        this_mpi_process(par_.this_mpi_process),
        compute_timer(par_.compute_timer),
        _traction_method(par_) {}


  template<int dim>
  Regularization<dim>::~Regularization() = default;


  template<int dim>
  void Regularization<dim>::RunTractionMethod(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                              AffineConstraints<double> &shape_constraints,
                                              Vector<double> &sensitivities, const Vector<double> &vertex_normals) {

    TimerOutput::Scope t(this->compute_timer,
                         "Regularization<dim>::RunTractionMethod");

    if (_raw_sensitivities.size() == 0)
      _raw_sensitivities = sensitivities;

    if (_data.sensitivity_projection != "none")
      _Projection(dof_handler_shape, sensitivities);

    _traction_method.Run(dof_handler_shape, shape_constraints, sensitivities, vertex_normals);

    if (_data.dual_descent_smoothing)
      _DualDescentSmoothing(dof_handler_shape, shape_constraints, sensitivities);

    if (_data.sensitivity_projection != "none")
      _Projection(dof_handler_shape, sensitivities);

    // release memory
    _raw_sensitivities.reinit(0);
  }

  template<int dim>
  void Regularization<dim>::RunSimpleTractionMethod(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                              AffineConstraints<double> &shape_constraints,
                                              Vector<double> &sensitivities, const Vector<double> &vertex_normals) {

    TimerOutput::Scope t(this->compute_timer,
                         "Regularization<dim>::RunSimpleTractionMethod");

    _raw_sensitivities = sensitivities;

    _traction_method.RunSimple(dof_handler_shape, shape_constraints, sensitivities, vertex_normals);

    Vector<double> before_sens(sensitivities);

    if (_data.dual_descent_smoothing)
      _DualDescentSmoothing(dof_handler_shape, shape_constraints, sensitivities);

    _raw_sensitivities.reinit(0);
  }

  template<int dim>
  void Regularization<dim>::DirectShapeSensitivityFiltering(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                            AffineConstraints<double> &shape_constraints,
                                            Vector<double> &sensitivities, const Vector<double> &vertex_normals) {

    if (_raw_sensitivities.size() == 0)
      _raw_sensitivities = sensitivities;
    if (_vertex_normals.size() == 0)
      _vertex_normals = vertex_normals;

    if (_data.sensitivity_weighting) {
      if (_data.sensitivity_weighting_type == "field")
        _FieldWeighting(dof_handler_shape, shape_constraints, sensitivities);
      else if (_data.sensitivity_weighting_type == "component")
        _ComponentWeighting(dof_handler_shape, shape_constraints, sensitivities);
      else
        std::cerr << "Wrong sensitivity weighting type." << std::endl;
    }

    if (_data.dual_descent_smoothing)
      _DualDescentSmoothing(dof_handler_shape, shape_constraints, sensitivities);

    if (_data.sensitivity_projection != "none")
      _Projection(dof_handler_shape, sensitivities);

    // release memory
    _raw_sensitivities.reinit(0);
    _vertex_normals.reinit(0);

  }

  template<int dim>
  void Regularization<dim>::DensitySensitivityFiltering(DoFHandler<dim> &dof_handler_base,
                                                        std::map<CellId, double> &sensitivities,
                                                        std::map<CellId, double> &filtered_sensitivity) {

    // CDEV:
    // For mesh independent filter based on radius.
    // Creating neighbour list:
    //    Gather a list of cell center and CellId from all process.
    //    For each cell loop over the list and check which cells are inside the radius.
    //    Create a neighbour list of CellId and their distance form current cell within the radius.
    // Based on this list smoothing operation is performed.

    // code to create neighbour list ----------------------
    if (_neighbour_list.empty()) { // Assemble the list only once

      std::map<CellId, Point<dim>> cellId_center_proc;
      for (const auto &cell : dof_handler_base.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;
        cellId_center_proc[cell->id()] = cell->center();
      }

      std::vector<std::map<CellId, Point<dim>>> temp_center = Utilities::MPI::all_gather(
          mpi_communicator, cellId_center_proc);

      std::map<CellId, Point<dim>> cellId_center;

      for(const auto& center_map : temp_center)
        for (const auto&[cellid, val] : center_map)
          if(cellId_center.find(cellid) == cellId_center.end())
            cellId_center[cellid] = val;

      for (const auto &cell : dof_handler_base.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;

        for (const auto &[cellid, val] : cellId_center) {

          Point<dim> center = cell->center();
          double distance = center.distance(val);
          if (distance < _data.initial_filter_radius)
            _neighbour_list[cell->id()][cellid] = _data.initial_filter_radius - distance;
        }
      }
    }
    // end of neighbour list creation ----------------------

    // If neighbour list is still empty that means the filtering radius is smaller than the element size
    // In that case throw an error otherwise the topology optimization would produce checkboard patterns
    if (_neighbour_list.empty()) {
      throw std::runtime_error("Neighbour list for density filtering is empty. "
                               "That means the filtering radius was chosen smaller than the base element size. "
                               "Try increasing the filtering radius.");
    }

      for (const auto &cell : dof_handler_base.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;
      if (cell->material_id() != e_inside_cell)
        continue;

      // Gab: do not consider outside and boundary cells for filtering.
      filtered_sensitivity[cell->id()] = 0;
      double sum_neighbour_weight=0.0;
      for(const auto &neighbour : _neighbour_list[cell->id()]){
        if (sensitivities.find(neighbour.first) == sensitivities.end())
          continue;
        filtered_sensitivity[cell->id()] += sensitivities[neighbour.first] * neighbour.second;
        sum_neighbour_weight += neighbour.second;
      }

      filtered_sensitivity[cell->id()] /= (sum_neighbour_weight);

    } // end of loop over cells
  }

  template<int dim>
  void Regularization<dim>::_Projection(DoFHandler<dim - 1, dim> &dof_handler_shape, Vector<double> &sensitivities) {

    int comp = -1;

    if (_data.sensitivity_projection == "x") comp = 0;
    else if (_data.sensitivity_projection == "y") comp = 1;
    else if (_data.sensitivity_projection == "z") comp = 2;

    if (comp != -1) {

      FESystem<dim - 1, dim> fe_shape(FE_Q<dim - 1, dim>(1), dim);
      unsigned int dofs_per_cell;
      std::vector<unsigned int> local_dof_indices;

      for (auto &cell : dof_handler_shape.active_cell_iterators()) {

        dofs_per_cell = cell->get_fe().dofs_per_cell;

        local_dof_indices.clear();
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {

          unsigned int dof_index = local_dof_indices[i];
          int component_index = fe_shape.system_to_component_index(i).first;

          if (component_index == comp)
            sensitivities[dof_index] = 0.0;
        }
      }
    }
  }

  template<int dim>
  void Regularization<dim>::_FieldWeighting(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                            AffineConstraints<double> &shape_constraints,
                                            Vector<double> &sensitivities) {

    FE_Q<dim - 1, dim> fe_manifold(1);
    DoFHandler<dim - 1, dim> dof_handler_manifold(dof_handler_shape.get_triangulation());
    dof_handler_manifold.distribute_dofs(fe_manifold);

    AffineConstraints<double> manifold_constraints;
    manifold_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_manifold, manifold_constraints);
    manifold_constraints.close();

    const unsigned int n_dofs = dof_handler_manifold.n_dofs();
    Vector<double> weighting_rhs(n_dofs);
    Vector<double> weighting_sol(n_dofs);
    FullMatrix<double> weighting_matrix(n_dofs, n_dofs);
    Vector<double> inverse_weights_vector(n_dofs);
    DiagonalMatrix<Vector<double>> inverse_weighting_matrix;
    Tensor<1, dim> _tmp;
    double val;

    const QGauss<dim - 1> quadrature(1 + 1);
    FEValues<dim - 1, dim> fe_values(fe_manifold, quadrature, update_values | update_quadrature_points |
                                                              update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe_manifold.dofs_per_cell;
    const unsigned int n_q_points = quadrature.size();
    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    assert(_raw_sensitivities.size() == dof_handler_manifold.n_dofs() * dim);

    // Get sensitivity field values
    for (unsigned int i = 0; i < weighting_rhs.size(); ++i) {
      _tmp = 0;
      val = 0;
      for (unsigned int d = 0; d < dim; ++d) {
        _tmp[d] = _vertex_normals[i * dim + d];
        val += _vertex_normals[i * dim + d] * _raw_sensitivities[i * dim + d];
      }
      weighting_rhs[i] = val / _tmp.norm();
    }

    // Assemble weighting matrix
    for (const auto &cell : dof_handler_manifold.active_cell_iterators()) {
      cell_matrix = 0;
      fe_values.reinit(cell);

      // Cell weighting matrix
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          cell_matrix(i, i) += fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
        }
      }


      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      manifold_constraints.distribute_local_to_global(cell_matrix, local_dof_indices, weighting_matrix);
    }
    // Inverse the values so that we can perform multiplication operation with initial sensitivities
    for (unsigned int i = 0; i < n_dofs; ++i) {
      inverse_weights_vector[i] = 1.0 / weighting_matrix(i, i);
    }

    // Store the components to diagonal matrix
    inverse_weighting_matrix.reinit(inverse_weights_vector);

    // Contraction of raw sensitivity and weighting matrix
    inverse_weighting_matrix.vmult(weighting_sol, weighting_rhs);

    sensitivities = _vertex_normals;
    for (unsigned int i = 0; i < weighting_sol.size(); ++i) {
      _tmp = 0;
      for (unsigned int d = 0; d < dim; ++d) {
        _tmp[d] = _vertex_normals[i * dim + d];
      }
      for (unsigned int d = 0; d < dim; ++d) {
        if (_tmp.norm() > 1.e-12)
          sensitivities[i * dim + d] /= _tmp.norm();
      }
      for (unsigned int d = 0; d < dim; ++d) {
        sensitivities[i * dim + d] *= weighting_sol[i];
      }
    }
    shape_constraints.distribute(sensitivities);

    dof_handler_manifold.clear();
  }

  template<int dim>
  void Regularization<dim>::_ComponentWeighting(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                                AffineConstraints<double> &shape_constraints,
                                                Vector<double> &sensitivities) {

    FESystem<dim - 1, dim> fe_shape(FE_Q<dim - 1, dim>(1), dim);

    const unsigned int n_dofs = dof_handler_shape.n_dofs();
    FullMatrix<double> weighting_matrix(n_dofs, n_dofs);
    Vector<double> inverse_weights_vector(n_dofs);
    DiagonalMatrix<Vector<double>> inverse_weighting_matrix;

    const QGauss<dim - 1> quadrature(1 + 1);
    FEValues<dim - 1, dim> fe_values(fe_shape, quadrature, update_values | update_quadrature_points |
                                                           update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe_shape.dofs_per_cell;
    const unsigned int n_q_points = quadrature.size();
    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    assert(_raw_sensitivities.size() == dof_handler_shape.n_dofs());

    // Assemble weighting matrix
    for (const auto &cell : dof_handler_shape.active_cell_iterators()) {
      cell_matrix = 0;
      fe_values.reinit(cell);

      // Cell weighting matrix
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          cell_matrix(i, i) += fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
        }
      }


      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      shape_constraints.distribute_local_to_global(cell_matrix, local_dof_indices, weighting_matrix);
    }
    // Inverse the values so that we can perform multiplication operation with initial sensitivities
    for (unsigned int i = 0; i < n_dofs; ++i) {
      inverse_weights_vector[i] = 1.0 / weighting_matrix(i, i);
    }

    // Store the components to diagonal matrix
    inverse_weighting_matrix.reinit(inverse_weights_vector);

    // Contraction of raw sensitivity and weighting matrix
    inverse_weighting_matrix.vmult(sensitivities, _raw_sensitivities);
  }

  template<int dim>
  void Regularization<dim>::_DualDescentSmoothing(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                                  AffineConstraints<double> &shape_constraints,
                                                  Vector<double> &sensitivities) {

    double sufficient_decrease = _data.sufficient_decrease_coeff;

    shape_constraints.distribute(sensitivities);

    //Compute the nodal averaging_vector
    _NodalAveraging(dof_handler_shape, shape_constraints);

    int vec_size = sensitivities.size();

    Vector<double> theta(vec_size);
    double nrm_raw_sen, nrm_norm_sen, nrm_theta, alpha_coeff, cos_theta_norm, cos_theta;

    nrm_raw_sen = _raw_sensitivities.l2_norm();
    nrm_norm_sen = sensitivities.l2_norm();

    cos_theta_norm = _raw_sensitivities * sensitivities / nrm_raw_sen / nrm_norm_sen;

    // Find the optimal alpha value, which defines the proportion of weighted sensitivities and averaging vector
    alpha_coeff = 0.5;
    for (unsigned int i = 0; i <= 10; ++i) {
      theta = 0;
      theta.add(alpha_coeff, sensitivities, 1 - alpha_coeff, _averaging_vector); // *this += a*V + b*W

      nrm_theta = theta.l2_norm();

      cos_theta = _raw_sensitivities * theta / nrm_raw_sen / nrm_theta;

      //If the dual descent direction meets the sufficient decrease criterion, then break
      if (cos_theta > sufficient_decrease * cos_theta_norm)
        break;

      alpha_coeff += .05;
    }
    sensitivities = theta;

    nrm_theta = sensitivities.l2_norm();
  }

  template<int dim>
  void Regularization<dim>::_NodalAveraging(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                            AffineConstraints<double> &shape_constraints) {

    _averaging_vector.reinit(dof_handler_shape.n_dofs());

    unsigned int vertex_index, dof_index;
    Point<dim> vertex, averaged_vertex;
    Tensor<1, dim> cell_contribution;

    // Vertices belong to more than one cell. We want to call then just once.
    std::vector<bool> vertex_touched(dof_handler_shape.get_triangulation().n_vertices());

    // Adjacent cells to current vertex
    std::vector<typename DoFHandler<dim - 1, dim>::active_cell_iterator> adjacent_cells;

    for (auto cell: dof_handler_shape.active_cell_iterators()) {

      for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v) {
        vertex_index = cell->vertex_index(v);

        if (int(cell->material_id()) == _data.symmetry_id) {
          dof_index = cell->vertex_dof_index(v, 0);
          cell_contribution = 0.0;
          for (unsigned int d = 0; d < dim; ++d)
            _averaging_vector[dof_index + d] = cell_contribution[d];
          vertex_touched[vertex_index] = true;
          continue;
        }

        if (!vertex_touched[vertex_index]) {
          dof_index = cell->vertex_dof_index(v, 0);
          vertex = cell->vertex(v);

          adjacent_cells = GridTools::find_cells_adjacent_to_vertex(dof_handler_shape, vertex_index);

          if(_data.bvp_type == "bimorphPiezoNormalModes" || _data.bvp_type == "bimorphPiezoStatic") {
            std::vector<typename DoFHandler<dim - 1, dim>::active_cell_iterator> adjacent_corrected_cells;

            for (auto& adj_cell : adjacent_cells) {
              if(std::abs(adj_cell->center()[2] - adj_cell->vertex(0)[2]) < 1e-4
              || std::abs(adj_cell->center()[0] - adj_cell->vertex(0)[0]) < 1e-4
              || int(adj_cell->material_id()) == _data.symmetry_id)
                continue;
              else
                adjacent_corrected_cells.push_back(adj_cell);
            }
            if (!adjacent_corrected_cells.empty())
              adjacent_cells = adjacent_corrected_cells;
          }

          else {
            std::vector<typename DoFHandler<dim - 1, dim>::active_cell_iterator> adjacent_corrected_cells;

            for (auto& adj_cell : adjacent_cells) {
              if(int(adj_cell->material_id()) == _data.symmetry_id)
                continue;
              else
                adjacent_corrected_cells.push_back(adj_cell);
            }
            if (!adjacent_corrected_cells.empty())
              adjacent_cells = adjacent_corrected_cells;
          }

          _ComputeAveragedVertex(vertex_index, adjacent_cells, averaged_vertex);
          cell_contribution = (averaged_vertex - vertex);

          for (unsigned int d = 0; d < dim; ++d)
            _averaging_vector[dof_index + d] = cell_contribution[d];

          vertex_touched[vertex_index] = true;
        }
      }
    }
    // Reset the values at constrained dofs to zero
    shape_constraints.distribute(_averaging_vector);

    if(_data.sensitivity_projection != "none")
      _Projection(dof_handler_shape, _averaging_vector);
  }

  template<int dim>
  void Regularization<dim>::_ComputeAveragedVertex(unsigned int vertex_index,
                                                   std::vector<typename DoFHandler<
                                                       dim - 1, dim>::active_cell_iterator> adjacent_cells,
                                                   Point<dim> &averaged_vertex) {

    //All vertices that belong to adjacent cells except the averaged vertex
    std::vector<Point<dim> > others;
    std::vector<unsigned int> others_indices;
    typename std::vector<unsigned int>::iterator oth_beg, oth_end;
    bool is_duplicate;
    // For umbrella method
    std::vector<unsigned int> tria_indices;
    std::vector<Point<dim>> tria_points;
    bool target_is_part_of_tria;
    (void) target_is_part_of_tria;

    others.clear();
    others_indices.clear();

    if (dim == 2) {
      for (unsigned int c = 0; c < adjacent_cells.size(); ++c) {
        for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v) {
          if (adjacent_cells[c]->vertex_index(v) != vertex_index)
            others.push_back(adjacent_cells[c]->vertex(v));
        }
      }
    }
      // respect all adjoining vertices
    else if (dim == 3) {
      for (unsigned int c = 0; c < adjacent_cells.size(); ++c) {
        for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v) {
          if (adjacent_cells[c]->vertex_index(v) != vertex_index) {
            is_duplicate = false;

            oth_beg = others_indices.begin();
            oth_end = others_indices.end();

            if (std::find(oth_beg, oth_end, adjacent_cells[c]->vertex_index(v)) != oth_end)
              is_duplicate = true;

            if (!is_duplicate) {
              others.push_back(adjacent_cells[c]->vertex(v));
              others_indices.push_back(adjacent_cells[c]->vertex_index(v));
            }
          }
        }
      }
    }

    for (unsigned int d = 0; d < dim; ++d) {
      averaged_vertex[d] = 0.0;
    }

    for (unsigned int v = 0; v < others.size(); ++v) {
      averaged_vertex += others[v];
    }
    averaged_vertex /= others.size();
  }

} // End of StructuralOptimization namespace


template
class StructuralOptimization::Regularization<2>;

template
class StructuralOptimization::Regularization<3>;
