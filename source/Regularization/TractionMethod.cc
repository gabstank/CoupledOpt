#include <Regularization.h>

namespace StructuralOptimization {

  template<int dim>
  TractionMethod<dim>::TractionMethod(Parameter &par_)
      : _data(par_.data),
        compute_timer(par_.compute_timer),
        _fe_shape(FE_Q<dim - 1, dim>(1), dim),
        _mapping(1 + 1) {}

  template<int dim>
  void TractionMethod<dim>::Run(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                AffineConstraints<double> &shape_constraints, Vector<double> &sensitivities,
                                const Vector<double> &vertex_normals) {

    _initial_sensitivities = sensitivities;
    _vertex_normals = vertex_normals;

    _SolveLoadStepNR(dof_handler_shape, shape_constraints);


    sensitivities = _solution;
  }


  template<int dim>
  void TractionMethod<dim>::RunSimple(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                AffineConstraints<double> &shape_constraints, Vector<double> &sensitivities,
                                const Vector<double> &vertex_normals) {

    _initial_sensitivities = sensitivities;
    _vertex_normals = vertex_normals;

    _SetupSystem(dof_handler_shape, shape_constraints);
    _AssembleSpringSmoothingMatrix(dof_handler_shape, shape_constraints);

    _system_rhs = _initial_sensitivities;
    shape_constraints.distribute(_system_rhs);
    _system_matrix = 0.0;
    _system_matrix.copy_from(_spring_matrix);

    // reset the vector newton update
    _newton_update = 0.0;
    _SolveLinearSystem(shape_constraints);
    // Add the newton increment to the solution
    _solution += _newton_update;

    sensitivities = _solution;
    sensitivities *= _initial_sensitivities.linfty_norm() / sensitivities.linfty_norm();
/*
    for(unsigned int i = 0; i < _initial_sensitivities.size(); ++i)
      if(_initial_sensitivities[i] == 0)
        sensitivities[i] = 0;*/
  }

  template<int dim>
  void TractionMethod<dim>::_SetupSystem(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                         AffineConstraints<double> &shape_constraints) {

    DynamicSparsityPattern dsp(dof_handler_shape.n_dofs(), dof_handler_shape.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_shape,
                                    dsp,
                                    shape_constraints,
        /*keep_constrained_dofs = */ false);

    _sparsity_pattern.copy_from(dsp);

    _system_matrix.reinit(_sparsity_pattern);
    _spring_matrix.reinit(_sparsity_pattern);

    _system_rhs.reinit(dof_handler_shape.n_dofs());
    _solution.reinit(dof_handler_shape.n_dofs());
    _residual.reinit(dof_handler_shape.n_dofs());
    _newton_update.reinit(dof_handler_shape.n_dofs());

    // Constraints are already setup
  }

  template<int dim>
  void TractionMethod<dim>::_AssembleSpringSmoothingMatrix(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                                           AffineConstraints<double> &shape_constraints) {

    QTrapezoid<dim - 1> quadrature_formula;
    //QGauss<dim-1> quadrature_formula(_fe_shape.degree + 1);
    FEValues<dim - 1, dim> fe_values(_mapping, _fe_shape, quadrature_formula,
                                     update_values | update_gradients | update_quadrature_points
                                     | update_normal_vectors | update_JxW_values);

    const FEValuesExtractors::Vector u_fe(0);

    const unsigned int dofs_per_cell = this->_fe_shape.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> dummy_cell_rhs(dofs_per_cell);
    dummy_cell_rhs = 0;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    double normal_spring_c = _data.tm_normal_spring_constant;
    double tangent_spring_c = _data.tm_tangent_spring_constant;
    double normal_smoothing_c = _data.tm_normal_smoothing_constant;
    double tangent_smoothing_c = _data.tm_tangent_smoothing_constant;

    for (const auto &cell : dof_handler_shape.active_cell_iterators()) {
      cell_matrix = 0.0;
      fe_values.reinit(cell);

      std::vector<Tensor<2, dim> > normal_grads_at_gauss(n_q_points);
      fe_values[u_fe].get_function_gradients(_vertex_normals, normal_grads_at_gauss);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        //The quadrature weight for the current quadrature point
        const double JxW = fe_values.JxW(q);

        Tensor<1, dim> Normal = fe_values.normal_vector(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {

          Tensor<1, dim> N_i = fe_values[u_fe].value(i, q);
          SymmetricTensor<2, dim> Grad_N_i = fe_values[u_fe].symmetric_gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            Tensor<1, dim> N_j = fe_values[u_fe].value(j, q);
            SymmetricTensor<2, dim> Grad_N_j = fe_values[u_fe].symmetric_gradient(j, q);

            // Normal spring contribution
            cell_matrix(i, j) += normal_spring_c * (N_i * Normal) * (N_j * Normal) * JxW;

            // Tangent spring contribution
            cell_matrix(i, j) += tangent_spring_c * (N_i * N_j - (N_i * Normal) * (N_j * Normal)) * JxW;

            // Normal smoothing contribution
            cell_matrix(i, j) += normal_smoothing_c * (Normal * Grad_N_i) * (Normal * Grad_N_j) * JxW;

            // Tangent (in-plane) smoothing contribution
            cell_matrix(i, j) +=
                tangent_smoothing_c * (Grad_N_i * Grad_N_j - (Normal * Grad_N_i) * (Normal * Grad_N_j)) * JxW;

          } // end of j loop
        } // end of i loop
      } // end of loop over quadrature points

      cell->get_dof_indices(local_dof_indices);

      shape_constraints
          .distribute_local_to_global(cell_matrix, dummy_cell_rhs, local_dof_indices, _spring_matrix,
                                      _system_rhs);

    }// end of loop over cells
  }

  template<int dim>
  void TractionMethod<dim>::_AssembleSpringSmoothingRhs(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                                        AffineConstraints<double> &shape_constraints) {

    QTrapezoid<dim - 1> quadrature_formula;
    FEValues<dim - 1, dim> fe_values(_mapping, _fe_shape, quadrature_formula,
                                     update_values | update_gradients | update_quadrature_points |
                                     update_normal_vectors | update_JxW_values);

    const FEValuesExtractors::Vector u_fe(0);

    const unsigned int dofs_per_cell = this->_fe_shape.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> dummy_cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    dummy_cell_matrix = 0;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    double normal_spring_c = _data.tm_normal_spring_constant;
    double tangent_spring_c = _data.tm_tangent_spring_constant;
    double normal_smoothing_c = _data.tm_normal_smoothing_constant;
    double tangent_smoothing_c = _data.tm_tangent_smoothing_constant;

    for (const auto &cell : dof_handler_shape.active_cell_iterators()) {
      cell_rhs = 0.0;
      fe_values.reinit(cell);

      std::vector<Tensor<1, dim> > solution_at_gauss(n_q_points);
      fe_values[u_fe].get_function_values(_solution, solution_at_gauss);

      std::vector<SymmetricTensor<2, dim> > solution_grads_at_gauss(n_q_points);
      fe_values[u_fe].get_function_symmetric_gradients(_solution, solution_grads_at_gauss);

      std::vector<Tensor<2, dim> > normal_grads_at_gauss(n_q_points);
      fe_values[u_fe].get_function_gradients(_vertex_normals, normal_grads_at_gauss);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        //The quadrature weight for the current quadrature point
        const double JxW = fe_values.JxW(q);

        Tensor<1, dim> Normal = fe_values.normal_vector(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          Tensor<1, dim> N_i = fe_values[u_fe].value(i, q);
          SymmetricTensor<2, dim> Grad_N_i = fe_values[u_fe].symmetric_gradient(i, q);

          // Normal spring contribution
          cell_rhs(i) -= normal_spring_c * (solution_at_gauss[q] * Normal) * (N_i * Normal) * JxW;

          // Tangent spring contribution
          cell_rhs(i) -= tangent_spring_c *
                         (solution_at_gauss[q] - (solution_at_gauss[q] * Normal) * Normal)
                         * N_i * JxW;

          // Normal smoothing contribution
          cell_rhs(i) -= normal_smoothing_c * (solution_grads_at_gauss[q] * Normal) * (Grad_N_i * Normal) * JxW;

          // Tangent (in-plane) smoothing contribution
          cell_rhs(i) -= tangent_smoothing_c *
                         (solution_grads_at_gauss[q] * Grad_N_i
                          - (solution_grads_at_gauss[q] * Normal) * (Grad_N_i * Normal)) * JxW;

        } // end of i loop
      } // end of loop over quadrature points

      cell->get_dof_indices(local_dof_indices);

      shape_constraints
          .distribute_local_to_global(dummy_cell_matrix, cell_rhs, local_dof_indices, _system_matrix,
                                      _system_rhs);

    }// end of loop over cells
  }

  template<int dim>
  void TractionMethod<dim>::_AssemblePenalties(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                               AffineConstraints<double> &shape_constraints) {

    // At this point we assume that the bounding box is always defined
    // A bounding box constraint is a pair of a space component that it bounds and its value
    // At this point we also assume that the upper bounds are positive and lower bounds are negative
    std::vector<std::pair<unsigned int, double>> bb_constraints;
    bb_constraints.emplace_back(0, _data.bb_x[0]);
    bb_constraints.emplace_back(0, _data.bb_x[1]);
    bb_constraints.emplace_back(1, _data.bb_y[0]);
    bb_constraints.emplace_back(1, _data.bb_y[1]);
    if(dim == 3)
    {
      bb_constraints.emplace_back(2, _data.bb_z[0]);
      bb_constraints.emplace_back(2, _data.bb_z[1]);
    }
    Tensor<1, dim> mat_unit_vector;
    double penalty_constant = _data.tm_penalty_constant;

    QTrapezoid<dim - 1> quadrature_formula;
    FEValues<dim - 1, dim> fe_values(_mapping, _fe_shape, quadrature_formula,
                                     update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const FEValuesExtractors::Vector u_fe(0);

    const unsigned int dofs_per_cell = _fe_shape.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler_shape.active_cell_iterators()) {

      cell_matrix = 0.0;
      cell_rhs = 0.0;
      fe_values.reinit(cell);

      //Vector to store the solution at n_q_points quadrature points
      std::vector<Tensor<1, dim> > solution_at_int_point(n_q_points);
      std::vector<Point<dim> > position_at_int_point(n_q_points);

      //Fill the previous vector using get_function_gradients
      fe_values[u_fe].get_function_values(_solution, solution_at_int_point);
      position_at_int_point = fe_values.get_quadrature_points();

      for (unsigned int q = 0; q < n_q_points; ++q) {

        // Check if a bounding box constraint is active - for that we need a integration point position and its
        // current displacement: g = (X + u - h)*sgn(h) -> if g > 0 then is active
        for (unsigned int bb_constr = 0; bb_constr < dim * 2; ++bb_constr) {

          unsigned int comp = bb_constraints[bb_constr].first;
          double current_pos = position_at_int_point[q][comp] + solution_at_int_point[q][comp];
          double current_bb = bb_constraints[bb_constr].second;

          double g = (current_pos - current_bb) * sgn(current_bb);

          if (g > 0.0) {

            // Set the E_a unit vector
            mat_unit_vector = 0;
            mat_unit_vector[comp] = 1;

            //The quadrature weight for the current quadrature point
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              Tensor<1, dim> N_i = fe_values[u_fe].value(i, q);

              cell_rhs(i) -= penalty_constant * g * N_i * mat_unit_vector * sgn(current_bb) * JxW;

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                Tensor<1, dim> N_j = fe_values[u_fe].value(j, q);

                cell_matrix(i, j) += penalty_constant * (N_i * mat_unit_vector) * (N_j * mat_unit_vector) * JxW;
              } // end of j loop
            } // end of i loop
          } // end of PenaltyIsActive condition
        } // end of bb_constr loop
      } // end of loop over quadrature points

      cell->get_dof_indices(local_dof_indices);

      shape_constraints
          .distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, _system_matrix,
                                      _system_rhs);

    }// end of loop over cells
  }

  template<int dim>
  void TractionMethod<dim>::_SolveLoadStepNR(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                             AffineConstraints<double> &shape_constraints) {

    _SetupSystem(dof_handler_shape, shape_constraints);
    _AssembleSpringSmoothingMatrix(dof_handler_shape, shape_constraints);

    if(!_rhs_scale_set)
      _SetRhsScale(dof_handler_shape, shape_constraints);

    // A vector used for all the newton increments
    _newton_update = 0.0;
    _error_NR.Reset();

    _newton_iteration = 0;

    for (; _newton_iteration < 10; ++_newton_iteration) {
      // Reset the tangent matrix and the rhs vector
      _system_matrix = 0.0;
      _system_rhs = 0.0;

      _system_matrix.copy_from(_spring_matrix);
      _system_rhs = _initial_sensitivities;
      _system_rhs *= _rhs_scale;
      shape_constraints.distribute(_system_rhs);

      // Add penalty stiffness and rhs
      _AssembleSpringSmoothingRhs(dof_handler_shape, shape_constraints);
      _AssemblePenalties(dof_handler_shape, shape_constraints);

      if (_newton_iteration == 0) {
        _error_NR.Initialize(_GetErrorResidual(dof_handler_shape, shape_constraints));
      }
      double error_residual_norm = _error_NR.GetNormalizedError(_GetErrorResidual(dof_handler_shape,
                                                                                  shape_constraints));

      // Problem has to be solved at least once
      if (_newton_iteration > 0 && error_residual_norm <= 1e-8) {
        break;
      }

      // reset the vector newton update
      _newton_update = 0.0;

      _SolveLinearSystem(shape_constraints);

      // Add the newton increment to the solution
      _solution += _newton_update;
    }
  }

  template<int dim>
  void TractionMethod<dim>::_SolveLinearSystem(AffineConstraints<double> &shape_constraints) {
    /*reset the vector newton update*/
    _newton_update = 0.0;

    const int solver_its = _system_matrix.m() * _data.max_iter;
    const double tol_sol = _data.solver_tol * _system_rhs.l2_norm();
    SolverControl solver_control(solver_its, tol_sol);
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(_system_matrix, 1.2);
    cg.solve(_system_matrix, _newton_update, _system_rhs, preconditioner);

    if(std::isnan(_newton_update.l1_norm())) {
      _newton_update = 0.0;
      SparseDirectUMFPACK A_direct;
      A_direct.initialize(_system_matrix);
      A_direct.vmult(_newton_update, _system_rhs);
    }

    shape_constraints.distribute(_newton_update);

  }

  template<int dim>
  double TractionMethod<dim>::_GetErrorResidual(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                                AffineConstraints<double> &shape_constraints) {
    /*This step is necessary if the entry of the vector
    at a constrained dof is not zero - this depends on
    the way constraints are imposed; To be sure it is
    safer to only consider the unconstrained entries anyway*/
    for (unsigned int i = 0; i < dof_handler_shape.n_dofs(); ++i) {
      if (!shape_constraints.is_constrained(i)) {
        _residual(i) = _system_rhs(i);
      }
    }
    return _residual.l2_norm();
  }

  template<int dim>
  void TractionMethod<dim>::_SetRhsScale(DoFHandler<dim - 1, dim> &dof_handler_shape,
                                        AffineConstraints<double> &shape_constraints) {
    _rhs_scale_set = true;
    _system_matrix.copy_from(_spring_matrix);
    _system_rhs = _initial_sensitivities;
    _AssembleSpringSmoothingRhs(dof_handler_shape, shape_constraints);
    _AssemblePenalties(dof_handler_shape, shape_constraints);
    _SolveLinearSystem(shape_constraints);

    double max_val_solution = 0.0;
    for (unsigned int i = 0; i < _newton_update.size(); ++i) {
      if (std::abs(_newton_update[i]) > max_val_solution)
        max_val_solution = std::abs(_newton_update[i]);
      _rhs_scale = this->_data.initial_shape_step_length / max_val_solution;
      //_rhs_scale = _initial_sensitivities.linfty_norm() / max_val_solution;
    }
  }

} // End of StructuralOptimization namespace

template
class StructuralOptimization::TractionMethod<2>;

template
class StructuralOptimization::TractionMethod<3>;