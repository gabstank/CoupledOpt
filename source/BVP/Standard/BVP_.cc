// C++ headers

// Deal.II headers

// Project headers
#include <BVP_.h>

namespace StructuralOptimization {

  template<int dim>
  BVP_<dim>::BVP_(Parameter &par_, Mesh<dim> &mesh_)
      :
      mpi_communicator(par_.mpi_communicator),
      n_mpi_processes(par_.n_mpi_processes),
      this_mpi_process(par_.this_mpi_process),
      pcout(par_.pcout),
      compute_timer(par_.compute_timer),
      data(par_.data),
      _smoothing_fe(FE_Q<dim>(1)),
      _smoothing_dof_handler(*mesh_.domain_parallel),
      _tria_domain(*mesh_.domain),
      _tria_domain_parallel(*mesh_.domain_parallel),
      _dof_handler(*mesh_.domain_parallel){

  }

  template<int dim>
  Vector<double> BVP_<dim>::ReturnPointSolution(Point<dim> pt) {
    Vector<double> locally_relevant_solution(_solution);
    Functions::FEFieldFunction<dim, DoFHandler<dim>> fe_eval(_dof_handler, locally_relevant_solution);
    Vector<double> pt_solution(dim);
    fe_eval.vector_value(pt, pt_solution);
    return pt_solution;
  }

  template<int dim>
  void BVP_<dim>::_SolveLinearSystem(LA::MPI::SparseMatrix &system_matrix,
                                     LA::MPI::Vector &solution,
                                     LA::MPI::Vector &system_rhs) {

    SolverControl solver_control(_dof_handler.n_dofs(),
                                 data.solver_tol * system_rhs.l2_norm());

    LA::MPI::PreconditionJacobi preconditioner;
    LA::MPI::PreconditionJacobi::AdditionalData additional_data;

    preconditioner.initialize(system_matrix, additional_data);

    if (data.solver_name == "CG") {

#ifdef USE_PETSC_LA
      LA::SolverCG solver(solver_control, mpi_communicator);
#else
      LA::SolverCG solver(solver_control);
#endif
      solver.solve(system_matrix, solution, system_rhs, preconditioner);

    } else if (this->data.solver_name == "Mumps") {

#ifdef USE_PETSC_LA
      PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
      solver.initialize(preconditioner);
      solver.set_symmetric_mode(true);
#else
      TrilinosWrappers::SolverDirect::AdditionalData additional_data;
      additional_data.solver_type = "Amesos_Mumps";
      TrilinosWrappers::SolverDirect solver(solver_control, additional_data);
#endif
      solver.solve(system_matrix, solution, system_rhs);
    }
    else if (this->data.solver_name == "Superludist") {

#ifdef USE_PETSC_LA
      PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
      solver.initialize(preconditioner);
      solver.set_symmetric_mode(true);
#else
      TrilinosWrappers::SolverDirect::AdditionalData additional_data;
      additional_data.solver_type = "Amesos_Superludist";
      TrilinosWrappers::SolverDirect solver(solver_control, additional_data);
#endif
      solver.solve(system_matrix, solution, system_rhs);
    }
    else if (this->data.solver_name == "Umfpack") {

#ifdef USE_PETSC_LA
      PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
      solver.initialize(preconditioner);
      solver.set_symmetric_mode(true);
#else
      TrilinosWrappers::SolverDirect::AdditionalData additional_data;
      additional_data.solver_type = "Amesos_Umfpack";
      TrilinosWrappers::SolverDirect solver(solver_control, additional_data);
#endif
      solver.solve(system_matrix, solution, system_rhs);
    }

    if (this->data.verbose)
      this->pcout << "Solver " << this->data.solver_name << ": Solved Linear system: n_dofs=" << solution.size()
                  << " | n_iteration=" << solver_control.last_step()
                  << " | residual=" << solver_control.last_value() << std::endl;

  }

  template<int dim>
  void BVP_<dim>::_SetupSmoothingSystem() {

    _smoothing_dof_handler.distribute_dofs(this->_smoothing_fe);

    _locally_owned_smooth_dofs = _smoothing_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(_smoothing_dof_handler, _locally_relevant_smooth_dofs);

    _smoothing_solution.reinit(_locally_owned_smooth_dofs, _locally_relevant_smooth_dofs, mpi_communicator);
    _smoothing_rhs.reinit(_locally_owned_smooth_dofs, mpi_communicator);

    _smoothing_constraints.clear();
    _smoothing_constraints.reinit(_locally_relevant_smooth_dofs);
    DoFTools::make_hanging_node_constraints(_smoothing_dof_handler, _smoothing_constraints);

#ifdef DEBUG
// We did not think about hp constraints on ghost cells yet.
// Thus, we are content with verifying their consistency for now.
    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(_smoothing_dof_handler, locally_active_dofs);
    AssertThrow(_smoothing_constraints.is_consistent_in_parallel(
        _smoothing_dof_handler.locally_owned_dofs_per_processor(),
        locally_active_dofs,
        mpi_communicator,
/*verbose=*/true),
                ExcMessage(
                    "AffineConstraints object contains inconsistencies!"));
#endif

    _smoothing_constraints.close();

    DynamicSparsityPattern dsp(_locally_relevant_smooth_dofs);
    DoFTools::make_sparsity_pattern(_smoothing_dof_handler, dsp, _smoothing_constraints, true);
    SparsityTools::distribute_sparsity_pattern(
        dsp,
        _locally_owned_smooth_dofs,
        mpi_communicator,
        _locally_relevant_smooth_dofs);

    _smoothing_matrix.reinit(_locally_owned_smooth_dofs, _locally_owned_smooth_dofs, dsp, mpi_communicator);

  }

  template<int dim>
  void BVP_<dim>::AssembleSmoothingMatrix() {

    _smoothing_matrix = 0;

    QGauss<dim> quadrature_formula(_smoothing_fe.degree + 1);
    FEValues<dim> fe_values(_smoothing_fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    double n_i, n_j;

    FullMatrix<double> cell_matrix;
    std::vector<unsigned int> local_dof_indices;
    const unsigned int dofs_per_cell = _smoothing_fe.dofs_per_cell;

    for (auto &cell : _smoothing_dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_matrix.reinit(cell->get_fe().dofs_per_cell, cell->get_fe().dofs_per_cell);

      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
        // phi_phi assembly
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
          n_i = fe_values.shape_value(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; j++) {
            n_j = fe_values.shape_value(j, q);
            cell_matrix(i, j) += n_i * n_j * fe_values.JxW(q);
          } // end of loop over j
        } // end of loop over i

      } // end of loop over q points

      // add to system smoothing matrix
      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      _smoothing_constraints.distribute_local_to_global(cell_matrix,
                                                        local_dof_indices,
                                                        _smoothing_matrix);
    }
    _smoothing_matrix.compress(VectorOperation::add);
  }

  template<int dim>
  void BVP_<dim>::SolveSmoothingSystem() {

    SolverControl solver_control(_dof_handler.n_dofs(),
                                 data.solver_tol * _smoothing_rhs.l2_norm());

    LA::MPI::PreconditionJacobi preconditioner;
    LA::MPI::PreconditionJacobi::AdditionalData additional_data;

    preconditioner.initialize(_smoothing_matrix, additional_data);

    LA::MPI::Vector completely_distributed_solution(_smoothing_rhs);
    completely_distributed_solution = 0.0;

    _smoothing_constraints.set_zero(completely_distributed_solution);

    if(data.pp_solver_name == "CG") {
#ifdef USE_PETSC_LA
      LA::SolverCG solver(solver_control, mpi_communicator);
#else
      LA::SolverCG solver(solver_control);
#endif
      solver.solve(_smoothing_matrix,
                   completely_distributed_solution,
                   _smoothing_rhs,
                   preconditioner);
    }
    else{
#ifdef USE_PETSC_LA
      PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
      solver.initialize(preconditioner);
      solver.set_symmetric_mode(true);
#else
      TrilinosWrappers::SolverDirect::AdditionalData additional_data;
      if(data.pp_solver_name == "Mumps")
        additional_data.solver_type = "Amesos_Mumps";
      else if(data.pp_solver_name == "Superludist")
        additional_data.solver_type = "Amesos_Superludist";
      else
        additional_data.solver_type = "Amesos_Umfpack";
      TrilinosWrappers::SolverDirect solver(solver_control, additional_data);
#endif
      solver.solve(_smoothing_matrix,
                   completely_distributed_solution,
                   _smoothing_rhs);

    }

    _smoothing_constraints.distribute(completely_distributed_solution);

    _smoothing_solution = completely_distributed_solution;

  }

  template<int dim>
  void BVP_<dim>::GetAllOutputData(DataOutput<dim, DomainParallelTriaType<dim>, DoFHandler<dim>> &output) {

    std::vector<std::string> sol_names(dim, "sol_u"), rhs_names(dim, "rhs_u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        sol_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector),
        rhs_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    if(data.bvp_type == "electroElasticityLin")
    {
      rhs_names.push_back("rhs_charge");
      sol_names.push_back("sol_phi");
      rhs_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar); // for charge
      sol_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar); // for potential
    }
    output.template PushDataName<LA::MPI::Vector>(_system_rhs, rhs_names,
                                                  rhs_data_component_interpretation,
                                                  &_dof_handler);
    output.template PushDataName<LA::MPI::Vector>(_solution, sol_names,
                                                  sol_data_component_interpretation,
                                                  &_dof_handler);

  }// end of function

  template<int dim>
  void BVP_<dim>::SetPseudoDensitiesToOne(){
    for(const auto& cell : _dof_handler.active_cell_iterators())
      if(cell->is_locally_owned())
        pseudo_densities[cell->id()] = 1.0;
  }


} // end of StructuralOptimization namespace

template
class StructuralOptimization::BVP_<2>;

template
class StructuralOptimization::BVP_<3>;
