// C++ headers

// Deal.II headers

// Project headers
#include <EddBVP_.h>
#include <Mesh.h>

#include <deal.II/lac/slepc_solver.h>

namespace StructuralOptimization {

  template<int dim>
  EddBVP_<dim>::EddBVP_(Parameter &par_, Mesh<dim> &mesh_)
      :
      mpi_communicator(par_.mpi_communicator),
      this_mpi_process(par_.this_mpi_process),
      pcout(par_.pcout),
      compute_timer(par_.compute_timer),
      data(par_.data),
      _tria_shape(mesh_.shape),
      _dof_handler_shape(mesh_.shape),
      _tria_base(*mesh_.base),
      _dof_handler_base(*mesh_.base),
      min_face_length_base(mesh_.mesh_tracking->min_face_length_base),
      boundary_cell_assignment(mesh_.mesh_tracking->boundary_cell_assignment),
      _smoothing_fe_void(FE_Nothing<dim>()),
      _smoothing_fe_solid(FE_Q<dim>(data.poly_degree)),
      _smoothing_dof_handler(*mesh_.base),
      smoothing_output(par_.data, par_.mpi_communicator, par_.compute_timer, "edd-smoothing", *mesh_.base){
  }

  template<int dim>
  void EddBVP_<dim>::SetupSmoothingSystem() {
    _smoothing_fe_collection.push_back(_smoothing_fe_void);
    _smoothing_fe_collection.push_back(_smoothing_fe_solid);

    // Setting active fe index
    /** Loop over all the cells in hp_dof_handler_hold_all and check if cell
    * is outside or inside using the _cell_is_outside(cell) or _cell_is_inside(cell)
    * function and set the flag cell->set_active_fe_index(0) or cell->set_active_fe_index(1)*/
    for (const auto &cell : _smoothing_dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;
      if (EddTools::CellIsOutside<dim>(cell))
        cell->set_active_fe_index(0);
      else if (EddTools::CellIsInside<dim>(cell))
        cell->set_active_fe_index(1);
      else
        std::cerr << __FILE__ << ":" << __LINE__ << "| Cell is neither inside nor outside" << std::endl;
    }

    _smoothing_dof_handler.distribute_dofs(_smoothing_fe_collection);
    // DoFRenumbering::Cuthill_McKee(_smoothing_dof_handler);

    // Now we get the locally owned dofs, that is the dofs that our local
    // to this processor. These dofs corresponding entries in the
    // matrix and vectors that we will write to.
    _locally_owned_smooth_dofs = _smoothing_dof_handler.locally_owned_dofs();

    // In additon to the locally owned dofs, we also need the the locally
    // relevant dofs.  These are the dofs that have read access to and we
    // need in order to do computations on our processor, but, that
    // we do not have the ability to write to.
    DoFTools::extract_locally_relevant_dofs(_smoothing_dof_handler, _locally_relevant_smooth_dofs);

    // Postprocessing smoothing vectors
    _smoothing_solution.reinit(_locally_owned_smooth_dofs, _locally_relevant_smooth_dofs, mpi_communicator);
    _smoothing_rhs.reinit(_locally_owned_smooth_dofs, mpi_communicator);

    // Setup constraints
    _smoothing_constraints.clear();
    _smoothing_constraints.reinit(_locally_relevant_smooth_dofs);

    DoFTools::make_hanging_node_constraints(_smoothing_dof_handler, _smoothing_constraints);

    // From deal.ii test- trillinos_step-27.cc
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
    DoFTools::make_sparsity_pattern(_smoothing_dof_handler, dsp, _smoothing_constraints, false);
    SparsityTools::distribute_sparsity_pattern(
        dsp,
        _locally_owned_smooth_dofs,
        mpi_communicator,
        _locally_relevant_smooth_dofs);

    _smoothing_matrix.reinit(_locally_owned_smooth_dofs, _locally_owned_smooth_dofs, dsp, mpi_communicator);

  }

  template<int dim>
  void EddBVP_<dim>::AssembleSmoothingMatrix() {

    _smoothing_matrix = 0;

    const QGauss<dim> void_quadrature(_smoothing_fe_solid.degree + 1);
    const QGauss<dim> solid_quadrature(_smoothing_fe_solid.degree + 1);

    hp::QCollection<dim> q_collection;
    q_collection.push_back(void_quadrature);
    q_collection.push_back(solid_quadrature);

    hp::FEValues<dim> fe_values_hp(_smoothing_fe_collection, q_collection,
                                   update_values | update_quadrature_points | update_JxW_values);


    double n_i, n_j;

    FullMatrix<double> cell_matrix;
    std::vector<unsigned int> local_dof_indices;

    for (auto &cell : _smoothing_dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      fe_values_hp.reinit(cell);
      cell_matrix.reinit(cell->get_fe().dofs_per_cell, cell->get_fe().dofs_per_cell);

      if (EddTools::CellIsInside<dim>(cell)) {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
          for (unsigned int i = 0; i < dofs_per_cell; i++) {
            n_i = fe_values.shape_value(i, q);

            for (unsigned int j = 0; j < dofs_per_cell; j++) {
              n_j = fe_values.shape_value(j, q);
              cell_matrix(i, j) += n_i * n_j * fe_values.JxW(q);
            }
          }
        }
      }

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
  void EddBVP_<dim>::SolveSmoothingSystem() {

    SolverControl solver_control(_smoothing_dof_handler.n_dofs(),
                                 data.solver_tol * _smoothing_rhs.l2_norm());

    LA::MPI::Vector completely_distributed_solution(_smoothing_rhs);
    completely_distributed_solution = 0.0;

    LA::MPI::PreconditionJacobi preconditioner;
    LA::MPI::PreconditionJacobi::AdditionalData additional_data;
    preconditioner.initialize(_smoothing_matrix, additional_data);

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
    else if(data.pp_solver_name == "Mumps") {

#ifdef USE_PETSC_LA
      PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
      solver.initialize(preconditioner);
      solver.set_symmetric_mode(true);
#else
      TrilinosWrappers::SolverDirect::AdditionalData additional_data;
      additional_data.solver_type = "Amesos_Mumps";
      TrilinosWrappers::SolverDirect solver(solver_control, additional_data);
#endif
      solver.solve(_smoothing_matrix, completely_distributed_solution, _smoothing_rhs);
    }
    else if(data.pp_solver_name == "Umfpack") {

#ifdef USE_PETSC_LA
      PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
      solver.initialize(preconditioner);
      solver.set_symmetric_mode(true);
#else
      TrilinosWrappers::SolverDirect::AdditionalData additional_data;
      additional_data.solver_type = "Amesos_Umfpack";
      TrilinosWrappers::SolverDirect solver(solver_control, additional_data);
#endif
      solver.solve(_smoothing_matrix, completely_distributed_solution, _smoothing_rhs);
    }
    else if(data.pp_solver_name == "Superludist") {

#ifdef USE_PETSC_LA
      PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
      solver.initialize(preconditioner);
      solver.set_symmetric_mode(true);
#else
      TrilinosWrappers::SolverDirect::AdditionalData additional_data;
      additional_data.solver_type = "Amesos_Superludist";
      TrilinosWrappers::SolverDirect solver(solver_control, additional_data);
#endif
      solver.solve(_smoothing_matrix, completely_distributed_solution, _smoothing_rhs);
    }
    else
      throw std::runtime_error("Wrong solver type.");

    // Handling a mysterious error that appears when PETSc solvers are used.
#ifdef USE_PETSC_LA
    {
      if (std::isinf(completely_distributed_solution.l1_norm()) || std::isnan(completely_distributed_solution.l1_norm()) || std::isinf(completely_distributed_solution.linfty_norm()) ||
          std::isnan(completely_distributed_solution.linfty_norm())) {

        if(this_mpi_process == 0)
          std::cout << "Solution of a PETSc Solver for L2 projection failed. Trying SolverCG. ";

        completely_distributed_solution.reinit(_smoothing_rhs);
        LA::MPI::PreconditionAMG prec_A;
        LA::MPI::PreconditionAMG::AdditionalData data;
        data.symmetric_operator = true;
        prec_A.initialize(_smoothing_matrix, data);
        SolverCG<LA::MPI::Vector> solver_cg(solver_control);
        solver_cg.solve(_smoothing_matrix, completely_distributed_solution, _smoothing_rhs, prec_A);

        if (std::isinf(completely_distributed_solution.l1_norm()) || std::isnan(completely_distributed_solution.l1_norm()) || std::isinf(completely_distributed_solution.linfty_norm()) ||
            std::isnan(completely_distributed_solution.linfty_norm())) {
          if (this_mpi_process == 0)
            std::cout << "Failed. " << std::endl;
          throw std::runtime_error("Solving L2 projection failed.");
        } else
          if(this_mpi_process == 0)
            std::cout << "Success! " << std::endl;
      }
    }
#endif

    _smoothing_constraints.distribute(completely_distributed_solution);

    _smoothing_solution = completely_distributed_solution;

    if(std::isinf(_smoothing_solution.linfty_norm()) || std::isnan(_smoothing_solution.linfty_norm()) ) {
      std::vector<std::string> sol_names(1, "sol"), rhs_names(1, "rhs");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          sol_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar),
          rhs_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);

      smoothing_output.template PushDataName<LA::MPI::Vector>(_smoothing_rhs, rhs_names,
                                                             rhs_data_component_interpretation,
                                                             &this->_smoothing_dof_handler);
      smoothing_output.template PushDataName<LA::MPI::Vector>(_smoothing_solution, sol_names,
                                                             sol_data_component_interpretation,
                                                             &this->_smoothing_dof_handler);

      _OutputSmoothing(cycle);
      ++cycle;
/*
      dealii::SolverControl eigensolver_control(this->data.eps_iter_limit,
                                                this->data.eps_tol, false, false);
      SLEPcWrappers::SolverKrylovSchur eigensolver(eigensolver_control, this->mpi_communicator);
      PETScWrappers::set_option_value("-st_type", "sinvert");
      PETScWrappers::set_option_value("-st_ksp_type", this->data.ksp_type);
      PETScWrappers::set_option_value("-st_pc_type", this->data.preconditioner);
      PETScWrappers::set_option_value("-st_ksp_max_it", this->data.ksp_max_it);
      PETScWrappers::set_option_value("-st_ksp_tol", this->data.ksp_tol);
      eigensolver.set_which_eigenpairs(EPS_TARGET_MAGNITUDE);
      eigensolver.set_problem_type(EPS_GHEP);
      this->_eigenvectors.resize(10);
      this->_eigenvalues.resize(10);
      for (unsigned int i = 0; i < 10; ++i) {
        this->_eigenvectors[i].reinit(_smoothing_rhs);
      }
      eigensolver.solve(_smoothing_matrix,
                       this->_eigenvalues,
                       this->_eigenvectors,
                       10);

      if (this_mpi_process == 0)
        for (unsigned int i = 0; i < this->_eigenvalues.size(); ++i)
          std::cout << "      Eigenvalue " << i << " : " << this->_eigenvalues[i]
                    << std::endl;
*/
    }
  }

  template<int dim>
  void EddBVP_<dim>::GetAllOutputData(DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> &output) {

    Vector<float> mat_id(_tria_base.n_active_cells());
    unsigned int counter = 0;
    for (const auto &cell : _tria_base.active_cell_iterators()) {
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
    output.template PushDataName<Vector<float>>(mat_id, data_names, data_component_interpretation,
                                                &_dof_handler_base);

    std::vector<std::string> sol_names(dim, "sol_u"), rhs_names(dim, "rhs_u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        sol_data_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector),
        rhs_data_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);
    if(data.bvp_type == "electroElasticityLin")
    {
      rhs_names.push_back("rhs_charge");
      sol_names.push_back("sol_phi");
      rhs_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar); // for charge
      sol_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar); // for potential
    }

    output.template PushDataName<LA::MPI::Vector>(_system_rhs,
                            rhs_names,
                            rhs_data_component_interpretation,
                            &_dof_handler_base);
    output.template PushDataName<LA::MPI::Vector>(_solution,
                            sol_names,
                            sol_data_component_interpretation,
                            &_dof_handler_base);

  }// end of function

  template<int dim>
  void EddBVP_<dim>::SolveLinearSystem(LA::MPI::SparseMatrix &system_matrix,
                                       LA::MPI::Vector &solution,
                                       LA::MPI::Vector &system_rhs,
                                       std::string solver_name) {

    SolverControl solver_control(_dof_handler_base.n_dofs(),
                                 data.solver_tol * system_rhs.l2_norm());

    LA::MPI::PreconditionJacobi preconditioner;
    LA::MPI::PreconditionJacobi::AdditionalData additional_data;

    preconditioner.initialize(system_matrix, additional_data);

    if (solver_name == "")
      solver_name = data.solver_name;

    if (solver_name == "CG") {
#ifdef USE_PETSC_LA
      LA::SolverCG solver(solver_control, mpi_communicator);
#else
      LA::SolverCG solver(solver_control);
#endif
      solver.solve(system_matrix, solution, system_rhs, preconditioner);

    } else if (solver_name == "Mumps") {
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

    } else if (solver_name == "GMRES") {
#ifdef USE_PETSC_LA
      LA::SolverGMRES solver(solver_control, mpi_communicator);
#else
      LA::SolverGMRES solver(solver_control);
#endif
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
    } else if (solver_name == "Umfpack") {
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

    } else if (solver_name == "Superludist") {
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
    else
      throw std::runtime_error("Wrong BVP type.");

    // Handling a mysterious error that appears when PETSc solvers are used.
#ifdef USE_PETSC_LA
    {
      if (std::isinf(solution.l1_norm()) || std::isnan(solution.l1_norm()) || std::isinf(solution.linfty_norm()) ||
          std::isnan(solution.linfty_norm())) {

        if(this_mpi_process == 0)
          std::cout << "Solution of a PETSc Solver for the BVP failed. Trying SolverCG. ";

        solution.reinit(system_rhs);
        LA::MPI::PreconditionAMG prec_A;
        LA::MPI::PreconditionAMG::AdditionalData data;
        data.symmetric_operator = true;
        prec_A.initialize(system_matrix, data);
        SolverCG<LA::MPI::Vector> solver_cg(solver_control);
        solver_cg.solve(system_matrix, solution, system_rhs, prec_A);

        if (std::isinf(solution.l1_norm()) || std::isnan(solution.l1_norm()) || std::isinf(solution.linfty_norm()) ||
            std::isnan(solution.linfty_norm())) {
          if (this_mpi_process == 0)
            std::cout << "Failed. " << std::endl;
          throw std::runtime_error("Solving the BVP failed.");
        } else
          if(this_mpi_process == 0)
            std::cout << "Success! " << std::endl;
      }
    }
#endif

    if (data.verbose)
      pcout << "Solver " << solver_name << ": Solved Linear system: n_dofs=" << solution.size()
                  << " | n_iteration=" << solver_control.last_step()
                  << " | residual=" << solver_control.last_value() << std::endl;

  }

  template<int dim>
  Vector<double> EddBVP_<dim>::ReturnPointSolution(Point<dim> pt) {
    Vector<double> locally_relevant_solution(_solution);
    Functions::FEFieldFunction<dim, DoFHandler<dim>> fe_eval(_dof_handler_base, locally_relevant_solution);
    Vector<double> pt_solution(dim);
    fe_eval.vector_value(pt, pt_solution);
    return pt_solution;
  }

  template<int dim>
  Vector<double> EddBVP_<dim>::GetShapeSolution() {

    Vector<double> shape_solution(_dof_handler_shape.n_dofs());

    std::vector<bool> vertex_touched(_tria_shape.n_vertices(), false);
    std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells;

    auto base_cell = _dof_handler_base.begin_active();
    for (auto shape_cell: _dof_handler_shape) {
      for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v) {
        unsigned int vertex_index = shape_cell.vertex_index(v);

        if (!vertex_touched[vertex_index]) {
          vertex_touched[vertex_index] = true;
          Point<dim> shape_vertex = shape_cell.vertex(v);
          unsigned int base_index = GridTools::find_closest_vertex(_dof_handler_base, shape_vertex);
          adjacent_cells = GridTools::find_cells_adjacent_to_vertex(_dof_handler_base, base_index);

          // establish hda cell that contains shape_vertex
          bool vertex_inside = false;
          unsigned int target_cell = 0;
          for (unsigned int c = 0; c < adjacent_cells.size(); ++c) {
            base_cell = adjacent_cells[c];
            if (!base_cell->is_locally_owned())
              continue;
            if ((base_cell->material_id() == e_boundary_cell || base_cell->material_id() == e_boundary_electrode_cell)  && base_cell->point_inside(shape_vertex)) {
              vertex_inside = true;
              target_cell = c;
            }
          }

          if (!vertex_inside)
            continue;

          base_cell = adjacent_cells[target_cell];
          Point<dim> unit_cell_coords;
          // determine local coordinates for shape_vertex within target hda_cell
          for (unsigned int d = 0; d < dim; ++d) {
            double cell_ext = 2 * std::abs((base_cell->center()[d] - base_cell->vertex(0)[d]));
            unit_cell_coords[d] = (shape_vertex[d] - base_cell->vertex(0)[d]) / cell_ext;
          }

          // interpolate hda values with cell shape functions
          types::global_dof_index n_dof_per_vertex = base_cell->get_fe().n_dofs_per_vertex();
          for (unsigned int d = 0; d < n_dof_per_vertex; ++d) {
            std::vector<unsigned int> local_dof_indices(base_cell->get_fe().dofs_per_cell);
            base_cell->get_dof_indices(local_dof_indices);
            double shape_sol = 0;

            for (unsigned int i = 0; i < std::pow(data.poly_degree + 1, dim); ++i) {
              types::global_dof_index base_dof = local_dof_indices[i * n_dof_per_vertex + d];
              double N_i = base_cell->get_fe().shape_value(i, unit_cell_coords);
              double base_sol = _solution[base_dof];
              shape_sol += N_i * base_sol;
            } // loop over nodes per cell
            types::global_dof_index vertex_dof_index = shape_cell.vertex_dof_index(v, d);
            shape_solution[vertex_dof_index] = shape_sol;
          } // loop over dim
        } // if vertex not touched
      } // end of loop over vertices per shape cell
    }// end of loop over shape

    std::vector<Vector<double>> all_shape_solution = Utilities::MPI::all_gather(mpi_communicator, shape_solution);

    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i) {
      if (i == this_mpi_process)
        break;
      for (unsigned int index = 0; index < all_shape_solution[i].size(); ++index) {
        if (all_shape_solution[i][index] != 0.0) {
          shape_solution[index] = all_shape_solution[i][index];
        }
      }
    }

    return shape_solution;
  }

  template<int dim>
  void EddBVP_<dim>::_OutputSmoothing(const unsigned int cycle) {
    smoothing_output.WriteDataOutput(cycle);
  }


} // end of StructuralOptimization namespace


template
class StructuralOptimization::EddBVP_<2>;

template
class StructuralOptimization::EddBVP_<3>;
