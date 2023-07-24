// C++ headers

// Deal.II headers

// Project headers
#include <ElasticityLin.h>

namespace StructuralOptimization {

  template<int dim>
  ElasticityLin<dim>::ElasticityLin(Parameter &par_, Mesh<dim> &mesh_)
      :
      BVP_<dim>(par_, mesh_),
      _fe(FE_Q<dim>(par_.data.poly_degree), dim),
      _mat(par_.data),
      bvp_output(par_.data, par_.mpi_communicator, par_.compute_timer, "bvp-LE",  *mesh_.domain_parallel){

  }

  template<int dim>
  ElasticityLin<dim>::~ElasticityLin() {
    this->_dof_handler.clear();
    this->_smoothing_dof_handler.clear();
  }

  template<int dim>
  void ElasticityLin<dim>::_SetupSystem() {
    TimerOutput::Scope t(this->compute_timer, "ElasticityLin<dim>::_SetupSystem");

    this->_dof_handler.distribute_dofs(_fe);

    this->_locally_owned_dofs = this->_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(this->_dof_handler, this->_locally_relevant_dofs);

    this->_solution.reinit(this->_locally_owned_dofs,
                           this->_locally_relevant_dofs,
                           this->mpi_communicator);
    this->_system_rhs.reinit(this->_locally_owned_dofs, this->mpi_communicator);

    this->_constraints.clear();
    this->_constraints.reinit(this->_locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(this->_dof_handler, this->_constraints);

    for (unsigned int i = 0; i < this->data.dbc_id.size(); ++i) {
      std::vector<bool> comp_vec(dim);
      comp_vec[this->data.dbc_comp[i]] = true;
      ComponentMask comp_mask(comp_vec);
      VectorTools::interpolate_boundary_values(this->_dof_handler,
                                               this->data.dbc_id[i],
                                               Functions::ConstantFunction<dim>(this->data.dbc_value[i], dim),
                                               this->_constraints,
                                               comp_mask);
    } // loop over all the given dbc id

    this->_constraints.close();

    DynamicSparsityPattern dsp(this->_locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(this->_dof_handler,
                                    dsp,
                                    this->_constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               this->_dof_handler.locally_owned_dofs(),
                                               this->mpi_communicator,
                                               this->_locally_relevant_dofs);
    this->_system_matrix.reinit(this->_locally_owned_dofs,
                                this->_locally_owned_dofs,
                                dsp,
                                this->mpi_communicator);

    this->_SetupSmoothingSystem();
    _SetupQPointHistory();

    if ( this->data.problem_type == "simp" )
      this->SetPseudoDensitiesToOne();

  } // end of _SetupSystem() function

  template<int dim>
  void ElasticityLin<dim>::_SetupQPointHistory() {
    std::vector<QPointHistory> tmp;
    tmp.swap(quadrature_point_history);

    QGauss<dim> solid_quadrature(_fe.degree + 1);
    quadrature_point_history.resize(this->_tria_domain_parallel.n_active_cells() * solid_quadrature.size());

    unsigned int history_index = 0;
    for (auto &cell : this->_tria_domain_parallel.active_cell_iterators()) {
      if (!cell->is_locally_owned()){
        history_index += solid_quadrature.size();
        continue;
      }
      cell->set_user_pointer(&quadrature_point_history[history_index]);
      history_index += solid_quadrature.size();
    }
  }

  template<int dim>
  void ElasticityLin<dim>::_UpdateQPointHistory(const unsigned int mode) {

    QGauss<dim> quadrature_formula(_fe.degree + 1);
    FEValues<dim> fe_values(_fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    std::vector<SymmetricTensor<2, dim>> local_displacement_gradients(quadrature_formula.size());

    const FEValuesExtractors::Vector u_fe(0);

    SymmetricTensor<4, dim> elasticity_tensor;

    if(mode != 3 )
      elasticity_tensor = _mat.GetElasticityTensor();

    for (auto &cell : this->_dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

//      std::cout << __FILE__ << ":" << __LINE__ << " " << mode << " " << elasticity_tensor << std::endl;

      auto *local_quadrature_points_history = reinterpret_cast<QPointHistory *>(cell->user_pointer());

      fe_values.reinit(cell);
      fe_values[u_fe].get_function_symmetric_gradients(this->_solution, local_displacement_gradients);

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q) {
        SymmetricTensor<2, dim> sym_grad_U_ = local_displacement_gradients[q];
        local_quadrature_points_history[q].sym_grad_U = sym_grad_U_;
        double strain_energy_at_q = 0.5 * contract3(sym_grad_U_, elasticity_tensor, sym_grad_U_);
        local_quadrature_points_history[q].strain_energy = strain_energy_at_q;
      }
    }
  }

  template<int dim>
  void ElasticityLin<dim>::_AssembleSystem(const unsigned int mode) {
    TimerOutput::Scope t(this->compute_timer, "ElasticityLin<dim>::_AssembleSystem");

    this->_system_matrix = 0.0;
    this->_system_rhs = 0.0;
    this->cell_simp_density_sens_matrices.clear();
    this->cell_simp_fract_sens_matrices.clear();

    const QGauss<dim> quadrature_formula(_fe.degree + 1);
    FEValues<dim> fe_values(_fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const FEValuesExtractors::Vector u_fe(0);

    const unsigned int dofs_per_cell = _fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_density_sens(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_fraction_sens(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    SymmetricTensor<4, dim> elasticity_tensor, elasticity_tensor_density_sens, elasticity_tensor_fraction_sens;
    if(mode != 3 ) {
      elasticity_tensor = _mat.GetElasticityTensor();
      elasticity_tensor_density_sens = 0.0;
      elasticity_tensor_fraction_sens = 0.0;
    }

    for (const auto &cell : this->_dof_handler.active_cell_iterators()) {
      // if cell belong to some other domain, do nothing
      if (!cell->is_locally_owned())
        continue;

      cell_matrix = 0.0;

      fe_values.reinit(cell);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          SymmetricTensor<2, dim> Grad_N_i = symmetrize(fe_values[u_fe].gradient(i, q));
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            SymmetricTensor<2, dim> Grad_N_j = symmetrize(fe_values[u_fe].gradient(j, q));
            cell_matrix(i, j) += contract3(Grad_N_i, elasticity_tensor, Grad_N_j) * fe_values.JxW(q);
          } // end of j loop
        } // end of i loop
      } // end of loop over quadrature points

      cell->get_dof_indices(local_dof_indices);

      this->_constraints.distribute_local_to_global(cell_matrix,
                                              local_dof_indices,
                                              this->_system_matrix);


    }// end of loop over cells

    this->_system_matrix.compress(VectorOperation::add);

  } // end of _AssembleSystem function

  template<int dim>
  void ElasticityLin<dim>::_AssembleRhs() {
    TimerOutput::Scope t(this->compute_timer,
                         "ElasticityLin<dim>::_AssembleRhs");


    // if no neumann boundary condition, do nothing.
    if (this->data.nbc_id.size() == 0)
      return;

    const QGauss<dim - 1> face_quadrature(_fe.degree * 2 + 1);

    FEFaceValues<dim> fe_face_values(_fe, face_quadrature,
                                     update_values | update_quadrature_points | update_JxW_values);
    unsigned int dofs_per_cell = _fe.dofs_per_cell;
    unsigned int n_q_points = face_quadrature.size();

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    unsigned int neumann_id = 0;
    int neumann_comp = 0;

    // create a list of neumann_id with no repetition of neumann id.
    std::vector<int> unique_neumann_id = this->data.nbc_id;
    sort(unique_neumann_id.begin(), unique_neumann_id.end());
    unique_neumann_id.erase(std::unique(unique_neumann_id.begin(), unique_neumann_id.end()),
                            unique_neumann_id.end());

    // create a map with neumann id vs load area
    std::map<int, double> neumann_id_area;
    for (auto id: unique_neumann_id)
      neumann_id_area[id] = 0.0;

    for (const auto& cell : this->_dof_handler.active_cell_iterators()) {
      // if cell belong to some other domain, do nothing
      if (!cell->is_locally_owned())
        continue;

      if (cell->at_boundary()) {
        for (unsigned int f = 0; f<GeometryInfo<dim>::faces_per_cell; ++f) {
          fe_face_values.reinit(cell, f);
          for (auto id: unique_neumann_id) {
            if (cell->face(f)->boundary_id()==(unsigned int) id) {
              for (unsigned int q_point = 0; q_point<n_q_points; ++q_point) {
                neumann_id_area[id] += fe_face_values.JxW(q_point);
              }
            }
          }
        }
      }
    }

    // create a map with id vs load area
    std::map<int, double> total_neumann_id_area;
    for (auto id: unique_neumann_id) {
      total_neumann_id_area[id] = Utilities::MPI::sum(neumann_id_area[id], this->mpi_communicator);
    }

    //actual run to assemble rhs vector
    for (const auto &cell : this->_dof_handler.active_cell_iterators()) {
      // if cell belong to some other domain, do nothing
      if (!cell->is_locally_owned())
        continue;
      cell_rhs = 0;

      if (cell->at_boundary()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          fe_face_values.reinit(cell, f);
          for (unsigned int id = 0; id < this->data.nbc_id.size(); ++id) {
            neumann_id = this->data.nbc_id[id];
            neumann_comp = this->data.nbc_comp[id];
            if (cell->face(f)->boundary_id() == neumann_id) {
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const int component_i = this->_fe.system_to_component_index(i).first;
                  if (component_i == neumann_comp)
                    cell_rhs(i) +=
                        this->data.nbc_value[id] * fe_face_values.shape_value(i, q_point) *
                        fe_face_values.JxW(q_point); // /total_neumann_id_area[neumann_id];
                }
              }
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);

      // we need to change this in order to allow for inhomogeneous dirichlet bc
      //constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
      this->_constraints.distribute_local_to_global(cell_rhs,
                                              local_dof_indices,
                                              this->_system_rhs);
    }

   this->_system_rhs.compress(VectorOperation::add);

  }

  template<int dim>
  void ElasticityLin<dim>::_SolveSystem() {

    TimerOutput::Scope t(this->compute_timer, "ElasticityLin<dim>::_SolveSystem");

    LA::MPI::Vector completely_distributed_solution(this->_locally_owned_dofs,
                                                    this->mpi_communicator);

    this->_constraints.set_zero(completely_distributed_solution);

    this->_SolveLinearSystem(this->_system_matrix,
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
                                                                    &this->_dof_handler);
    bvp_output.template PushDataName<LA::MPI::Vector>(this->_solution, sol_names,
                                                                    sol_data_component_interpretation,
                                                                    &this->_dof_handler);

  }

  template<int dim>
  void ElasticityLin<dim>::_Postprocess(const unsigned int mode) {

    TimerOutput::Scope t(this->compute_timer, "ElasticityLin<dim>::_Postprocess");

    _UpdateQPointHistory(mode);
    this->AssembleSmoothingMatrix();

    if (dim == 2) {
      for (unsigned int i = e_sym_grad_u_11; i <= e_sym_grad_u_22; ++i) {
        _AssembleSmoothingRhs(i);
        this->SolveSmoothingSystem();
        _UpdatePostprocessingData(i);
      }
    } else if (dim == 3) {
      for (unsigned int i = e_sym_grad_u_11; i <= e_sym_grad_u_33; ++i) {
        _AssembleSmoothingRhs(i);
        this->SolveSmoothingSystem();
        _UpdatePostprocessingData(i);
      }
    }

    for (unsigned int i = e_strain_energy; i <= e_strain_energy; ++i) {
      _AssembleSmoothingRhs(i);
      this->SolveSmoothingSystem();
      _UpdatePostprocessingData(i);
    }
  }

  template<int dim>
  void ElasticityLin<dim>::_AssembleSmoothingRhs(unsigned int assembly_flag) {
    this->_smoothing_rhs = 0;

    QGauss<dim> quadrature_formula(this->_smoothing_fe.degree + 1);
    FEValues<dim> fe_values(this->_smoothing_fe, quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs;
    std::vector<unsigned int> local_dof_indices;
    double q_value;

    for (auto &cell : this->_smoothing_dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_rhs.reinit(cell->get_fe().dofs_per_cell);

      auto *local_quadrature_points_history = reinterpret_cast<QPointHistory *>(cell->user_pointer());

      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;

      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
        q_value = 0;

        if (assembly_flag == e_sym_grad_u_11)
          q_value = local_quadrature_points_history[q].sym_grad_U[0][0];
        else if (assembly_flag == e_sym_grad_u_12)
          q_value = local_quadrature_points_history[q].sym_grad_U[0][1];
        else if (assembly_flag == e_sym_grad_u_22)
          q_value = local_quadrature_points_history[q].sym_grad_U[1][1];
        else if (assembly_flag == e_sym_grad_u_13)
          q_value = local_quadrature_points_history[q].sym_grad_U[0][2];
        else if (assembly_flag == e_sym_grad_u_23)
          q_value = local_quadrature_points_history[q].sym_grad_U[1][2];
        else if (assembly_flag == e_sym_grad_u_33)
          q_value = local_quadrature_points_history[q].sym_grad_U[2][2];
        else if (assembly_flag == e_strain_energy)
          q_value = local_quadrature_points_history[q].strain_energy;

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          cell_rhs(i) += q_value * fe_values.shape_value(i, q) * fe_values.JxW(q);
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
  void ElasticityLin<dim>::_UpdatePostprocessingData(unsigned int assembly_flag) {

    Vector<double> reduced_smoothing_solution(this->_smoothing_solution);

    if (assembly_flag == e_sym_grad_u_11) {
      postprocessing_data.domain_strain11.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_strain11 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "sym_grad_U_11");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_12) {
      postprocessing_data.domain_strain12.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_strain12 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "sym_grad_U_12");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_22) {
      postprocessing_data.domain_strain22.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_strain22 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "sym_grad_U_22");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_13) {
      postprocessing_data.domain_strain13.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_strain13 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "sym_grad_U_13");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_23) {
      postprocessing_data.domain_strain23.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_strain23 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "sym_grad_U_23");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_sym_grad_u_33) {
      postprocessing_data.domain_strain33.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_strain33 = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "sym_grad_U_33");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    } else if (assembly_flag == e_strain_energy) {
      postprocessing_data.domain_strain_energy.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_strain_energy = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "strain_energy");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    }
    else if (assembly_flag == e_von_mises_s) {
      postprocessing_data.domain_vm_stress.reinit(this->_smoothing_dof_handler.n_dofs());
      postprocessing_data.domain_vm_stress = reduced_smoothing_solution;

      std::vector<std::string> vec_names(1, "vm_stress");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vec_data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      bvp_output.template PushDataName<Vector<double>>(reduced_smoothing_solution, vec_names,
                                                       vec_data_component_interpretation,
                                                       &this->_smoothing_dof_handler);

    }
  }

  template<int dim>
  void ElasticityLin<dim>::Run(const unsigned int mode) {

    if(mode != 0)
      bvp_output.SwitchOffOutput();

    _SetupSystem();
    _AssembleSystem(mode);
    _AssembleRhs();
    _SolveSystem();
    _Postprocess(mode);
    _OutputResults(0);

    if (this->data.verbose) {
      this->pcout << "Solved: ElasticityLin<dim> " << std::endl;
      this->pcout << "\n \n \t ElasticityLin max disp : " << this->_solution.linfty_norm() << std::endl;
    }
  } // end of run

  template<int dim>
  void ElasticityLin<dim>::_OutputResults(const unsigned int cycle) {
    TimerOutput::Scope t(this->compute_timer,
                         "ElasticityLin<dim>::_OutputResults");

    bvp_output.WriteDataOutput(cycle);

  }

  template<int dim>
  Vector<double> ElasticityLin<dim>::GetPostprocessingData(unsigned int &data_flag){
    Vector<double> postprocessing_vector;
    if (data_flag == e_sym_grad_u_11) postprocessing_vector = postprocessing_data.domain_strain11;
    else if (data_flag == e_sym_grad_u_12) postprocessing_vector = postprocessing_data.domain_strain12;
    else if (data_flag == e_sym_grad_u_22) postprocessing_vector = postprocessing_data.domain_strain22;
    else if (data_flag == e_sym_grad_u_13) postprocessing_vector = postprocessing_data.domain_strain13;
    else if (data_flag == e_sym_grad_u_23) postprocessing_vector = postprocessing_data.domain_strain23;
    else if (data_flag == e_sym_grad_u_33) postprocessing_vector = postprocessing_data.domain_strain33;
    else if (data_flag == e_cauchy_s_11) postprocessing_vector = postprocessing_data.domain_stress11;
    else if (data_flag == e_cauchy_s_12) postprocessing_vector = postprocessing_data.domain_stress12;
    else if (data_flag == e_cauchy_s_22) postprocessing_vector = postprocessing_data.domain_stress22;
    else if (data_flag == e_cauchy_s_13) postprocessing_vector = postprocessing_data.domain_stress13;
    else if (data_flag == e_cauchy_s_23) postprocessing_vector = postprocessing_data.domain_stress23;
    else if (data_flag == e_cauchy_s_33) postprocessing_vector = postprocessing_data.domain_stress33;
    else if (data_flag == e_strain_energy) postprocessing_vector = postprocessing_data.domain_strain_energy;
    else if (data_flag == e_von_mises_s) postprocessing_vector = postprocessing_data.domain_vm_stress;
    else throw std::runtime_error( "Incorrect postprocessing data request.");
    return postprocessing_vector;

  }

} // end of StructuralOptimization namespace

template
class StructuralOptimization::ElasticityLin<2>;

template
class StructuralOptimization::ElasticityLin<3>;
