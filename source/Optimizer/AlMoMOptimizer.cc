// C++ headers

// Deal.II headers

// Project headers
#include <AlMoMOptimizer.h>

namespace StructuralOptimization {
  using namespace dealii;

  template<int dim>
  AlMoMOptimizer<dim>::AlMoMOptimizer(Parameter &par_, Mesh<dim> &mesh_, EddBVP_<dim> &bvp_)
      : Optimizer_<dim>(par_, mesh_, bvp_),
        mpi_communicator(par_.mpi_communicator),
        this_mpi_process(par_.this_mpi_process),
        pcout(par_.pcout),
        compute_timer(par_.compute_timer),
        coupled_opt(par_.data.problem_type == "couple"),
        _contraction_factor(par_.data.contraction_step_length),
        _initial_shape_step_length(par_.data.initial_shape_step_length),
        _initial_density_step_length(par_.data.initial_density_step_length)
        { }

  template<int dim>
  void AlMoMOptimizer<dim>::_InitializeAlParameters() {
    // Initialize shape parameters
    _current_shape_step_length = _initial_shape_step_length;
    contraction_level_shape = 0;

    if(coupled_opt) {
      // Initialize density parameters
      _current_density_step_length = _initial_density_step_length;
      contraction_level_density = 0;

      // Initial pseudo density values
      for(const auto& cell : this->_mesh.base->active_cell_iterators())
        if(cell->is_locally_owned())
          this->_pseudo_densities[cell->id()] = this->_data.initial_density;
      this->_bvp.SetPseudoDensities(this->_pseudo_densities);
    }
    this->_bvp.Run();
    this->_response_handler->RunAdjointBVP();

    // Pseudo densities adjusted according to whether they are inner, boundary or outside cells
    if(coupled_opt)
      this->_pseudo_densities = this->_bvp.GetPseudoDensities();
    opt_iteration = 0;

    convergence_history.clear();

    _n_constraints = this->_response_handler->GetNResponses() - this->_data.objectives_id.size();

    if (_n_constraints != this->_data.constraints_id.size())
      throw std::runtime_error("Number of constraints specified in constraints_id is less than number of constraints");
    if (_n_constraints != this->_data.almom_lambda.size())
      throw std::runtime_error("Number of augmented lagrange values is less than number of constraints");

    // Initialize the map lambda with key = response_id and value = lambda
    for (unsigned int idx = 0; idx < _n_constraints; ++idx) {
      lambda[this->_data.constraints_id[idx]] = this->_data.almom_lambda[idx];
      _macauly_theta[this->_data.constraints_id[idx]] = 0.0;
    }

    al_penalty_c = this->_data.almom_penalty_c;

    _raw_initial_response_values.clear();
    _initial_response_values.clear();
    _raw_initial_response_values = this->_response_handler->GetValues();
    _initial_response_values = _raw_initial_response_values;

    pcout << "Initial response value : " ;
    for(const auto &iter : _initial_response_values)
      pcout << iter.second << "    " ;
    pcout << std::endl;

    // modify constraints.
    // the constraints are in the form g(x) = g_0(x)
    // bring it to the form g(x) - g_0(x) = 0
    for (unsigned int i = 0; i < this->_data.constraints_eq_id.size(); ++i)
      _initial_response_values[this->_data.constraints_eq_id[i]] *= this->_data.constraints_eq[i];

    // modify constraints.
    // the constraints are in the form g(x) <= g_0(x) * constraint_ub.
    // Now we just got the value of g_0(x). So we multiply with constraint_ub
    for (unsigned int i = 0; i < this->_data.constraints_ub_id.size(); ++i)
      _initial_response_values[this->_data.constraints_ub_id[i]] *= this->_data.constraints_ub[i];

    // modify constraints.
    // the constraints are in the form g(x) >= g_0(x) * constraint_lb.
    // Now we just got the value of g_0(x). So we multiply with constraint_ub
    for (unsigned int i = 0; i < this->_data.constraints_lb_id.size(); ++i)
      _initial_response_values[this->_data.constraints_lb_id[i]] *= this->_data.constraints_lb[i];
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_UpdateAlParameters() {

    for (const auto &id : this->_data.constraints_id)
      lambda[id] = _macauly_theta[id];

    al_penalty_c *= 2.0;

    // Now we are updating the step length.
    // If we set the step length to initial step length we have to do a lot of contractions and this involves solving a lot of BVPs
    // We have decided to have a update scheme where the level of contraction is half of the contraction from the previous major.
    n_contraction_shape = 0;

    contraction_level_shape = std::ceil(contraction_level_shape / 2);
    contraction_level_shape_during_convergence = contraction_level_shape;
    _current_shape_step_length = _initial_shape_step_length * std::pow(_contraction_factor, contraction_level_shape);

    if(coupled_opt) {
      n_contraction_density = 0;

      if(contraction_level_density > 0)
        contraction_level_density = std::ceil(contraction_level_density / 2);
      contraction_level_density_during_convergence = contraction_level_density;
      _current_density_step_length = _initial_density_step_length * std::pow(_contraction_factor, contraction_level_density);
    }
  }


  template<int dim>
  void AlMoMOptimizer<dim>::_ComputeAlValue() {

    PROJ_MPI_BARRIER

    // clear data from previous iterations
    _current_raw_response_values.clear();
    _modified_response_values.clear();

    // get response values
    _current_raw_response_values = this->_response_handler->GetValues();

    _modified_response_values = _current_raw_response_values;

    // normalizing wrt to initial response values
    for (auto &[id, mod_value] : _modified_response_values)
      mod_value /= _initial_response_values[id];

    for (auto const &id: this->_data.constraints_eq_id)
      _modified_response_values[id] -= 1.0;

    for (auto const &id: this->_data.constraints_ub_id)
      _modified_response_values[id] -= 1.0;

    for (auto const &id: this->_data.constraints_lb_id)
      _modified_response_values[id] = 1.0 - _modified_response_values[id];

    // _macauly_theta is the evaluation of Macaulay bracket operation for inequality constraint
    for (auto &iter : _macauly_theta)
      iter.second = 0.0;

    for (auto const &id: this->_data.constraints_eq_id) {
       double macauly_operation = lambda[id] + al_penalty_c * _modified_response_values[id];
      _macauly_theta[id] = macauly_operation;
    }

    for (auto const &id: this->_data.constraints_ub_id) {
      double macauly_operation = lambda[id] + al_penalty_c * _modified_response_values[id];
      if (macauly_operation > 0)
        _macauly_theta[id] = macauly_operation;
    }

    for (auto const &id: this->_data.constraints_lb_id) {
      double macauly_operation = lambda[id] + al_penalty_c * _modified_response_values[id];
      if (macauly_operation > 0)
        _macauly_theta[id] = macauly_operation;
    }

    _al_value = 0.0;
    // computing the augmented lagrange function value
    //_al_value = _modified_response_values[this->_data.objective_id];
    unsigned int count = 0;
    for (auto const &id: this->_data.objectives_id) {
      _al_value += _modified_response_values[id] * this->_data.objectives_weights[count];
      ++count;
    }

    for (auto const &id: this->_data.constraints_eq_id) {
      _al_value += lambda[id] * _modified_response_values[id]
                   + al_penalty_c * 0.5 * _modified_response_values[id] * _modified_response_values[id];
    }

    for (auto const &id: this->_data.constraints_ub_id) {
      _al_value += (1.0 / (2.0 * al_penalty_c)) * (_macauly_theta[id] * _macauly_theta[id]
                                                   - lambda[id] * lambda[id]);
    }

    for (auto const &id: this->_data.constraints_lb_id) {
      _al_value += (1.0 / (2.0 * al_penalty_c)) * (_macauly_theta[id] * _macauly_theta[id]
                                                   - lambda[id] * lambda[id]);
    }

  }

  template<int dim>
  void AlMoMOptimizer<dim>::_ComputeAlShapeSensitivity() {
    TimerOutput::Scope t(this->compute_timer,
                         "AlMoMOptimizer<dim>::_ComputeShapeSensitivity");
    // adjoint problems are required only for computation of sensitivity.
    this->_response_handler->RunAdjointBVP();
    PROJ_MPI_BARRIER

    // clear gradient data
    this->_current_raw_shape_gradients.clear();
    this->_modified_shape_gradients.clear();

    // get gradient data
    this->_current_raw_shape_gradients = this->_response_handler->GetRawShapeGradients();

    // normalizing wrt to initial response values
    this->_modified_shape_gradients = this->_current_raw_shape_gradients;

    for(auto &[id, shape_gradient] : this->_modified_shape_gradients)
      shape_gradient /= _initial_response_values[id];

    for (const auto &id : this->_data.constraints_lb_id)
      this->_modified_shape_gradients[id] *= -1;

    _al_shape_sensitivity.reinit(this->_bvp._dof_handler_shape.n_dofs());
    // computing the augmented lagrange sensitivity

    unsigned int count = 0;
    for(auto &[id, shape_gradient] : this->_modified_shape_gradients) {

      if (std::find(this->_data.objectives_id.begin(), this->_data.objectives_id.end(), (int) id) != this->_data.objectives_id.end()) {
        Vector<double> weighted_shape_gradient(shape_gradient);
        weighted_shape_gradient *= this->_data.objectives_weights[count];
        _al_shape_sensitivity += weighted_shape_gradient;
        ++count;
        continue;
      }
      // even equality constraints works here, because has the equality constraints accounted for.
      Vector<double> constraint_contribution;
      constraint_contribution = shape_gradient; // now in this loop only constraints are considered
      constraint_contribution *= _macauly_theta[id];

      _al_shape_sensitivity += constraint_contribution;
    }
    this->_bvp.shape_constraints.distribute(_al_shape_sensitivity);

    PROJ_MPI_BARRIER

  }

  template<int dim>
  void AlMoMOptimizer<dim>::_ComputeAlDensitySensitivity() {
    TimerOutput::Scope t(this->compute_timer,
                         "AlMoMOptimizer<dim>::_ComputeDensitySensitivity");

    // clear gradient data
    this->_current_raw_density_gradients.clear();
    this->_modified_density_gradients.clear();

    // get gradient data
    this->_current_raw_density_gradients = this->_response_handler->GetRawDensityGradients();

    // normalizing wrt to initial response values
    this->_modified_density_gradients = this->_current_raw_density_gradients;

    for(auto &[id, density_gradient] : this->_modified_density_gradients)
      for(auto &[cellid, grad_val] : density_gradient)
        grad_val /= _initial_response_values[id];

    for (const auto &id : this->_data.constraints_lb_id)
      for(auto &[cellid, grad_val] : this->_modified_density_gradients[id])
        grad_val *= -1;

    _al_density_sensitivity.clear();
    // computing the augmented lagrange sensitivity

    unsigned int count = 0;
    for(auto &[id, density_gradient] : this->_modified_density_gradients) {

      if (std::find(this->_data.objectives_id.begin(), this->_data.objectives_id.end(), (int) id) != this->_data.objectives_id.end()) {
        for (auto &[cellid, grad_val] : density_gradient)
          _al_density_sensitivity[cellid] += grad_val *  this->_data.objectives_weights[count];
        ++count;
        continue;
      }
      // even equality constraints works here, because has the equality constraints accounted for.
      std::map<CellId, double> constraint_contribution;
      constraint_contribution = density_gradient; // now in this loop only constraints are considered
      for (auto &[cellid, grad_val] : constraint_contribution) {
        grad_val *= _macauly_theta[id];
        _al_density_sensitivity[cellid] += grad_val;
      }

    }
    PROJ_MPI_BARRIER
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_SmoothingAlShapeSensitivity() {
    TimerOutput::Scope t(this->compute_timer,
                         "AlMoMOptimizer<dim>::_SmoothingAlShapeSensitivity");

    // Direct Filtering - apply if traction method is not requested
    if (!this->_data.traction_method) {
      this->_regularization.DirectShapeSensitivityFiltering(this->_bvp._dof_handler_shape, this->_bvp.shape_constraints,
                                            _al_shape_sensitivity, this->_response_handler->GetVertexNormals());

      PROJ_MPI_BARRIER
    }

    // Sensitivity indicates ascent direction, so we reverse it
    // Normalization and scaling of shape design update
    this->_shape_descent_direction = _al_shape_sensitivity;
    this->_shape_descent_direction *= -1.0;

  }

  template<int dim>
  void AlMoMOptimizer<dim>::_SmoothingAlDensitySensitivity() {
    TimerOutput::Scope t(this->compute_timer,
                         "AlMoMOptimizer<dim>::_SmoothingAlDensitySensitivity");

    _smooth_al_density_sensitivity.clear();

    // Direct Filtering
    std::vector<std::map<CellId, double>> tmp_vec_sens = Utilities::MPI::all_gather(mpi_communicator, _al_density_sensitivity);
    std::map<CellId, double> _smooth_al_density_sensitivity_all_proc;
    for(const auto& sens_map : tmp_vec_sens)
      for (const auto&[cellid, val] : sens_map)
        _smooth_al_density_sensitivity_all_proc[cellid] = val;

    std::vector<std::map<CellId, double>> tmp_vec_dens = Utilities::MPI::all_gather(mpi_communicator, this->_pseudo_densities);
    std::map<CellId, double> pseudo_densities_all_proc;
    for(const auto& dens_map : tmp_vec_dens)
      for (const auto&[cellid, val] : dens_map)
        pseudo_densities_all_proc[cellid] = val;

    this->_regularization.DensitySensitivityFiltering(this->_bvp._dof_handler_base,
                                                      _smooth_al_density_sensitivity_all_proc,
                                                      _smooth_al_density_sensitivity);

    PROJ_MPI_BARRIER
    this->_density_descent_direction.clear();
    this->_density_descent_direction = _smooth_al_density_sensitivity;

    // Sensitivity indicates an ascend direction, so we invert it.
    for (auto &[cellid, grad_val] : this->_density_descent_direction)
      grad_val *= -1.0;
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_MinorConvergenceCheck() {

    if (optimality_shape_check < this->_data.al_convergence_tol * max_optimality_shape_check)
      shape_minor_convergence = 1;
    else
      shape_minor_convergence = 0;

    // Density convergence check
    if (coupled_opt) {
      if (optimality_density_check < this->_data.al_convergence_tol * max_optimality_density_check)
        density_minor_convergence = 1;
      else
        density_minor_convergence = 0;
    }

    if (minor_iteration == 0) {
      if (major_iteration == 0)
        _PrintToConsole("major_header");
      _PrintToConsole("minor_header");
    }
    _PrintToConsole("minor_data");

    if (shape_minor_convergence == 0 || (coupled_opt && density_minor_convergence == 0)) {
      for (auto &[id, mod_value] : _modified_response_values) {

        std::string response_name;

        if (std::count(this->_data.constraints_lb_id.begin(), this->_data.constraints_lb_id.end(), id)) {
          response_name = this->_data.responses[id] + "_LB";
        }
        else if (std::count(this->_data.constraints_ub_id.begin(), this->_data.constraints_ub_id.end(), id)) {
          response_name = this->_data.responses[id] + "_UB";
        }
        else if (std::count(this->_data.constraints_eq_id.begin(), this->_data.constraints_eq_id.end(), id)) {
          response_name = this->_data.responses[id] + "_EQ";
        }
        else
          response_name = this->_data.responses[id] + "_OBJ";

        convergence_history[response_name].push_back(mod_value);

        response_name += "_raw";

        convergence_history[response_name].push_back(_current_raw_response_values[id]);
      }

      for (auto &[id, val] : lambda)
        convergence_history["lambda_" + std::to_string(id)].push_back(val);

      for (auto &[id, val] : _macauly_theta)
        convergence_history["macauly_theta_" + std::to_string(id)].push_back(val);

      convergence_history["iteration"].push_back(opt_iteration);
      convergence_history["al_value"].push_back(_al_value);
      convergence_history["al_penalty_c"].push_back(al_penalty_c);
      convergence_history["optimality_shape_check"].push_back(optimality_shape_check / max_optimality_shape_check);
      convergence_history["shape_step_length"].push_back(_current_shape_step_length);
      if(coupled_opt) {
        convergence_history["optimality_density_check"].push_back(optimality_density_check / max_optimality_density_check);
        convergence_history["density_step_length"].push_back(_current_density_step_length);
        convergence_history["n_grey_cells"].push_back((100.0*this->n_grey_cells) / this->n_design_cells);
      }
    }

  }

  template<int dim>
  void AlMoMOptimizer<dim>::_MajorConvergenceCheck() {

    _PrintToConsole("minor_footer");

    if (major_iteration != 0) {
      ++major_convergence;

      for (auto const &id: this->_data.constraints_eq_id) {
        if (std::abs(_modified_response_values[id]) > this->_data.constraint_violation_tol)
          major_convergence = 0;
      }

      for (auto const &id: this->_data.constraints_ub_id) {
        if (_modified_response_values[id] > this->_data.constraint_violation_tol)
          major_convergence = 0;
      }

      for (auto const &id: this->_data.constraints_lb_id) {
        if (_modified_response_values[id] > this->_data.constraint_violation_tol)
          major_convergence = 0;
      }
    }
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_BacktrackingArmijoLineSearchShape() {

    double armijo_factor_mu = this->_data.armijo_factor_mu;
    unsigned int max_contractions = this->_data.max_contractions;
    n_contraction_shape = 0;

    Vector<double> before_shape;
    Vector<double> descent_dir_theta;
    double old_al_value = _al_value;
    double armijo_al_value;

    bool stop_w_decrease = false, stop_wo_decrease = false;

    this->_shape_descent_direction /= this->_shape_descent_direction.l2_norm();

    // Reference values to restore the previous state
    std::map<CellId, double> ref_pseudo_densities = this->_pseudo_densities;
    std::map<CellId, double> ref_density_descent_direction = this->_density_descent_direction;

    while (true) {
      // Raw design update is a descent direction scaled with the step size
      current_shape_design_update.reinit(this->_shape_descent_direction.size());
      current_shape_design_update = this->_shape_descent_direction;
      current_shape_design_update *= _current_shape_step_length; // / _shape_step_length_multiplier;

      // Regularize the design update
      if (this->_data.traction_method)
        this->_regularization.RunTractionMethod(this->_bvp._dof_handler_shape, this->_bvp.shape_constraints,
                                                current_shape_design_update,
                                                this->_response_handler->GetVertexNormals());

      // Update the shape and track it
      this->_mesh.MoveShape(this->_bvp._dof_handler_shape, current_shape_design_update, before_shape);
      PROJ_MPI_BARRIER
      this->_mesh.ResetBase(this->_bvp._dof_handler_base);
      this->_mesh.mesh_tracking->RunTracking();

      // Track the pseudo densities and sensitivities
      if (coupled_opt) {
        this->_TrackDensities();
        this->_bvp.SetPseudoDensities(this->_pseudo_densities);
      }
      // Run BVP
      this->_bvp.Run();
      this->_response_handler->RunAdjointBVP();
      if (coupled_opt)
        this->_pseudo_densities = this->_bvp.GetPseudoDensities();

      // Compute AL value
      _ComputeAlValue();

      optimality_shape_check = std::abs(_al_shape_sensitivity * current_shape_design_update);

      // Armijo value
      armijo_al_value = armijo_factor_mu * (_al_shape_sensitivity * current_shape_design_update);

      // Backtracking stopping criteria
      if (_al_value <= old_al_value + armijo_al_value) { // desired stopping criterion - sufficient decrease
        if(contraction_level_shape > 0)
          --contraction_level_shape;
        _current_shape_step_length = _initial_shape_step_length * std::pow(_contraction_factor, contraction_level_shape);
        stop_w_decrease = true;
      }

      // Safeguard stopping criteria - no decrease
      else if (n_contraction_shape >= max_contractions // maximum number of contractions within backtracking (safeguard)
               || contraction_level_shape >= 15 // step size limit from bottom (safeguard)
               || optimality_shape_check < this->_data.al_convergence_tol * max_optimality_shape_check
        // Stop when the sufficient decrease condition is not met but the AL subproblem converges
          )
        stop_wo_decrease = true;

      // Reset the state if there is no decrease
      if(!stop_w_decrease) {
        _al_value = old_al_value;
        this->_mesh.UpdateShape(this->_bvp._dof_handler_shape, before_shape);
        if (coupled_opt) {
          this->_pseudo_densities.clear();
          this->_pseudo_densities = ref_pseudo_densities;
          this->_density_descent_direction.clear();
          this->_density_descent_direction = ref_density_descent_direction;
        }
      }

      // Rerun the problem for the reset state to obtain the previous solution
      if(stop_wo_decrease) {
        this->_mesh.ResetBase(this->_bvp._dof_handler_base);
        this->_mesh.mesh_tracking->RunTracking();
        // Track the pseudo densities and sensitivities
        if (coupled_opt) {
          this->_TrackDensities();
          this->_bvp.SetPseudoDensities(this->_pseudo_densities);
        }
        this->_bvp.Run();
        this->_response_handler->RunAdjointBVP();
        if (coupled_opt)
          this->_pseudo_densities = this->_bvp.GetPseudoDensities();
        // Compute AL value
        _ComputeAlValue();
      }

      // If any of the stopping criteria is met, break the loop
      if(stop_w_decrease || stop_wo_decrease) {
        contraction_level_shape_during_convergence = contraction_level_shape;
        // Update max optimality value if necessary
        if (max_optimality_shape_check < optimality_shape_check)
          max_optimality_shape_check = optimality_shape_check;
        // Output shape
        _AddShapeDataToOutput();
        this->_WriteShapeOutput(total_iteration);
        // Output base here if it is only shape optimization
        if(!this->coupled_opt)
          this->_WriteBaseOutput(total_iteration);

        break;
      }

      // Contract and continue line search
      ++contraction_level_shape;
      ++n_contraction_shape;
      _current_shape_step_length = _initial_shape_step_length * std::pow(_contraction_factor, contraction_level_shape);

    }// end of while loop
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_BacktrackingArmijoLineSearchDensity() {

    // L2 normalize the descent direction
    this->_MapL2Normalize(this->_density_descent_direction);

    double armijo_factor_mu = this->_data.armijo_factor_mu;
    unsigned int max_contractions = this->_data.max_contractions;
    n_contraction_density = 0;

    double old_al_value = _al_value;
    double armijo_al_value;

    bool stop_w_decrease = false, stop_wo_decrease = false;

    // Reference pseudo densities, used to restore previous state
    std::map<CellId, double> ref_pseudo_densities = this->_pseudo_densities;

    // Test update - estimate max change.
    // The step length multiplier is needed to ensure that the requested initial step size is maintained
    if (_density_step_length_multiplier == 0.0) {
      double test_val = 0.0;
      for (auto &[cellid, val]: this->_pseudo_densities) {

        // We only want to update inside cells
        if (this->_density_descent_direction.find(cellid) == this->_density_descent_direction.end())
          continue;

        test_val = val;
        test_val += this->_density_descent_direction[cellid];
        if (test_val > 1.0) test_val = 1.0;
        if (test_val < this->_data.min_density) test_val = this->_data.min_density;
        if (std::abs(val - test_val) > _density_step_length_multiplier)
          _density_step_length_multiplier = std::abs(val - test_val);
      }
      _density_step_length_multiplier = Utilities::MPI::max(_density_step_length_multiplier, mpi_communicator);
    }

    while (true) {
      // The density descent direction is already regularized. Now we apply the step size.
      current_density_design_update.clear();
      current_density_design_update = this->_density_descent_direction;
      for (auto &[cellid, val] : current_density_design_update)
        val *= _current_density_step_length / _density_step_length_multiplier;

      // Design update vector ensures the bounding box constraints are fulfilled
      for (auto &[cellid, val]: this->_pseudo_densities) {

        // We only want to update inside cells
        if (this->_density_descent_direction.find(cellid) == this->_density_descent_direction.end())
          continue;

        double old_val = val;
        val += current_density_design_update[cellid];

        if (val > 1.0) val = 1.0;
        if (val < this->_data.min_density) val = this->_data.min_density;
        current_density_design_update[cellid] = val - old_val;

      }

      // Design update of the topology
      this->_bvp.SetPseudoDensities(this->_pseudo_densities);
      this->_bvp.Run();
      this->_response_handler->RunAdjointBVP();
      this->_pseudo_densities = this->_bvp.GetPseudoDensities();
      _ComputeAlValue();

      // Optimality check
      double density_check_per_proc = 0;
      for (const auto &[cellid, val] : current_density_design_update)
        density_check_per_proc += val * _al_density_sensitivity[cellid];
      optimality_density_check = Utilities::MPI::sum(density_check_per_proc, mpi_communicator);

      // Sufficient decrease factor
      armijo_al_value = armijo_factor_mu * optimality_density_check;
      optimality_density_check = std::abs(optimality_density_check);

      // Backtracking stopping criteria
      if (_al_value <= old_al_value + armijo_al_value) { // desired stopping criterion - sufficient decrease
        if(contraction_level_density > -10)
          --contraction_level_density;
        _current_density_step_length =
            _initial_density_step_length * std::pow(_contraction_factor, contraction_level_density);
        stop_w_decrease = true;
      }

      // Safeguard stopping criteria - no decrease
      else if (n_contraction_density >= max_contractions // maximum number of contractions within backtracking (safeguard)
      || contraction_level_density >= 15 // step size limit from bottom (safeguard)
      || optimality_density_check < this->_data.al_convergence_tol * max_optimality_density_check
      // Stop when the sufficient decrease condition is not met but the AL subproblem converges
      )
        stop_wo_decrease = true;

      // Reset the state if there is no decrease
      if(!stop_w_decrease) {
        this->_pseudo_densities = ref_pseudo_densities;
        _al_value = old_al_value;
      }

      // Rerun the problem for the reset state to obtain the previous solution
      if(stop_wo_decrease) {
        this->_bvp.SetPseudoDensities(this->_pseudo_densities);
        this->_bvp.Run();
        this->_response_handler->RunAdjointBVP();
        this->_pseudo_densities = this->_bvp.GetPseudoDensities();
        _ComputeAlValue();
      }

      // If any stopping criteria met, break the line search
      if(stop_w_decrease || stop_wo_decrease) {
        contraction_level_density_during_convergence = contraction_level_density;
        // Update max optimality value if necessary
        if (max_optimality_density_check < optimality_density_check)
          max_optimality_density_check = optimality_density_check;
        // Write base output
        _AddDensityDataToOutput();
        this->_WriteBaseOutput(total_iteration);
        break;
      }

      // Contract and continue line search
      ++contraction_level_density;
      ++n_contraction_density;
      _current_density_step_length = _initial_density_step_length * std::pow(_contraction_factor, contraction_level_density);
    }
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_AddShapeDataToOutput() {
    for (auto &[id, raw_shape_grad] : this->_current_raw_shape_gradients) {

      std::string response_name;
      if (std::count(this->_data.constraints_lb_id.begin(), this->_data.constraints_lb_id.end(), id)) {
        response_name = this->_data.responses[id] + "_LB";
      }
      else if (std::count(this->_data.constraints_ub_id.begin(), this->_data.constraints_ub_id.end(), id)) {
        response_name = this->_data.responses[id] + "_UB";
      }
      else if (std::count(this->_data.constraints_eq_id.begin(), this->_data.constraints_eq_id.end(), id)) {
        response_name = this->_data.responses[id] + "_EQ";
      }
      else
        response_name = this->_data.responses[id] + "_OBJ";

      std::vector<std::string> grad_names(dim, response_name + "_grad_raw");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          grad_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
      this->_shape_output->template PushDataName<Vector<double>>(raw_shape_grad, grad_names,
                                                                 grad_component_interpretation,
                                                                 &this->_bvp._dof_handler_shape);
    }

    for (auto &[id, mod_shape_grad] : this->_modified_shape_gradients) {

      std::string response_name;
      if (std::count(this->_data.constraints_lb_id.begin(), this->_data.constraints_lb_id.end(), id)) {
        response_name = this->_data.responses[id] + "_LB";
      }
      else if (std::count(this->_data.constraints_ub_id.begin(), this->_data.constraints_ub_id.end(), id)) {
        response_name = this->_data.responses[id] + "_UB";
      }
      else if (std::count(this->_data.constraints_eq_id.begin(), this->_data.constraints_eq_id.end(), id)) {
        response_name = this->_data.responses[id] + "_EQ";
      }
      else
        response_name = this->_data.responses[id] + "_OBJ";

      std::vector<std::string> grad_names(dim, response_name + "_grad_mod");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          grad_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
      this->_shape_output->template PushDataName<Vector<double>>(mod_shape_grad, grad_names,
                                                                 grad_component_interpretation,
                                                                 &this->_bvp._dof_handler_shape);
    }

    unsigned int i = 0;
    Vector<double> boundary_ids(this->_mesh.shape.n_active_cells());
    for (auto & cell : this->_mesh.shape.active_cell_iterators()) {
      boundary_ids[i] = cell->material_id();
      ++i;
    }

    std::vector<std::string> boundary_names(1, "boundary_id");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        boundary_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_shape_output->template PushDataName<Vector<double>>(boundary_ids, boundary_names,
                                                               boundary_component_interpretation,
                                                               &this->_bvp._dof_handler_shape);


    std::vector<std::string> al_sens_names(dim, "al_sensitivity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        al_sens_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    this->_shape_output->template PushDataName<Vector<double>>(_al_shape_sensitivity, al_sens_names,
                                                               al_sens_component_interpretation,
                                                               &this->_bvp._dof_handler_shape);

    if (!this->_data.traction_method) {
      std::vector<std::string> descent_dir_names(dim, "al_sensitivity_smooth");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          descent_dir_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
      this->_shape_output->template PushDataName<Vector<double>>(this->_shape_descent_direction, descent_dir_names,
                                                                 descent_dir_component_interpretation,
                                                                 &this->_bvp._dof_handler_shape);
    }

    std::vector<std::string> descent_dir_names(dim, "design_update");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        descent_dir_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    this->_shape_output->template PushDataName<Vector<double>>(current_shape_design_update, descent_dir_names,
                                                               descent_dir_component_interpretation,
                                                               &this->_bvp._dof_handler_shape);

  }

  template<int dim>
  void AlMoMOptimizer<dim>::_AddDensityDataToOutput() {

    Vector<double> design_update = this->GenerateVectorFromMap(current_density_design_update);
    Vector<double> al_sensitivity = this->GenerateVectorFromMap(_al_density_sensitivity);
    Vector<double> smooth_al_sensitivity = this->GenerateVectorFromMap(_smooth_al_density_sensitivity);
    Vector<double> pseudo_densities = this->GenerateVectorFromMap(this->_pseudo_densities);
    Vector<double> penalized_densities = pseudo_densities;
    for (auto & val : penalized_densities)
      val = std::pow(val, this->_bvp.penalty);
    Vector<double> descent_direction =this->GenerateVectorFromMap(this->_density_descent_direction);


    /*Vector<double> design_update(this->_mesh.base->n_active_cells()),
    al_sensitivity(this->_mesh.base->n_active_cells()),
    smooth_al_sensitivity(this->_mesh.base->n_active_cells()),
    pseudo_densities(this->_mesh.base->n_active_cells()),
    penalized_densities(this->_mesh.base->n_active_cells()),
    descent_direction(this->_mesh.base->n_active_cells());
    int i = 0;
    for (auto cell : this->_bvp._dof_handler_base.active_cell_iterators()) {
      if(!cell->is_locally_owned()) {
        ++i;
        continue;
      }
      design_update[i] = current_density_design_update[cell->id()];
      al_sensitivity[i] = _al_density_sensitivity[cell->id()];
      smooth_al_sensitivity[i] = _smooth_al_density_sensitivity[cell->id()];
      pseudo_densities[i] = this->_pseudo_densities[cell->id()];
      penalized_densities[i] = std::pow(this->_pseudo_densities[cell->id()], this->_bvp.penalty);
      descent_direction[i] = this->_density_descent_direction[cell->id()];
      ++i;
    }*/

    std::map<unsigned int, Vector<double>> raw_density_gradients, mod_density_gradients;
    for (auto &[id, raw_grad] : this->_current_raw_density_gradients) {

      raw_density_gradients[id] = this->GenerateVectorFromMap(raw_grad);
      mod_density_gradients[id] = this->GenerateVectorFromMap(this->_modified_density_gradients[id]);
    }

    // Reduction of the sensitivities
    std::vector<std::map<CellId, double>> tmp_vec_sens = Utilities::MPI::all_gather(mpi_communicator, this->_density_descent_direction);
    std::map<CellId, double> _density_descent_direction_reduced;
    for(const auto& sens_map : tmp_vec_sens)
      for (const auto&[cellid, val] : sens_map)
        _density_descent_direction_reduced[cellid] = val;

    Vector<double> descent_direction_reduced = this->GenerateVectorFromMap(_density_descent_direction_reduced);

    // Reduction of the densities
    std::vector<std::map<CellId, double>> tmp_vec_dens = Utilities::MPI::all_gather(mpi_communicator, this->_pseudo_densities);
    std::map<CellId, double> _pseudo_densities_reduced;
    for(const auto& dens_map : tmp_vec_dens)
      for (const auto&[cellid, val] : dens_map)
        _pseudo_densities_reduced[cellid] = val;

    Vector<double> pseudo_densities_reduced = this->GenerateVectorFromMap(_pseudo_densities_reduced);

    for (auto &[id, raw_density_grad] : raw_density_gradients) {

      std::string response_name;
      if (std::count(this->_data.constraints_lb_id.begin(), this->_data.constraints_lb_id.end(), id)) {
        response_name = this->_data.responses[id] + "_LB";
      }
      else if (std::count(this->_data.constraints_ub_id.begin(), this->_data.constraints_ub_id.end(), id)) {
        response_name = this->_data.responses[id] + "_UB";
      }
      else if (std::count(this->_data.constraints_eq_id.begin(), this->_data.constraints_eq_id.end(), id)) {
        response_name = this->_data.responses[id] + "_EQ";
      }
      else
        response_name = this->_data.responses[id] + "_OBJ";

      std::vector<std::string> grad_names(1, response_name + "_grad_raw");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          grad_component_interpretation( DataComponentInterpretation::component_is_scalar);
      this->_base_output->template PushDataName<Vector<double>>(raw_density_grad, grad_names,
                                                                 grad_component_interpretation,
                                                                 &this->_bvp._dof_handler_base);
    }

    for (auto &[id, mod_density_grad] : mod_density_gradients) {

      std::string response_name;
      if (std::count(this->_data.constraints_lb_id.begin(), this->_data.constraints_lb_id.end(), id)) {
        response_name = this->_data.responses[id] + "_LB";
      }
      else if (std::count(this->_data.constraints_ub_id.begin(), this->_data.constraints_ub_id.end(), id)) {
        response_name = this->_data.responses[id] + "_UB";
      }
      else if (std::count(this->_data.constraints_eq_id.begin(), this->_data.constraints_eq_id.end(), id)) {
        response_name = this->_data.responses[id] + "_EQ";
      }
      else
        response_name = this->_data.responses[id] + "_OBJ";

      std::vector<std::string> grad_names(1, response_name + "_grad_mod");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          grad_component_interpretation( DataComponentInterpretation::component_is_scalar);
      this->_base_output->template PushDataName<Vector<double>>(mod_density_grad, grad_names,
                                                                 grad_component_interpretation,
                                                                 &this->_bvp._dof_handler_base);
    }

    std::vector<std::string> al_sens_names(1, "al_sensitivity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        al_sens_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(al_sensitivity, al_sens_names,
                                                               al_sens_component_interpretation,
                                                               &this->_bvp._dof_handler_base);

    std::vector<std::string> smooth_al_sens_names(1, "smooth_al_sensitivity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        smooth_al_sens_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(smooth_al_sensitivity, smooth_al_sens_names,
                                                              smooth_al_sens_component_interpretation,
                                                              &this->_bvp._dof_handler_base);

    std::vector<std::string> design_update_names(1, "design_update");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        esign_update_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(design_update, design_update_names,
                                                              esign_update_component_interpretation,
                                                               &this->_bvp._dof_handler_base);

    std::vector<std::string> densities_dir_names(1, "pseudo_densities");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        densities_dir_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(pseudo_densities, densities_dir_names,
                                                              densities_dir_component_interpretation,
                                                              &this->_bvp._dof_handler_base);

    std::vector<std::string> densities_reduced_dir_names(1, "pseudo_densities_reduced");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        densities_reduced_dir_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(pseudo_densities_reduced, densities_reduced_dir_names,
                                                              densities_reduced_dir_component_interpretation,
                                                              &this->_bvp._dof_handler_base);

    std::vector<std::string> pen_densities_dir_names(1, "penalized_densities");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        pen_densities_dir_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(penalized_densities, pen_densities_dir_names,
                                                              pen_densities_dir_component_interpretation,
                                                              &this->_bvp._dof_handler_base);

    std::vector<std::string> descent_dir_names(1, "descent_direction");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        descent_dir_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(descent_direction, descent_dir_names,
                                                              descent_dir_component_interpretation,
                                                              &this->_bvp._dof_handler_base);

    std::vector<std::string> descent_dir_red_names(1, "descent_direction_reduced");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        descent_dir_red_component_interpretation(DataComponentInterpretation::component_is_scalar);
    this->_base_output->template PushDataName<Vector<double>>(descent_direction_reduced, descent_dir_red_names,
                                                              descent_dir_red_component_interpretation,
                                                              &this->_bvp._dof_handler_base);
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_PrintToConsole(const std::string& type) {
    PROJ_MPI_BARRIER
    if (this_mpi_process == 0) {
      // Float numbers data format
      std::cout << std::setprecision(3) << std::scientific << std::setfill(' ');

      int short_width = 7;
      int long_width = 11;
      int n_long_entries = 3 + 2 * this->_data.responses.size();
      int n_short_entries = 4;

      if(coupled_opt) {
        n_long_entries += 3;
        n_short_entries += 1;
      }
      int width = 2 * (n_short_entries + n_long_entries) + short_width * n_short_entries
                  + long_width * n_long_entries - 1;

      if (type == "major_header") {
        std::cout << "  ";
        for (int i = 0; i < width; ++i)
          std::cout << "_";
        std::cout << std::endl;

        int major_short_width = short_width * n_short_entries + 2 * (n_short_entries - 1);
        int major_long_width = width - major_short_width - 2 - 1;

        std::cout << " |" << center("Refinement level = " + std::to_string(shape_refine_level), major_short_width);

        std::cout << " |" << center("Augmented Lagrangian - Method of Multipliers", major_long_width);

        std::cout << " |" << std::endl;
      }

      // Print iteration header
      if (type == "minor_header") {

        std::cout << " |";
        for (int i = 0; i < width; ++i)
          std::cout << "_";
        std::cout << "|" << std::endl;

        std::cout << std::setfill(' ')
                  << " |" << center("TOTAL", short_width)
                  << " |" << center("MAJOR", short_width)
                  << " |" << center("MINOR", short_width)
                  << " |" << center("CONTR_S", short_width);

        if(coupled_opt)
          std::cout << std::setfill(' ')
                  << " |" << center("CONTR_D", short_width);

        for (unsigned int i = 0; i < this->_data.objectives_id.size(); ++i)
          std::cout << std::setfill(' ')
                    << " |" << center("OBJ_" + std::to_string(i), long_width)
                    << " |" << center("OBJ_" + std::to_string(i) + "_RAW", long_width);

        for (unsigned int i = 0; i < this->_data.constraints_id.size(); ++i)
          std::cout << std::setfill(' ')
                    << " |" << center("CON_" + std::to_string(i), long_width)
                    << " |" << center("CON_" + std::to_string(i) + "_RAW", long_width);

        std::cout << std::setfill(' ')
                  << " |" << center("AL_VAL", long_width)
                  << " |" << center("OPT_CON_S", long_width);
        if(coupled_opt)
          std::cout << std::setfill(' ')
                  << " |" << center("OPT_CON_D", long_width);

        std::cout << std::setfill(' ')
                  << " |" << center("SH_STEP", long_width);
        if(coupled_opt)
          std::cout << " |" << center("D_STEP", long_width);
        if(coupled_opt)
          std::cout << " |" << center("N_GREY", long_width);

        std::cout << " |" << std::endl;

        std::cout << " |";
        for (int i = 0; i < width; ++i)
          std::cout << "_";
        std::cout << "|" << std::endl;
      }

      if (type == "minor_data") {

        std::stringstream major_it_stream;
        if (major_convergence != 0)
          major_it_stream << "*";
        major_it_stream << major_iteration;
        std::cout << " |" << std::setw(short_width) << total_iteration;

        if (minor_iteration == 0)
          std::cout << " |" << std::setw(short_width) << major_it_stream.str();
        else
          std::cout << " |" << std::setw(short_width) << " ";

        std::stringstream it_stream, als_kkt_stream_s;
        std::stringstream als_kkt_stream_d;
        std::map<unsigned int, std::stringstream> constr_stream, constr_stream_raw;

        if (shape_minor_convergence != 0)
          als_kkt_stream_s << "*";
        if(coupled_opt && density_minor_convergence != 0)
          als_kkt_stream_d << "*";
        if (shape_minor_convergence != 0 && !(coupled_opt && density_minor_convergence == 0)) {
          it_stream << "*";
        }
        it_stream << minor_iteration;
        als_kkt_stream_s << std::setprecision(3) << std::scientific << optimality_shape_check / max_optimality_shape_check;
        if(coupled_opt)
          als_kkt_stream_d << std::setprecision(3) << std::scientific << optimality_density_check / max_optimality_density_check;

        for (unsigned int i = 1; i < this->_data.responses.size(); ++i) {
          if (std::count(this->_data.constraints_eq_id.begin(), this->_data.constraints_eq_id.end(), i)) {
            if (_modified_response_values[i] > -this->_data.constraint_violation_tol &&
                _modified_response_values[i] < this->_data.constraint_violation_tol) {
              constr_stream[i] << "*";
              constr_stream_raw[i] << "*";
            }
          }
          else {
            if (_modified_response_values[i] < this->_data.constraint_violation_tol) {
              constr_stream[i] << "*";
              constr_stream_raw[i] << "*";
            }
          }
          constr_stream[i] << std::setprecision(3) << std::scientific << _modified_response_values[i];
          constr_stream_raw[i] << std::setprecision(3) << std::scientific << _current_raw_response_values[i];
        }

        std::cout << " |" << std::setw(short_width) << it_stream.str()
                  << " |" << std::setw(short_width) << n_contraction_shape;
        if(coupled_opt)
          std::cout << " |" << std::setw(short_width) << n_contraction_density;

        std::cout << " |" << std::setw(long_width) << _modified_response_values[0];
        std::cout << " |" << std::setw(long_width) << _current_raw_response_values[0];
        for (unsigned int i = 1; i < this->_data.responses.size(); ++i) {
          std::cout << " |" << std::setw(long_width) << constr_stream[i].str();
          std::cout << " |" << std::setw(long_width) << constr_stream_raw[i].str();
        }

        std::cout << " |" << std::setw(long_width) << _al_value
                  << " |" << std::setw(long_width) << als_kkt_stream_s.str();
        if(coupled_opt)
          std::cout << " |" << std::setw(long_width) << als_kkt_stream_d.str();

        std::cout << " |" << std::setw(long_width) << _initial_shape_step_length * std::pow(_contraction_factor, contraction_level_shape_during_convergence);
        if(coupled_opt)
          std::cout << " |" << std::setw(long_width) << _initial_density_step_length * std::pow(_contraction_factor, contraction_level_density_during_convergence);

        if(coupled_opt)
          std::cout << " |" << std::setw(long_width-1) << (100.0*this->n_grey_cells) / this->n_design_cells << "%";

        std::cout << " |" << std::endl;
      }

      if (type == "minor_footer") {

        std::cout << " |";
        for (int i = 0; i < width; ++i)
          std::cout << "_";
        std::cout << "|" << std::endl;

        int major_short_width = short_width * n_short_entries + 2 * (n_short_entries - 1);
        int major_long_width = width - major_short_width - 2 - 1;

        if (minor_iteration < this->_data.max_al_iterations)
          std::cout << " |" << center("AL subproblem converged", major_short_width);
        else
          std::cout << " |" << center("AL iter limit reached", major_short_width);

        std::stringstream des_change_stream;
        des_change_stream << "Shape optimality criterion = " << std::setprecision(3) << std::scientific
                          << optimality_shape_check / max_optimality_shape_check;
        if(coupled_opt) {
          des_change_stream << " | ";
          des_change_stream << "Density optimality criterion = " << std::setprecision(3) << std::scientific
                            << optimality_density_check / max_optimality_density_check;
        }

        std::string des_change_string = des_change_stream.str();

        std::cout << " |" << center(des_change_string, major_long_width);

        std::cout << " |" << std::endl;
      }

      if (type == "major_footer") {

        std::cout << " |";
        for (int i = 0; i < width; ++i)
          std::cout << "_";
        std::cout << "|" << std::endl;

        if (major_iteration < this->_data.max_lagrange_updates)
          std::cout << " |" << center("Constrained optimization problem converged", width - 1);
        else
          std::cout << " |" << center("Lagrange multiplier updates limit reached", width - 1);

        std::cout << " |" << std::endl;

        std::cout << " |";
        for (int i = 0; i < width; ++i)
          std::cout << "_";
        std::cout << "|" << std::endl;
      }

      if(type == "final_response"){
        pcout << "Final response value : " ;
        for(const auto &iter : _final_response_values)
          pcout << iter.second << "    " ;
        pcout << std::endl;

        pcout << "Ratio final / initial response values : " ;
        for(unsigned int it=0; it< _final_response_values.size(); ++it)
          pcout << _final_response_values[it] / _raw_initial_response_values[it] << "    " ;
        pcout << std::endl;
      }

    }
    PROJ_MPI_BARRIER
    std::cout << std::defaultfloat;
  }

  template<int dim>
  void AlMoMOptimizer<dim>::_WriteConvergenceHistory() {
    // sanity check, check if all the values in map have same size.
    unsigned int size = convergence_history["iteration"].size();
    for (const auto &entry: convergence_history) {
      if (entry.second.size() != size) {
        std::cerr << "Size error: " << std::endl;
        for (const auto &iter: convergence_history) {
          std::cerr << iter.first << " : " << std::to_string(iter.second.size()) << std::endl;
        }
        throw std::runtime_error("Error in collecting logging details: look the above size output on the screen");
      }
    }

    PROJ_MPI_BARRIER
    if (this_mpi_process == 0) {
      std::string logging_filename = this->_data.destination_path + this->_data.analysis_name
                                     + "-P" + std::to_string(this_mpi_process)
                                     + "-refine-l" + std::to_string(shape_refine_level)
                                     + "-convergence.csv";
      std::ofstream convergence_log(logging_filename, std::ios_base::out);
      // first write the names
      for (const auto &entry: convergence_history)
        convergence_log << entry.first << " , ";
      convergence_log << std::endl;
      // write the entries.
      for (unsigned int i = 0; i < size; ++i) {
        for (const auto &entry: convergence_history)
          convergence_log << entry.second[i] << " , ";
        convergence_log << std::endl;
      }
      convergence_log.close();
    }
    PROJ_MPI_BARRIER
  }

  template<int dim>
  void AlMoMOptimizer<dim>::Run() {

    unsigned int n_shape_refinements = this->_data.n_shape_refinements,
        n_lagrange_updates = this->_data.max_lagrange_updates,
        n_al_iterations = this->_data.max_al_iterations;
    _InitializeAlParameters();
    for (shape_refine_level = 0; shape_refine_level <= n_shape_refinements; ++shape_refine_level) {

      // Rerun BVP after shape refinement
      if(shape_refine_level != 0) {
        if(coupled_opt)
          this->_bvp.SetPseudoDensities(this->_pseudo_densities);
        this->_bvp.Run();
        this->_response_handler->RunAdjointBVP();
        if(coupled_opt)
          this->_pseudo_densities = this->_bvp.GetPseudoDensities();
      }

      // Reset convergence flags
      shape_minor_convergence = 0, density_minor_convergence = 0, major_convergence = 0;

      // Loop over augmented lagrange subproblems
      for (major_iteration = 0, opt_iteration = 0; major_iteration < n_lagrange_updates
      && total_iteration < this->_data.max_total_iters; ++major_iteration) {
        // Loop over iterations of an augmented lagrange subproblem
        for (minor_iteration = 0; minor_iteration < n_al_iterations
        && total_iteration < this->_data.max_total_iters; ++minor_iteration) {

          // Compute AL value
          _ComputeAlValue();

          // Compute raw sensitivities
          _ComputeAlShapeSensitivity();

          // Smoothing of the sensitivities by direct filtering. It is applied when the traction method is off.
          _SmoothingAlShapeSensitivity();

          if(coupled_opt) {
            _ComputeAlDensitySensitivity();
            _SmoothingAlDensitySensitivity();
          }

          // Determine the design update vector
          _BacktrackingArmijoLineSearchShape();

          if(coupled_opt) {
            _BacktrackingArmijoLineSearchDensity();
            this->_ComputeDensityStatistics();
            //if((100.0*this->n_grey_cells)/this->n_design_cells < 1.0)
              //coupled_opt = false;
          }

          // Check for minor convergence
          _MinorConvergenceCheck();
          // Stop the augmented lagrange subproblem if the minor convergence is reached
          if (shape_minor_convergence == 1 && !(coupled_opt && density_minor_convergence == 0)) {
            shape_minor_convergence = 0, density_minor_convergence = 0;
            ++opt_iteration;
            ++total_iteration;
            break;
          }

          // Open topological void elimination
          // Only for coupled opt; if returns true and the design converged, then reduce the contraction level
          if (coupled_opt) {
            if (this->ProjectShapeToTopologyFeature() && shape_minor_convergence > 0)
              --contraction_level_shape;

            // Shape hole generation and merging of overlapped holes
            if (dim == 2 && this->n_grey_cells < (this->_data.grey_threshold * this->n_design_cells)) {
              this->GenerateShapeHoles();
              this->MergeOverlappingHoles();
            }
          }

          // Adaptive shape refinement, not supported in 3D due to hanging nodes
          if(dim == 2 && this->_data.adaptive_shape_refinement)
            // Curvature refinement takes place not straight from the beginning.
            // The reason is that we do not want to refine shape cells at corners that are going to be smoothed out
            if(this->_mesh.AdaptiveShapeRefinement(this->n_grey_cells < 0.5*this->n_design_cells))
              // The function returns true if at least one shape element was refined
              // Then we need to redefine the constraints and boundary conditions
              this->_SetupShape();

          ++opt_iteration;
          ++total_iteration;
        } // End of augmented lagrange subproblem

        // Check for major convergence
        _MajorConvergenceCheck();

        // Stop the constrained optimization problem if the major convergence is reached
        if (major_convergence == 1) {
          major_convergence = 0;
          break;
        }

        std::cout << "17" << std::endl;
        // Update lagrange mulpipliers etc.
        _UpdateAlParameters();
        std::cout << "18" << std::endl;
      }// end of lagrange_update_iter

      _PrintToConsole("major_footer");
      // refine shape
      _WriteConvergenceHistory();

      if (shape_refine_level != n_shape_refinements) { // refinement for last iteration
        this->_ExecuteRefinement();
      }
    }// end of shape shape_refinement_iter

    // write the final response values.
    _final_response_values = this->_response_handler->GetValues();
    _PrintToConsole("final_response");

    // write a geo file, check if a refined geo file is required.
    this->_mesh.WriteGeoFile(this->_bvp._dof_handler_shape);

    // write a geo file, check if a refined geo file is required.
    if(this->_data.write_refine_geo ){
        this->_ExecuteRefinement();
        this->_bvp.Run();
    }
    this->_mesh.WriteGeoFile(this->_bvp._dof_handler_shape);

    if(dim ==2 && this_mpi_process == 0)
      WriteAbaqusScript(this->_bvp._dof_handler_shape, this->_data.destination_path, this->_data.analysis_name);

  }

}// End of StructuralOptimization namespace


template
class StructuralOptimization::AlMoMOptimizer<2>;

template
class StructuralOptimization::AlMoMOptimizer<3>;