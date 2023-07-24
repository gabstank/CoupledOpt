#pragma once

// C++ headers

#include <Optimizer_.h>

namespace StructuralOptimization {

  /**
   * @brief Optimizer of constrained minimization problem. 
   * Augmented Lagrangian Methods of Multipliers (Rockafellar, R.T.: The multiplier method of Hestenes and Powell applied to convex programming. Journal of Optimization Theory and applications 12(6), 555–562 (1973).). 
   * This optimizer can solve min / max problem with inequality constraints. In this algorithm the lagrange multipliers and the penalty terms are initialized by the user, using which
   * an unconstrained optimization problem is formed. The unconstrained optimization problem is solved using Armijo backtracking line seach method.
   * To use the optimizer specify the convergence tolerances, penalty parametes and step length parametes.
   * 
   * These are the recommended choices.
   * AL subproblem convergence tolerance       = 1e-3 <br>
   * Constraint violation tolerance            = 1e-5 <br>
   * Max number of AL subproblem iterations    = 50 <br>
   * Max number of lagrange multiplier updates = 20 <br>
   * Augmented lagrange penalty                = 1.0 <br>
   * Lagrange multiplier initial value         = 0.1 <br>
   * 
   * Armijo line search mu                     = 1e-4 <br>
   * Step length contraction factor            = 0.5 <br>
   * Max number of contraction                 = 10 <br>
   * 
   */
  template<int dim>
  class AlMoMOptimizer : public Optimizer_<dim> {
  public:

    /**
     * @brief Construct a new AlMoMOptimizer object.
     * 
     * @param par_ 
     * @param mesh_ 
     * @param bvp_ 
     */
    AlMoMOptimizer(Parameter &par_, Mesh<dim> &mesh_, EddBVP_<dim> &bvp_);

    ~AlMoMOptimizer() = default;

    virtual void Run() override;

  private:

    /**
     * @brief Function to initialize the Optimizer parameters, here help variables such as
     * AlMoMOptimizer::opt_iteration, AlMoMOptimizer::_n_constraints are initialized.
     * Logging variables such as AlMoMOptimizer::convergence_history is initialized with header.
     * Optimization penalty variables such as AlMoMOptimizer::lambda , AlMoMOptimizer::_macauly_theta and AlMoMOptimizer::al_penalty_c are initialized from the paramter file.
     * Optimization response variables such as AlMoMOptimizer::_initial_response_values are initialized from the BVP solution.
     * Converge check variable AlMoMOptimizer::previous_shape_design_major is initialized form the shape tria.
     */
    void _InitializeAlParameters();

    /**
     * @brief Function to update the Al Parametes. This is called after each minor interation. In this function,
     * penalty variables AlMoMOptimizer::lambda and AlMoMOptimizer::al_penalty_c gets updated. Also the design change is updated in the variable AlMoMOptimizer::previous_shape_design_major.
     * Finally the setplength parameter AlMoMOptimizer::_current_shape_step_length is updated based the number of contraction in the backtracking.
     */
    void _UpdateAlParameters();

    /**
     * @brief Function to compute the Al response values from the AlMoMOptimizer::_response_handler. Here the response values
     * are normalized and the variabled AlMoMOptimizer::_modified_response_values and AlMoMOptimizer::_macauly_theta are updated. Finally the
     * AlMoMOptimizer::_al_value is computed.
     *  This function is called in AlMoMOptimizer::_BacktrackingArmijoLineSearchShape() and in each minor iteration.
     */
    void _ComputeAlValue();

    /**
     * @brief Function to compute the Al shape sensitivity. Here first the sensitivity data is retrieved form the AlMoMOptimizer::this->_response_handler,
     * then they are normalized wrt to AlMoMOptimizer::_initial_response_values and updated in AlMoMOptimizer::_modified_shape_gradients.
     * Finally the AlMoMOptimizer::_al_shape_sensitivity is computed. This function is called in each minor iteration.
     */
    void _ComputeAlShapeSensitivity();

    /**
     * @brief Function to set the AlMoMOptimizer::_shape_descent_direction. If traction method is used AlMoMOptimizer::_shape_descent_direction = AlMoMOptimizer::_al_shape_sensitivity. Else the
     * AlMoMOptimizer::_shape_descent_direction is computed using DirectShapeSensitivityFiltering in Optimizer_::_regularization.
     */
    void _SmoothingAlShapeSensitivity();

    /**
     * @brief Function to perform Armijo Line seach to determine step length parameter.
     * AlMoMOptimizer::_current_shape_step_length is reduced until the Armijo condition is satisfied or till AlMoMOptimizer::max_contractions is reached.
     * The shape is also updated at the end of the function.
     */
    void _BacktrackingArmijoLineSearchShape();

    /**
     * @brief Function to compute the Al density sensitivity. Here first the sensitivity data is retrieved form the AlMoMOptimizer::this->_response_handler,
     * then they are normalized wrt to AlMoMOptimizer::_initial_response_values and updated in AlMoMOptimizer::_modified_density_gradients.
     * Finally the AlMoMOptimizer::_al_density_sensitivity is computed. This function is called in each minor iteration.
     */
    void _ComputeAlDensitySensitivity();

    /**
     * @brief Function to set the AlMoMOptimizer::_density_descent_direction.
     * It is computed using filtering in Optimizer_::_regularization.
     */
    void _SmoothingAlDensitySensitivity();

    /**
     * @brief Function to perform Armijo Line seach to determine step length parameter for density update.
     * AlMoMOptimizer::_current_density_step_length is reduced until the Armijo condition is satisfied or till AlMoMOptimizer::max_contractions is reached.
     * The density is also updated at the end of the function.
     */
    void _BacktrackingArmijoLineSearchDensity();

    /**
     * @brief Function to check minor(optimization sub problem) convergence. The problem is converged when first order stationary
     * condition (i.e. AlMoMOptimizer::_al_shape_sensitivity.l2_norm() * AlMoMOptimizer::_current_shape_step_length < tol) is satisfied for 2 consecutive iterations
     * This function also does the convergence history logging.
     */
    void _MinorConvergenceCheck();

    /**
     * @brief Function to check the major convergence. Check is performed based on design change.
     */
    void _MajorConvergenceCheck();

    /**
     * @brief Function to add shape sensitivity data to output.
     */
    void _AddShapeDataToOutput();

    /**
     * @brief Function to add density-based sensitivity data to output.
     */
    void _AddDensityDataToOutput();

    /**
     * @brief Function to print convergence to terminal.
     */
    void _PrintToConsole(const std::string&);

    /**
     * @brief Function to write convergence log.
     */
    void _WriteConvergenceHistory();

    MPI_Comm &mpi_communicator;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    //-------------COMMON DATA----------------------------------
    bool coupled_opt = false;
    const double _contraction_factor;

    unsigned int major_convergence = 0; /// major convergence flag

    unsigned int opt_iteration = 0, /// lagrange_update_iter+al_iter. This will be reset for each refinement
    total_iteration = 0,/// this will include also the shape ref iterations.
    shape_refine_level = 0, /// to keep track of shape refinement level.
    major_iteration = 0, /// lagrange update iterations
    minor_iteration = 0, /// augmented lagrange problem iterations
    n_contraction_shape = 0,
    n_contraction_density = 0;

    unsigned int _n_constraints = 0;
    std::map<unsigned int, double> lambda; /// lagrange multipliers for constraints, map<response_id, value>
    double al_penalty_c = 0.0; /// penalty for augmented lagrange function

    std::map<unsigned int, double> _initial_response_values;
    std::map<unsigned int, double> _raw_initial_response_values;
    std::map<unsigned int, double> _final_response_values;
    std::map<unsigned int, double> _current_raw_response_values;

    std::map<unsigned int, double> _modified_response_values;
    std::map<unsigned int, double> _macauly_theta; /// part of al algorithm to handle constraints.
    double _al_value = 0.0;

    /// information to print convergence history
    /// the keys are: {"major_iteration", "minor_iteration", this->_data.responses[i],
    ///                "al_value", "step_length", "n_contractions",
    ///               "lambda_"+std::to_string(i), al_penalty_c,
    ///                "macauly_theta_" + std::to_string(i),
    ///                "kkt_l2_norm" , "kkt_infinity_norm"}
    std::map<std::string, std::vector<double>> convergence_history;

    //--------------------------------SHAPE DATA-----------------------------------------
    // Convergence flags
    unsigned int shape_minor_convergence = 0;

    double optimality_shape_check = 0.0;
    double max_optimality_shape_check = 0.0;
    const double _initial_shape_step_length;
    double _current_shape_step_length;
    int contraction_level_shape;
    int contraction_level_shape_during_convergence;

    /// Shape design update vector
    Vector<double> current_shape_design_update;

    Vector<double> _al_shape_sensitivity;
    //-----------------------------------------------------------------------------------

    //-----------------------------------DENSITY DATA------------------------------------
    // Convergence flags
    unsigned int density_minor_convergence = 0;

    double optimality_density_check = 0.0;
    double max_optimality_density_check = 0.0;

    const double _initial_density_step_length;
    double _current_density_step_length;
    double _density_step_length_multiplier = 0.0;
    int contraction_level_density;
    int contraction_level_density_during_convergence;

    /// Density design update vector
    std::map<CellId, double> current_density_design_update;

    std::map<CellId, double> _al_density_sensitivity; /// this is al density sensitivity
    std::map<CellId, double> _smooth_al_density_sensitivity; /// this is smooth al density sensitivity
    //-----------------------------------------------------------------------------------
  };

}
