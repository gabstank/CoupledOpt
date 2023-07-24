#pragma once

// C++ headers
#include <iostream>
#include <fstream>
#include <assert.h>

// Deal.II headers
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

// Project headers
#include <Utilities.h>

namespace StructuralOptimization {
  using namespace dealii;

// struct_parameters
  struct Data {
    /**
     * @brief Struct with all the run time parametes of the program.
     * 
     */

    // general
    std::string analysis_name; /// Any name to prefix the output files
    std::string destination_path; /// Path to write the output files
    int dim = 2; /// dimension of the problem
    std::string problem_type; /// std | edd | opt; i.e standard or embedded domain or optimization
    std::string bvp_type; /// elasticityLin | elasticityNonLin | electroElasticity etc

    // Output flags
    bool output_mesh = false; /// Do you want mesh output false == no output; true == yes.

    // Verbose output, this will supress all the solver output and narrow output
    bool verbose = false; // false == no output; true == yes.

    /**Base geometry is always a hyper hyper_rectangle.
     * On this geometry a the domain will be tracked/mapped.
     */
    std::vector<double> base_point1, // bottom left corner
    base_point2; // top right corner
    std::vector<int> base_subdivision = {1, 1, 1};
    unsigned int base_refinements = 0;
    bool anisotropic_refinement = false;
    bool extended_refinement = true;
    std::string refinement_criterion;
    unsigned int tracking_refinements;
    unsigned int tracking_precision;
    double narrow = 0.1;

    /**Domain geometry
     * Here we can any geometry from deal.ii or an input from abaqus
     */
    std::string geometry_name; // abaqus | hyper_rectangle | hyper_cube_with_hole
    std::vector<int> boundary_ids; // ID to assign to a bounary
    std::vector<int> boundary_comp; // component to decide the boundary
    std::vector<double> boundary_coord; // coordinates of the boundary
    std::vector<double> boundary_tol; // tolerance within which a point should be in relation to the boundary_coord to be assigned a particular boundary ID.

    // Abaqus
    std::string abaqus_input;
    unsigned int abaqus_refinements = 0;

    // Hyper Rectangle
    std::vector<double> hyper_rectangle_point1, // bottom left corner
    hyper_rectangle_point2; // top right corner
    std::vector<int> hyper_rectangle_subdivision = {1, 1, 1};
    unsigned int hyper_rectangle_refinements = 0;

    // Graded Hyper Rectangle
    std::vector<double> graded_hyper_rectangle_point1, // bottom left corner
    graded_hyper_rectangle_point2; // top right corner
    std::vector<double> steps_x;
    std::vector<double> steps_y;
    std::vector<double> steps_z;
    unsigned int graded_hyper_rectangle_refinements = 0;

    // Hyper Cube with Hole
    double hyper_cube_with_hole_inner_radius;
    double hyper_cube_with_hole_outer_radius;
    double hyper_cube_with_hole_L;              // 3D: Extrusion depth
    unsigned int hyper_cube_with_hole_repetitions;    // 3D: Subdivisions along extrusion direction
    unsigned int hyper_cube_with_hole_refinements;    // global refinements
    bool hyper_cube_with_hole_rotate = false;
    double hyper_cube_with_hole_rotation_angle = 0.0; // will be used in GridTools::rotate
    unsigned int hyper_cube_with_hole_rotation_axis; // will be used in GridTools::rotate

    // Abaqus
    bool restart_from_geo = false;
    std::string geo_input;
    bool refine_after_restart = false;

    // BVP
    int poly_degree = 1;
    // dirichlet_bc
    std::vector<int> dbc_id;
    std::vector<int> dbc_comp;
    std::vector<double> dbc_value;
    // neumann_bc
    std::vector<int> nbc_id;
    std::vector<int> nbc_comp;
    std::vector<double> nbc_value;
    // symmetry
    int symmetry_id = 200;
    // non-linear bvp
    int max_NR_iter = 1;
    double tol_residual_NR = 1e-8;
    int number_load_steps = 1;
    bool output_NR = false;

    // EDD BVP
    int cut_cell_quadrature = 5;
    double weak_material = 1e-3;
    double penalty = 1e9;

    // Material
    std::string material_law;
    struct MatElasticIsotropic {
      double lambda = 115000.0, mu = 77000;
      double density = 7.8e-9; // t/mm3
    } mat_elastic_isotropic;

    // Solver
    std::string solver_name = "CG"; // CG, Direct, Amesos_Mumps
    std::string pp_solver_name = "CG"; // CG, Direct, Amesos_Mumps
    int max_iter = 100; // actual iter = no_of_rows * max_iter
    double solver_tol = 1e-6; // actual_tol = _system_rhs.l2_norm() * solver_tol

    // Optimization

    std::string optimizer_type;
    bool write_refine_geo=false;
    // AlMoM
    unsigned int max_total_iters = 500, max_lagrange_updates = 20, max_al_iterations = 50, max_contractions = 1;
    std::vector<double> almom_lambda;
    double almom_penalty_c = 0.0;
    double initial_shape_step_length = 0.1, initial_density_step_length = 0.1;
    double contraction_step_length = 0.5, armijo_factor_mu = 1e-1;
    double al_convergence_tol = 5e-3, constraint_violation_tol = 1e-5;

    // Node-based
    int n_shape_refinements;
    // Adaptive shape refinement
    bool adaptive_shape_refinement;
    double adaptive_shape_ref_size_threshold;
    double adaptive_shape_ref_cos_threshold;
    double adaptive_shape_ref_min_size_ratio;
    std::vector<int> non_design_id;
    std::vector<int> non_design_comp;
    // optimization_bounding_box
    std::vector<double> bb_x; /// {low, upp}
    std::vector<double> bb_y; /// {low, upp}
    std::vector<double> bb_z; /// {low, upp}
    std::string sensitivity_projection;
    // Responses
    std::vector<std::string> responses; /// compliance | volume ...
    //int objective_id;
    std::vector<int> objectives_id;
    std::vector<double> objectives_weights;
    std::vector<int> constraints_id;
    std::vector<int> constraints_ub_id;
    std::vector<int> constraints_lb_id;
    std::vector<int> constraints_eq_id;
    std::vector<double> constraints_lb;
    std::vector<double> constraints_ub;
    std::vector<double> constraints_eq;
    std::string gradient_normalization;
    // Regularization
    bool sensitivity_weighting;
    std::string sensitivity_weighting_type;
    bool dual_descent_smoothing;
    double sufficient_decrease_coeff;
    // Traction method
    bool traction_method;
    double tm_normal_spring_constant;
    double tm_tangent_spring_constant;
    double tm_normal_smoothing_constant;
    double tm_tangent_smoothing_constant;
    double tm_penalty_constant;

    // Density-based
    double initial_density;
    double initial_filter_radius;
    double initial_penalty;
    double min_density;
    double quasi_void_threshold;
    double quasi_solid_threshold;
    double grey_threshold;
    double hole_area_threshold;
    double open_void_elim_factor;
    bool surface_reconstruction;

  };
// End of struct_parameters

  class Parameter {
  public:
    MPI_Comm mpi_communicator;
    const unsigned int this_mpi_process;
    const unsigned int n_mpi_processes;
    ConditionalOStream pcout;
    mutable TimerOutput compute_timer;

    /**
     * @brief Construct a new Parameter object
     * 
     * @param filename 
     */
    Parameter(std::string filename = "stop.prm");

    /**
     * @brief Destroy the Parameter object
     * 
     */
    ~Parameter();

    /**
     * @brief Function to declare all the parameters that can be specified in the .prm file. 
     * Further the data type of the parameter and also their default values are specified in this function.
     */
    void DeclareParameters();

    /**
     * @brief Function to parse the parameter file and populate the #data object.
     */
    void ParseParameters();

    Data data;

    /**
     * @brief This function writes a parameter file with default parameters. The output file is default.prm.
     * 
     */
    void OutputDefaultParameters();

  private:
    std::string parameter_file;
    ParameterHandler prm;

  };

} // End of StructuralOptimization namespace
