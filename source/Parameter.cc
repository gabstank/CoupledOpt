// C++ headers

// Deal.II headers

// Project headers
#include <Parameter.h>

namespace StructuralOptimization {
  Parameter::Parameter(std::string parameter_file_)
      :
      mpi_communicator(MPI_COMM_WORLD),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      pcout(std::cout, this_mpi_process == 0),
      compute_timer(pcout, TimerOutput::summary, TimerOutput::wall_times),
      parameter_file(parameter_file_) {
  }

// destructor
  Parameter::~Parameter() {
  }

// DeclareParameters
  void Parameter::DeclareParameters() {

// general
    prm.declare_entry("Analysis name", "stop", Patterns::Anything(),
                      "Name of the analysis to prefix the output files.");
    prm.declare_entry("Destination path", "./results/", Patterns::DirectoryName(),
                      "Destination path to write the output files.");
    prm.declare_entry("dim", "2", Patterns::Integer(2, 3),
                      "Number of space dimensions.");
    prm.declare_entry("Problem Type", "std", Patterns::Anything(),
                      "std | edd | shape | simp | couple");
    prm.declare_entry("BVP Type", "elasticityLin", Patterns::Anything(),
                      "elasticityLin | elasticityNonLin | electroElasticityLin | normalModes |"
                      " bimporphPiezoStatic | bimporphPiezoNormalModes");
    prm.declare_entry("Mesh Output", "false", Patterns::Bool(),
                      "Do you want to output mesh?");
    prm.declare_entry("Verbose Output", "false", Patterns::Bool(),
                      "Do you want to string output?");

    // Base Geometry
    prm.enter_subsection("Base");
    {
      prm.declare_entry("Base Point1", "-1,-1,-1",
                        Patterns::List(Patterns::Double()),
                        "Point 1: x,y,(z) (hyper_rectangle)");
      prm.declare_entry("Base Point2", "1,1,1",
                        Patterns::List(Patterns::Double()),
                        "Point 2: x,y,(z) (hyper_rectangle)");
      prm.declare_entry("Base Subdivisions", "1,1,1",
                        Patterns::List(Patterns::Integer()),
                        "Subdivision in : x,y,(z) (hyper_rectangle)");
      prm.declare_entry("Base refinements", "1", Patterns::Integer(0, 10),
                        "Number of global refinements");
      prm.declare_entry("Anisotropic refinement", "false", Patterns::Bool(),
                        "Used for energy harvesters - no refinement in the thickness direction");
      prm.declare_entry("Extended refinement", "true", Patterns::Bool(),
                        "Refine boundary neighbours? true | false");
      prm.declare_entry("Refinement criterion", "fixed", Patterns::Anything(),
                        "fixed | narrow");
      prm.declare_entry("Desired narrow", "0.1", Patterns::Double(1e-3, 1),
                        "Narrow criterion: Desired narrow for mesh tracking.");
      prm.declare_entry("Tracking refinements", "3", Patterns::Integer(0, 10),
                        "Fixed criterion: Number of tracking refinements");
      prm.declare_entry("Tracking precision", "2", Patterns::Integer(1, 4),
                        "Levels: [1,4]; 1 - use just vertices, 2:4 - additional test points");

    }
    prm.leave_subsection();

    // Domain Geometry
    prm.enter_subsection("Domain");
    {
      prm.declare_entry("Geometry name", "hyper_rectangle", Patterns::Anything(),
                        "abaqus | hyper_rectangle | graded_hyper_rectangle | hyper_cube_with_hole");

      prm.enter_subsection("Abaqus");
      {
        prm.declare_entry("Input file", "./abq.inp", Patterns::Anything(),
                          "path to abaqus input file");
        prm.declare_entry("Abaqus refinements", "0", Patterns::Integer(0, 10),
                          "Number of global refinements after importing the mesh");
      }
      prm.leave_subsection();

      prm.enter_subsection("Hyper Rectangle");
      {
        prm.declare_entry("Hyper Rectangle Point1", "-1,-1,-1",
                          Patterns::List(Patterns::Double()),
                          "Point 1: x,y,(z) (hyper_rectangle)");
        prm.declare_entry("Hyper Rectangle Point2", "1,1,1",
                          Patterns::List(Patterns::Double()),
                          "Point 2: x,y,(z) (hyper_rectangle)");

        prm.declare_entry("Hyper Rectangle Subdivisions", "1,1,1",
                          Patterns::List(Patterns::Integer()),
                          "Subdivision in : x,y,(z) (hyper_rectangle)");

        prm.declare_entry("Hyper Rectangle refinements", "1", Patterns::Integer(0, 10),
                          "Number of global refinements");
      }
      prm.leave_subsection(); // leaving the Hyper Rectangle subsection.

      prm.enter_subsection("Graded Hyper Rectangle");
      {
        prm.declare_entry("Graded Hyper Rectangle Point1", "-1,-1,-1",
                          Patterns::List(Patterns::Double()),
                          "Point 1: x,y,(z) (graded_hyper_rectangle)");
        prm.declare_entry("Graded Hyper Rectangle Point2", "1,1,1",
                          Patterns::List(Patterns::Double()),
                          "Point 2: x,y,(z) (graded_hyper_rectangle)");

        prm.declare_entry("Steps x", "0.5,0.5",
                          Patterns::List(Patterns::Double()),
                          "Steps in x (graded_hyper_rectangle)");
        prm.declare_entry("Steps y", "0.1,0.8,0.1",
                          Patterns::List(Patterns::Double()),
                          "Steps in y (graded_hyper_rectangle)");
        prm.declare_entry("Steps z", "0.5,0.5",
                          Patterns::List(Patterns::Double()),
                          "Steps in z (graded_hyper_rectangle)");

        prm.declare_entry("Graded Hyper Rectangle refinements", "1", Patterns::Integer(0, 10),
                          "Number of global refinements");
      }
      prm.leave_subsection(); // leaving the Graded Hyper Rectangle subsection.

      prm.enter_subsection("Hyper Cube With Hole");
      {
        prm.declare_entry("Hyper Cube With Hole Inner Radius", "0.25",
                          Patterns::Double(), "Inner radius of the cylindrical hole");
        prm.declare_entry("Hyper Cube With Hole Outer Radius", "0.5",
                          Patterns::Double(), "Outer dimension of the cube");
        prm.declare_entry("Hyper Cube With Hole L", "0.5",
                          Patterns::Double(), "3D: Extrusion depth");
        prm.declare_entry("Hyper Cube With Hole Repetitions", "1",
                          Patterns::Integer(), "3D: Subdivisions along extrusion direction");
        prm.declare_entry("Hyper Cube With Hole Rotate", "false", Patterns::Bool(),
                          "Do you want to rotate the geometry.");
        prm.declare_entry("Hyper Cube With Hole Rotation angle", "0.0", Patterns::Double(),
                          "Rotation angle in radians");
        prm.declare_entry("Hyper Cube With Hole Refinements", "1", Patterns::Integer(0, 10),
                          "Number of global refinements");
        prm.declare_entry("Hyper Cube With Hole Rotation axis", "0", Patterns::Integer(0, 2),
                          "Rotation axis.");
      }
      prm.leave_subsection(); // leaving the Hyper Cube With Hole subsection.

      prm.enter_subsection("Restart");
      {
        prm.declare_entry("Geo Input file", "./restart.geo", Patterns::Anything(),
                          "path to geo input file");
        prm.declare_entry("Restart from Geo", "false", Patterns::Bool(),
                          "Do you want to restart from geo?");
        prm.declare_entry("Refine after restart", "false", Patterns::Bool(),
            "Do you want to apply one refinement step to the shape?");
      }
      prm.leave_subsection(); // leaving the Restart section

      prm.enter_subsection("Boundary");
      {
        prm.declare_entry("Boundary IDs", "0,0", Patterns::List(Patterns::Integer(0, 50)),
                          "List of boundary IDs to assign to a boundary, if two boundaries satisfy the condition only the first shall survive");
        prm.declare_entry("Boundary comp", "0,1", Patterns::List(Patterns::Integer(0, 2)),
                          "List of boundary components to be considered while deciding the boundary ID");
        prm.declare_entry("Boundary coord", "0,0", Patterns::List(Patterns::Double()),
                          "List of coordinates of the boundary, each value correspond to the component prescribed above.");
        prm.declare_entry("Boundary tol", "1e-4,1e-4", Patterns::List(Patterns::Double()),
                          "List of tolerance for each value to satisfy");
      }
      prm.leave_subsection(); // leaving boundary subsection.

    }
    prm.leave_subsection(); // leaving the Domain subsection

    prm.enter_subsection("EDD BVP");
    {
      prm.declare_entry("Cut cell quadrature", "5", Patterns::Integer(5, 10),
                        "Number of quadrature points to consider for boundary cells");
      prm.declare_entry("Weak material contribution", "1e-4", Patterns::Double(),
                        "Penalty for boundary cells gauss points outside domain");
      prm.declare_entry("Dirchlet Penalty", "1e9", Patterns::Double(),
                        "Penalty parameter for weak enforcement of essential boundary conditions");
    }
    prm.leave_subsection();

    prm.enter_subsection("BVP");
    {
      prm.declare_entry("Polynomial degree", "1", Patterns::Integer(1, 3),
                        "Basis function polynomial degree");

      // dirichlet_bc
      prm.enter_subsection("Dirichlet BC");
      {
        prm.declare_entry("Dirichlet ID", "0", Patterns::List(Patterns::Integer(0, 50)),
                          "List of Dirichlet IDs");

        // in range of (0,3) considering potential
        prm.declare_entry("Dirichlet comp", "0", Patterns::List(Patterns::Integer(0, 3)),
                          "List of Dirichlet comp");

        prm.declare_entry("Dirichlet value", "0", Patterns::List(Patterns::Double()),
                          "List of Dirichlet value");
      }
      prm.leave_subsection(); // leaving Dirichlet BC subsection.

      // neumann_bc
      prm.enter_subsection("Neumann BC");
      {
        prm.declare_entry("Neumann ID", "1", Patterns::List(Patterns::Integer(0, 50)),
                          "List of Neumann IDs");

        prm.declare_entry("Neumann comp", "0", Patterns::List(Patterns::Integer(0, 3)),
                          "List of Dirichlet comp");

        prm.declare_entry("Neumann value", "0", Patterns::List(Patterns::Double()),
                          "List of Neumann value");
      }
      prm.leave_subsection(); // leaving Neumann BC subsection.

      // symmetry
      prm.enter_subsection("Symmetry");
      {
        prm.declare_entry("Symmetry ID", "200", Patterns::Integer(),
            "Boudary ID responsible for symmetry constraint");
      }
      prm.leave_subsection();

      // Newton Raphson
      prm.enter_subsection("Newton Raphson");
      {
        prm.declare_entry("Max Newton Iteration", "1", Patterns::Integer(1, 999),
                          "Max Newton Raphson Iteraions.");
        prm.declare_entry("Tolerance Newton Raphson", "1e-6", Patterns::Double(),
                          "Tolerance for error in Newton Raphson method.");
        prm.declare_entry("Number of Load Steps", "1", Patterns::Integer(1, 999),
                          "Number of Load Steps.");
        prm.declare_entry("Output Newton Raphson", "false", Patterns::Bool(),
                          "Do you want to output each Newton Raphson iteration?");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection(); // leaving BVP subsection.

    // Material
    prm.enter_subsection("Material");
    {
      prm.declare_entry("Material Law", "StVenant",
                        Patterns::Anything(), "Name of Material law: StVenant, NeoHooke, orthotropic");
      // Elastic isotropic
      prm.enter_subsection("Elastic isotropic");
      {
        prm.declare_entry("lambda", "115",
                          Patterns::Double(), "Lamé's first parameter");
        prm.declare_entry("mu", "77",
                          Patterns::Double(), "Lamé's second parameter, shear modulus sometimes denoted as G");
        prm.declare_entry("density", "7.8e-6", Patterns::Double(), "Density");
      }
      prm.leave_subsection();

    }
    prm.leave_subsection(); // leaving material section

    // Solver
    prm.enter_subsection("Solver");
    {
      prm.declare_entry("Solver name", "CG",
                        Patterns::Anything(), "Name of solver: CG, Direct, Amesos_Mumps, Amesos_Umfpack");
      prm.declare_entry("Postprocessor Solver name", "Amesos_Umfpack",
                        Patterns::Anything(), "Name of direct solver(use only direct sovler): Direct, Amesos_Mumps, Amesos_Umfpack");
      prm.declare_entry("Max iterations", "100",
                        Patterns::Integer(1, 9999),
                        "Max number of iteration, actual max iter = no_of_rows * max_iter");
      prm.declare_entry("Solver Tolerance", "1e-9",
                        Patterns::Double(1e-15, 1e15),
                        "Tolerance for iterative solver: actual_tol = _system_rhs.l2_norm() * solver_tol");
    }
    prm.leave_subsection(); // Leaving solver section.

    // Optimization
    prm.enter_subsection("Optimization");
    {
      prm.declare_entry("Optimizer type", "AlMoM", Patterns::Anything(),
                        "AlMoM");
      prm.declare_entry("Write refined geo", "false", Patterns::Bool(),
                        "Write refined geo file? default = false");

      prm.enter_subsection("AlMoM");
      {
        prm.declare_entry("Max total iterations", "500", Patterns::Integer(1, 2000),
                          "Maximum allowed number of total iterations [1, 2000]");
        prm.declare_entry("Max number of lagrange multiplier updates", "20", Patterns::Integer(1, 100),
                          "Number of times lagrange multiplier should be updated [1, 50]");
        prm.declare_entry("Max number of AL subproblem iterations", "50", Patterns::Integer(1, 500),
                          "Maximum number of augmented lagrange subproblem iterations");
        prm.declare_entry("Lagrange multiplier initial value", "0.0", Patterns::List(Patterns::Double(0, 1e20)),
                          "Initial value of Lagrange multiplier corresponding to inequality constraint");
        prm.declare_entry("Augmented lagrange penalty", "0.0", Patterns::Double(0, 1e20),
                          "Initial value of penalty in augmented lagrange function.");
        prm.declare_entry("Initial shape step length", "0.1", Patterns::Double(1e-10, 10.0),
                          "Initial shape step length for line search.");
        prm.declare_entry("Initial density step length", "0.1", Patterns::Double(1e-10, 0.5),
                          "Initial density step length for line search.");
        prm.declare_entry("Step length contraction factor", "0.5", Patterns::Double(0.1, 1),
                          "Contraction factor for step length in backtracking algorithm.");
        prm.declare_entry("Armijo line search mu", "1e-4", Patterns::Double(0, 1),
                          "Armijo step length factor in backtracking algorithm.");
        prm.declare_entry("Max number of contraction", "1", Patterns::Integer(1, 20),
                          "Max number of backtracking contractions");
        prm.declare_entry("AL subproblem convergence tolerance", "5e-3", Patterns::Double(1e-6, 1e-1),
                          "Convergence tolerance of an augmented lagrange subproblem (stopping criterion)");
        prm.declare_entry("Constraint violation tolerance", "1e-5", Patterns::Double(1e-6, 1e-2),
                          "Constraint violation tolerance (stopping criterion)");
      }
      prm.leave_subsection();

      prm.enter_subsection("Density-based optimization");
      {
        prm.declare_entry("Initial density", "1.0", Patterns::Double(0.001,1.0),
            "Initial value of design variables");
        prm.declare_entry("Initial filter radius", "1.0", Patterns::Double(0.0,1e6),
                          "Starting value of the filter radius, should not be smaller than the target radius");
        prm.declare_entry("Initial penalty", "3.0", Patterns::Double(1.0,10.0),
                          "Initial penalization factor for intermediate densities");
        prm.declare_entry("Minimum density", "1.0e-3", Patterns::Double(0.0,1.0),
                          "Minimum density value.");
        prm.declare_entry("Quasi void threshold", "0.01", Patterns::Double(1.0e-20,1.0),
                          "Void threshold value for heuristics.");
        prm.declare_entry("Quasi solid threshold", "0.99", Patterns::Double(1.0e-20,1.0),
                          "Solid threshold value for heuristics.");
        prm.declare_entry("Grey cells threshold", "1.0e-3", Patterns::Double(1.0e-20,1.0),
                          "At what %of grey cells do you want to check for clusters.");
        prm.declare_entry("Hole area threshold", "1.0e-3", Patterns::Double(1.0e-20,1.0e20),
                          "At what area of cluster should it be considered as a hole.");
        prm.declare_entry("Open void elimination factor", "0.2", Patterns::Double(0.1,0.5),
                          "Scaling factor for shape moving step, which projects the shape onto solid feature. [0.1, 0.5]");
        prm.declare_entry("Reconstruct surface after optimization", "false", Patterns::Bool(),
                          "Do you want to reconstruct surface after optimization? ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Responses");
      {

        prm.declare_entry("Responses", "compliance", Patterns::List(Patterns::Anything()),
                          "compliance | volume");
        //prm.declare_entry("Objective ID", "0", Patterns::Integer(0, 10),
        //"Which response is the objective?");
        prm.declare_entry("Objectives IDs", "", Patterns::List(Patterns::Integer(0, 1)),
                          "Which responses are the objectives?");
        prm.declare_entry("Objectives weights", "", Patterns::List(Patterns::Double(0, 10)),
                          "Weights of the objectives");

        prm.declare_entry("Constraints IDs", "1", Patterns::List(Patterns::Integer(0, 10)),
                          "Which responses are the constraints?");

        prm.declare_entry("UB Constraints IDs", "", Patterns::List(Patterns::Integer(0, 10)),
                          "Which responses are the constraints?");
        prm.declare_entry("LB Constraints IDs", "", Patterns::List(Patterns::Integer(0, 10)),
                          "Which responses are the constraints?");
        prm.declare_entry("EQ Constraints IDs", "", Patterns::List(Patterns::Integer(0, 10)),
                          "Which responses are the constraints?");

        prm.declare_entry("Constraints LB", "", Patterns::List(Patterns::Double(-1e20, 1e20)),
                          "Lower bounds for each constraint. Unbounded: -1e20");
        prm.declare_entry("Constraints UB", "", Patterns::List(Patterns::Double(-1e20, 1e20)),
                          "Upper bounds for each constraint. Unbounded: 1e20");
        prm.declare_entry("Constraints EQ", "", Patterns::List(Patterns::Double(-1e20, 1e20)),
                          "RHS of equality constraint.");

        prm.declare_entry("Gradient normalization", "magnitude", Patterns::Anything(),
                          "l2_norm | magnitude");

      }
      prm.leave_subsection();


      prm.enter_subsection("Node-based optimization");
      {
        prm.declare_entry("Number of shape refinements", "0", Patterns::Integer(0, 5),
                          "Number of repeated shape optimization runs with refined shape [0, 5]");
        prm.declare_entry("Adaptive shape refinement", "true", Patterns::Bool(),
                          "Apply adaptive shape refinement?");

        prm.enter_subsection("Adaptive shape refinement options");
        {
          prm.declare_entry("Size threshold", "1.5", Patterns::Double(1.0, 10.0),
                            "Expressed as a multiplication of average element size of the initial design. [1.0, 10.0]");
          prm.declare_entry("Curvature threshold", "0.9", Patterns::Double(0.5, 1.0),
                            "Expressed as a cosinus of normal vectors between two adjacent shape cells. [0.5, 1.0]");
          prm.declare_entry("Minimum size ratio", "0.25", Patterns::Double(0.01, 1.0),
                            "Minimum allowable size of a shape element: minimum size ratio x avg element size");
        }
        prm.leave_subsection();

        prm.declare_entry("Non-design ID", "0", Patterns::List(Patterns::Integer(0, 50)),
                          "List of non-design IDs");
        prm.declare_entry("Non-design comp", "0", Patterns::List(Patterns::Integer(0, 3)),
                          "List of non-design comp");
        prm.declare_entry("Bounding box X values", "-1,1", Patterns::List(Patterns::Double(-1e3, 1e3)),
                          "Bounding values for X {low, upp}");
        prm.declare_entry("Bounding box Y values", "-1,1", Patterns::List(Patterns::Double(-1e3, 1e3)),
                          "Bounding values for Y {low, upp}");
        prm.declare_entry("Bounding box Z values", "-1,1", Patterns::List(Patterns::Double(-1e3, 1e3)),
                          "Bounding values for Z {low, upp}");
        prm.declare_entry("Sensitivity projection", "none", Patterns::Anything(),
                          "none | x | y | z");

        prm.enter_subsection("Regularization Options");
        {
          prm.declare_entry("Sensitivity Weighting", "true", Patterns::Bool(),
                            "Apply sensitivity weighting?: true or false");
          prm.declare_entry("Sensitivity Weighting Type", "field", Patterns::Anything(),
                            "field | component");
          prm.declare_entry("Dual Descent Smoothing", "true", Patterns::Bool(),
                            "Apply dual descent smoothing?: true or false");
          prm.declare_entry("Sufficient Decrease Coefficient", "0.85", Patterns::Double(0.0, 1.0),
                            "Sufficient decrease coefficient for dual descent smoothing: [0.0, 1.0]");

          prm.enter_subsection("Traction Method");
          {
            prm.declare_entry("Traction Method", "true", Patterns::Bool(),
                              "Apply regularization using traction method?: true or false");
            prm.declare_entry("Normal spring constant", "1", Patterns::Double(),
                              "Normal spring constant (stiffness)");
            prm.declare_entry("Tangent spring constant", "1", Patterns::Double(),
                              "Tangent spring constant (stiffness)");
            prm.declare_entry("Normal smoothing constant", "0.01", Patterns::Double(),
                              "Normal smoothing constant");
            prm.declare_entry("Tangent smoothing constant", "0.01", Patterns::Double(),
                              "Tangent smoothing constant");
            prm.declare_entry("Penalty constant", "1e9", Patterns::Double(),
                              "Penalty constant (geometrical constraints)");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

    }
    prm.leave_subsection(); // Leaving optimization section

  } // End of DeclareParameters function

  void Parameter::ParseParameters() {

    prm.parse_input(parameter_file);
    std::string arg;

    data.analysis_name = prm.get("Analysis name");
    data.destination_path = prm.get("Destination path");
    data.dim = prm.get_integer("dim");

    data.problem_type = prm.get("Problem Type");
    data.bvp_type = prm.get("BVP Type");

    data.output_mesh = prm.get_bool("Mesh Output");
    data.verbose = prm.get_bool("Verbose Output");

    prm.enter_subsection("Base");
    {
      arg = prm.get("Base Point1");
      string_to_vector_of_double(arg, data.base_point1);
      arg = prm.get("Base Point2");
      string_to_vector_of_double(arg, data.base_point2);
      arg = prm.get("Base Subdivisions");
      string_to_vector_of_int(arg, data.base_subdivision);
      data.base_refinements = prm.get_integer("Base refinements");
      data.anisotropic_refinement = prm.get_bool("Anisotropic refinement");
      data.extended_refinement = prm.get_bool("Extended refinement");
      data.refinement_criterion = prm.get("Refinement criterion");
      data.narrow = prm.get_double("Desired narrow");
      data.tracking_refinements = prm.get_integer("Tracking refinements");
      data.tracking_precision = prm.get_integer("Tracking precision");
    }
    prm.leave_subsection();

    prm.enter_subsection("Domain");
    {
      data.geometry_name = prm.get("Geometry name");

      prm.enter_subsection("Abaqus");
      {
        data.abaqus_input = prm.get("Input file");
        data.abaqus_refinements = prm.get_integer("Abaqus refinements");
      }
      prm.leave_subsection();

      prm.enter_subsection("Hyper Rectangle");
      {
        arg = prm.get("Hyper Rectangle Point1");
        string_to_vector_of_double(arg, data.hyper_rectangle_point1);

        arg = prm.get("Hyper Rectangle Point2");
        string_to_vector_of_double(arg, data.hyper_rectangle_point2);

        arg = prm.get("Hyper Rectangle Subdivisions");
        string_to_vector_of_int(arg, data.hyper_rectangle_subdivision);

        data.hyper_rectangle_refinements = prm.get_integer("Hyper Rectangle refinements");
      }
      prm.leave_subsection();

      prm.enter_subsection("Graded Hyper Rectangle");
      {
        arg = prm.get("Graded Hyper Rectangle Point1");
        string_to_vector_of_double(arg, data.graded_hyper_rectangle_point1);
        arg = prm.get("Graded Hyper Rectangle Point2");
        string_to_vector_of_double(arg, data.graded_hyper_rectangle_point2);

        arg = prm.get("Steps x");
        string_to_vector_of_double(arg, data.steps_x);
        arg = prm.get("Steps y");
        string_to_vector_of_double(arg, data.steps_y);
        arg = prm.get("Steps z");
        string_to_vector_of_double(arg, data.steps_z);

        data.graded_hyper_rectangle_refinements = prm.get_integer("Graded Hyper Rectangle refinements");
      }
      prm.leave_subsection();

      prm.enter_subsection("Hyper Cube With Hole");
      {
        data.hyper_cube_with_hole_inner_radius = prm.get_double("Hyper Cube With Hole Inner Radius");
        data.hyper_cube_with_hole_outer_radius = prm.get_double("Hyper Cube With Hole Outer Radius");
        data.hyper_cube_with_hole_L = prm.get_double("Hyper Cube With Hole L");
        data.hyper_cube_with_hole_repetitions = prm.get_integer("Hyper Cube With Hole Repetitions");
        data.hyper_cube_with_hole_refinements = prm.get_integer("Hyper Cube With Hole Refinements");
        data.hyper_cube_with_hole_rotate = prm.get_bool("Hyper Cube With Hole Rotate");
        data.hyper_cube_with_hole_rotation_angle = prm.get_double("Hyper Cube With Hole Rotation angle");
        data.hyper_cube_with_hole_rotation_axis = prm.get_integer("Hyper Cube With Hole Rotation axis");

      }
      prm.leave_subsection();

      prm.enter_subsection("Restart");
      {
        data.geo_input = prm.get("Geo Input file");
        data.restart_from_geo = prm.get_bool("Restart from Geo");
        data.refine_after_restart = prm.get_bool("Refine after restart");
      }
      prm.leave_subsection(); // leaving the Restart section

      prm.enter_subsection("Boundary");
      {
        arg = prm.get("Boundary IDs");
        string_to_vector_of_int(arg, data.boundary_ids);

        arg = prm.get("Boundary comp");
        string_to_vector_of_int(arg, data.boundary_comp);

        arg = prm.get("Boundary coord");
        string_to_vector_of_double(arg, data.boundary_coord);

        arg = prm.get("Boundary tol");
        string_to_vector_of_double(arg, data.boundary_tol);

      }
      prm.leave_subsection();

    }
    prm.leave_subsection();

    prm.enter_subsection("EDD BVP");
    {
      data.cut_cell_quadrature = prm.get_integer("Cut cell quadrature");
      data.weak_material = prm.get_double("Weak material contribution");
      data.penalty = prm.get_double("Dirchlet Penalty");
    }
    prm.leave_subsection();

    prm.enter_subsection("BVP");
    {
      data.poly_degree = prm.get_integer("Polynomial degree");

      // dirichlet_bc
      prm.enter_subsection("Dirichlet BC");
      {
        arg = prm.get("Dirichlet ID");
        string_to_vector_of_int(arg, data.dbc_id);

        arg = prm.get("Dirichlet comp");
        string_to_vector_of_int(arg, data.dbc_comp);

        // A check to ensure the correct component is passed to Dirichlet BC
        if (data.dbc_comp.size() > 0) {
          if (data.bvp_type == "elasticityLin") {
            Assert(*(std::max_element(data.dbc_comp.begin(), data.dbc_comp.end())) < data.dim,
                   ExcMessage("Dirichlet comp > dim for elasticity problem, which is stupid!"));
          }
        }

        arg = prm.get("Dirichlet value");
        string_to_vector_of_double(arg, data.dbc_value);
      }
      prm.leave_subsection(); // leaving Dirichlet BC subsection.

      // neumann_bc
      prm.enter_subsection("Neumann BC");
      {
        arg = prm.get("Neumann ID");
        string_to_vector_of_int(arg, data.nbc_id);

        arg = prm.get("Neumann comp");
        string_to_vector_of_int(arg, data.nbc_comp);

        // A check to ensure the correct component is passed to Dirichlet BC
        if (data.nbc_comp.size() > 0) {
          if (data.bvp_type == "elasticityLin") {
            Assert(*(std::max_element(data.nbc_comp.begin(), data.nbc_comp.end())) < data.dim,
                   ExcMessage("Neumann comp > dim for elasticity problem, which is stupid!"));
          }
        }

        arg = prm.get("Neumann value");
        string_to_vector_of_double(arg, data.nbc_value);
      }
      prm.leave_subsection(); // leaving Neumann BC subsection.

      prm.enter_subsection("Symmetry");
      {
        data.symmetry_id = prm.get_integer("Symmetry ID");
      }
      prm.leave_subsection();

      // Newton Raphson
      prm.enter_subsection("Newton Raphson");
      {
        data.max_NR_iter = prm.get_integer("Max Newton Iteration");
        data.tol_residual_NR = prm.get_double("Tolerance Newton Raphson");
        data.number_load_steps = prm.get_integer("Number of Load Steps");
        data.output_NR = prm.get_bool("Output Newton Raphson");
      }
      prm.leave_subsection();

    }
    prm.leave_subsection(); // leaving BVP subsection.

    // Material
    prm.enter_subsection("Material");
    {
      data.material_law = prm.get("Material Law");
      prm.enter_subsection("Elastic isotropic");
      {
        data.mat_elastic_isotropic.lambda = prm.get_double("lambda");
        data.mat_elastic_isotropic.mu = prm.get_double("mu");
        data.mat_elastic_isotropic.density = prm.get_double("density");
      }
      prm.leave_subsection();

    }
    prm.leave_subsection(); // leaving material section

    // Solver
    prm.enter_subsection("Solver");
    {
      data.solver_name = prm.get("Solver name");
      data.pp_solver_name = prm.get("Postprocessor Solver name");
      data.max_iter = prm.get_integer("Max iterations");
      data.solver_tol = prm.get_double("Solver Tolerance");
    }
    prm.leave_subsection(); // leaving solver section

    // Optimization
    prm.enter_subsection("Optimization");
    {

      data.optimizer_type = prm.get("Optimizer type");
      data.write_refine_geo = prm.get_bool("Write refined geo");

      prm.enter_subsection("AlMoM");
      {
        data.max_total_iters = prm.get_integer("Max total iterations");
        data.max_lagrange_updates = prm.get_integer("Max number of lagrange multiplier updates");
        data.max_al_iterations = prm.get_integer("Max number of AL subproblem iterations");

        arg = prm.get("Lagrange multiplier initial value");
        string_to_vector_of_double(arg, data.almom_lambda);

        data.almom_penalty_c = prm.get_double("Augmented lagrange penalty");
        data.initial_shape_step_length = prm.get_double("Initial shape step length");
        data.initial_density_step_length = prm.get_double("Initial density step length");
        data.contraction_step_length = prm.get_double("Step length contraction factor");
        data.armijo_factor_mu = prm.get_double("Armijo line search mu");
        data.max_contractions = prm.get_integer("Max number of contraction");
        data.al_convergence_tol = prm.get_double("AL subproblem convergence tolerance");
        data.constraint_violation_tol = prm.get_double("Constraint violation tolerance");
      }
      prm.leave_subsection();

      prm.enter_subsection("Density-based optimization");
      {
        data.initial_density = prm.get_double("Initial density");
        data.initial_filter_radius = prm.get_double("Initial filter radius");
        data.initial_penalty = prm.get_double("Initial penalty");
        data.min_density = prm.get_double("Minimum density");
        data.quasi_void_threshold = prm.get_double("Quasi void threshold");
        data.quasi_solid_threshold = prm.get_double("Quasi solid threshold");
        data.grey_threshold = prm.get_double("Grey cells threshold");
        data.hole_area_threshold = prm.get_double("Hole area threshold");
        data.open_void_elim_factor = prm.get_double("Open void elimination factor");
        data.surface_reconstruction = prm.get_bool("Reconstruct surface after optimization");
      }
      prm.leave_subsection();

      prm.enter_subsection("Responses");
      {
        arg = prm.get("Responses");
        string_to_vector_of_strings(arg, data.responses);

        arg = prm.get("Objectives IDs");
        string_to_vector_of_int(arg, data.objectives_id);

        arg = prm.get("Objectives weights");
        string_to_vector_of_double(arg, data.objectives_weights);

        arg = prm.get("Constraints IDs");
        string_to_vector_of_int(arg, data.constraints_id);

        arg = prm.get("UB Constraints IDs");
        string_to_vector_of_int(arg, data.constraints_ub_id);

        arg = prm.get("LB Constraints IDs");
        string_to_vector_of_int(arg, data.constraints_lb_id);

        arg = prm.get("EQ Constraints IDs");
        string_to_vector_of_int(arg, data.constraints_eq_id);

        arg = prm.get("Constraints LB");
        string_to_vector_of_double(arg, data.constraints_lb);

        arg = prm.get("Constraints UB");
        string_to_vector_of_double(arg, data.constraints_ub);

        arg = prm.get("Constraints EQ");
        string_to_vector_of_double(arg, data.constraints_eq);

        data.gradient_normalization = prm.get("Gradient normalization");

      }
      prm.leave_subsection();


      prm.enter_subsection("Node-based optimization");
      {
        data.n_shape_refinements = prm.get_integer("Number of shape refinements");

        data.adaptive_shape_refinement = prm.get_bool("Adaptive shape refinement");

        prm.enter_subsection("Adaptive shape refinement options");
        {
          data.adaptive_shape_ref_size_threshold = prm.get_double("Size threshold");
          data.adaptive_shape_ref_cos_threshold = prm.get_double("Curvature threshold");
          data.adaptive_shape_ref_min_size_ratio = prm.get_double("Minimum size ratio");
        }
        prm.leave_subsection();

        arg = prm.get("Non-design ID");
        string_to_vector_of_int(arg, data.non_design_id);

        arg = prm.get("Non-design comp");
        string_to_vector_of_int(arg, data.non_design_comp);

        arg = prm.get("Bounding box X values");
        string_to_vector_of_double(arg, data.bb_x);

        arg = prm.get("Bounding box Y values");
        string_to_vector_of_double(arg, data.bb_y);

        arg = prm.get("Bounding box Z values");
        string_to_vector_of_double(arg, data.bb_z);

        data.sensitivity_projection = prm.get("Sensitivity projection");

        prm.enter_subsection("Regularization Options");
        {
          data.sensitivity_weighting = prm.get_bool("Sensitivity Weighting");
          data.sensitivity_weighting_type = prm.get("Sensitivity Weighting Type");
          data.dual_descent_smoothing = prm.get_bool("Dual Descent Smoothing");
          data.sufficient_decrease_coeff = prm.get_double("Sufficient Decrease Coefficient");

          prm.enter_subsection("Traction Method");
          {
            data.traction_method = prm.get_bool("Traction Method");
            data.tm_normal_spring_constant = prm.get_double("Normal spring constant");
            data.tm_tangent_spring_constant = prm.get_double("Tangent spring constant");
            data.tm_normal_smoothing_constant = prm.get_double("Normal smoothing constant");
            data.tm_tangent_smoothing_constant = prm.get_double("Tangent smoothing constant");
            data.tm_penalty_constant = prm.get_double("Penalty constant");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

    }
    prm.leave_subsection();

  } // End of ParseParameters fucntion


  void Parameter::OutputDefaultParameters() {
    const std::string out_file = "default.prm";
    std::ofstream parameter_out(out_file);
    prm.print_parameters(parameter_out, ParameterHandler::Text);
  }

} // End of namespace StructuralOptimization
