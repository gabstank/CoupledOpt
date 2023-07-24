#pragma once

// C++ headers
#include <iostream>

// Deal.II headers
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>

// Project headers
#include <Parameter.h>
#include <Utilities.h>


namespace StructuralOptimization {

  using namespace dealii;

  /**
   * The traction method is meant to regularize the sensitivities by solving a fictitious BVP.
   * The raw sensitivities are incorporated as external loads (tractions)
   * and the solution vector of the fictitious BVP provides the sought-after, regularized sensitivities.
   * The internal energy is defined by means of the spring energy and a simple smoothing energy, which is based
   * on the vector laplacian. Additionally, the bounding box constraints are introduced by employing the penalty method.
   * The user is given the control to specify a series of constants for the fictitious BVP of the traction method.
   *
   * These are:
   * - Normal spring stiffness - the higher the stiffness, the smaller the design update in the direction normal to shape.
   * - Tangent spring stiffness - as above, but in the direction tangent to shape (controlling the distribution of the nodes).
   * - Normal smoothing constant - the higher the constant, the stronger the smoothing in the direction normal to shape.
   * - Tangent smoothing constant - as above, but in the direction tangent to shape (in-plane regularization).
   * - Penalty parameter - for the bounding box constraints, usually 10e9.
   *
   * Recommended choice:
   *
   * \f$ c_{spr,N} = 1.0 \f$
   *
   * \f$ c_{spr,T} = \left[ 0.5, 1.0 \right] \f$
   *
   * \f$ c_{sm,N} = \left[ 0.001, 0.1 \right] \f$
   *
   * \f$ c_{sm,T} = \left[ 0.001, 0.1 \right] \f$
   *
   * \f$ c_p = 10^9 \f$
   *
   * The tangent spring set to a smaller value than the normal spring allows for enhanced in-plane regularization
   * of the nodes. In some examples this has shown to improve not only the distribution of the nodes but also exhibits
   * a slightly faster convergence.
   *
   * @tparam dim
   */
  template<int dim>
  class TractionMethod {
  public:
    /// Constructor
    TractionMethod(Parameter &par_);

    /// Destructor
    ~TractionMethod() = default;

    /**
     * The only public interface of the class. Executes the ficititious BVP of the traction method.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     * @param[in,out] sensitivities - passed as reference, the solution of traction method is there
     * @param vertex_normals
     */
    void Run(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints,
        Vector<double> &sensitivities, const Vector<double> &vertex_normals);

    void RunSimple(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints,
             Vector<double> &sensitivities, const Vector<double> &vertex_normals);

  private:
    /**
     * Usual setup system function as used in dealii.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     */
    void _SetupSystem(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints);

    /**
     * Assembly of the spring and smoothing matrices.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     */
    void _AssembleSpringSmoothingMatrix(DoFHandler<dim - 1, dim> &dof_handler_shape,
        AffineConstraints<double> &shape_constraints);

    /**
     * Assembly of the spring and smoothing rhs. Necessary since it is a nonlinear BVP.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     */
    void _AssembleSpringSmoothingRhs(DoFHandler<dim - 1, dim> &dof_handler_shape,
        AffineConstraints<double> &shape_constraints);

    /**
     * Assembly of the penalty matrices and rhs.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     */
    void _AssemblePenalties(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints);

    /**
     * Solving a load step of the nonlinear BVP. In the current implementation, there is just one load step.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     */
    void _SolveLoadStepNR(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints);

    /**
     * Calls a linear solver.
     *
     * @param shape_constraints
     */
    void _SolveLinearSystem(AffineConstraints<double> &shape_constraints);

    /**
     * The function returns the residual.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     * @return Residual value
     */
    double _GetErrorResidual(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints);

    void _SetRhsScale(DoFHandler<dim - 1, dim> &dof_handler_shape,
                      AffineConstraints<double> &shape_constraints);

    Data &_data;
    TimerOutput &compute_timer;

    FESystem<dim - 1, dim> _fe_shape;
    MappingQ<dim - 1, dim> _mapping;
    SparsityPattern _sparsity_pattern;
    int _newton_iteration = 0;
    Errors _error_NR;

    SparseMatrix<double> _system_matrix;
    Vector<double> _system_rhs;
    Vector<double> _solution;

    SparseMatrix<double> _spring_matrix;

    Vector<double> _initial_sensitivities;
    Vector<double> _vertex_normals;

    Vector<double> _newton_update;
    Vector<double> _residual;

    double _rhs_scale;
    bool _rhs_scale_set = false;
  };


  template<int dim>
  class Regularization {
  public:
    /// Constructor
    Regularization(Parameter &par_);

    /// Destructor
    ~Regularization();

    /**
     * The first way to regularize the sensitivities.
     * The traction method is called when the parameter "Traction Method" is set true.
     * This function calls the TractionMethod.Run() method of the TractionMethod class.
     * The sensitivities are regularized by running a fictitious BVP, in which the sensitivities are used
     * external loading and the solution of which are the desired, regularized sensitivities.
     * If requested so in the parameter file, the _DualDescentSmoothing() and _Projection() are called inside
     * this function.
     *
     * The following regularization procedure is invoked:
     * 1. _Projection() (optional)
     * 2. TractionMethod.Run()
     * 3. _DualDescentSmoothing() (optional)
     * 4. _Projection() (optional)
     *
     * For more details regarding the traction method, see the description of the TractionMethod class.
     *
     * @param dof_handler_shape
     * @param shape_constraints
     * @param[in,out] sensitivities
     * @param vertex_normals
     *
     * @see TractionMethod
     * @see _DualDescentSmoothing()
     * @see _Projection()
     */
    void
    RunTractionMethod(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints,
        Vector<double> &sensitivities, const Vector<double> &vertex_normals);

    void
    RunSimpleTractionMethod(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints,
                      Vector<double> &sensitivities, const Vector<double> &vertex_normals);

    /**
     * The second way to regularize the sensitivities.
     * Directly "filter" the sensitivity vector.
     * Direct filtering is an alternative approach to the traction method to regularize the sensitivities
     * and is called when the parameter "Traction Method" is set to false. Then, depending on
     * further parameter options, either _FieldWeighting() or _ComponentWeighting() filter function is called,
     * which gets rid of mesh size dependency of the sensitivities. It also calls for _DualDescentSmoothing()
     * and _Projection() if requested so in the parameter file.
     *
     * The following regularization procedure is invoked:
     * 1. _FieldWeighting() or _ComponentWeighting() (optional)
     * 2. _DualDescentSmoothing() (optional)
     * 3. _Projection() (optional)
     *
     * @param dof_handler_shape
     * @param shape_constraints
     * @param[in,out] sensitivities
     * @param vertex_normals
     *
     * @see _ComponentWeighting()
     * @see _FieldWeighting()
     * @see _DualDescentSmoothing()
     * @see _Projection()
     */
    void
    DirectShapeSensitivityFiltering(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints,
        Vector<double> &sensitivities, const Vector<double> &vertex_normals);

    /**
     * Sensitivity filter
     * @param dof_handler_base
     * @param pseudo_densities
     * @param sensitivities
     * @param filtered_sensitivity
     */
    void
    DensitySensitivityFiltering(DoFHandler<dim> &dof_handler_base, std::map<CellId, double> &sensitivities, std::map<CellId, double> &filtered_sensitivity);

  private:
    /**
     * This function can be used to project the sensitivities along the axis as specified
     * in the parameter file under "Sensitivity projection". This function is useful when the goal of optimization
     * is to restrict the design change to dim-1 dimensions, i.e. x,y-plane optimization of a 3D model of a cantilever
     * energy harvester.
     *
     * @param dof_handler_shape - iteration over shape cells
     * @param[in,out] sensitivities - passed as reference and modified directly, no return type
     */
    void _Projection(DoFHandler<dim - 1, dim> &dof_handler_shape, Vector<double> &sensitivities);

    /**
     * Field weighting is called by DirectShapeSensitivityFiltering() function if the flag for the weighting is set to true.
     * It is an alternative approach to what is done by the traction method.
     * The magnitude of the raw sensitivities is dependent on the element size, therefore it needs to be scaled.
     * For that, diagonal weighting matrix is setup using shape functions and element jacobians.
     * The scaling is done by contraction of raw sensitivity vector and diagonal weighting matrix.
     *
     * @param dof_handler_shape - needed to access the shape triangulation
     * @param shape_constraints - needed to distribute constraints over modified sensitivities
     * @param[in,out] sensitivities - passed as reference and modified directly, no return type
     *
     * @see _ComponentWeighting() - alternative implementation
     */
    void _FieldWeighting(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints,
        Vector<double> &sensitivities);

    /**
     * Component weighting function is just an alternative implementation of the _FieldWeighting() function.
     * The functionality is the same and leads to the same result, but the way the sensitivities are weighted differs.
     * Instead of weighting the normal field values of the sensitivities, the weighting process is performed
     * componentwise on the vector-valued sensitivities.
     *
     * @param dof_handler_shape - needed to access the shape triangulation
     * @param shape_constraints - needed to distribute constraints over modified sensitivities
     * @param[in,out] sensitivities - passed as reference and modified directly, no return type
     *
     * @see _FieldWeighting() - alternative implementation
     */
    void _ComponentWeighting(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints,
        Vector<double> &sensitivities);

    /**
     * This function is called by both the RunTractionMethod() and DirectShapeSensitivityFiltering() functions if the flag
     * for dual descent smoothing is set to true.
     * Dual descent is a method for sensitivity smoothing in the direction normal to surface of the shape.
     * This is a geometry-based smoothing, which means that the smoothed sensitivities compensate for the
     * irregularities in the geometry itself rather than the irregularities in the sensitivity values.
     * The resulting smoothed sensitivities are a weighted combination of initial sensitivities and the vector
     * corresponding to averaged nodal positions of the shape.
     *
     * @param dof_handler_shape - passed directly to NodalAveraging() function
     * @param shape_constraints - passed directly to NodalAveraging() function
     * @param[in,out] sensitivities - passed as reference and modified directly, no return type
     *
     * @see _ComputeAveragedVertex()
     * @see _NodalAveraging() - function directly called by the dual descent smoothing
     */
    void _DualDescentSmoothing(DoFHandler<dim - 1, dim> &dof_handler_shape,
        AffineConstraints<double> &shape_constraints, Vector<double> &sensitivities);

    /**
     * Nodal Averaging is called by _DualDescentSmoothing() function.
     * We compute averaged nodal positions as an average of the positions of adjacent nodes.
     * The averaging vector then stores the differences between the averaged and the actual positions of the nodes.
     *
     * @param dof_handler_shape - used to iterate of shape cells
     * @param shape_constraints - used to distributed constraints over averaged nodes vector
     *
     * @see _DualDescentSmoothing()
     * @see _ComputeAveragedVertex()
     */
    void _NodalAveraging(DoFHandler<dim - 1, dim> &dof_handler_shape, AffineConstraints<double> &shape_constraints);

    /**
     * Function called by _NodalAveraging().
     * Compute an averaged position of the vertices of the given adjacent cells.
     *
     * @param adjacent_cells - the cells that share the vertex for which an averaged position needs to be computed
     * @param[in,out] averaged_vertex - a point to store the averaged vertex
     *
     * @see _DualDescentSmoothing()
     * @see _NodalAveraging()
     */
    void _ComputeAveragedVertex(unsigned int,
                                std::vector<typename DoFHandler<dim - 1, dim>::active_cell_iterator> adjacent_cells,
                                Point<dim> &averaged_vertex);

    Vector<double> _averaging_vector;
    Vector<double> _raw_sensitivities;
    Vector<double> _vertex_normals;

    /// Neighbour list for mesh independent filter for density-based sensitivities. Assembled only once.
    /// The key is the cell value, and the value is the list of neighbours and their distance from the key.
    std::map<CellId, std::map<CellId, double>> _neighbour_list;

    Data &_data;
    MPI_Comm &mpi_communicator;
    const unsigned int &this_mpi_process;
    TimerOutput &compute_timer;

    TractionMethod<dim> _traction_method;
  };


} //End of StructuralOptimization Namespace
