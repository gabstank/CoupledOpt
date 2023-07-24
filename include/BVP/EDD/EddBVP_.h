#pragma once

// C++ headers
#include <iostream>

// Deal.II headers
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/data_postprocessor.h>

#include <deal.II/numerics/fe_field_function.h>
// Project headers
#include <Parameter.h>
#include <Mesh.h>
#include <Material.h>
#include <EddTools.h>

namespace StructuralOptimization {
  /**
   * @brief Abstract Base class for all the embedded domain BVP.
   * Embedded domain BVP operates on a non body confirming mesh. This method differs from the standard BVP wrt to application of the boundary condition
   * and the treatment of the integration of the boundary cells.
   *
   * Base class has implementation of common function such as the solver and the post processing functions.
   *
   */
  template<int dim>
  class EddBVP_ {
  public:
    /**
     * @brief Construct a new EddBVP object.
     * In the constructor the following variables are initialized.
     * MPI related functions: EddBVP_::mpi_communicator , EddBVP_::this_mpi_process, EddBVP_::pcout
     * Compute time logger: EddBVP_::compute_timer
     * List of parameters: EddBVP_::data
     * Shape and base mesh: EddBVP_::_tria_shape , EddBVP_::_tria_base
     * Shape and base DoFHandlers: EddBVP_::_dof_handler_shape , EddBVP_::_dof_handler_base
     * Smoothing related variables: EddBVP_::_smoothing_dof_handler, EddBVP_::_smoothing_fe_void, EddBVP_::_smoothing_fe_solid
     * Help variables: EddBVP_::min_face_length_base and EddBVP_::boundary_cell_assignment
     * @param par_ 
     * @param mesh_ 
     */
    EddBVP_(Parameter &par_, Mesh<dim> &mesh_);

    /// Destructor of base class must always be virtual
    virtual ~EddBVP_() = default;

    /**
     * @brief Virtual run function.
     */
    virtual void Run() = 0;

    /// Pure virtual function.
    virtual Vector<double> GetPostprocessingData(unsigned int &) = 0;

    /**
     * @brief Function to get the base output data that are displayed in optimization.
     * 
     * @param output 
     */
    virtual void GetAllOutputData(DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> &output);

    virtual hp::FECollection<dim> GetBaseFE() = 0;

    /**
     * @brief Function to setup smoothing related variables: EddBVP_::_smoothing_constraints,
     * EddBVP_::_smoothing_solution , EddBVP_::_smoothing_rhs and EddBVP_::_smoothing_matrix
     * 
     */
    void SetupSmoothingSystem();

    /**
     * @brief Function to assemble smoothing matrix. EddBVP_::_smoothing_matrix.
     * \f$ K_{i,j} = \int n_i n_j dV \f$
     */
    void AssembleSmoothingMatrix();

    /**
     * @brief Function to solve smoothing system. 
     * Takes in a solver which is initialized and factorised with EddBVP_::_smoothing_matrix.
     * @param solver 
     */
    void SolveSmoothingSystem();

    /**
     * @brief Function to solve linear system of equations.
     * 
     * @param system_matrix 
     * @param solution 
     * @param system_rhs 
     * @param solver 
     */
    void SolveLinearSystem(LA::MPI::SparseMatrix &system_matrix,
                           LA::MPI::Vector &solution,
                           LA::MPI::Vector &system_rhs,
                           std::string solver = "");

    /**
     * @brief To be used in compliance computation.
     * 
     * @return Vector<double> 
     */
    virtual Vector<double> GetForceVector() {
      throw std::runtime_error ("This function should be called by only derived class.");
    }

    /**
     * @brief Function to get solution at a point pt.
     * 
     * @param pt 
     * @return Vector<double> 
     */
    Vector<double> ReturnPointSolution(Point<dim> pt);

    /**
     * @brief Function to project base solution on to the shape. 
     * 
     * @return Vector<double> 
     */
    Vector<double> GetShapeSolution();

    MPI_Comm &mpi_communicator;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    Data &data;

    ShapeTriaType<dim> &_tria_shape;
    DoFHandler<dim - 1, dim> _dof_handler_shape;

    BaseTriaType<dim> &_tria_base;
    DoFHandler<dim> _dof_handler_base;

    double min_face_length_base;

    /// Used to apply dirichlet boundary condition.
    std::map<unsigned int, std::vector<typename BaseTriaType<dim>::active_cell_iterator>> &boundary_cell_assignment;
    /**
     * Used for open cluster elimination.
     * Each shape cell is assigned with all the base cells that are intersected by this shape cell
     */
    std::map<CellId, std::vector<CellId>> shape_to_base_cell_assignment;

    IndexSet _locally_owned_dofs, _locally_relevant_dofs;
    AffineConstraints<double> _constraints;

    AffineConstraints<double> shape_constraints;

    //----------Density-based optimization data----------
    std::map<CellId, FullMatrix<double>> cell_stiffness_matrices;
    std::map<CellId, double> pseudo_densities;
    double penalty;

    inline FullMatrix<double> GetCellStiffnessMatrix(CellId id) {return cell_stiffness_matrices[id];}
    inline void SetPseudoDensities(const std::map<CellId, double>& pseudo_densities_) {pseudo_densities = pseudo_densities_;}
    inline double GetPseudoDensity(CellId id) {return pseudo_densities[id];}
    inline std::map<CellId, double> GetPseudoDensities() {return pseudo_densities;}
    //---------------------------------------------------

    //-------------Static Elasticity data----------
    LA::MPI::SparseMatrix _system_matrix;
    LA::MPI::Vector _solution;
    LA::MPI::Vector _system_rhs;
    //---------------------------------------------

    //-----------------Mech Normal Modes data------------
    LA::MPI::SparseMatrix _stiffness_matrix;
    LA::MPI::SparseMatrix _mass_matrix;
    std::vector<double> _eigenvalues;
    std::vector<LA::MPI::Vector> _eigenvectors;
    unsigned int _n_modes;

    inline Vector<double> GetEigenvector(unsigned int n) { return Vector<double>(_eigenvectors[n]); }
    //---------------------------------------------------

    //-----------------Piezo Normal Modes data-----------
    LA::MPI::SparseMatrix _piezo_stiffness_matrix;
    LA::MPI::SparseMatrix _piezo_mass_matrix;

    std::vector<double> _piezo_resonance_eigenvalues;
    std::vector<LA::MPI::Vector> _piezo_resonance_eigenvectors;
    std::vector<double> _piezo_antiresonance_eigenvalues;
    std::vector<LA::MPI::Vector> _piezo_antiresonance_eigenvectors;

    inline Vector<double> GetPiezoResonanceVector(unsigned int n) { return Vector<double>(_piezo_resonance_eigenvectors[n]); }
    inline Vector<double> GetPiezoAntiresonanceVector(unsigned int n) { return Vector<double>(_piezo_antiresonance_eigenvectors[n]); }
    //---------------------------------------------------

    //-----------------Postprocessing data---------------
    // System of equations to smooth the strain and stress output
    IndexSet _locally_owned_smooth_dofs, _locally_relevant_smooth_dofs;
    AffineConstraints<double> _smoothing_constraints;
    FESystem<dim> _smoothing_fe_void;
    FESystem<dim> _smoothing_fe_solid;
    hp::FECollection<dim> _smoothing_fe_collection;
    DoFHandler<dim> _smoothing_dof_handler;
    LA::MPI::SparseMatrix _smoothing_matrix;
    LA::MPI::Vector _smoothing_solution;
    LA::MPI::Vector _smoothing_rhs;

    void _OutputSmoothing(const unsigned int cycle);
    DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> smoothing_output;
    int cycle = 0;
    //---------------------------------------------------

  }; // end of EddBVP_ class definition
} // end of StructuralOptimization namespace
