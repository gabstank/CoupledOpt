#pragma once

// C++ headers
#include <iostream>

// Deal.II headers
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

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
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/data_postprocessor.h>

#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>

// Project headers
#include <Parameter.h>
#include <Mesh.h>
#include <Material.h>
#include <DataOutput.h>

namespace StructuralOptimization {
  template<int dim>
  class BVP_ {
  public:
    BVP_(Parameter &, Mesh<dim> &);

    virtual ~BVP_() = default;

    /**
     * mode = 0 -> std bvp evaluation
     * mode = 1 -> simp isotropic optimization case
     * mode = 3 ->
     * @param mode
     */
    virtual void Run(const unsigned int mode = 0) = 0;

    //-------------------------------------------------------------

    /**
     * @brief Function to get the base output data that are displayed in optimization.
     *
     * @param output
     */
    virtual void GetAllOutputData(DataOutput<dim, DomainParallelTriaType<dim>, DoFHandler<dim>> &output);

    /// TODO: make this Pure virtual function.
    virtual Vector<double> GetPostprocessingData(unsigned int &data_flag){
      (void) data_flag;
      throw std::runtime_error("BVP_::GetPostprocessingData should never be called in base class. "
                               "TODO: Once this funcion is implemented in all the classes this pure virtual.");
    }

  public:
    void _SolveLinearSystem(LA::MPI::SparseMatrix &system_matrix,
                            LA::MPI::Vector &solution,
                            LA::MPI::Vector &system_rhs);

    void _SetupSmoothingSystem();

    void AssembleSmoothingMatrix();

    void SolveSmoothingSystem();

    Vector<double> ReturnPointSolution(Point<dim> pt);

    MPI_Comm &mpi_communicator;
    const unsigned int &n_mpi_processes;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    Data &data;

    IndexSet _locally_owned_dofs, _locally_relevant_dofs;
    //-----------------Postprocessing data-----------------
    // System of equations to smooth the strain and stress output
    IndexSet _locally_owned_smooth_dofs , _locally_relevant_smooth_dofs;
    FESystem<dim> _smoothing_fe;
    AffineConstraints<double> _smoothing_constraints;
    LA::MPI::SparseMatrix _smoothing_matrix;
    LA::MPI::Vector _smoothing_solution;
    LA::MPI::Vector _smoothing_rhs;
    DoFHandler<dim> _smoothing_dof_handler;
    //---------------------------------------------------

    DomainTriaType<dim> &_tria_domain;
    DomainParallelTriaType<dim> &_tria_domain_parallel;
//    DomainParallelTriaTypeDistributed<dim> _tria_domain_parallel;

    DoFHandler<dim> _dof_handler;
    AffineConstraints<double> _constraints;
    LA::MPI::Vector _solution;

    LA::MPI::SparseMatrix _system_matrix;
    LA::MPI::Vector _system_rhs;

    //----------Density-based optimization data----------
    std::map<CellId, FullMatrix<double>> cell_simp_density_sens_matrices;
    std::map<CellId, double> pseudo_densities;
    std::map<CellId, FullMatrix<double>> cell_simp_fract_sens_matrices;
    std::map<CellId, double> pseudo_fractions;

    inline FullMatrix<double> GetCellSIMPSensMatrix(CellId id) {return cell_simp_density_sens_matrices[id];}
    inline FullMatrix<double> GetCellSIMPFracSensMatrix(CellId id) {return cell_simp_fract_sens_matrices[id];}
    void SetPseudoDensitiesToOne();
    inline void SetPseudoDensities(const std::map<CellId, double>& pseudo_densities_) {pseudo_densities = pseudo_densities_;}
    inline void SetPseudoFractions(const std::map<CellId, double>& pseudo_fractions_) {pseudo_fractions = pseudo_fractions_;}
    inline double GetPseudoDensity(CellId id) {return pseudo_densities[id];}
    inline double GetPseudoFraction(CellId id) {return pseudo_fractions[id];}
    inline std::map<CellId, double> GetPseudoDensities() {return pseudo_densities;}
    inline std::map<CellId, double> GetPseudoFractions() {return pseudo_fractions;}
    //---------------------------------------------------


  }; // end of BVP_ class definition
} // end of StructuralOptimization namespace
