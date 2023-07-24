#pragma once

// C++ headers

// Deal.II headers

// Project headers
#include <EddBVP_.h>
#include <Material.h>

namespace StructuralOptimization {

  template<int dim>
  class EddElasticityLin : public EddBVP_<dim> {
    /**
     * @brief Linear elasticity BVP solver in embedded domain setting. Strong form of Linear elasticity BVP
     * \f$ -div(\mathbf{\sigma}) = \mathbf{f} \qquad in \ \Omega \qquad Balance \ Equation\f$
     * \f$ \mathbf{\sigma} = \mathbf{C} : \mathbf{\epsilon} \qquad Constitutive \ relation\f$
     * \f$ \mathbf{\epsilon} = 0.5 \left[ \nabla \mathbf{u} + \nabla \mathbf{u} ^\top \right] \f$
     * \f$ \mathbf{u} = \overline{\mathbf{u}} \qquad on \ \partial \Omega_D \qquad Dirichlet \ BC \f$
     * \f$ \mathbf{\sigma} \cdot \mathbf{n} = \overline{\mathbf{t}} \qquad on \ \partial \Omega_N \qquad \ Neumann BC \f$
     */
  public:
    /// Constructor
    EddElasticityLin(Parameter &, Mesh<dim> &);

    /// Destructor
    ~EddElasticityLin();

    /// Solves the BVP.
    void Run() override;

    /**
     * Given an enumerator index corresponding to the requested scalar quantity,
     * this function returns a a Vector<double> of size corresponding to the number of nodes,
     * for which this quantity is available. 
     */
    Vector<double> GetPostprocessingData(unsigned int &) override;

    virtual hp::FECollection<dim> GetBaseFE() override;

  private:

    /**
     * @brief Function to setup all the finite element and the linear algebra objects.
     */
    void _SetupSystem();

    /**
     * @brief Sets up #quadrature_point_history object to log the quadrature point data.
     */
    void _SetupQPointHistory();

    /**
     * @brief Function to compute the quadrature point data specified in #QPointHistory.
     */
    void _UpdateQPointHistory();

    /**
     * @brief Assemble the stiffness matrix.
     * \f$ K_{i,j} = \int_{V^e} \nabla \mathbf{N}_i : \mathbf{C} : \nabla \mathbf{N}_j dV \f$
     */
    void _AssembleSystem();

    /**
     * @brief Function to weakly impose Dirichlet boundary condition. Adds a penalty matrix with entry
     * \f$ L_{i,j} = \beta \int_{V^e} \mathbf{N}_i \cdot \mathbf{N}_j \mathbf{S} dV \f$
     * and rhs penalty vector with entry
     * \f$ l_{i} = \beta \int_{V^e} \mathbf{N}_i \cdot \mathbf{S} \cdot \overline{\mathbf{u}}  dV \f$
     */
    void _DirichletPenaltyMatrix();

    /**
     * @brief Assemble force vector. Performs surface integral over the shape segment cutting the base cells. 
     */
    void _AssembleRhs();

    /**
     * @brief Function to perform surface integral over the segment formed from segment_p1 and segment_p2 in case of 2D.
     * or surface integral over the surface formed by subtriangulation. Finally the results are in cell_rhs.
     * 
     * @param cell 
     * @param segment_p1 
     * @param segment_p2 
     * @param subtriangulation 
     * @param id 
     * @param cell_rhs 
     */
    void _IntegrateCellRhs(typename DoFHandler<dim>::active_cell_iterator &cell,
                           Point<dim> &segment_p1, Point<dim> &segment_p2,
                           std::vector<std::vector<Point<dim>>> &subtriangulation,
                           unsigned int &id,
                           Vector<double> &cell_rhs);

    /**
     * @brief Solves linear system, internally calls the solve linear system from the base class. 
     * Also Pushes the solution and rhs to the DataOutput object.
     */
    void _SolveSystem();

    /**
     * @brief Function to write the output.
     * 
     * @param cycle 
     */
    void _OutputResults(const unsigned int cycle);

    /**
     * @brief Function to perform post processing. This function loops over the post processing flags
     * and calls the projction functions to compute and finally updates the data.
     */
    void _Postprocess();

    /**
     * @brief Assemble RHS with data corresponding to assembly_flag.
     * @param assembly_flag 
     */
    void _AssembleSmoothingRhs(unsigned int assembly_flag);

    /**
     * @brief Updates data in #postprocessing_data object based on the assembly_flag.
     * @param assembly_flag 
     */
    void _UpdatePostprocessingData(unsigned int assembly_flag);

    MPI_Comm &mpi_communicator;
    const unsigned int &n_mpi_processes;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    FESystem<dim> _fe_void;
    FESystem<dim> _fe_solid;
    FESystem<dim - 1, dim> _fe_shape;

    hp::FECollection<dim> _fe_collection;

    MaterialLinElastic<dim> _mat;
    DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> bvp_output;

    /// Struct that stores the information about stress, strain, etc. at a specific quadrature point
    struct QPointHistory {
      SymmetricTensor<2, dim> strain;
      SymmetricTensor<2, dim> stress;
      double strain_energy;
    };

    unsigned int cycle = 0;

    /// Vector of quadrature point history for all quadrature points of the base
    std::vector<QPointHistory> quadrature_point_history;

    /**
     * @brief Struct to hold all the post processing data.
     */
    struct PostprocessingData {
      Vector<double> base_strain11;
      Vector<double> base_strain12;
      Vector<double> base_strain22;
      Vector<double> base_strain13;
      Vector<double> base_strain23;
      Vector<double> base_strain33;
      Vector<double> base_stress11;
      Vector<double> base_stress12;
      Vector<double> base_stress22;
      Vector<double> base_stress13;
      Vector<double> base_stress23;
      Vector<double> base_stress33;
      Vector<double> base_strain_energy;
      Vector<double> base_vm_stress;
      Vector<double> base_comp_topo;
    } postprocessing_data;

  }; // end of ElasticityLin class

} // end of StructuralOptimization namespace
