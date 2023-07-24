#pragma once

#include <iostream>
#include <fstream>

#include <boost/algorithm/string.hpp> // to trim string

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/lac/generic_linear_algebra.h>

//#define FORCE_USE_OF_TRILINOS
#ifdef STOP_USE_TRILINOS
#define FORCE_USE_OF_TRILINOS
#endif

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

namespace StructuralOptimization {
#define CERR_DEBUG std::cerr <<__FILE__<< ":" <<__LINE__<<" | "
#ifdef DEBUG
#  define COUT_DEBUG std::cout <<__FILE__<< ":" <<__LINE__<<" | "
#else
#  define COUT_DEBUG if (false) std::cout
#endif

#define WAIT_STOP { double d=0.0; for(int i=0; i<1e6; ++i) d+=i; if(d>0.0) d=0.0; }

/// Macro to have an MPI_Barrier
#define PROJ_MPI_BARRIER {int ierr = MPI_Barrier(this->mpi_communicator); AssertThrowMPI(ierr);}

  using namespace dealii;

  template<int dim>
  using DomainTriaType = Triangulation<dim>; // Standard domain mesh

  template<int dim>
  using ShapeTriaType = Triangulation<dim - 1, dim>; // shape mesh

  template<int dim>
  using DomainParallelTriaType = parallel::TriangulationBase<dim>; // base background mesh
  template<int dim>
  using DomainParallelTriaTypeDistributed = parallel::distributed::Triangulation<dim>;
  template<int dim>
  using DomainParallelTriaTypeTypeShared = parallel::shared::Triangulation<dim>;

  template<int dim>
  using BaseTriaType = parallel::TriangulationBase<dim>; // base background mesh
  template<int dim>
  using BaseTriaTypeDistributed = parallel::distributed::Triangulation<dim>;
  template<int dim>
  using BaseTriaTypeShared = parallel::shared::Triangulation<dim>;

// standard tensors
  template<int dim>
  class StandardTensors {
  public:
    static const SymmetricTensor<2, dim> I;

    static const SymmetricTensor<4, dim> IxI();

    static const SymmetricTensor<4, dim> II();
  };

  template<int dim>
  inline SymmetricTensor<4, dim> GetIsoElasticityTensorLameParameters(const double lambda, const double mu){
    SymmetricTensor<4, dim> elasticity_tensor = lambda * StandardTensors<dim>::IxI() + 2 * mu * StandardTensors<dim>::II();
    return elasticity_tensor;
  }

  template<int dim>
  SymmetricTensor<4, dim> GetIsoElasticityTensorYoungsPoission(const double E, const double nu){
    double lambda = E*nu/((1+nu)*(1-2*nu));
    double mu = 0.5*E/(1+nu);
    SymmetricTensor<4, dim> elasticity_tensor = GetIsoElasticityTensorLameParameters<dim>(lambda,  mu);
    return elasticity_tensor;
  }

  template<int dim>
  SymmetricTensor<4, dim> GetIsoElasticityTensorBulkShear(const double kappa, const double mu){
    double lambda = kappa - (2./3)*mu;
    SymmetricTensor<4, dim> elasticity_tensor = GetIsoElasticityTensorLameParameters<dim>(lambda,  mu);
    return elasticity_tensor;
  }

  /**
   * To construct orthotropic elasticity tensor.
   * First I determine compliance tensor based on the expression in https://en.wikipedia.org/wiki/Orthotropic_material
   * @tparam dim
   * @param Es First value correspond to the fiber dir, 2nd value to the trans dir
   * @param Nus First value correspond to the fiber dir, 2nd value to the trans dir
   * @param Mus
   * @param fiber_dir
   * @return
   */
  template<int dim>
  SymmetricTensor<4, dim> GetOrthoElasticityTensorBulkShear(const std::vector<double> Es, const std::vector<double> Nus, const std::vector<double> Mus, const unsigned int fiber_dir = 0){
    unsigned int trans_dir_1, trans_dir_2;
    if(dim ==2){
      trans_dir_1 = (fiber_dir == 0) ? 1 : 0;
    } else { // dim ==3
      trans_dir_1 = (fiber_dir == 0) ? 1 : 0;
      trans_dir_2 = (fiber_dir == 2) ? 1 : 2;
    }
    // compliance tensor is constructed based on
    // https://en.wikipedia.org/wiki/Orthotropic_material
    SymmetricTensor<4, dim> compliance_tensor;
    if(dim == 2){
      compliance_tensor[fiber_dir][fiber_dir][fiber_dir][fiber_dir] = 1.0 / Es[0];
      compliance_tensor[trans_dir_1][trans_dir_1][trans_dir_1][trans_dir_1] = 1.0 / Es[1];

      compliance_tensor[fiber_dir][fiber_dir][trans_dir_1][trans_dir_1] = - Nus[0] / Es[0];
      compliance_tensor[trans_dir_1][trans_dir_1][fiber_dir][fiber_dir] = - Nus[0] / Es[0];

      compliance_tensor[fiber_dir][trans_dir_1][fiber_dir][trans_dir_1] = 0.5 / Mus[0];
      compliance_tensor[fiber_dir][trans_dir_1][trans_dir_1][fiber_dir] = 0.5 / Mus[0];
      compliance_tensor[trans_dir_1][fiber_dir][fiber_dir][trans_dir_1] = 0.5 / Mus[0];
      compliance_tensor[trans_dir_1][fiber_dir][trans_dir_1][fiber_dir] = 0.5 / Mus[0];

    } else { // dim ==3
      (void) trans_dir_2;
      throw std::runtime_error("Not implemented in 3D. But it is possible to implement in 3D");
    }

    return dealii::invert(compliance_tensor);

  }

  /// Enum for FESystems in electro-elasticity problems
  enum {
    e_u_dof = 0,
    e_phi_dof = 1
  };

  /// Enum for base cells. Enum specified if the cells position in relation to shape.
  enum {
    e_outside_cell = 0,
    e_boundary_cell = 1,
    e_inside_cell = 2,
    e_boundary_electrode_cell = 3
  };

  /// Strain 
  enum {
    e_sym_grad_u_11 = 100,
    e_sym_grad_u_12 = 101,
    e_sym_grad_u_22 = 102,
    e_sym_grad_u_13 = 103,
    e_sym_grad_u_23 = 104,
    e_sym_grad_u_33 = 105
  };

  /// Stress
  enum {
    e_cauchy_s_11 = 200,
    e_cauchy_s_12 = 201,
    e_cauchy_s_22 = 202,
    e_cauchy_s_13 = 203,
    e_cauchy_s_23 = 204,
    e_cauchy_s_33 = 205
  };
  enum {
    e_von_mises_s = 210
  };

  /// Electric potential
  enum {
    e_grad_potential_1 = 300,
    e_grad_potential_2 = 301,
    e_grad_potential_3 = 302
  };

  /// Offset value used to distinguish between output request for e.g. resonance and antiresonance EVPs in _EddBimorphPiezoNormalModes
  enum {
    e_post_offset = 50
  };

  /// Eigen vector
  enum {
    e_eigvec_1 = 400,
    e_eigvec_2 = 401,
    e_eigvec_3 = 402,
    e_sym_grad_eigvec_11 = 403,
    e_sym_grad_eigvec_12 = 404,
    e_sym_grad_eigvec_22 = 405,
    e_sym_grad_eigvec_13 = 406,
    e_sym_grad_eigvec_23 = 407,
    e_sym_grad_eigvec_33 = 408,

    e_pseudo_stress_11 = 410,
    e_pseudo_stress_12 = 411,
    e_pseudo_stress_22 = 412,
    e_pseudo_stress_13 = 413,
    e_pseudo_stress_23 = 414,
    e_pseudo_stress_33 = 415,

    e_eigpotential = 420,
    e_grad_eigpotential_1 = 421,
    e_grad_eigpotential_2 = 422,
    e_grad_eigpotential_3 = 423
  };

  /// Energies
  enum {
    e_strain_energy = 1000,
    e_electric_energy = 1001,
    e_total_energy = 1002,
  };

  /// Other nodal data related
  enum {
    e_compliance_topo = 2000,
  };


/// string to vector of int
  void string_to_vector_of_int(const std::string &s,
                               std::vector<int> &int_vec);

/// string to vector of double
  void string_to_vector_of_double(const std::string &s,
                                  std::vector<double> &double_vec);

/// string to vector of strings
  void string_to_vector_of_strings(const std::string &s,
                                   std::vector<std::string> &string_vec);

  /// Type-safe signum function, returns -1,0,1
  template<typename T>
  inline int sgn(T val) { return (T(0) < val) - (val < T(0)); }

  std::string center(const std::string, const int);

  class Errors {
    /**
     * @brief This class is to compute the relative error, which is used in Newton Raphson scheme. 
     * The class is Initialized with error in the first iteration, subsequently in later iterations given an error value, it can return
     * normalized error wrt the error in first iteration.
     */
  private:
    double error_first_iter = 0.0;
    bool initialized = false;
  public:
  /**
   * @brief Initialize error.
   * 
   * @param error 
   */
    inline void Initialize(double error) {
      if (error == 0.0)
        throw std::runtime_error ("First iteration error cannot be 0.0 ");
      else {
        if (!initialized) {
          error_first_iter = error;
          initialized = true;
        } else
          std::cerr << "Already the error is initialized." << std::endl;
      }
    }

    /**
     * @brief Function to get the Normalized Error. 
     * 
     * @param error 
     * @return double 
     */
    inline double GetNormalizedError(double error) {
      if (initialized)
        return error / error_first_iter;
      else {
        std::cerr << "First iteration error not initialized, so cannot Normalize." << std::endl;
        return 1e9;
      }
    }

    /**
     * @brief Reset the error.
     * 
     */
    inline void Reset() {
      error_first_iter = 0.0;
      initialized = false;
    }
  };

  namespace DebugUtilities {
    template<typename VecType>
    void PrintVector(const VecType &, const std::string &);

    void PrintVectorDouble(const Vector<double> &, const std::string &);

  }

  /**
    * @brief Function to write an Abacus script to generate inp file for the next optimization run.
    * To run the script use `<Abaqus binary> cae noGUI=<script.py>`
    *
    * @param dof_handler_shape - DoFHandler object containing shape information
    * @param dest_path - path to the destination folder
    * @param analysis_name - user defined name
    */
  template<int dim>
  void WriteAbaqusScript(const DoFHandler<dim-1, dim> &dof_handler_shape, std::string dest_path, std::string analysis_name);

}// end of namespace StructuralOptimization

//int ierr = MPI_Barrier(mpi_communicator);
//AssertThrowMPI(ierr);
//std::cout << __FILE__<< ":" << __LINE__ << "|" << "P:" << this_mpi_process<<" CDEV " << std::endl;
//WAIT_STOP
//ierr = MPI_Barrier(mpi_communicator);
//AssertThrowMPI(ierr);

//{
//std::vector<std::string> solution_names;
//switch (dim)
//{
//case 2:
//solution_names.push_back("u_x");
//solution_names.push_back("u_y");
//break;
//
//case 3:
//solution_names.push_back("u_x");
//solution_names.push_back("u_y");
//solution_names.push_back("u_z");
//break;
//
//default:
//Assert(false, ExcNotImplemented() );
//}
//
//DataOut<dim, DoFHandler<dim>> data_out;
//data_out.attach_dof_handler(_dof_handler_base);
//
//data_out.add_data_vector(_solution,  solution_names);
//
//Vector<float>   subdomain(_tria_base.n_active_cells());
//for (unsigned int i=0; i<subdomain.size(); i++)
//subdomain(i) = _tria_base.locally_owned_subdomain();
//
//data_out.add_data_vector(subdomain,"subdomain");
//
//data_out.build_patches();
//
//const std::string filename = ("solution."   +
//        Utilities::int_to_string(
//                _tria_base.locally_owned_subdomain(),4));
//std::ofstream output((filename + ".vtu").c_str());
//data_out.write_vtu(output);
//
//if (this_mpi_process == 0 )
//{
//std::vector<std::string>    filenames;
//for (unsigned int i=0; i < n_mpi_processes; i++)
//{
//filenames.push_back("solution." +
//Utilities::int_to_string(i,4) +
//".vtu");
//}
//std::ofstream master_output("solution.pvtu");
//data_out.write_pvtu_record(master_output, filenames);
//}
//}
//
//std::filebuf fb_mat;
//fb_mat.open ("matrix"+Utilities::int_to_string(this->this_mpi_process)+".mat",std::ios::out);
//std::ostream os_mat(&fb_mat);
//_system_matrix.print(os_mat);
//fb_mat.close();


//{
//  if(newton_iteration == 0)
//  {
//    std::cout << "P:" << this_mpi_process <<" Doing the _residual == 0, DEBUG output: Resedual = " << result << std::endl;
//    if(this_mpi_process==0){
//      DataOut<dim-1, DoFHandler<dim - 1, dim>> data_out;
//      data_out.attach_dof_handler(this->_dof_handler_shape);
//
//      data_out.build_patches();
//
//      const std::string filename = (this->data.destination_path + "shape_with_0_res");
//      std::ofstream output((filename + ".vtu").c_str());
//      data_out.write_vtu(output);
//    }
//
//    DataOutput<dim, BaseTriaType<dim>> base_output(this->data, mpi_communicator, this->compute_timer, "_with_0_res", this->_tria_base);
//
//    std::vector<std::string> rhs_names(dim, "rhs"), res_names(dim, "res"), dof_names(dim,"dof");
//    std::vector<DataComponentInterpretation::DataComponentInterpretation>
//            rhs_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector),
//            res_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector),
//            dof_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
//
//    base_output.template PushDataName<TrilinosWrappers::MPI::Vector>(this->_system_rhs, rhs_names,
//                                                                    rhs_data_component_interpretation);
//
//    base_output.template PushDataName<TrilinosWrappers::MPI::Vector>(_residual, res_names,
//                                                                    res_data_component_interpretation);
//
//    base_output.template PushDataName<TrilinosWrappers::MPI::Vector>(_dirichlet_dofs, dof_names,
//                                                                    dof_data_component_interpretation);
//
//    base_output.template WriteDataOutput<DoFHandler<dim>>(this->_dof_handler_base);
//
//  }
//}
//}
