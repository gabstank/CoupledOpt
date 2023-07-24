// C++ headers

// Deal.II headers

// Project headers
#include <Driver.h>

namespace StructuralOptimization {

  template<int dim>
  Driver<dim>::Driver(Parameter &_par)
      :
      mpi_communicator(_par.mpi_communicator),
      n_mpi_processes(_par.n_mpi_processes),
      this_mpi_process(_par.this_mpi_process),
      pcout(_par.pcout),
      compute_timer(_par.compute_timer),
      par(_par),
      mesh(_par) {
  }


// initializes BVP
  template<int dim>
  void Driver<dim>::InitializeBVP(const std::string &bvp_name) {
    if (bvp_name == "elasticityLin")
      bvp = std::make_unique<ElasticityLin<dim>>(par, mesh);
    else
      std::cerr << __FILE__ << ":" << __LINE__ << "WRONG BVP NAME" << std::endl;
  }

  template<int dim>
  void Driver<dim>::InitializeEddBVP(const std::string &bvp_name) {
    if (bvp_name == "elasticityLin")
      edd_bvp = std::make_unique<EddElasticityLin<dim>>(par, mesh);
    else
      std::cerr << __FILE__ << ":" << __LINE__ << "WRONG BVP NAME" << std::endl;
  }

  template<int dim>
  void Driver<dim>::InitializeOptimizer(const std::string &optimizer_name) {
    if (optimizer_name == "AlMoM")
      opt = std::make_unique<AlMoMOptimizer<dim>>(par, mesh, *edd_bvp);
    else
      std::cerr << __FILE__ << ":" << __LINE__ << "WRONG OPTIMIZER NAME" << std::endl;
  }

  template<int dim>
  void Driver<dim>::Run() {
    bool solved_anything = false;
    // Generate mesh and output mesh.
    mesh.SetupMesh();
    if (par.data.output_mesh)
      mesh.OutputMesh();
    // Standard analysis
    if (par.data.problem_type == "std") {
      solved_anything = true;
      // Initialize BVP
      InitializeBVP(par.data.bvp_type);
      //  bvp->showType();
      bvp->Run();
    }
    // EDD analysis
    if (par.data.problem_type == "edd") {
      solved_anything = true;
      InitializeEddBVP(par.data.bvp_type);
      edd_bvp->Run();
    }
    // Node-based optimization
    if (par.data.problem_type == "shape" || par.data.problem_type == "couple") {
      solved_anything = true;
      InitializeEddBVP(par.data.bvp_type);
      InitializeOptimizer(par.data.optimizer_type);
      opt->Run();
    }
    if (!solved_anything)
      std::cerr << __FILE__ << __LINE__ << " Did not solve any problem, wrong problem or optimization type."
                << std::endl;
  }
} // end of StructuralOptimization namespace

template
class StructuralOptimization::Driver<2>;

template
class StructuralOptimization::Driver<3>;
