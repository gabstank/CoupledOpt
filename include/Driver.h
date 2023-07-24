#pragma once

// C++ headers
#include <iostream>
#include <memory> // smart pointers

// Deal.II headers

// Project headers
#include <Parameter.h>
#include <DataOutput.h>

#include <Mesh.h>

#include <BVP_.h>
#include <ElasticityLin.h>

#include <EddBVP_.h>
#include <EddElasticityLin.h>

#include <ResponseHandler.h>

#include <Optimizer_.h>
#include <AlMoMOptimizer.h>

namespace StructuralOptimization {

  template<int dim>
  class Driver {
    /**
     * @brief Interface class to setup all the run time objects based on the parameter file.
     * 
     */
  public:
    Driver(Parameter &_par);

    ~Driver() = default;

    /// Main function that runs the problem.
    void Run();

  private:
    MPI_Comm &mpi_communicator;
    const unsigned int &n_mpi_processes;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    /**
     * @brief Based on the bvp_name corresponding BVP is initialized in this function.
     * 
     * @param bvp_name 
     */
    void InitializeBVP(const std::string &bvp_name);

    /**
     * @brief Based on the bvp_name corresponding BVP is initialized in this function.
     * 
     * @param bvp_name 
     */
    void InitializeEddBVP(const std::string &bvp_name);

    /**
     * @brief Based on the optimizer_name corresponding Optimizer is initialized in this function.
     * 
     * @param optimizer_name 
     */
    void InitializeOptimizer(const std::string &optimizer_name);

    Parameter &par;

    Mesh<dim> mesh;

    // InitiliseBVP will initilize this pointer
    std::unique_ptr<BVP_<dim>> bvp;
    std::unique_ptr<EddBVP_<dim>> edd_bvp;

    std::unique_ptr<Optimizer_<dim>> opt;

    //------------------------------------------------------------

  };
} // end of StructuralOptimization namespace
