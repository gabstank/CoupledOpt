#pragma once

// C++ headers

// Deal.II headers

// Project headers
#include <BVP_.h>
#include <Material.h>

namespace StructuralOptimization {

  template<int dim>
  class ElasticityLin : public BVP_<dim> {
  public:
    ElasticityLin(Parameter &, Mesh<dim> &);

    ~ElasticityLin();

    virtual void Run(const unsigned int mode = 0) override;

    virtual Vector<double> GetPostprocessingData(unsigned int &data_flag) override;

  private:

    void _SetupSystem();

    void _AssembleSystem(const unsigned int mode = 0);

    void _AssembleRhs();

    void _SolveSystem();

    void _OutputResults(const unsigned int cycle);

    FESystem<dim> _fe;

    MaterialLinElastic<dim> _mat;

    void _SetupQPointHistory();

    void _UpdateQPointHistory(const unsigned int mode = 0);

    void _AssembleSmoothingRhs(unsigned int);

    void _UpdatePostprocessingData(unsigned int);

    void _Postprocess(const unsigned int mode = 0);

    // Struct that stores the information at a specific quadrature point
    struct QPointHistory {
      SymmetricTensor<2, dim> sym_grad_U;
      double strain_energy;
    };
    std::vector<QPointHistory> quadrature_point_history;
    //---------------------------------------------------

    DataOutput<dim, DomainParallelTriaType<dim>, DoFHandler<dim>> bvp_output;

    /**
    * @brief Struct to hold all the post processing data.
    */
    struct PostprocessingData {
      Vector<double> domain_strain11;
      Vector<double> domain_strain12;
      Vector<double> domain_strain22;
      Vector<double> domain_strain13;
      Vector<double> domain_strain23;
      Vector<double> domain_strain33;
      Vector<double> domain_stress11;
      Vector<double> domain_stress12;
      Vector<double> domain_stress22;
      Vector<double> domain_stress13;
      Vector<double> domain_stress23;
      Vector<double> domain_stress33;
      Vector<double> domain_strain_energy;
      Vector<double> domain_vm_stress;
    } postprocessing_data;

  }; // end of ElasticityLin class

} // end of StructuralOptimization namespace
