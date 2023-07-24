#pragma once

// C++ headers

// Deal.II headers
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>

// Project headers
#include <Parameter.h>
#include <Utilities.h>

namespace StructuralOptimization {

  template<int dim>
  class MaterialLinElastic {
  public:
    MaterialLinElastic(Data &data_);

    inline SymmetricTensor<4, dim> GetElasticityTensor() { return _elasticity_tensor; }

    SymmetricTensor<4, dim> GetElasticityTensorSym();

    SymmetricTensor<2, dim> GetCauchyStress(SymmetricTensor<2, dim> &e, double pseudo_density=1.0, double penalty=1.0);

    double GetVonMisesStress(SymmetricTensor<2, dim> &);

    inline double GetDensity() { return _density; }

    double GetComplianceSens(SymmetricTensor<2, dim> &e);

    inline SymmetricTensor<4, dim> GetSIMPSens(const double pseudo_density){
      SymmetricTensor<4, dim> eleasticity_tensor_sensitivity = _penalty*std::pow(pseudo_density, _penalty-1.0) * _elasticity_tensor;
      return eleasticity_tensor_sensitivity;
    }

  private:

    void _PrintElasticityTensor();

    const double _lambda, _mu, _density, _penalty;
    SymmetricTensor<4, dim> _elasticity_tensor;

  };

} // end of namespace StructuralOptimization
