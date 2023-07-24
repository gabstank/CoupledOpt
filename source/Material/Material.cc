// C++ headers

// Deal.II headers

// Project headers
#include <Material.h>

namespace StructuralOptimization {

  template<int dim>
  MaterialLinElastic<dim>::MaterialLinElastic(Data &data_)
      :_lambda(data_.mat_elastic_isotropic.lambda), _mu(data_.mat_elastic_isotropic.mu),
       _density(data_.mat_elastic_isotropic.density), _penalty(data_.initial_penalty) {

      _elasticity_tensor = _lambda * StandardTensors<dim>::IxI() + 2 * _mu * StandardTensors<dim>::II();

//    _PrintElasticityTensor();
//    throw std::runtime_error("cdev error");
  }

  template<int dim>
  SymmetricTensor<4, dim> MaterialLinElastic<dim>::GetElasticityTensorSym() {
    return  _elasticity_tensor;
  }

  template<int dim>
  SymmetricTensor<2, dim> MaterialLinElastic<dim>::GetCauchyStress(SymmetricTensor<2, dim> &e, double pseudo_density, double penalty) {
    return std::pow(pseudo_density, penalty) * _elasticity_tensor * e;
  }

  template<>
  void MaterialLinElastic<2>::_PrintElasticityTensor() {
    std::cout << __FILE__ << ":" << __LINE__ << " MaterialLinElastic<2>::_PrintElasticityTensor" << std::endl;
    std::cout << _elasticity_tensor << std::endl;
  }

  template<>
  void MaterialLinElastic<3>::_PrintElasticityTensor() {
    SymmetricTensor<4, 3> elasticity_invert = invert(_elasticity_tensor);

    std::cout << "Young Modulus: " << 1.0 / elasticity_invert[0][0][0][0] << std::endl;
    std::cout << "Poisson's ratio: " << -elasticity_invert[0][0][1][1] / elasticity_invert[0][0][0][0] << std::endl;
  }

  template<int dim>
  double MaterialLinElastic<dim>::GetVonMisesStress(SymmetricTensor<2, dim> &e){
    SymmetricTensor<2, dim> stress = GetCauchyStress(e);
    double vm_stress = std::sqrt( std::pow(first_invariant(stress),2) - 3 * second_invariant(stress) );
    return vm_stress;
  }

  template<int dim>
  double MaterialLinElastic<dim>::GetComplianceSens(SymmetricTensor<2, dim> &e){
    SymmetricTensor<2, dim> S = GetCauchyStress(e);

    double result=0;
    double a,b;
    double tr_e, tr_S, S_e;

    if(dim==2){
      a = (2*_mu + _lambda) / (_lambda + _mu);
      b = (_mu - _lambda) / (4*_mu);

      tr_e = e[0][0] + e[1][1];
      tr_S = S[0][0] + S[1][1];
      S_e = S[0][0]*e[0][0] + S[0][1]*e[0][1] + S[1][0]*e[1][0] + S[1][1]*e[1][1];

      result = a*(S_e - b*tr_e*tr_S);
    }

    return result;

  }

} // end of namespace StructuralOptimization

template
class StructuralOptimization::MaterialLinElastic<2>;

template
class StructuralOptimization::MaterialLinElastic<3>;