// C++ headers

// Deal.II headers

// Project headers
#include <Response_.h>

namespace StructuralOptimization {

  template<int dim>
  Response_<dim>::Response_(Parameter &par_, EddBVP_<dim> &bvp_)
      :
      _data(par_.data),
      _bvp(bvp_) {

  }


} // End of StructuralOptimization namespace

template
class StructuralOptimization::Response_<2>;

template
class StructuralOptimization::Response_<3>;
