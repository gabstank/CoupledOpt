#pragma once

// C++ headers
#include <iostream>

// Deal.II headers
#include <deal.II/base/timer.h>

// Project headers
#include <EddBVP_.h>
#include <Utilities.h>
#include <Parameter.h>

namespace StructuralOptimization {
  using namespace dealii;

  /**
   * Abstract base class. Defines common interface for the responses in shape optimization.
   * The responses are managed by the ResponseHandler class. For more details, see the documentation of the
   * ResponseHandler.
   * See also the documentation for the derived classes.
   *
   * @tparam dim
   */
  template<int dim>
  class Response_ {
  public:
    Response_(Parameter &, EddBVP_<dim> &);

    virtual ~Response_() = default;

    /**
     * Pure virtual function. Returns the value of the response.
     *
     * @return Value of the response.
     */
    virtual double GetFunction() = 0;

    /**
     * Pure virtual function. Defined only in those derived classes, for which running an adjoint BVP is necessary.
     */
    virtual void RunAdjointBVP() = 0;

    /**
     * Function to get output data of adjoint problem.
     * @param output
     */
    virtual void GetAdjointOutputData(DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> &output){
//      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      (void) output;
      return;
    }

    /**
     * Pure virtual function.
     */
    virtual void IntegrateBaseCellShapeGradient(typename DoFHandler<dim>::active_cell_iterator &,
                                           Point<dim> &, Point<dim> &,
                                           std::vector<std::vector<Point<dim>>> &,
                                           Point<dim> &, Point<dim> &, Point<dim> &,
                                           Point<dim> &, double &, double &) = 0;

    /**
     * Pure virtual function.
     */
    virtual void ComputeCellDensityGradient(const typename DoFHandler<dim>::active_cell_iterator &, double &) {};

    /**
     * Pure virtual function. Defined only in the eigenvalue response class.
     */
    virtual void ParseCommonData() = 0;

  protected:

    Data &_data;
    EddBVP_<dim> &_bvp;
  };

} //End of StructuralOptimization namespace
