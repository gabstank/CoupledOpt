#pragma once

// C++ headers

// Deal.II headers

// Project headers
#include <Response_.h>

namespace StructuralOptimization {

  /**
   * Volume response class for shape optimization.
   *
   * The volume functional is defined as:
   *
   * \f$
   * \mathcal{F}^{vol} = \int_{\mathcal{B}} 1 {\: dV}
   * \f$
   *
   * The material derivative of the volume functional is given by:
   *
   * \f$
   * \dot{\mathcal{F}}^{vol} = \int_{\Gamma} 1 \left[ \mathbf{\theta} \cdot \mathbf{n} \right] \: dA
   * \f$
   *
   * @tparam dim
   */
  template<int dim>
  class Volume : public Response_<dim> {
  public:
    /// Constructor
    Volume(Parameter &, EddBVP_<dim> &);

    /// Destructor
    ~Volume();

    /// Volume is self adjoint.
    void RunAdjointBVP() override { /*volume is self adjoint;*/}

    /**
     * The volume response is computed elementwise, as a sum of the volume of the base cells, returned by cell->measure() method.
     * For inner base elements, simply the total cell volume is taken. For the volume of the boundary base elements,
     * each gauss point given by the cut cell quadrature is checked whether it lays inside or outside the shape.
     * If the gauss point lays inside the shape, a cell measure divided by the total number of the gauss points in the cut cell
     * is added to the total volume. If the gauss point lays outside the shape, no contribution to the total volume is
     * added.
     *
     * @return Volume response value
     */
    double GetFunction() override;

    /**
     * Integration of the sensitivity density for a given base cell over the intersected shape geometry.
     *
     * @param cell - base cell for which the gradient has to be integrated
     * @param segment_p1 - (2D) first point of the integration segment
     * @param segment_p2 - (2D) second point of the integration segment
     * @param subtriangulation - (3D) a collection of subtriangles that define the intersection of the
     * considered adjacent shape cell and the base cell
     * @param v1 - (3D) first vertex of the adjacent shape cell tria ("half" of the adjacent shape cell).
     * Does not consider intersection
     * @param v2 - (3D) second vertex, see above.
     * @param v3 - (3D) third vertex, see above.
     * @param anchor - the node for which sensitivity is computed, equivalent to:
     *  2D: one of the segment points
     *  3D: one of the adjacent shape cell tria vertices
     * @param shape_cell_measure - (2D) length of the shape cell (line), used for computation of the N_anchor shape function
     * @param[out] base_cell_gradient_density - value to be computed. Scalar sensitivity contribution of:
     *  2D: segment line
     *  3D: subtriangulation
     *
     *  @see ResponseHandler._ComputeShapeGradients()
     */
    void IntegrateBaseCellShapeGradient(typename DoFHandler<dim>::active_cell_iterator &cell,
                                   Point<dim> &segment_p1, Point<dim> &segment_p2,
                                   std::vector<std::vector<Point<dim>>> &subtriangulation,
                                   Point<dim> &v1, Point<dim> &v2, Point<dim> &v3,
                                   Point<dim> &anchor, double &shape_cell_measure,
                                   double &base_cell_gradient_density) override;

    /**
     * Computation of the density sensitivity for a given base cell.
     *
     * @param cell - base cell for which the gradient has to be integrated
     * @param base_cell_gradient_density - value to be computed
     */
    void ComputeCellDensityGradient(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    double &base_cell_gradient_density);

    /// Empty for the volume response.
    void ParseCommonData() override;

  protected:

    MPI_Comm &mpi_communicator;
    const unsigned int &n_mpi_processes;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    double _mpilocal_function_value;
    double _function_value;

    bool first_call = true;
  };

} // End of StructuralOptimization namespace
