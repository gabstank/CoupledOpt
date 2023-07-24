#pragma once

// C++ headers

// Deal.II headers

// Project headers
#include <Response_.h>

namespace StructuralOptimization {

  /**
   * Compliance response class for shape optimization.
   *
   * The compliance functional is defined as:
   *
   * \f$
   * \mathcal{F}^{com} = \int_{\mathcal{B}} \mathbf{b} \cdot \mathbf{u} \: dV + \int_{\Gamma^N} \bar{\mathbf{t}}
   * \cdot \mathbf{u} \: dA
   * \f$
   *
   * The material derivative of the compliance functional is given by:
   *
   * \f{eqnarray*}{
   * \dot{\mathcal{F}}^{com} &=& \int_{\Gamma} \left[ 2 \left[ \mathbf{b} \cdot \mathbf{u} \right] - \mathbf{\sigma} :
   * \mathbf{\epsilon} \right] \left[ \mathbf{\theta} \cdot \mathbf{n} \right] \: dA \\
   * &+& \int_{\Gamma^N} \left[ 2 \nabla \left[ \bar{\mathbf{t}} \cdot \mathbf{u} \right] \cdot \mathbf{n} + 2 \left[
   * \bar{\mathbf{t}} \cdot \mathbf{u} \right] H \right] \left[ \mathbf{\theta} \cdot \mathbf{n} \right] \: dA
   * \f}
   *
   * @tparam dim
   */
  template<int dim>
  class Compliance : public Response_<dim> {
  public:
    Compliance(Parameter &, EddBVP_<dim> &);

    ~Compliance();

    /**
     * The total compliance is computed as scalar product of the global rhs and global solution vector.
     *
     * @return Compliance response value
     */
    double GetFunction() override;

    /**
     * Compliance sensitivity is self-adjoint.
     */
    void RunAdjointBVP() override { /*compliance is self adjoint;*/}

    /**
     * Integration of the shape sensitivity density for a given base cell over the intersected shape geometry.
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

    /// Empty for the compliance response.
    void ParseCommonData();

  protected:

    MPI_Comm &mpi_communicator;
    const unsigned int &n_mpi_processes;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    double _mpilocal_strain_energy = 0.0;
    double _strain_energy = 0.0;
    double _function_value;
  };

} // End of StructuralOptimization namespace
