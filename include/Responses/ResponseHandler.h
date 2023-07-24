#pragma once

// C++ headers
#include <iostream>

// Deal.II headers
#include <deal.II/base/timer.h>

// Project headers
#include <EddBVP_.h>
#include <Utilities.h>
#include <Parameter.h>
#include <Regularization.h>

#include <Response_.h>
#include <Volume.h>
#include <Compliance.h>

namespace StructuralOptimization {

  using namespace dealii;

  /**
   * This class is meant to construct and store the response objects deriving from the _ShapeResponse class and
   * store and return the response values and response gradients. It provides a public functionality to let
   * the optimizer class interact with the shape responses classes.
   *s
   * @tparam dim
   */
  template<int dim>
  class ResponseHandler {
  public:
    /**
     * Upon construction of an instance of this class, the information about requested responses is read from the
     * parameter data and the corresponding instances of the derived classes of the _ShapeResponse_ type are
     * constructed and stored using the _AddResponse() method.
     *
     * @see _AddResponse()
     */
    ResponseHandler(Parameter &, EddBVP_<dim> &);

    /// Destructor
    virtual ~ResponseHandler() = default;

    /**
     * Public function that calls the adjoint BVP of each response, for which it is necessary.
     */
    void RunAdjointBVP();

    /**
    * @brief Function to get the base output data that are displayed in optimization.
    *
    * @param output
    */
    void GetAdjointOutputData(DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> &output);

    /**
     * Public getter function. Calls _ComputeValues() and returns the values.
     *
     * @see _ComputeValues()
     * @return map of response values
     */
    std::map<unsigned int, double> GetValues();

    /**
     * Public getter function. Calls _ComputeShapeGradients() and returns the raw gradients.
     *
     * @see _ComputeShapeGradients()
     * @return map of gradient vectors
     *
     */
    std::map<unsigned int, Vector<double>> GetRawShapeGradients();

    /**
     * Public getter function. Calls _ComputeDensityGradients() and returns the raw gradients.
     *
     * @see _ComputeDensityGradients()
     * @return map of gradient vectors
     *
     */
    std::map<unsigned int, std::map<CellId, double>> GetRawDensityGradients();

    /**
     * Public getter function. Returns the vertex normals.
     *
     * @return
     */
    Vector<double> GetVertexNormals();

    /**
     * Public getter function. Returns the number of responses stored by this instance of the class.
     *
     * @return
     */
    unsigned int GetNResponses();

    /**
     * Normalization of gradients given as the first argument. The second argument depends on the
     * "Gradient normalization" parameter and decides whether the normalization should be done w.r.t. the
     * l2 norm of the gradient or w.r.t. the magnitude of the gradient values.
     *
     * @see _NormalizeGradientsByL2Norm()
     * @see _NormalizeGradientsByMagnitude()
     *
     * @param[in,out] gradients - map of gradient vectors
     * @param gradient_normalization - method of normalization
     */
    void NormalizeGradients(std::map<unsigned int, Vector<double>> &gradients, std::string gradient_normalization = "");

  protected:

    /**
     * Computes all the response values. For that, it calls each response class to compute its function value.
     */
    void _ComputeValues();

    /**
     * A common assembly function that handles the computation of all the requested shape gradients at once.
     *
     * Since the assembly procedure to integrate the shape gradients over the embedded shape is common for all shape
     * gradients and, on the top of that, being a time-consuming and complicated operation, the assembly procedure
     * takes place once during an optimization iteration and computes all shape gradients at once.
     *
     * At this point, we shall recall that the discretized sensitivities are vectors defined at the nodes,
     * integration of which involves contributions from all the adjacent shape cells (all shape cells that share this node).
     *
     * Therefore, the assembly structure for dim = 2 is organized as follows
     *
     *      Loop over all shape vertices:
     *          Find adjacent shape cells to the current vertex
     *          Compute vertex normal
     *          Loop over adjacent shape cells:
     *              Find the local shape cell vertex matching the sensitivity vertex
     *              Loop over base cells assigned to the same boundary ID as the adjacent shape cell:
     *                  Check if the base cell intersects with the adjacent shape cell:
     *                      Determine intersection segment
     *                      Loop over all requested responses:
     *                          Call the IntegrateBaseCellShapeGradient() method for each response, for the current segment
     *                          Add the segment contribution of the gradient density to the adjacent shape cell gradient density
     *
     *              Loop over all requested responses:
     *                  Multiply the adjacent shape cell gradient density with the vertex normal
     *
     * For dim = 3:
     *
     *      Loop over all shape vertices:
     *          Find adjacent shape cells to the current vertex
     *          Compute vertex normal
     *          Loop over adjacent shape cells:
     *              Find the local shape cell vertex matching the sensitivity vertex
     *              Loop over the two subtriangles of the adjacent shape quad:
     *                  Loop over base cells assigned to the same boundary ID as the adjacent shape cell:
     *                      Check if the base cell intersects with the adjacent shape cell:
     *                          Determine intersection subtriangulation of the shape cell with the base cell
     *                          Loop over all requested responses:
     *                              Call the IntegrateBaseCellShapeGradient() method for each response, for the current subtriangulation
     *                              Add the subtriangulation contribution of the gradient density to the adjacent shape cell gradient density
     *
     *                  Loop over all requested responses:
     *                      Multiply the adjacent shape cell gradient density with the vertex normal
     *
     * @see EddBVP_ - some of the geometry functions used here are defined in this class
     */
    void _ComputeShapeGradients();

    /**
     * Computation of density gradients involves iteration over all base cells (in a distributed manner). For each base
     * cell, sensivitity for each requested response in computed by calling corresponding a method in derived classes
     * of Response_ base class.
     */
    void _ComputeDensityGradients();

    /**
     * For each shape cell its normal vector is computed and stored in a map.
     *
     * @param fe_values_shape - FEValues object returns the cell normal
     * @param[out] shape_cell_normals - a map of shape cells to their respective normal vectors
     */
    void _SetupShapeCellNormals(FEValues<dim - 1, dim> &fe_values_shape,
                                std::map<typename DoFHandler<
                                    dim - 1, dim>::active_cell_iterator, Tensor<1, dim>> &shape_cell_normals);

    /**
     * A vertex normal is computed as an average of the adjacent cell normals. A vector of adjacent cells has to
     * be passed as argument.
     *
     * @param adjacent_shape_cell_normals - a vector of shape cells
     * @param[out] vertex_normal
     */
    void _ComputeCurrentVertexNormal(std::vector<Tensor<1, dim>> &adjacent_shape_cell_normals,
        Tensor<1, dim> &vertex_normal);

    /**
     * Normalization of gradients w.r.t. their L2Norm.
     *
     * @param[in,out] gradients
     */
    void _NormalizeGradientsByL2Norm(std::map<unsigned int, Vector<double>> &gradients);

    /**
     * Normalization of gradients w.r.t. the magnitude of their values.
     *
     * @param[in,out] gradients
     */
    void _NormalizeGradientsByMagnitude(std::map<unsigned int, Vector<double>> &gradients);

    /**
     * This function utilizes the factory pattern to construct the instance of the derived response class.
     *
     * @param id - id of a the response
     * @param response_name - a string indicating which response should be constructed
     */
    void _AddResponse(const unsigned int &id, const std::string &response_name);

    /**
     * Write the shape output for Paraview.
     */
    void _OutputResults();

    MPI_Comm &mpi_communicator;
    const unsigned int &n_mpi_processes;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    // Map of values: scalar values of the response functions
    std::map<unsigned int, double> _values;
    // Map of shape gradients: response gradients (Vector<double>) stored in an std::map
    std::map<unsigned int, Vector<double>> _shape_gradients;
    // Map of density gradients
    std::map<unsigned int, std::map<CellId, double>> _density_gradients;

    Vector<double> _vertex_normals;
    Vector<double> _mpilocal_vertex_normals;

    std::map<unsigned int, std::unique_ptr<Response_<dim>>> _responses;
    std::map<unsigned int, std::string> _responses_names;

    Parameter &_par;
    EddBVP_<dim> &_bvp;
    Regularization<dim> _regularization;
    DataOutput<dim - 1, ShapeTriaType<dim>, DoFHandler<dim - 1, dim>> _shape_output;
  };

} //End of StructuralOptimization namespace
