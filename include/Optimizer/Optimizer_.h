#pragma once

// C++ headers
#include <iostream>
#include <memory> // smart pointers

// Deal.II headers

// Project headers
#include <Parameter.h>
#include <DataOutput.h>
#include <Mesh.h>
#include <EddBVP_.h>
#include <EddElasticityLin.h>
#include <ResponseHandler.h>

namespace StructuralOptimization {
  /**
   * @brief Optimizer base class.
   */

  template<int dim>
  class Optimizer_ {
  public:

    /**
     * @brief Construct a new Optimizer_ object. In the constructor the following objects are initialized: 
     * @param par_ 
     * @param mesh_ 
     * @param bvp_ 
     */
    Optimizer_(Parameter &par_, Mesh<dim> &mesh_, EddBVP_<dim> &bvp_);

    /**
     * @brief Destroy the Optimizer_ object
     */
    virtual ~Optimizer_() = default;

    /**
     * @brief Pure virtual Run function.
     */
    virtual void Run() = 0;

  protected:

    void _SetupShape();

    /**
    * @brief Function to write the output to file. Also the base output data is retrived here.
    * @param iter
    */
    void _WriteShapeOutput(const unsigned int &iter);
    void _WriteBaseOutput(const unsigned int &iter);

    /**
     * @brief Function to refine shape mesh and perform remeshing of base triangulation.
     */
    void _ExecuteRefinement();

    /**
     * TODO: write description
     */
    void _TrackDensities();

    /**
     * TODO: write description
     */
    void GenerateShapeHoles();

    /**
    * Function to merge two holes that happen to overlap each other.
    */
    void MergeOverlappingHoles();

    /**
   * Function to move shape to next solid base cell.
   *
   * @return
   */
    bool ProjectShapeToTopologyFeature();

    /**
   *  @brief Function to group cells with pseudo_density less than threshold into islands.
   *  @return Vector of Islands. Each Island is a vector of Point.
   */
    std::vector<std::vector<Point<dim>>> _SegregateCellsToIslands();

    /**
    * Function to form polygons given a cluster of points cluster.
    * @param vec_clusters
    * @param vec_clusters
    * @return polygon_expansion_length
    */
    std::vector<std::vector<Point<dim>>> _ConvertClusterToPolygons(const std::vector<std::vector<Point<dim>>>& vec_clusters, double polygon_expansion_length);

    /**
     * Function to generate convex hull. Given the points in cluster, the function
     * return the vector of points which enclose (form the outline of) the cluster
     * Based on https://www.geeksforgeeks.org/quickhull-algorithm-convex-hull/
     * @param cluster
     * @param hull_expansion_length
     * @return
     */
    std::vector<Point<dim>> _ConvexHullGeneration(const std::vector<Point<dim>> &cluster, double hull_expansion_length, double segment_size);

    void _MapL2Normalize(std::map<CellId, double> &map);

    Vector<double> GenerateVectorFromMap(std::map<CellId, double> data);

    /**
     * Function to compute number of design cells, number of solid cells, number of void cells, and number of grey cells.
     */
    void _ComputeDensityStatistics();

    MPI_Comm &mpi_communicator;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    Data &_data;
    Mesh<dim> &_mesh;
    EddBVP_<dim> &_bvp;
    Regularization<dim> _regularization;

    std::unique_ptr<ResponseHandler<dim>> _response_handler;
    std::unique_ptr<DataOutput<dim - 1, ShapeTriaType<dim>, DoFHandler<dim - 1, dim>>> _shape_output;

    // ------------------- Shape data ------------------------
    std::map<unsigned int, Vector<double>> _current_raw_shape_gradients;
    std::map<unsigned int, Vector<double>> _modified_shape_gradients;
    Vector<double> _shape_descent_direction;
    // -------------------------------------------------------

    // ----------------- Density data ------------------------
    std::map<unsigned int, std::map<CellId, double>> _current_raw_density_gradients;
    std::map<unsigned int, std::map<CellId, double>> _modified_density_gradients;
    std::map<CellId, double> _density_descent_direction;

    std::map<CellId, std::map<CellId, double>> _neighbour_list; /// neighbour list for density approximation
    // -------------------------------------------------------

    unsigned int n_design_cells=1, n_solid_cells=0, n_void_cells=0, n_grey_cells=0;
    std::map<CellId, double> _pseudo_densities; /// pseudo density design variables vector

  public:
    std::unique_ptr<DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>>> _base_output;

  };

} // End of StructuralOptimization namespace
