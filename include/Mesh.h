#pragma once

// C++ headers
#include <iostream>
#include <typeinfo>

// Deal.II headers
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/base/data_out_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/dof_handler.h>

// Project headers
#include <Utilities.h>
#include <GeometryAlgorithms.h>
#include <Parameter.h>
#include <DataOutput.h>

namespace StructuralOptimization {

  /**
   * All the functionality for the mesh tracking as well as setting the boundary indicators is implemented in this class.
   *
   * The tracking prodecure loops over the base mesh (embedded domain) and queries a set of
   * representative points of each base cell to check whether they are laying inside the shape (embedded domain)
   * or outside the shape. In the current implementation, the representative points of the cell are the nodes,
   * the center and the midpoints between the nodes and the center. For quad elements, that would result in
   * 9 representative points for each cell (4 nodes, 1 center, 4 midpoints) and for hex element in 17
   * representative points (8 nodes, 1 center, 8 midpoints). A cell is marked for refinement if the number
   * of points in the shape neither 0 nor 9 for quad or 17 for hex, i.e. it is neither fully inside the shape
   * nor fully outside the shape.
   *
   * The introduction of midpoints should reduce the risk of having a very thin member of the shape passing
   * through relatively large base cell just between its nodes and a center and not being marked for refinement.
   *
   * The tracking procedure is recursive, which means that the _Tracking() function called recursively until
   * a desired narrow value is obtained.
   *
   * @tparam dim
   */
  template<int dim>
  class MeshTracking {
  public:
    /// Constructor
    MeshTracking(Data &data_,
                 MPI_Comm &mpi_communicator_,
                 const unsigned int &this_mpi_process_,
                 ConditionalOStream &pcout_,
                 TimerOutput &compute_timer_,
                 BaseTriaType<dim> &base_,
                 ShapeTriaType<dim> &shape_,
                 DoFHandler<dim> &dof_handler_);

    /// The only public function. Executes the tracking procedure.
    void RunTracking();

    inline virtual void AddPointsToBeRemoved(const std::vector<Point<dim>>& vec) final{
      for(const auto &p : vec)
        points_to_be_removed.push_back(p);
    }

    unsigned int n_tracking = 0;
    unsigned int refinement_step = 0;
    double min_face_length_base = 0.0;
    double min_shape_length = 0.0;
    std::map<unsigned int, std::vector<typename BaseTriaType<dim>::active_cell_iterator>> boundary_cell_assignment;
  private:
    /**
     * The private method that exucutes the tracking procedure described in the class description.
     */
    void _Tracking();

    /**
     * Shape measure is either a total length of the shape in 2D space or total area of the shape in 3D space.
     * Shape measure is used to compute the narrow, a stopping criterion for the recursive tracking procedure.
     */
    void _CalculateShapeMeasure();

    /**
     * A function that executes the base mesh refinement, based on the refinement flags set in the tracking procedure.
     */
    void _RefineBoundary();

    /**
     * This function is not useful currently. This function was meant to decide upon anisotropic refinement case
     * for the _RefineBoundary() function. Since neither the parallel::shared::Triangulation nor
     * parallel::distributed::Triangulation allows for anisotropic refinement, this function has no use.
     */
    void _GetRefinementCase(unsigned int &);

    /**
     * Sets the boundary indicator for each base cell. That is, builds the boundary_cell_assignment map
     * (a map of boundary IDs to vectors of cells).
     */
    void _SetBoundaryIndicator();

    /**
     * Checks if there is any shape vertex inside the given base cell.
     *
     * @param cell - base cell, for which the check has to be done
     * @return boolean
     */
    bool _CellContainsShapeVertex(typename BaseTriaType<dim>::active_cell_iterator &cell);

    /**
     * Only for dim = 3. Checks if any shape edge interval or shape diagonal interval is contained with the
     * given base cell. This function also builds the vector tmp_edge_boundary_ids, which contains boundary
     * assignment for each shape cell that is cut but this base cell.
     *
     * @param cell - base cell
     * @return
     */
    bool _CellContainsShapeEdge(typename BaseTriaType<dim>::active_cell_iterator &cell);

    /**
     * The boundary indicator is assigned either by the projection of point p onto the shape,
     * in case this projection lays within the shape cell, onto which the p is projected
     * or it is assigned based on the closest vertex of the shape (sharp corner scenario -
     * see _ClosestVertexAtShape()).
     *
     * @param p - point representing a base cell, for which the boundary ID has to be found
     *
     * @see _ClosestVertexAtShape()
     *
     * @return
     */
    std::vector<unsigned int> _GetBoundaryIndicator(Point<dim> &p);

    /**
     * Finds the closest point within the shape to the given point. The closest point at shape could be
     * either a shape vertex (imagine a case in which the given point is close to a sharp corner of the shape)
     * or a projection of the given point onto a shape cell.
     *
     * @param point - the given point, i.e. can be a center of a base cell
     * @return
     */
    Point<dim> _ClosestVertexAtShape(Point<dim> &point);

    /**
     * Tracking output. Prints the mesh at all loops of refinement.
     */
    void _OutputMesh();

    std::vector<unsigned int> tmp_edge_boundary_ids;
    std::vector<unsigned int> tmp_interior_vertices;

    Data &data;
    MPI_Comm &mpi_communicator;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;
    BaseTriaType<dim> &tria_base;
    ShapeTriaType<dim> &tria_shape;
    DoFHandler<dim> &base_dof_handler;

    std::vector<Point<dim>> points_to_be_removed;

    std::map<unsigned int,
        std::vector<typename BaseTriaType<dim>::active_cell_iterator> >
        tracked_cells_per_level;

    Vector<float> boundary_ids_base;

    double _shape_measure = 0.0;

    DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> tracking_output;

  };

  /**
   * The Mesh class owns all the triangulations used for optimization, i.e. base, shape and domain meshes.
   * It is central class that is able to invoke any task with respect to creation, querying or manipulation
   * of the mesh.
   *
   * The functionality for shape update in the optimization process is implemented in this class and consists of
   * GetShape(), UpdateShape(), RefineShape() functions.
   *
   * All of the functionality for mesh creation is gathered in MeshSetup struct, which is a part Mesh class. The
   * MeshSetup class is used by the public function SetupMesh() to generate all the necessary triangulations and
   * setup the boundary IDs.
   *
   * The class contains an instance of the MeshTracking class \a mesh_tracking and calls its public method
   * to perform the tracking procedure.
   *
   * The class also writes the mesh output if requested so.
   *
   * @tparam dim
   */
  template<int dim>
  class Mesh {
  public:
    /**
     * Constructor.
     *
     * @param _par
     */
    Mesh(Parameter &_par);

    /**
     * Destructor.
     */
    ~Mesh();

    /**
     * Invokes the functionality of MeshSetup struct to create the domain, shape and base triangulations. Invokes
     * the tracking procedure as well.
     */
    void SetupMesh();

    /**
     * Writes the domain and shape output if requested.
     */
    void OutputMesh();

    /**
     * Stores the positions of the DoFs of the current shape in the \a current_shape vector.
     *
     * @param dof_handler_shape
     * @param[out] current_shape - the current shape is written there
     */
    void GetShape(DoFHandler<dim - 1, dim> &dof_handler_shape, Vector<double> &current_shape);

    /**
     * Stores the positions of the DoFs of the zero level cells of the current shape in the \a current_shape vector.
     *
     * @param dof_handler_shape
     * @param[out] current_shape - the current shape is written there
     */
    void GetZeroLevelShape(DoFHandler<dim - 1, dim> &dof_handler_shape, Vector<double> &current_shape);

    /**
     * Given a vector of desired DoF positions in the \a desired_shape vector, the function assigns these positions
     * to the shape. Used for resetting the shape with previous configuration.
     *
     * @param dof_handler_shape
     * @param[in] desired_shape - vector of DoF positions
     * @param iter - iteration counter needed for .geo file
     * @param write_geo - boolean - should the restart .geo file be written
     */
    void UpdateShape(DoFHandler<dim - 1, dim> &dof_handler_shape, const Vector<double> &desired_shape, unsigned int iter = 0, bool write_geo = false);

    /**
     * It takes the design update vector and moves the shape in this direction. This function also
     * returns the before state of shape values. When direct filtering regularization is used, this function
     * also checks the bounding box constriants and projects the nodes that violate these constraints.
     *
     * @param dof_handler_shape
     * @param[in] design_update - the design update vector (scaled the step length)
     * @param[out] before_state - the state before design update is stored there
     * @param write_geo - boolean - should the restart .geo file be written
     */
    void MoveShape(DoFHandler<dim - 1, dim> &dof_handler_shape, const Vector<double> &design_update,
        Vector<double> &before_state, bool write_geo = false);

    /**
     * Writes the restart .prm file, which can be used to resume the optimization from this design state.
     *
     * @param dof_handler_shape
     * @param iter
     */
    void WriteGeoFile(DoFHandler<dim - 1, dim> &dof_handler_shape, int iter = -1);

    /**
     * Refines the shape triangulation once.
     */
    void RefineShape();

    /**
     * Clears the DoFHandler of the base mesh, clears the base triangulation and creates a new base triangulation
     * by calling a proper MeshSetup function.
     *
     * @param dof_handler_base
     */
    void ResetBase(DoFHandler<dim> &dof_handler_base);

    /**
     *
     * @param vec_polygons
     */
    void MergePolygonToShapeMesh(const std::vector<std::vector<Point<dim>>> &vec_polygons);

    /**
     * This function will remove polygons with material id mentioned in matid_to_remove and all polygons given in vec_polygons_to_merge.
     * @param vec_polygons_to_merge
     * @param matid_to_remove
     */
    void RemoveAndMergePolygonToShapeMesh(const std::vector<std::vector<Point<dim>>> &vec_polygons_to_merge, const std::vector<unsigned int> &matid_to_remove);

    double DistanceFromShape(Point<dim>& point);

    /**
     *
     *
     * @param cos_refinement: indicates whether curvature refinement should be applied
     * @return
     */
    bool AdaptiveShapeRefinement(bool cos_refinement);

    /**
     * Function to flatten the refined shape triangulation. Flattened mesh will not include material id mentioned in the matid_to_exclude
     * @param shape_tria
     * @param result
     * @param matid_to_exclude
     */
    void FlattenMesh(const ShapeTriaType<dim> &shape_tria, ShapeTriaType<dim> &result, std::vector<unsigned int> matid_to_exclude = {});

    bool CheckHoleIntersectionAndMerge();

    MPI_Comm &mpi_communicator;
    const unsigned int &n_mpi_processes;
    const unsigned int &this_mpi_process;
    ConditionalOStream &pcout;
    TimerOutput &compute_timer;

    Data &data;

    std::unique_ptr<DomainTriaType<dim>> domain; // Standard domain mesh
    std::unique_ptr<DomainParallelTriaType<dim>> domain_parallel; // Parallel domain mesh
    ShapeTriaType<dim> shape; // shape mesh
    std::unique_ptr<BaseTriaType<dim>> base;

    DoFHandler<dim> domain_dof_handler;
    DoFHandler<dim> domain_parallel_dof_handler;
    DoFHandler<dim - 1, dim> shape_dof_handler;
    DoFHandler<dim> base_dof_handler;

    DataOutput<dim, DomainTriaType<dim>, DoFHandler<dim>> domain_output;
    std::unique_ptr<DataOutput<dim, DomainParallelTriaType<dim>, DoFHandler<dim>>> domain_prarallel_output;
    DataOutput<dim - 1, ShapeTriaType<dim>, DoFHandler<dim - 1, dim>> shape_output;

    double avg_shape_el_size = 0.0;

    unsigned int hole_material_id = 100;
    std::vector<unsigned int> existing_hole_material_id;

  public:
    /**
     * MeshSetup is a struct with only functions to create mesh of required geometry.
     * This struct has function to setup mesh for base, domain and shape triangulation.
     * It includes also functionality to setup boundary IDs.
     */
    struct MeshSetup {

    public:
      /**
       * Creates the base mesh.
       *
       * @note Special treatment of the the _BimorphPiezoStatic case. the piezoelectric layers are perfectly
       * aligned with their geometry given in the parameter file, so that the interface between piezoelectric and
       * purely elastic domain is passing exactly between the base elements not through the base elements.
       * @param mesh
       */
      void CreateHyperRectangleBase(Mesh<dim> &mesh);

      /**
       * Creates the hyper rectangle domain mesh. Sets up the boundary IDs.
       *
       * @see _SetupBoundaryIDs()
       *
       * @param mesh
       */
      void CreateHyperRectangleDomain(Mesh<dim> &mesh);

      /**
       * Creates the graded hyper rectangle domain mesh. Sets up the boundary IDs.
       *
       * @see _SetupBoundaryIDs()
       *
       * @note In debug mode, this function often returns a numerical error that the step sizes do not sum to the total
       * dimension. This is a dealii assertion that checks the sum of step sizes with very restrictive tolerance 1e-12,
       * which for some combination of sums of double precision numbers cannot be met.
       *
       * @param mesh
       */
      void CreateGradedHyperRectangleDomain(Mesh<dim> &mesh);

      /**
       * Creates the hyper cube with hole domain mesh. Sets up the boundary IDs.
       *
       * @see _SetupBoundaryIDs()
       *
       * @param mesh
       */
      void CreateHyperCubeWithHoleDomain(Mesh<dim> &mesh);

      /**
       * Reads the Abaqus input file and generates a domain mesh based on it. Sets up the boundary IDs.
       *
       * @see _SetupBoundaryIDs()
       *
       * @param mesh
       */
      void CreateAbaqusDomain(Mesh<dim> &mesh);

      /**
       * Creates the shape triangulation by extracting a boundary triangulation from the domain triangulation.
       * Sets up the boundary IDs of the shape based on the domain boundary IDs.
       *
       * @param mesh
       */
      void CreateShape(Mesh<dim> &mesh);

      /**
       * Updates the shape from the restart .geo file.
       *
       * @param mesh
       */
      void UpdateShapeFromGeo(Mesh<dim> &mesh);

    private:
      /**
       * Returns the boundary ID, which matches the geometry of the given point. Returns 0 if not boundary
       * matches the geometry of the given point.
       *
       * @param data
       * @param[in] bid_counter - a map storing a multiplicity of a boundary ID
       * @param[in] point - a point for which the boundary ID has to be returned
       * @return - boundary ID
       */
      int _GetBoundaryID(Data &data, std::map<int, int> bid_counter, const Point<dim> &point);

      /**
       * Sets up all boundary IDs to the faces of domain cells.
       *
       * @param mesh
       */
      void _SetupBoundaryIDs(Mesh<dim> &mesh);

      /**
       * Conversion from std::vector to Point.
       *
       * @param vec[in]
       * @param p[out]
       */
      void _VectorToPoint(const std::vector<double> &vec, Point<dim> &p);

      ParameterHandler geo_prm;

    } mesh_setup;

    std::unique_ptr<MeshTracking<dim>> mesh_tracking;

  }; // end of Mesh class definition

} // end of StructuralOptimization namespace
