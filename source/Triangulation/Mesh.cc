// C++ headers

// Deal.II headers

// Project headers
#include <Mesh.h>

namespace StructuralOptimization {

  template<int dim>
  Mesh<dim>::Mesh(Parameter &_par)
      :
      mpi_communicator(_par.mpi_communicator),
      n_mpi_processes(_par.n_mpi_processes),
      this_mpi_process(_par.this_mpi_process),
      pcout(_par.pcout),
      compute_timer(_par.compute_timer),
      data(_par.data),
      domain(std::make_unique<DomainTriaType<dim>>() ),
      domain_dof_handler(*domain),
      shape_dof_handler(shape),
      domain_output(_par.data, _par.mpi_communicator, _par.compute_timer, "domain", *domain),
      shape_output(_par.data, _par.mpi_communicator, _par.compute_timer, "shape", shape) {

    if (_par.data.bvp_type == "bimorphPiezoStatic" || _par.data.bvp_type == "bimorphPiezoNormalModes" || _par.data.bvp_type == "electroElasticityLin") {
      base = std::make_unique<BaseTriaTypeShared<dim>>(mpi_communicator,
                                                       typename Triangulation<dim>::MeshSmoothing(
                                                           Triangulation<dim>::smoothing_on_refinement
                                                           | Triangulation<dim>::smoothing_on_coarsening));
      domain_parallel = std::make_unique<DomainParallelTriaTypeTypeShared<dim>>(mpi_communicator,
                                                       typename Triangulation<dim>::MeshSmoothing(
                                                           Triangulation<dim>::smoothing_on_refinement
                                                           | Triangulation<dim>::smoothing_on_coarsening));
    }
    else {
      base = std::make_unique<BaseTriaTypeDistributed<dim>>(mpi_communicator,
                                                            typename Triangulation<dim>::MeshSmoothing(
                                                                Triangulation<dim>::smoothing_on_refinement
                                                                | Triangulation<dim>::smoothing_on_coarsening));
      domain_parallel = std::make_unique<DomainParallelTriaTypeDistributed<dim>>(mpi_communicator,
                                                            typename Triangulation<dim>::MeshSmoothing(
                                                                Triangulation<dim>::smoothing_on_refinement
                                                                | Triangulation<dim>::smoothing_on_coarsening));
    }
    domain_prarallel_output = std::make_unique<DataOutput<dim, DomainParallelTriaType<dim>, DoFHandler<dim>>>(_par.data,
                                                                                                    _par.mpi_communicator,
                                                                                                    _par.compute_timer,
                                                                                                    "domain-parallel",
                                                                                                    *domain_parallel);
    mesh_tracking = std::make_unique<MeshTracking<dim>>(_par.data,
                                                        _par.mpi_communicator, _par.this_mpi_process, _par.pcout,
                                                        _par.compute_timer, *base, shape, base_dof_handler);

  }

// destructor
  template<int dim>
  Mesh<dim>::~Mesh() {
    domain_dof_handler.clear();
    domain_parallel_dof_handler.clear();
    shape_dof_handler.clear();
    base_dof_handler.clear();
  }

  template<int dim>
  void Mesh<dim>::SetupMesh() {
    if (data.geometry_name == "hyper_rectangle")
      mesh_setup.CreateHyperRectangleDomain(*this);
    else if (data.geometry_name == "graded_hyper_rectangle")
      mesh_setup.CreateGradedHyperRectangleDomain(*this);
    else if (data.geometry_name == "hyper_cube_with_hole")
      mesh_setup.CreateHyperCubeWithHoleDomain(*this);
    else if (data.geometry_name == "abaqus")
      mesh_setup.CreateAbaqusDomain(*this);
    else throw std::runtime_error("Invalid geometry, please fill the geometry parameter correctly");

    domain_parallel_dof_handler.reinit(*domain_parallel);
    domain_parallel_dof_handler.distribute_dofs(FE_Q<dim>(1));

    mesh_setup.CreateShape(*this);

    mesh_setup.CreateHyperRectangleBase(*this);

    mesh_tracking->RunTracking();

    // printing mesh statistics
    pcout << std::endl;

    pcout << "Domain: No of cells: " << domain->n_active_cells() << " | No of vertices: " << domain->n_vertices()
          << std::endl;
    pcout << "Shape: No of cells: " << shape.n_active_cells() << " | No of vertices: " << shape.n_vertices()
          << std::endl;

    std::vector<unsigned int> domain_cells_info = Utilities::MPI::gather(mpi_communicator, domain_parallel->n_active_cells());
    std::vector<unsigned int> domain_vertices_info = Utilities::MPI::gather(mpi_communicator, domain_parallel->n_vertices());

    pcout << "Domain Parallel: No of cells: ";
    for(const auto& cell_info : domain_cells_info)
      pcout << cell_info << " , ";
    pcout <<  " | No of vertices: ";
    for(const auto& vert_info : domain_vertices_info)
      pcout << vert_info << " , ";
    pcout << std::endl;

    std::vector<unsigned int> base_cells_info = Utilities::MPI::gather(mpi_communicator, base->n_active_cells());
    std::vector<unsigned int> base_vertices_info = Utilities::MPI::gather(mpi_communicator, base->n_vertices());

    pcout << "Base: No of cells: ";
    for(const auto& cell_info : base_cells_info)
      pcout << cell_info << " , ";
    pcout <<  " | No of vertices: ";
    for(const auto& vert_info : base_vertices_info)
      pcout << vert_info << " , ";
    pcout << std::endl;

    pcout << std::endl;

    PROJ_MPI_BARRIER

  }

  template<int dim>
  void Mesh<dim>::OutputMesh() {
    TimerOutput::Scope t(compute_timer, "Mesh<dim>::OutputMesh");
    domain_output.WriteDataOutput();
    domain_prarallel_output->WriteDataOutput();
    shape_output.WriteDataOutput();
  }

  template<int dim>
  void Mesh<dim>::GetShape(DoFHandler<dim - 1, dim> &dof_handler_shape, Vector<double> &current_shape) {
    unsigned int dof_index, vertex_index;
    current_shape.reinit(dof_handler_shape.n_dofs());
    std::vector<bool> vertex_touched(dof_handler_shape.n_dofs());

    for (const auto &cell: dof_handler_shape.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = cell->vertex_index(v);

        if (!vertex_touched[vertex_index]) {
          for (unsigned int d = 0; d < dim; ++d) {
            dof_index = cell->vertex_dof_index(v, d);

            current_shape[dof_index] = cell->vertex(v)[d];
          }
          vertex_touched[vertex_index] = true;
        }
      }
    }
  }

  template<int dim>
  void Mesh<dim>::GetZeroLevelShape(DoFHandler<dim - 1, dim> &dof_handler_shape, Vector<double> &current_shape) {
    unsigned int dof_index, vertex_index;
    current_shape.reinit(dof_handler_shape.n_dofs());
    std::vector<bool> vertex_touched(dof_handler_shape.n_dofs());

    for (const auto &cell: dof_handler_shape.active_cell_iterators_on_level(0)) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = cell->vertex_index(v);

        if (!vertex_touched[vertex_index]) {
          for (unsigned int d = 0; d < dim; ++d) {
            dof_index = cell->vertex_dof_index(v, d);

            current_shape[dof_index] = cell->vertex(v)[d];
          }
          vertex_touched[vertex_index] = true;
        }
      }
    }
  }

  template<int dim>
  void Mesh<dim>::WriteGeoFile(DoFHandler<dim - 1, dim> &dof_handler_shape, int iter) {

    unsigned int vertex_index;
    std::vector<bool> vertex_touched(dof_handler_shape.n_dofs(), false);
    Point<dim> vertex;

    // To write a geo file to restart the simulation.
    std::ostringstream filename;
    if (iter == -1)
      filename << data.destination_path << data.analysis_name << "_restart.prm";
    else
      filename << data.destination_path << "VTUs/" << data.analysis_name << "_restart_" << iter << ".prm";
    std::ofstream output(filename.str().c_str());

    for (const auto &cell: dof_handler_shape.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = cell->vertex_index(v);
        vertex = cell->vertex(v);

        if (!vertex_touched[vertex_index]) {

          // writing vertex to geo file.
          // the input is written such that it serves as a parameter file during restart. Hence the prefix set.
          if (dim == 2)
            output << "set vertex_index " << vertex_index << " = " << vertex[0] << "," << vertex[1]
                   << std::endl;
          else if (dim == 3)
            output << "set vertex_index " << vertex_index << " = " << vertex[0] << "," << vertex[1] << ","
                   << vertex[2] << std::endl;

          vertex_touched[vertex_index] = true;
        }
      }
    }
    output.close(); // close the .geo file

  }


  template<int dim>
  void Mesh<dim>::MoveShape(DoFHandler<dim - 1, dim> &dof_handler_shape,
                            const Vector<double> &design_update,
                            Vector<double> &before_state,
                            bool write_geo) {

    unsigned int dof_index, vertex_index;
    double perturbation = 0;

    std::vector<bool> vertex_touched(dof_handler_shape.n_dofs());
    before_state.reinit(dof_handler_shape.n_dofs());

    // Move shape to new position based on step length and design_update. Also save the old state
    for (auto &cell: dof_handler_shape.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = cell->vertex_index(v);

        if (!vertex_touched[vertex_index]) {
          for (unsigned int d = 0; d < dim; ++d) {
            dof_index = cell->vertex_dof_index(v, d);

            perturbation = design_update[dof_index];
            before_state[dof_index] = cell->vertex(v)[d];

            cell->vertex(v)[d] += perturbation;
          }
          vertex_touched[vertex_index] = true;
        }
      }
    }

    // If traction method isn't used
    // check if a shape has moved out of bounding box and project it back to bounding box.
    if (!data.traction_method) {
      vertex_touched.assign(dof_handler_shape.n_dofs(), false);
      for (auto &cell: dof_handler_shape.active_cell_iterators()) {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
          vertex_index = cell->vertex_index(v);
          if (!vertex_touched[vertex_index]) {
            // x
            if (cell->vertex(v)[0] > data.bb_x[1]) {
              cell->vertex(v)[0] = data.bb_x[1];
              dof_index = cell->vertex_dof_index(v, 0);
            } else if (cell->vertex(v)[0] < data.bb_x[0]) {
              cell->vertex(v)[0] = data.bb_x[0];
              dof_index = cell->vertex_dof_index(v, 0);
            }
            // y
            if (cell->vertex(v)[1] > data.bb_y[1]) {
              cell->vertex(v)[1] = data.bb_y[1];
              dof_index = cell->vertex_dof_index(v, 1);
            } else if (cell->vertex(v)[1] < data.bb_y[0]) {
              cell->vertex(v)[1] = data.bb_y[0];
              dof_index = cell->vertex_dof_index(v, 1);
            }
            // z
            if (dim == 3) {
              if (cell->vertex(v)[2] > data.bb_z[1]) {
                cell->vertex(v)[2] = data.bb_z[1];
                dof_index = cell->vertex_dof_index(v, 2);
              } else if (cell->vertex(v)[2] < data.bb_z[0]) {
                cell->vertex(v)[2] = data.bb_z[0];
                dof_index = cell->vertex_dof_index(v, 2);
              }
            }
            vertex_touched[vertex_index] = true;
          }
        }
      }
    }

    PROJ_MPI_BARRIER
    if (write_geo && this->this_mpi_process == 0)
      WriteGeoFile(dof_handler_shape);
    PROJ_MPI_BARRIER

  }

  template<int dim>
  void Mesh<dim>::UpdateShape(DoFHandler<dim - 1, dim> &dof_handler_shape,
                              const Vector<double> &desired_shape,
                              unsigned int iter,
                              bool write_geo) {
    unsigned int vertex_index, dof_index;
    std::vector<bool> vertex_touched(dof_handler_shape.n_dofs());
    Point<dim> vertex;

    for (const auto &cell: dof_handler_shape.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = cell->vertex_index(v);
        vertex = cell->vertex(v);

        if (!vertex_touched[vertex_index]) {

          for (unsigned int d = 0; d < dim; ++d) {
            dof_index = cell->vertex_dof_index(v, d);
            cell->vertex(v)[d] = desired_shape[dof_index];
          }
          vertex_touched[vertex_index] = true;
        }
      }
    }
    PROJ_MPI_BARRIER
    if (write_geo && this->this_mpi_process == 0)
      WriteGeoFile(dof_handler_shape, iter);
    PROJ_MPI_BARRIER

  }


  template<int dim>
  void Mesh<dim>::RefineShape() {
    shape.refine_global(1);

    unsigned int cell_id = 0;

    // setup boundary id data to be used by the output function
    Vector<double> boundary_ids_shape(shape.n_active_cells());

    for (auto &shape_cell : shape.active_cell_iterators()) {
      boundary_ids_shape[cell_id] = shape_cell->material_id();
      ++cell_id;
    }

    shape.prepare_coarsening_and_refinement();
    shape.execute_coarsening_and_refinement();

    // pushing data to output vector
    std::vector<std::string> data_names(1, "boundary_id");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(DataComponentInterpretation::component_is_scalar);
    shape_output.template PushDataName<Vector<double>>(boundary_ids_shape, data_names,
                                                       data_component_interpretation,
                                                       &shape_dof_handler);
  }


  template<int dim>
  void Mesh<dim>::ResetBase(DoFHandler<dim> &dof_handler_base) {
    dof_handler_base.clear();
    base->clear();
    mesh_setup.CreateHyperRectangleBase(*this);
  }



  template<>
  void Mesh<2>::MergePolygonToShapeMesh(const std::vector<std::vector<Point<2>>> &vec_polygons) {

    std::vector<ShapeTriaType<2>> vec_poly_tria(vec_polygons.size());
    unsigned int poly_counter =0;
    for(const auto& polygon : vec_polygons) {
      ++hole_material_id;
      const unsigned int n_poly_points = polygon.size(), n_poly_cells = polygon.size();

      //Generate Triangulation description from polygon data to generate mesh
      std::vector<Point<2>> poly_vertices(n_poly_points);
      std::vector<CellData<1>> poly_cells(n_poly_cells, CellData<1>());
      for(unsigned int pi=0; pi < n_poly_points; ++pi){
        poly_vertices[pi] = polygon[pi];
        if(pi != (n_poly_points-1)){
          poly_cells[pi].vertices[0] = pi;
          poly_cells[pi].vertices[1] = pi+1;
        }
        else{
          poly_cells[pi].vertices[0] = pi;
          poly_cells[pi].vertices[1] = 0;
        }
        poly_cells[pi].material_id = hole_material_id;
      }
      ShapeTriaType<2> poly_tria;
      poly_tria.create_triangulation(poly_vertices, poly_cells, SubCellData());

      vec_poly_tria[poly_counter].copy_triangulation(poly_tria);
      ++poly_counter;
      existing_hole_material_id.push_back(hole_material_id);
    } // end of loop over vec polygons

    std::vector<const ShapeTriaType<2>*> vec_poly_tria_ptr;
    for(const auto &tria : vec_poly_tria)
      vec_poly_tria_ptr.push_back(&tria);

    ShapeTriaType<2> flat_shape_tria;
    FlattenMesh(shape, flat_shape_tria);
    vec_poly_tria_ptr.push_back(&flat_shape_tria);

    ShapeTriaType<2> merged_tria;
    GridGenerator::merge_triangulations(vec_poly_tria_ptr, merged_tria);

    shape.clear();
    shape.copy_triangulation(merged_tria);

    shape_dof_handler.clear();
    shape_dof_handler.reinit(shape);
    shape_dof_handler.distribute_dofs(FESystem<1, 2>(FESystem<1, 2>(FE_Q<1, 2>(data.poly_degree), 2)));

  }

  template<>
  void Mesh<3>::MergePolygonToShapeMesh(const std::vector<std::vector<Point<3>>> &vec_polygons) {
    (void) vec_polygons;
    std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
    if(true)
      throw std::runtime_error( message );
  }


  template<>
  void Mesh<2>::RemoveAndMergePolygonToShapeMesh(const std::vector<std::vector<Point<2>>> &vec_polygons_to_merge, const std::vector<unsigned int> &matid_to_remove){

    std::vector<ShapeTriaType<2>> vec_polygons_to_merge_tria(vec_polygons_to_merge.size());
    unsigned int poly_counter =0;
    for(const auto& polygon : vec_polygons_to_merge) {
      ++hole_material_id;
      const unsigned int n_poly_points = polygon.size(), n_poly_cells = polygon.size();

      //Generate Triangulation description from polygon data to generate mesh
      std::vector<Point<2>> poly_vertices(n_poly_points);
      std::vector<CellData<1>> poly_cells(n_poly_cells, CellData<1>());
      for(unsigned int pi=0; pi < n_poly_points; ++pi){
        poly_vertices[pi] = polygon[pi];
        if(pi != (n_poly_points-1)){
          poly_cells[pi].vertices[0] = pi;
          poly_cells[pi].vertices[1] = pi+1;
        }
        else{
          poly_cells[pi].vertices[0] = pi;
          poly_cells[pi].vertices[1] = 0;
        }
        poly_cells[pi].material_id = hole_material_id;
      }
      ShapeTriaType<2> poly_tria;
      poly_tria.create_triangulation(poly_vertices, poly_cells, SubCellData());

      vec_polygons_to_merge_tria[poly_counter].copy_triangulation(poly_tria);
      ++poly_counter;
      existing_hole_material_id.push_back(hole_material_id);
    } // end of loop over vec polygons


    std::vector<const ShapeTriaType<2>*> vec_polygons_to_merge_tria_ptr;
    for(const auto &tria : vec_polygons_to_merge_tria)
      vec_polygons_to_merge_tria_ptr.push_back(&tria);

    ShapeTriaType<2> flat_shape_tria;
    FlattenMesh(shape, flat_shape_tria, matid_to_remove);
    vec_polygons_to_merge_tria_ptr.push_back(&flat_shape_tria);

    ShapeTriaType<2> merged_tria;
    GridGenerator::merge_triangulations(vec_polygons_to_merge_tria_ptr, merged_tria);

    shape.clear();
    shape.copy_triangulation(merged_tria);

    shape_dof_handler.clear();
    shape_dof_handler.reinit(shape);
    shape_dof_handler.distribute_dofs(FESystem<1, 2>(FESystem<1, 2>(FE_Q<1, 2>(data.poly_degree), 2)));

  }

  template<>
  void Mesh<3>::RemoveAndMergePolygonToShapeMesh(const std::vector<std::vector<Point<3>>> &vec_polygons_to_merge, const std::vector<unsigned int> &matid_to_remove){
    (void) vec_polygons_to_merge;
    (void) matid_to_remove;
    std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
    if(true)
      throw std::runtime_error( message );
  }


  template<>
  double Mesh<2>::DistanceFromShape(Point<2>& point){
    double min_distance = 1e20;

    Point<2> p1, p2, candidate;
    std::vector<Point<2> > candidates;
    double s=0;
    double distance;

    candidates.clear();

    for(const auto& run_shape : shape.active_cell_iterators()){

      p1 = run_shape->vertex(0);
      p2 = run_shape->vertex(1);

      candidates.push_back(p1);
      candidates.push_back(p2);
      s = (point-p1)*(p2-p1) / ((p2-p1)*(p2-p1));

      if(s>0 && s<1){
        candidate = p1 + s*(p2-p1);
        candidates.push_back(candidate);
      }

    }

    candidate=candidates[0];
    for(unsigned int k=0; k<candidates.size(); ++k){
      distance = point.distance(candidates[k]);

      if(distance<=min_distance){
        candidate = candidates[k];
        min_distance = distance;
      }

    }

    return min_distance;
  }

  // distance_to_shape [dim=3]
  // http://stackoverflow.com/questions/9605556/how-to-project-a-3d-point-to-a-3d-plane
  // http://geomalgorithms.com/a06-_intersect-2.html
  template<>
  double Mesh<3>::DistanceFromShape(Point<3>& point){

    double min_distance;

    Point<3> candidate;
    Point<3> vert, proj;
    Point<3> base, edge_a, edge_b, ray, w;
    Tensor<1,3> unit_normal;
    std::vector<Point<3> > candidates;
    double distance, proj_dist;
    double uu, uv, vv, wu, wv, D;
    double bary_s, bary_t;
    const double tol = 1.e-3;
    unsigned int vertex_index;
    std::vector<bool> vertex_touched(shape.n_vertices());

    candidates.clear();

    // loop over all triangles - project point onto triangle plane - check if projection is within triangle
    for(const auto& run_shape : shape.active_cell_iterators()){

      // mark all vertices as candidates
      for(unsigned int v=0; v<GeometryInfo<3>::vertices_per_face; ++v){
        vertex_index = run_shape->vertex_index(v);

        if(!vertex_touched[vertex_index]){
          vert = run_shape->vertex(v);
          candidates.push_back(vert);
          vertex_touched[vertex_index] = true;
        }
      }

      base = run_shape->vertex(0);

      for(unsigned int tria=0; tria<2; ++tria){

        if(tria==0){
          edge_a = run_shape->vertex(1);
          edge_b = run_shape->vertex(3);
        }
        else{
          edge_a = run_shape->vertex(3);
          edge_b = run_shape->vertex(2);
        }

        edge_a -= base;
        edge_b -= base;

        //cross_product(unit_normal,edge_a,edge_b);
        unit_normal = cross_product_3d(edge_a,edge_b);
        unit_normal /= unit_normal.norm();

        ray = point;
        ray -= base;

        proj_dist = ray * unit_normal;
        proj = point;
        proj -= proj_dist * unit_normal;

        w = proj;
        w -= base;
        uu = edge_a * edge_a;
        uv = edge_a * edge_b;
        vv = edge_b * edge_b;
        wu = w * edge_a;
        wv = w * edge_b;
        D = uv * uv - uu * vv;

        bary_s = (uv * wv - vv * wu) / D;
        if(bary_s < -tol || bary_s > 1.0+tol)
          continue;

        bary_t = (uv * wu - uu * wv) / D;
        if(bary_t < -tol || (bary_s + bary_t) > 1.0+tol)
          continue;

        candidates.push_back(proj);

      }

    }

    // check minimum distance
    min_distance = 1.0e20;
    candidate=candidates[0];

    for(unsigned int k=0; k<candidates.size(); ++k){
      distance = point.distance(candidates[k]);

      if(distance<=min_distance){
        candidate = candidates[k];
        min_distance = distance;
      }

    }

    return min_distance;
  }

  template <int dim>
  bool Mesh<dim>::AdaptiveShapeRefinement(bool cos_refinement) {

    if(dim == 3)
      throw std::runtime_error("Adaptive shape refinement works only for dim == 2");

    bool refined = false;

    // In the first call, calculate the average shape element size
    if(avg_shape_el_size == 0.0) {
      double total_measure = 0.0;
      unsigned int count = 0;
      for (const auto& cell : shape.active_cell_iterators()) {
        total_measure += cell->measure();
        ++count;
      }
      avg_shape_el_size = total_measure / count;
    }

    double size_threshold = data.adaptive_shape_ref_size_threshold;
    double curvature_threshold = data.adaptive_shape_ref_cos_threshold;

    for (const auto& cell : shape.active_cell_iterators()) {

      // Skip non-design IDs
      if(std::find(data.non_design_id.begin(), data.non_design_id.end(), (int)cell->material_id()) != data.non_design_id.end())
        continue;

      // Check for size threshold
      if(cell->measure() > size_threshold * avg_shape_el_size) {
        cell->set_refine_flag();
        refined = true;
        continue;
      }

      if(!cos_refinement)
        continue;

      Tensor<1, dim> normal, adj_normal;
      if(dim == 2) {
        normal[0] = cell->vertex(0)[1] - cell->vertex(1)[1];
        normal[1] = cell->vertex(1)[0] - cell->vertex(0)[0];
        normal /= normal.norm();
      }
      if (dim == 3) {
        throw std::runtime_error("Curvature refinement not implemented for dim == 3");
      }
      std::vector<typename ShapeTriaType<dim>::active_cell_iterator> active_neighbors;
      GridTools::get_active_neighbors<ShapeTriaType<dim>>(cell, active_neighbors);

      // We refine only if all neighbors are at angle
      // Ver 2: Refine is at least one neighbor is at an angle
      bool cos_refine = false;

      for (const auto& adj_cell : active_neighbors) {

        // Skip non-design IDs
        if(std::find(data.non_design_id.begin(), data.non_design_id.end(), (int)adj_cell->material_id()) != data.non_design_id.end())
          continue;

        if (dim == 2) {
          adj_normal[0] = adj_cell->vertex(0)[1] - adj_cell->vertex(1)[1];
          adj_normal[1] = adj_cell->vertex(1)[0] - adj_cell->vertex(0)[0];
          adj_normal /= adj_normal.norm();
        }
        if (dim == 3) {
          throw std::runtime_error("Curvature refinement not implemented for dim == 3");
        }
        // Avoid too tiny elements
        //if (normal * adj_normal > curvature_threshold || cell->measure() < 0.4 * avg_shape_el_size) {
        if (normal * adj_normal < curvature_threshold && cell->measure() > data.adaptive_shape_ref_min_size_ratio * avg_shape_el_size) {
          cos_refine = true;
        }
      }
      if(cos_refine) {
        cell->set_refine_flag();
        refined = true;
      }
    }

    shape.prepare_coarsening_and_refinement();
    shape.execute_coarsening_and_refinement();

    return refined;
  }

  template <int dim>
  void Mesh<dim>::FlattenMesh(const ShapeTriaType<dim> &shape_tria, ShapeTriaType<dim> &result, std::vector<unsigned int> matid_to_exclude) {

    // Old v ID to New ID
    std::map<int, int> old_v_id_to_new_v_id;
    // New ID to coordinates
    std::vector<Point<dim>> vertices;
    // Actual ID to touch bool
    std::vector<bool> vertex_touched(shape_tria.n_vertices());


    unsigned int n_cells_in_new_tria = 0;
    for (auto & cell : shape_tria.active_cell_iterators()) {
      if (std::find(matid_to_exclude.begin(), matid_to_exclude.end(), cell->material_id()) != matid_to_exclude.end())
        continue;
      ++n_cells_in_new_tria;
    }

    // New Cell ID to new vertices IDs
    std::vector<std::vector<int>> cell_vertices(n_cells_in_new_tria);
    // New Cell ID with data
    std::vector<CellData<dim-1>> cells(n_cells_in_new_tria, CellData<dim-1>());

    unsigned int cell_new_id = 0;
    unsigned int vertex_new_id = 0;
    for (auto & cell : shape_tria.active_cell_iterators()) {
      if(std::find(matid_to_exclude.begin(), matid_to_exclude.end(), cell->material_id()) != matid_to_exclude.end())
        continue;

      // New IDs of vertices
      std::vector<int> vertices_per_cell;
      for (unsigned int j = 0; j < GeometryInfo<dim-1>::vertices_per_cell; ++j) {
        int old_v_id = cell->vertex_index(j);
        if(!vertex_touched[old_v_id]) {
          vertex_touched[old_v_id] = true;
          // Add to map
          old_v_id_to_new_v_id[old_v_id] = vertex_new_id;
          // Add the point
          vertices.push_back(cell->vertex(j));
          ++vertex_new_id;
        }
        vertices_per_cell.push_back(old_v_id_to_new_v_id[old_v_id]);
      }
      cell_vertices[cell_new_id] = vertices_per_cell;
      cells[cell_new_id].material_id = cell->material_id();
      ++cell_new_id;
    }

    for (unsigned int cell_id = 0; cell_id < cell_vertices.size(); ++cell_id) {
      for (unsigned int v_num = 0; v_num < cell_vertices[cell_id].size(); ++v_num) {
        cells[cell_id].vertices[v_num] = cell_vertices[cell_id][v_num];
      }
    }

//    std::cout << __FILE__ << ":" << __LINE__ << " |P:" << this_mpi_process << std::endl;

    result.clear();
    result.create_triangulation(vertices, cells, SubCellData());

//    std::cout << __FILE__ << ":" << __LINE__ << " |P:" << this_mpi_process << std::endl;

  }

  template <int dim>
  bool Mesh<dim>::CheckHoleIntersectionAndMerge(){

    std::map<unsigned int, std::vector<Point<dim>>> hole_id_points;

    return false;

  }

} // end of StructuralOptimization namespace

template
class StructuralOptimization::Mesh<2>;

template
class StructuralOptimization::Mesh<3>;
