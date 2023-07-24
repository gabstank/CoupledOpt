// C++ headers

// Deal.II headers

// Project headers
#include <Mesh.h>

namespace StructuralOptimization {

  template<int dim>
  MeshTracking<dim>::MeshTracking(Data &data_,
                                  MPI_Comm &mpi_communicator_,
                                  const unsigned int &this_mpi_process_,
                                  ConditionalOStream &pcout_,
                                  TimerOutput &compute_timer_,
                                  BaseTriaType<dim> &base_,
                                  ShapeTriaType<dim> &shape_,
                                  DoFHandler<dim> &dof_handler_)
      :data(data_),
       mpi_communicator(mpi_communicator_),
       this_mpi_process(this_mpi_process_),
       pcout(pcout_),
       compute_timer(compute_timer_),
       tria_base(base_),
       tria_shape(shape_),
       base_dof_handler(dof_handler_),
       tracking_output(data, mpi_communicator_, compute_timer_, "tracking", base_) {
    min_face_length_base = 2 / pow(2.0, data.base_refinements);
  }

  template<int dim>
  void MeshTracking<dim>::_Tracking() {

    if(data.refinement_criterion == "narrow")
      if (_shape_measure == 0.0)
        _CalculateShapeMeasure();

    // Get the min shape cell length
    double current_min_shape_length = tria_shape.begin()->diameter();
    for (const auto &shape_cell : tria_shape.active_cell_iterators()) {
      double current_shape_length = shape_cell->diameter();
      current_min_shape_length = (current_min_shape_length > current_shape_length) ? current_shape_length
                                                                                   : current_min_shape_length;
    }
    min_shape_length = current_min_shape_length;

    // Track boundary and interior. Calculate track measure.
    std::vector<bool> vertex_touched(tria_base.n_vertices());
    std::vector<bool> vertex_inside(tria_base.n_vertices());
    unsigned int total_counter;
    unsigned int v_counter;
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Track measure data
    double _track_measure = 0, _track_measure_thread = 0;
    std::vector<typename BaseTriaType<dim>::active_cell_iterator> level_cells;
    if(data.refinement_criterion == "narrow") {
      tracked_cells_per_level.clear();
      tracked_cells_per_level[tria_base.n_levels()] = level_cells;
    }

    FE_Q<dim> ref_el(data.tracking_precision );

    for (auto base_cell : tria_base.active_cell_iterators()) {
      if (!base_cell->is_locally_owned() && typeid(tria_base) == typeid(BaseTriaTypeDistributed<dim>))
        continue;

      v_counter = 0;

      // Estimate the extreme vertices which will be used for interpolation
      Point<dim> min_vertex(base_cell->vertex(0)), max_vertex(base_cell->vertex(vertices_per_cell-1));
      std::vector<Point<dim>> unit_support_points = ref_el.get_unit_support_points();
      for(const auto& unit_supp_point : unit_support_points) {
        Point<dim> sample_point;
        for (unsigned int dir = 0; dir < dim; ++dir)
          sample_point[dir] = min_vertex[dir] + unit_supp_point[dir]*(max_vertex[dir]-min_vertex[dir]);
        if(GeometryAlgorithms::PointInShape(sample_point, tria_shape,
                                            min_face_length_base))
          ++v_counter;
      }

      // Set boundary ID if only part of the set = (vertices + cell center) are inside shape;
      // Add measure
      if (v_counter > 0 && v_counter < unit_support_points.size()) {
        base_cell->set_material_id(e_boundary_cell);
        if(data.refinement_criterion == "narrow") {
          _track_measure_thread += base_cell->measure();
          tracked_cells_per_level[tria_base.n_levels()].push_back(base_cell);
        }
        ++total_counter;
      }
        // Set interior ID if all vertices and cell center are inside shape
      else if (v_counter == unit_support_points.size()) {
        base_cell->set_material_id(e_inside_cell);
      }
        // Else it is an outside cell
      else
        base_cell->set_material_id(e_outside_cell);
    }

    // A barrier here to ensure all the cells have set the material_id.
    int ierr = MPI_Barrier(mpi_communicator);
    AssertThrowMPI(ierr);
    // Check if any cells were set as boundary
    //int total = Utilities::MPI::sum(total_counter, mpi_communicator);
    //AssertThrow(total > 0, ExcMessage("0 user_flags_set, something wrong in _track_boundary function "));

    // Output mesh if requested
    if (data.output_mesh) {
      _OutputMesh();
    }

    // For coupled optimization, we do not refine boundary
    if(data.problem_type != "couple") {
      // Check narrow
      if (data.refinement_criterion == "narrow") {
        if (typeid(tria_base) == typeid(BaseTriaTypeShared<dim>))
          _track_measure = _track_measure_thread;
        else if (typeid(tria_base) == typeid(BaseTriaTypeDistributed<dim>))
          _track_measure = Utilities::MPI::sum(_track_measure_thread, mpi_communicator);

        // Print narrow
        if (data.verbose)
          pcout << " narrow = " << (_track_measure / _shape_measure) << " > desired narrow = " << data.narrow
                << " || min_face_length_base " << min_face_length_base << " > min_shape_length " << min_shape_length
                << std::endl;

        if ((_track_measure / _shape_measure) > data.narrow || min_face_length_base > 0.75 * min_shape_length) {
          ++n_tracking;
          _RefineBoundary();
          _Tracking();
        }
      } else if (data.refinement_criterion == "fixed") {
        if (refinement_step < data.tracking_refinements) {
          ++refinement_step;
          ++n_tracking;
          _RefineBoundary();
          _Tracking();
        } else if (min_face_length_base > 0.75 * min_shape_length) {
          pcout << "Mesh tracking: Min face length base condition not met." << std::endl;
        }
      } else
        throw std::runtime_error("Wrong refinement criterion.");
    }

    // TODO: Old bubble
    // We get rid of explicit cell removal as the intermediate densities are introduced
    if(false) {
      // mark all the cell surrounding points in vec as outside_cells
      if (points_to_be_removed.size() != 0) {

        for (auto base_cell : tria_base.active_cell_iterators()) {
          if (!base_cell->is_locally_owned() && typeid(tria_base) == typeid(BaseTriaTypeDistributed<dim>))
            continue;

          if (base_cell->material_id() != e_inside_cell)
            continue;

          for (const auto &p : points_to_be_removed) {
            if (base_cell->point_inside(p)) {
              base_cell->set_material_id(e_outside_cell);
//
//            std::cout << __FILE__ << ":" << __LINE__ << "|P:" << this_mpi_process
//                      << " For point p= " << p
//                      << " Setting cell with center: " << base_cell->center() << std::endl;
              break;
            } // if point is inside
          } // loop over points to be removed
        } // loop over base cells.
      } // if point_to_be_removed.size() != 0
    }

  }

  template<>
  void MeshTracking<2>::_CalculateShapeMeasure() {
    for (typename ShapeTriaType<2>::active_cell_iterator shape_cell : tria_shape.active_cell_iterators())
      _shape_measure += shape_cell->measure();
  }

  template<>
  void MeshTracking<3>::_CalculateShapeMeasure() {
    Point<3> v1, v2, tmp_v_x;
    Tensor<1, 3> v_x;
    for (typename ShapeTriaType<3>::active_cell_iterator shape_cell : tria_shape.active_cell_iterators()) {
      v1 = shape_cell->vertex(1);
      v1 -= shape_cell->vertex(0);
      v2 = shape_cell->vertex(2);
      v2 -= shape_cell->vertex(0);
      v_x = cross_product_3d(v1, v2);
      for (unsigned int d = 0; d < 3; ++d) {
        tmp_v_x[d] = v_x[d];
      }
      _shape_measure += pow(tmp_v_x.square(), 0.5); // shape_cell->measure seems not to work..
    }
  }

  template<int dim>
  void MeshTracking<dim>::_RefineBoundary() {

    unsigned int refinement_case = 0;
    _GetRefinementCase(refinement_case);

    for (auto cell: tria_base.active_cell_iterators()) {
      if (!cell->is_locally_owned() && typeid(tria_base) == typeid(BaseTriaTypeDistributed<dim>))
        continue;

      // User flag is set only in anisotropic refinement mode when a base cell is a surface cell
      if (cell->material_id() == e_boundary_cell) {
        cell->set_refine_flag(RefinementCase<dim>(refinement_case));
        if (data.extended_refinement)
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) {

              if (!(cell->neighbor(f)->used() && cell->neighbor(f)->is_active()))
                continue;

              cell->neighbor(f)->set_refine_flag(RefinementCase<dim>(refinement_case));
            }
          }
      }
    }

    // A barrier here to ensure flags have been set by all the proc.
    int ierr = MPI_Barrier(mpi_communicator);
    AssertThrowMPI(ierr);

    tria_base.prepare_coarsening_and_refinement();
    tria_base.execute_coarsening_and_refinement();

    min_face_length_base /= 2.0;
  }

  template<>
  void MeshTracking<2>::_GetRefinementCase(unsigned int &refinement_case) {
    refinement_case = RefinementCase<2>::isotropic_refinement;
  }

  template<>
  void MeshTracking<3>::_GetRefinementCase(unsigned int &refinement_case) {
    refinement_case = RefinementCase<3>::isotropic_refinement;
  }


// _closest_vertex_at_shape [dim=2]
  template<>
  Point<2> MeshTracking<2>::_ClosestVertexAtShape(Point<2> &point) {

    Point<2> candidate;

    Point<2> p1, p2;
    std::vector<Point<2> > candidates;
    double s = 0;
    double tol = 1.e-3;
    double distance, min_distance = 10.0;

    candidates.clear();

    // Loop over all shape cells to gather all candidate points to be the closest point at shape to the given point
    for (auto run_shape : tria_shape.active_cell_iterators()) {

      p1 = run_shape->vertex(0);
      p2 = run_shape->vertex(1);

      // Consider the shape vertices as potentially closest points with shape to the given point.
      candidates.push_back(p1);
      candidates.push_back(p2);
      // Calculate the fraction of the p1,p2 segment at which the projection of the given point onto p1,p2
      // segment takes place
      s = (point - p1) * (p2 - p1) / ((p2 - p1) * (p2 - p1));

      // Add the projection point as the candidate to be the closest point at shape
      if ((s > (0 + tol)) && (s < (1 - tol))) {
        candidate = p1 + s * (p2 - p1);
        candidates.push_back(candidate);
      }
    }

    // Check all candidates for being the closest point at shape to the given point.
    candidate = candidates[0];
    for (unsigned int k = 0; k < candidates.size(); ++k) {
      distance = point.distance(candidates[k]);

      if (distance <= min_distance) {
        candidate = candidates[k];
        min_distance = distance;
      }
    }
    return candidate;
  }

// _closest_vertex_at_shape [dim=3]
  template<>
  Point<3> MeshTracking<3>::_ClosestVertexAtShape(Point<3> &point) {

    Point<3> candidate; // The closest point at shape to be found

    Point<3> vert, proj; // proj - the given point projection onto a selected tria of the shape
    Point<3> base, edge_a, edge_b, ray, w; // Variables used to calculate the projection
    Tensor<1, 3> unit_normal;
    std::vector<Point<3> > candidates; // Vector of all potential candidates to be the closest point at shape
    double distance, min_distance, proj_dist;
    double uu, uv, vv, wu, wv, D;
    double bary_s, bary_t;
    const double tol = 1.e-3;
    unsigned int vertex_index;
    std::vector<bool> vertex_touched(tria_shape.n_vertices());

    candidates.clear();

    // Loop over all triangles - project point onto triangle plane - check if projection is within triangle
    for (auto run_shape : tria_shape.active_cell_iterators()) {

      // Mark all vertices as candidates
      // ToDo mark all shape edges to contain potential closest vertex, if map
      //  to triangle fails and closest shape vertex is not closest point..
      for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_face; ++v) {
        vertex_index = run_shape->vertex_index(v);

        if (!vertex_touched[vertex_index]) {
          vert = run_shape->vertex(v);
          candidates.push_back(vert);
          vertex_touched[vertex_index] = true;
        }
      }

      base = run_shape->vertex(0);

      // Add the given point projections onto two trias of the shape cell as potential candidates
      // only if their projection is within the shape cell
      for (unsigned int tria = 0; tria < 2; ++tria) {

        if (tria == 0) {
          edge_a = run_shape->vertex(1);
          edge_b = run_shape->vertex(3);
        } else {
          edge_a = run_shape->vertex(3);
          edge_b = run_shape->vertex(2);
        }

        edge_a -= base;
        edge_b -= base;

        // Calculate the projection of the given point on the current tria
        //cross_product(unit_normal,edge_a,edge_b);
        unit_normal = cross_product_3d(edge_a, edge_b);
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

        // Do the check is the projection is within the tria
        bary_s = (uv * wv - vv * wu) / D;
        if (bary_s < -tol || bary_s > 1.0 + tol)
          continue;

        bary_t = (uv * wu - uu * wv) / D;
        if (bary_t < -tol || (bary_s + bary_t) > 1.0 + tol)
          continue;

        // Push back the projection to the candidates
        candidates.push_back(proj);

      }

    }

    // Check which of the candidates is the closest point at the shape to the given point
    min_distance = 10.0;
    candidate = candidates[0];

    for (unsigned int k = 0; k < candidates.size(); ++k) {
      distance = point.distance(candidates[k]);

      if (distance <= min_distance) {
        candidate = candidates[k];
        min_distance = distance;
      }
    }
    return candidate;
  }

  template<int dim>
  bool MeshTracking<dim>::_CellContainsShapeVertex(typename BaseTriaType<dim>::active_cell_iterator &cell) {

    if (!cell->is_locally_owned() && typeid(tria_base) == typeid(BaseTriaTypeDistributed<dim>))
      return false;

    bool cell_contains_vertex = false;
    double dist, diameter = cell->diameter();
    unsigned int n_interior_vertices = 0;

    std::vector<Point<dim> > shape_poly = tria_shape.get_vertices();
    Point<dim> center = cell->center();

    tmp_interior_vertices.clear();

    for (unsigned int v = 0; v < shape_poly.size(); ++v) {

      dist = center.distance(shape_poly[v]);

      if (dist < diameter) {
        if (cell->point_inside(shape_poly[v])) {
          cell_contains_vertex = true;
          tmp_interior_vertices.push_back(v);
          ++n_interior_vertices;
        }
      }
    }
    return cell_contains_vertex;
  }

// _cell_contains_shape_edge [dim=3]
  template<>
  bool MeshTracking<3>::_CellContainsShapeEdge(typename Triangulation<3>::active_cell_iterator &cell) {

    bool tmp = false;
    bool cell_contains_edge = false;
    tmp_edge_boundary_ids.clear();

    unsigned int closest_vertex_index, boundary_id, step;
    double n_intervals, dist;

    std::vector<typename ShapeTriaType<3>::active_cell_iterator> adjacent_cells_shape;
    typename ShapeTriaType<3>::active_cell_iterator shape_cell;

    Point<3> edge_a, edge_b, base, check;
    Point<3> center = cell->center();
    // Returns either the closest vertex of the shape or a projection of the center onto the shape
    Point<3> proj = _ClosestVertexAtShape(center);

    // If it the projection returned by the function above, we find the closest shape vertex
    closest_vertex_index = GridTools::find_closest_vertex(tria_shape, proj);
    // Get the adjacent shape cells to this vertex
    adjacent_cells_shape = GridTools::find_cells_adjacent_to_vertex(tria_shape, closest_vertex_index);

    // Loop over these adjacent shape cells
    for (unsigned int c = 0; c < adjacent_cells_shape.size(); ++c) {

      shape_cell = adjacent_cells_shape[c];
      dist = shape_cell->center().distance(center);

      // TODO: Where does this come from?
      if (dist < 2. * shape_cell->diameter()) {

        // TODO: Where does this come from?
        n_intervals = 3 * floor(shape_cell->diameter() / min_face_length_base); // heavy load..!
        boundary_id = shape_cell->material_id();

        // diag
        // If any of the diagonal intervals is within the base, the diagonal is marked as inside the base
        for (unsigned int diag = 0; diag < 1; ++diag) {
          cell_contains_edge = false;
          base = shape_cell->vertex(0);

          edge_a = shape_cell->vertex(3);
          edge_a -= base;
          edge_a /= n_intervals;

          check = base;
          step = 0;

          while (step < n_intervals) {
            if (cell->point_inside(check))
              cell_contains_edge = true;
            check += edge_a;
            ++step;
          }

          if (cell_contains_edge) {
            tmp_edge_boundary_ids.push_back(boundary_id);
            tmp = true;
          }
        }

        // outer edges
        // If any of the edge intervals is within the base, the edge is marked as inside the base
        for (unsigned int sec = 0; sec < 2; ++sec) {
          cell_contains_edge = false;

          base = shape_cell->vertex(0);
          if (sec > 0)
            base = shape_cell->vertex(3);

          edge_a = shape_cell->vertex(1);
          edge_a -= base;
          edge_b = shape_cell->vertex(2);
          edge_b -= base;

          edge_a /= n_intervals;
          edge_b /= n_intervals;

          check = base;
          step = 0;

          while (step < n_intervals) {
            if (cell->point_inside(check))
              cell_contains_edge = true;
            check += edge_a;
            ++step;
          }

          if (cell_contains_edge && sec == 0) {
            tmp_edge_boundary_ids.push_back(boundary_id);
            if (!shape_cell->neighbor(2)->has_children())
              tmp_edge_boundary_ids.push_back(shape_cell->neighbor(2)->material_id());
            break;
          }

          if (cell_contains_edge && sec == 1) {
            tmp_edge_boundary_ids.push_back(boundary_id);
            if (!shape_cell->neighbor(1)->has_children())
              tmp_edge_boundary_ids.push_back(shape_cell->neighbor(1)->material_id());
            break;
          }

          check = base;
          step = 0;

          while (step < n_intervals) {
            if (cell->point_inside(check))
              cell_contains_edge = true;
            check += edge_b;
            ++step;
          }

          if (cell_contains_edge && sec == 0) {
            tmp_edge_boundary_ids.push_back(boundary_id);
            if (!shape_cell->neighbor(0)->has_children())
              tmp_edge_boundary_ids.push_back(shape_cell->neighbor(0)->material_id());
            break;
          }

          if (cell_contains_edge && sec == 1) {
            tmp_edge_boundary_ids.push_back(boundary_id);
            if (!shape_cell->neighbor(3)->has_children())
              tmp_edge_boundary_ids.push_back(shape_cell->neighbor(3)->material_id());
            break;
          }
        }
      }

      if (cell_contains_edge) {
        tmp = true;
        //break;
      }
    }

    return tmp;
  }

  // get_boundary_indicator [dim=2]
  template<>
  std::vector<unsigned int> MeshTracking<2>::_GetBoundaryIndicator(Point<2> &p) {

    std::vector<unsigned int> return_id;
    unsigned int boundary_id;
    Point<2> p1, p2, proj;
    Point<2> close_vertex;
    double s;
    double tol = 0.1;
    double dist = 0, min_dist = 10.0;
    unsigned int closest_vertex_index;

    std::vector<ShapeTriaType<2>::active_cell_iterator> adjacent_cells_shape;

    // If the projections of the p point are not within the shape cells, on which the p is projected,
    // then we need to consider the closest vertex of the shape (sharp corner situation)
    boundary_id = 99;

    // First case - a projection will indicate the material id
    for (auto shape_cell: tria_shape.active_cell_iterators()) {
      p1 = shape_cell->vertex(0);
      p2 = shape_cell->vertex(1);

      s = (p - p1) * (p2 - p1) / ((p2 - p1) * (p2 - p1));

      if ((s > (0 - tol)) && (s < (1 + tol))) {
        proj = p1 + s * (p2 - p1);
        dist = p.distance(proj);

        if (dist < min_dist) {
          min_dist = dist;
          boundary_id = shape_cell->material_id();

        }
      }
    }

    // Second case - we use the _ClosestVertexAtShape function
    if (boundary_id == 99) {
      close_vertex = _ClosestVertexAtShape(p);

      closest_vertex_index = GridTools::find_closest_vertex(tria_shape, close_vertex);
      adjacent_cells_shape = GridTools::find_cells_adjacent_to_vertex(tria_shape, closest_vertex_index);

      min_dist = 10.0;

      for (unsigned int c = 0; c < adjacent_cells_shape.size(); ++c) {

        p1 = adjacent_cells_shape[c]->vertex(0);
        p2 = adjacent_cells_shape[c]->vertex(1);

        s = (p - p1) * (p2 - p1) / ((p2 - p1) * (p2 - p1));
        proj = p1 + s * (p2 - p1);
        dist = proj.distance(p);

        if (dist < min_dist) {
          min_dist = dist;
          boundary_id = adjacent_cells_shape[c]->material_id();
        }
      }
    }
    return_id.push_back(boundary_id);
    return return_id;
  }

// get_boundary_indicator [dim=3]
  template<>
  std::vector<unsigned int> MeshTracking<3>::_GetBoundaryIndicator(Point<3> &p) {

    std::vector<unsigned int> return_id;

    unsigned int boundary_id;
    bool add_tmp_id = false;

    unsigned int closest_vertex_index;
    unsigned int tmp_id;
    Point<3> close_vertex;
    Point<3> base, edge_a, edge_b, proj, w, bb_vert;
    Point<3> ray;
    Tensor<1, 3> unit_normal;
    double uu, uv, vv, wu, wv, D;
    double bary_s, bary_t;
    double proj_dist;
    const double tol = 1.e-2;
    double dist = 0, min_dist = 10.0;

    std::vector<typename ShapeTriaType<3>::active_cell_iterator> adjacent_cells_shape;

    typename ShapeTriaType<3>::active_cell_iterator shape_cell = tria_shape.begin_active(),
        end_shape = tria_shape.end();

    // If the projections of the p point are not within the shape cells, on which the p is projected,
    // then we need to consider the closest vertex of the shape (sharp corner situation)
    boundary_id = 99;

    // First case - a projection will indicate the boundary id
    for (unsigned int safe_run = 0; safe_run < 2; ++safe_run) {

      if (safe_run == 1)
        shape_cell = tria_shape.begin_active();

      for (; shape_cell != end_shape; ++shape_cell) {

        for (unsigned int tria = 0; tria < 2; ++tria) {
          if (tria == 0) {
            edge_a = shape_cell->vertex(1);
            edge_b = shape_cell->vertex(3);
          } else {
            edge_a = shape_cell->vertex(3);
            edge_b = shape_cell->vertex(2);
          }

          base = shape_cell->vertex(0);

          edge_a -= base;
          edge_b -= base;

          //cross_product(unit_normal,edge_a,edge_b);
          unit_normal = cross_product_3d(edge_a, edge_b);
          unit_normal /= unit_normal.norm();

          ray = p;
          ray -= base;

          proj_dist = ray * unit_normal;
          proj = p;
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
          if (bary_s < -tol || bary_s > 1.0 + tol)
            continue;

          bary_t = (uv * wu - uu * wv) / D;
          if (bary_t < -tol || (bary_s + bary_t) > 1.0 + tol)
            continue;

          dist = proj.distance(p);

          if (std::abs(p[0] - .53) < 1.e-2 && std::abs(p[1] - .22) < 1.e-2 && std::abs(p[2] + .22) < 1.e-2)
            std::cout << "case 3..: " << dist << std::endl;

          if (dist < min_dist && dist < 1.42 * min_face_length_base) {
            min_dist = dist;
            boundary_id = shape_cell->material_id();
          }

          if (safe_run == 1 && dist < 1.42 * min_face_length_base && dist > min_dist) {
            tmp_id = shape_cell->material_id();
            if (tmp_id != boundary_id)
              add_tmp_id = true;
          }
        }
      }
    }

    // Second case - we need to use the ClosestVertexAtShape function
    if (boundary_id == 99) {
      close_vertex = _ClosestVertexAtShape(p);

      closest_vertex_index = GridTools::find_closest_vertex(tria_shape, close_vertex);
      adjacent_cells_shape = GridTools::find_cells_adjacent_to_vertex(tria_shape, closest_vertex_index);

      min_dist = 10.0;

      for (unsigned int c = 0; c < adjacent_cells_shape.size(); ++c) {

        for (unsigned int tria = 0; tria < 2; ++tria) {

          if (tria == 0) {
            edge_a = adjacent_cells_shape[c]->vertex(1);
            edge_b = adjacent_cells_shape[c]->vertex(3);
          } else {
            edge_a = adjacent_cells_shape[c]->vertex(3);
            edge_b = adjacent_cells_shape[c]->vertex(2);
          }

          base = adjacent_cells_shape[c]->vertex(0);

          edge_a -= base;
          edge_b -= base;

          //cross_product(unit_normal,edge_a,edge_b);
          unit_normal = cross_product_3d(edge_a, edge_b);
          unit_normal /= unit_normal.norm();

          ray = p;
          ray -= base;

          proj_dist = ray * unit_normal;

          if (std::abs(proj_dist) < min_dist) {
            min_dist = std::abs(proj_dist);
            boundary_id = adjacent_cells_shape[c]->material_id();
          }
        }
      }
    }

    return_id.push_back(boundary_id);

    if (add_tmp_id)
      return_id.push_back(tmp_id);

    return return_id;
  }

// _set_boundary_indicators [dim=2]
  template<>
  void MeshTracking<2>::_SetBoundaryIndicator() {
    boundary_ids_base.reinit(tria_base.n_active_cells());

    boundary_cell_assignment.clear();

    // Just a help variable.
    bool has_99_problem = false;

    Point<2> center;
    unsigned int cell_id = 0;
    unsigned int boundary_id = 0;
    unsigned int closest_vertex_index = 0;
    std::vector<unsigned int> return_id;

    std::vector<typename BaseTriaType<2>::active_cell_iterator> dummy_vector;
    std::vector<typename ShapeTriaType<2>::active_cell_iterator> adjacent_cells_shape;

    for (auto cell: tria_base.active_cell_iterators()) {
      if (!cell->is_locally_owned() && typeid(tria_base) == typeid(BaseTriaTypeDistributed<2>))
        continue;

      has_99_problem = false;

      // cell->cell->material_id() == msh->boundary
      if (cell->material_id() == e_boundary_cell || cell->material_id() == e_boundary_electrode_cell) {

        // special case: cell contains shape vertex -> at most 2 distinct boundary ids
        if (_CellContainsShapeVertex(cell)) {
          center = cell->center();
          closest_vertex_index = GridTools::find_closest_vertex(tria_shape, center);
          adjacent_cells_shape = GridTools::find_cells_adjacent_to_vertex(tria_shape, closest_vertex_index);

          for (unsigned int c = 0; c < adjacent_cells_shape.size(); ++c) {
            boundary_id = adjacent_cells_shape[c]->material_id();

            if (boundary_cell_assignment.find(boundary_id) == boundary_cell_assignment.end())
              boundary_cell_assignment[boundary_id] = dummy_vector;

            if (std::find(boundary_cell_assignment[boundary_id].begin(),
                          boundary_cell_assignment[boundary_id].end(), cell)
                == boundary_cell_assignment[boundary_id].end()) {

              boundary_cell_assignment[boundary_id].push_back(cell);

              if (boundary_id > boundary_ids_base[cell_id])
                boundary_ids_base[cell_id] = boundary_id;
            }
          }
        }
          // remaining cells: one and only one boundary id
        else {
          center = cell->center();
          return_id = _GetBoundaryIndicator(center);
          boundary_id = return_id[0];
          boundary_ids_base[cell_id] = boundary_id;

          if (boundary_id == 99)
            has_99_problem = true;

          if (boundary_cell_assignment.find(boundary_id) == boundary_cell_assignment.end())
            boundary_cell_assignment[boundary_id] = dummy_vector;

          if (std::find(boundary_cell_assignment[boundary_id].begin(),
                        boundary_cell_assignment[boundary_id].end(), cell)
              == boundary_cell_assignment[boundary_id].end()) {

            boundary_cell_assignment[boundary_id].push_back(cell);
          }
        }
      }

      if (has_99_problem) {
        std::cout << "cell has 99 problem.." << std::endl;
        boundary_ids_base[cell_id] = 99;
      }
      ++cell_id;
    }
  }

// _set_boundary_indicators [dim=3]
  template<>
  void MeshTracking<3>::_SetBoundaryIndicator() {

    boundary_ids_base.reinit(tria_base.n_active_cells());

    unsigned int cell_id = 0;
    unsigned int boundary_id = 0;

    std::vector<unsigned int> return_id;

    Point<3> center;
    bool has_99_problem = false;

    std::vector<typename BaseTriaType<3>::active_cell_iterator> dummy_vector;
    std::vector<typename ShapeTriaType<3>::active_cell_iterator> adjacent_cells_shape, tmp_adjacent_cells;

    typename Triangulation<3>::active_cell_iterator neighbor;

    unsigned int tmp_id;
    std::vector<unsigned int> tmp_boundary_ids;
    std::vector<unsigned int> current_ids, neighbor_ids;
    std::vector<unsigned int> all_neighbor_ids, missing_ids;
    std::map<typename Triangulation<3>::active_cell_iterator, std::vector<unsigned int>> hda_cell_to_boundary_ids;

    boundary_cell_assignment.clear();

    for (auto cell : tria_base.active_cell_iterators()) {

      if (!cell->is_locally_owned() && typeid(tria_base) == typeid(BaseTriaTypeDistributed<3>))
        continue;

      has_99_problem = false;

      if (cell->material_id() == e_boundary_cell || cell->material_id() == e_boundary_electrode_cell) {

        tmp_boundary_ids.clear();

        // case 1: cell contains shape vertex -> at most 4 distinct boundary ids
        if (_CellContainsShapeVertex(cell)) {

          adjacent_cells_shape.clear();
          for (unsigned int v = 0; v < tmp_interior_vertices.size(); ++v) {
            tmp_adjacent_cells = GridTools::find_cells_adjacent_to_vertex(tria_shape,
                                                                          tmp_interior_vertices[v]);
            for (unsigned int ac = 0; ac < tmp_adjacent_cells.size(); ++ac) {
              adjacent_cells_shape.push_back(tmp_adjacent_cells[ac]);
            }
          }

          for (unsigned int c = 0; c < adjacent_cells_shape.size(); ++c) {
            boundary_id = adjacent_cells_shape[c]->material_id();

            if (boundary_cell_assignment.find(boundary_id) == boundary_cell_assignment.end())
              boundary_cell_assignment[boundary_id] = dummy_vector;

            if (std::find(boundary_cell_assignment[boundary_id].begin(),
                          boundary_cell_assignment[boundary_id].end(), cell)
                == boundary_cell_assignment[boundary_id].end()) {
              boundary_cell_assignment[boundary_id].push_back(cell);
              tmp_boundary_ids.push_back(boundary_id);
              if (boundary_id > boundary_ids_base[cell_id])
                boundary_ids_base[cell_id] = boundary_id;
            }
          }
        }

          // case 2: cell contains shape edge -> at most 2 distinct boundary ids (not sure since it may contain more than one shape edge..)
        else if (_CellContainsShapeEdge(cell)) {

          for (unsigned int c = 0; c < tmp_edge_boundary_ids.size(); ++c) {
            boundary_id = tmp_edge_boundary_ids[c];

            if (boundary_cell_assignment.find(boundary_id) == boundary_cell_assignment.end())
              boundary_cell_assignment[boundary_id] = dummy_vector;

            if (std::find(boundary_cell_assignment[boundary_id].begin(),
                          boundary_cell_assignment[boundary_id].end(), cell)
                == boundary_cell_assignment[boundary_id].end()) {
              boundary_cell_assignment[boundary_id].push_back(cell);
              tmp_boundary_ids.push_back(boundary_id);
              if (boundary_id > boundary_ids_base[cell_id])
                boundary_ids_base[cell_id] = boundary_id;
            }
          }
        }

          // case 3: remaining -> one and only one boundary id (wrong assumption, since it may contain more than one shape cell)
        else {

          center = cell->center();
          return_id = _GetBoundaryIndicator(center);
          boundary_id = return_id[0];

          if (boundary_id > boundary_ids_base[cell_id])
            boundary_ids_base[cell_id] = boundary_id;

          if (boundary_id == 99)
            has_99_problem = true;

          if (boundary_cell_assignment.find(boundary_id) == boundary_cell_assignment.end())
            boundary_cell_assignment[boundary_id] = dummy_vector;

          if (std::find(boundary_cell_assignment[boundary_id].begin(),
                        boundary_cell_assignment[boundary_id].end(), cell)
              == boundary_cell_assignment[boundary_id].end()) {
            boundary_cell_assignment[boundary_id].push_back(cell);
            tmp_boundary_ids.push_back(boundary_id);
          }

          if (return_id.size() > 1) {
            boundary_id = return_id[1];

            if (boundary_id > boundary_ids_base[cell_id])
              boundary_ids_base[cell_id] = boundary_id;

            if (boundary_id == 99)
              has_99_problem = true;

            if (boundary_cell_assignment.find(boundary_id) == boundary_cell_assignment.end())
              boundary_cell_assignment[boundary_id] = dummy_vector;

            if (std::find(boundary_cell_assignment[boundary_id].begin(),
                          boundary_cell_assignment[boundary_id].end(), cell)
                == boundary_cell_assignment[boundary_id].end()) {
              boundary_cell_assignment[boundary_id].push_back(cell);
              tmp_boundary_ids.push_back(boundary_id);
            }
          }
        }
        hda_cell_to_boundary_ids[cell] = tmp_boundary_ids;
      }

      if (has_99_problem) {
        std::cout << "cell has 99 problem.." << std::endl;
        boundary_ids_base[cell_id] = 99;
      }
      ++cell_id;
    }

    // safeguard edge contributions
    for (auto cell : tria_base.active_cell_iterators()) {

      if (!cell->is_locally_owned() && typeid(tria_base) == typeid(BaseTriaTypeDistributed<3>))
        continue;

      if (cell->material_id() == e_boundary_cell || cell->material_id() == e_boundary_electrode_cell) {

        center = cell->center();

        current_ids = hda_cell_to_boundary_ids[cell];
        missing_ids.clear();
        all_neighbor_ids.clear();

        for (unsigned int n = 0; n < GeometryInfo<3>::faces_per_cell; ++n) {
          if(cell->neighbor_index(n) != -1 && !cell->neighbor(n)->has_children()) {
            neighbor = cell->neighbor(n);
            if (cell->neighbor_index(n) == -1)
              continue;
            if ((neighbor->material_id() == e_boundary_cell || neighbor->material_id() == e_boundary_electrode_cell) &&
                !neighbor->has_children()) {
              neighbor_ids = hda_cell_to_boundary_ids[neighbor];
              for (unsigned int i = 0; i < neighbor_ids.size(); ++i) {
                all_neighbor_ids.push_back(neighbor_ids[i]);
              }
            }
          }
        }

        for (unsigned int i = 0; i < all_neighbor_ids.size(); ++i) {
          tmp_id = all_neighbor_ids[i];
          if (std::find(current_ids.begin(), current_ids.end(), tmp_id) == current_ids.end()) {
            if (std::find(missing_ids.begin(), missing_ids.end(), tmp_id) == missing_ids.end()) {
              missing_ids.push_back(tmp_id);
            }
          }
        }

        for (unsigned int i = 0; i < missing_ids.size(); ++i) {
          boundary_cell_assignment[missing_ids[i]].push_back(cell);
        }
      }
    }
  }

  template<int dim>
  void MeshTracking<dim>::_OutputMesh() {
    Vector<double> flags(tria_base.n_active_cells()), materialid(tria_base.n_active_cells());
    int i = 0;
    for (auto cell : tria_base.active_cell_iterators()) {
      if (cell->material_id() == e_boundary_cell || cell->material_id() == e_inside_cell || cell->material_id() == e_boundary_electrode_cell) {
        materialid[i] = cell->material_id();
      }
      if (cell->material_id() == e_boundary_cell || cell->material_id() == e_boundary_electrode_cell) {
        flags[i] = 100;
      }
      ++i;
    }

    std::vector<std::string> flag_names(1, "flag"), mat_names(1, "mat");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        flag_data_component_interpretation(DataComponentInterpretation::component_is_scalar),
        mat_data_component_interpretation(DataComponentInterpretation::component_is_scalar);

    tracking_output.template PushDataName<Vector<double>>(flags, flag_names,
                                                          flag_data_component_interpretation, &base_dof_handler);
    tracking_output.template PushDataName<Vector<double>>(materialid, mat_names,
                                                          mat_data_component_interpretation, &base_dof_handler);

    DoFHandler<dim> base_dof_handler(tria_base);
    tracking_output.WriteDataOutput(n_tracking);
    base_dof_handler.clear();
  }

  template<int dim>
  void MeshTracking<dim>::RunTracking() {
    TimerOutput::Scope t(this->compute_timer, "MeshTracking<dim>::RunTracking");

    if(data.refinement_criterion == "fixed")
      refinement_step = 0;

    _Tracking();
    _SetBoundaryIndicator();
  }

} // end of StructuralOptimization namespace

template
class StructuralOptimization::MeshTracking<2>;

template
class StructuralOptimization::MeshTracking<3>;