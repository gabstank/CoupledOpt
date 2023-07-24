// C++ headers

// Deal.II headers

// Project headers
#include <Mesh.h>

namespace StructuralOptimization {

  template<int dim>
  int Mesh<dim>::MeshSetup::_GetBoundaryID(Data &data, std::map<int, int> bid_counter, const Point<dim> &point) {
    for (unsigned int i = 0; i < data.boundary_ids.size(); ++i) {
      if (std::abs(point[data.boundary_comp[i]] - data.boundary_coord[i]) < data.boundary_tol[i]) {
        bid_counter.find(data.boundary_ids[i])->second++;
        if (bid_counter.find(data.boundary_ids[i])->second == dim + 1) {
          return bid_counter.find(data.boundary_ids[i])->first;
        }
      }
    }

    return 0;
  }

  template<int dim>
  void Mesh<dim>::MeshSetup::_SetupBoundaryIDs(Mesh<dim> &mesh) {
    // Mark the boundaries.
    // Defining some help variable
    // Map of {ID, counter}
    /**
     * The reason for having this map.
     * The counter is initial set to dim for each unique ID.
     * If there are multiple conditions for an ID, each additional condition the counter is reduced by 1.
     *
     * During the check if a condition is satisfied the counter will be increased by 1 for corresponding ID.
     * Finally whichever ID has counter value dim + 1 is the Boundary ID of that point.
     */
    AssertThrow(mesh.data.boundary_ids.size() == mesh.data.boundary_comp.size(),
           ExcDimensionMismatch(mesh.data.boundary_ids.size(), mesh.data.boundary_comp.size()));
    AssertThrow(mesh.data.boundary_ids.size() == mesh.data.boundary_coord.size(),
           ExcDimensionMismatch(mesh.data.boundary_ids.size(), mesh.data.boundary_coord.size()));
    AssertThrow(mesh.data.boundary_ids.size() == mesh.data.boundary_tol.size(),
           ExcDimensionMismatch(mesh.data.boundary_ids.size(), mesh.data.boundary_tol.size()));

    std::map<int, int> bid_counter;
    for (unsigned int i = 0; i < mesh.data.boundary_ids.size(); ++i) {
      if (bid_counter.find(mesh.data.boundary_ids[i]) == bid_counter.end())
        bid_counter.insert(std::pair<int, int>(mesh.data.boundary_ids[i], dim));
      else
        bid_counter.find(mesh.data.boundary_ids[i])->second -= 1;
    }

    Vector<double> dom_boundary_ids(mesh.domain->n_active_cells() * GeometryInfo<dim>::faces_per_cell);
    Vector<double> out_dom_boundary_ids(mesh.domain->n_active_cells());

    // Loop over the cells. If the cell is at bounday, get boundary ID from the function _GetBoundaryID(...).
    // For serial domain
    unsigned int counter = 0, cell_counter = 0;
    for (auto cell: mesh.domain->active_cell_iterators()) {
      if (cell->at_boundary())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          if (cell->face(f)->at_boundary()) {
            int boundary_id = _GetBoundaryID(mesh.data, bid_counter, cell->face(f)->center());
            cell->face(f)->set_boundary_id(boundary_id);
            if (dom_boundary_ids[counter] == 0) {
              dom_boundary_ids[counter] = boundary_id;
              out_dom_boundary_ids[cell_counter] = boundary_id;
            }
          }
          ++counter;
        }
      ++cell_counter;
    }

    // For parallel domain
    // Loop over the cells. If the cell is at bounday, get boundary ID from the function _GetBoundaryID(...).
    Vector<double> p_dom_boundary_ids(mesh.domain_parallel->n_active_cells() * GeometryInfo<dim>::faces_per_cell);
    Vector<double> p_out_dom_boundary_ids(mesh.domain_parallel->n_active_cells());
    counter = 0, cell_counter = 0;
    for (auto cell: mesh.domain_parallel->active_cell_iterators()) {
//      if(!cell->is_locally_owned()){
//        ++cell_counter;
//        counter += GeometryInfo<dim>::faces_per_cell;
//        continue;
//      }
      if (cell->at_boundary())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          if (cell->face(f)->at_boundary()) {
            int boundary_id = _GetBoundaryID(mesh.data, bid_counter, cell->face(f)->center());
            cell->face(f)->set_boundary_id(boundary_id);
            if (p_dom_boundary_ids[counter] == 0) {
              p_dom_boundary_ids[counter] = boundary_id;
              p_out_dom_boundary_ids[cell_counter] = boundary_id;
            }
          }
          ++counter;
        }
      ++cell_counter;
    }

    // Check if all boundaries were set - if a boundary ID was not used then report error
    for (unsigned int i = 0; i < mesh.data.boundary_ids.size(); ++i) Assert(
        std::find(dom_boundary_ids.begin(), dom_boundary_ids.end(), mesh.data.boundary_ids[i]) !=
        dom_boundary_ids.end(),
        ExcMessage("None of the cells were assigned with boundary ID: " +
                   Utilities::int_to_string(mesh.data.boundary_ids[i])));

    std::vector<std::string> data_names(1, "boundary_id"), p_data_names(1, "boundary_id");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(DataComponentInterpretation::component_is_scalar),
        p_data_component_interpretation(DataComponentInterpretation::component_is_scalar);
    mesh.domain_output.template PushDataName<Vector<double>>(out_dom_boundary_ids,
                                                             data_names,
                                                             data_component_interpretation,
                                                             &mesh.domain_dof_handler);
    mesh.domain_prarallel_output->template PushDataName<Vector<double>>(p_out_dom_boundary_ids,
                                                             p_data_names,
                                                             p_data_component_interpretation,
                                                             &mesh.domain_parallel_dof_handler);

  }


  template<int dim>
  void Mesh<dim>::MeshSetup::CreateHyperRectangleDomain(Mesh<dim> &mesh) {

    // Gather data from .prm file to Generate grid
    Point<dim> p1;
    Point<dim> p2;
    std::vector<unsigned int> subdivisions;

    switch (dim) {
      case 2:
        p1 = Point<dim>(mesh.data.hyper_rectangle_point1[0], mesh.data.hyper_rectangle_point1[1]);
        p2 = Point<dim>(mesh.data.hyper_rectangle_point2[0], mesh.data.hyper_rectangle_point2[1]);
        subdivisions.push_back(mesh.data.hyper_rectangle_subdivision[0]);
        subdivisions.push_back(mesh.data.hyper_rectangle_subdivision[1]);
        break;
      case 3:
        p1 = Point<dim>(mesh.data.hyper_rectangle_point1[0], mesh.data.hyper_rectangle_point1[1],
                        mesh.data.hyper_rectangle_point1[2]);
        p2 = Point<dim>(mesh.data.hyper_rectangle_point2[0], mesh.data.hyper_rectangle_point2[1],
                        mesh.data.hyper_rectangle_point2[2]);
        subdivisions.push_back(mesh.data.hyper_rectangle_subdivision[0]);
        subdivisions.push_back(mesh.data.hyper_rectangle_subdivision[1]);
        subdivisions.push_back(mesh.data.hyper_rectangle_subdivision[2]);
        break;
    }

    mesh.domain->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::subdivided_hyper_rectangle(*mesh.domain, subdivisions, p1, p2,
                                              true); // Generate the hyper_rectangle mesh.
    mesh.domain->refine_global(mesh.data.hyper_rectangle_refinements);

    mesh.domain_parallel->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::subdivided_hyper_rectangle(*mesh.domain_parallel, subdivisions, p1, p2,
                                              true); // Generate the hyper_rectangle mesh.
    mesh.domain_parallel->refine_global(mesh.data.hyper_rectangle_refinements);

    //Assign all boundary IDs to the mesh
    _SetupBoundaryIDs(mesh);

  }


  template<int dim>
  void Mesh<dim>::MeshSetup::CreateGradedHyperRectangleDomain(Mesh<dim> &mesh) {

    // Gather data from .prm file to Generate grid
    Point<dim> p1;
    Point<dim> p2;
    std::vector<std::vector<double>> step_sizes;

    step_sizes.push_back(mesh.data.steps_x);
    step_sizes.push_back(mesh.data.steps_y);

    double sum_0 = 0;
    double sum_1 = 0;

    for (auto el : step_sizes[0])
      sum_0 += el;
    for (auto el : step_sizes[1])
      sum_1 += el;

    switch (dim) {
      case 2:
        p1 = Point<dim>(mesh.data.graded_hyper_rectangle_point1[0], mesh.data.graded_hyper_rectangle_point1[1]);
        p2 = Point<dim>(mesh.data.graded_hyper_rectangle_point2[0], mesh.data.graded_hyper_rectangle_point2[1]);
        break;
      case 3:
        p1 = Point<dim>(mesh.data.graded_hyper_rectangle_point1[0], mesh.data.graded_hyper_rectangle_point1[1],
                        mesh.data.graded_hyper_rectangle_point1[2]);
        p2 = Point<dim>(mesh.data.graded_hyper_rectangle_point2[0], mesh.data.graded_hyper_rectangle_point2[1],
                        mesh.data.graded_hyper_rectangle_point2[2]);
        step_sizes.push_back(mesh.data.steps_z);
        break;
    }

    mesh.domain->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::subdivided_hyper_rectangle(*mesh.domain, step_sizes, p1, p2,
                                              true); // Generate the graded_hyper_rectangle mesh.

    mesh.domain->refine_global(mesh.data.graded_hyper_rectangle_refinements);

    mesh.domain_parallel->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::subdivided_hyper_rectangle(*mesh.domain_parallel, step_sizes, p1, p2,
                                              true); // Generate the graded_hyper_rectangle mesh.

    mesh.domain_parallel->refine_global(mesh.data.graded_hyper_rectangle_refinements);

    //Assign all boundary IDs to the mesh
    _SetupBoundaryIDs(mesh);

  }

  template<>
  void Mesh<2>::MeshSetup::CreateHyperCubeWithHoleDomain(Mesh<2> &mesh) {
    mesh.domain->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::hyper_cube_with_cylindrical_hole(*mesh.domain, mesh.data.hyper_cube_with_hole_inner_radius,
                                                    mesh.data.hyper_cube_with_hole_outer_radius,
                                                    mesh.data.hyper_cube_with_hole_L,
                                                    mesh.data.hyper_cube_with_hole_repetitions, false);

    mesh.domain->refine_global(mesh.data.hyper_cube_with_hole_refinements);

    mesh.domain_parallel->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::hyper_cube_with_cylindrical_hole(*mesh.domain_parallel, mesh.data.hyper_cube_with_hole_inner_radius,
                                                    mesh.data.hyper_cube_with_hole_outer_radius,
                                                    mesh.data.hyper_cube_with_hole_L,
                                                    mesh.data.hyper_cube_with_hole_repetitions, false);

    mesh.domain_parallel->refine_global(mesh.data.hyper_cube_with_hole_refinements);

    //Assign all boundary IDs to the mesh
    _SetupBoundaryIDs(mesh);

  }
  template<>
  void Mesh<3>::MeshSetup::CreateHyperCubeWithHoleDomain(Mesh<3> &mesh) {
    mesh.domain->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::hyper_cube_with_cylindrical_hole(*mesh.domain, mesh.data.hyper_cube_with_hole_inner_radius,
                                                    mesh.data.hyper_cube_with_hole_outer_radius,
                                                    mesh.data.hyper_cube_with_hole_L,
                                                    mesh.data.hyper_cube_with_hole_repetitions, false);

    if(mesh.data.hyper_cube_with_hole_rotate)
      GridTools::rotate(mesh.data.hyper_cube_with_hole_rotation_angle,
                        mesh.data.hyper_cube_with_hole_rotation_axis,
                     *mesh.domain);
    mesh.domain->refine_global(mesh.data.hyper_cube_with_hole_refinements);

    mesh.domain_parallel->clear(); // Just to be sure I clear here, if deemed not useful can be removed.

    GridGenerator::hyper_cube_with_cylindrical_hole(*mesh.domain_parallel, mesh.data.hyper_cube_with_hole_inner_radius,
                                                    mesh.data.hyper_cube_with_hole_outer_radius,
                                                    mesh.data.hyper_cube_with_hole_L,
                                                    mesh.data.hyper_cube_with_hole_repetitions, false);

    if(mesh.data.hyper_cube_with_hole_rotate)
      GridTools::rotate(mesh.data.hyper_cube_with_hole_rotation_angle,
                        mesh.data.hyper_cube_with_hole_rotation_axis,
                        *mesh.domain_parallel);
    mesh.domain_parallel->refine_global(mesh.data.hyper_cube_with_hole_refinements);

    //Assign all boundary IDs to the mesh
    _SetupBoundaryIDs(mesh);

  }

  template<int dim>
  void Mesh<dim>::MeshSetup::CreateAbaqusDomain(Mesh<dim> &mesh) {

    GridIn<dim> grid_in, p_grid_in;
    std::ifstream input_file(mesh.data.abaqus_input);

    grid_in.attach_triangulation(*mesh.domain);
    grid_in.read_abaqus(input_file);
    mesh.domain->refine_global(mesh.data.abaqus_refinements);

    std::ifstream p_input_file(mesh.data.abaqus_input); // rereading the file because of dealii I/O error. Could not use the input_file again.
    p_grid_in.attach_triangulation(*mesh.domain_parallel);
    p_grid_in.read_abaqus(p_input_file);
    mesh.domain_parallel->refine_global(mesh.data.abaqus_refinements);

    //Assign all boundary IDs to the mesh
    _SetupBoundaryIDs(mesh);

  }

  template<int dim>
  void Mesh<dim>::MeshSetup::UpdateShapeFromGeo(Mesh<dim> &mesh) {
    // restart is false simply return
    if (!mesh.data.restart_from_geo)
      return;

    std::string base("vertex_index ");
    std::string full, index, arg;
    std::stringstream ss;
    unsigned int vertex_index;
    std::vector<bool> vertex_touched(mesh.shape.n_vertices());
    std::vector<double> vec_p;
    Point<dim> p;

    // this loop we setup geo_prm
    for (auto &cell : mesh.shape.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = cell->vertex_index(v);
        if (!vertex_touched[vertex_index]) {
          ss.str("");
          ss.clear();
          ss << vertex_index;
          index = ss.str();
          full = base;
          full += index;
          geo_prm.declare_entry(full, "0,0", Patterns::List(Patterns::Double()), "");
          vertex_touched[vertex_index] = true;
        }
      }
    }

    geo_prm.parse_input(mesh.data.geo_input);
    vertex_touched.flip();

    // Update the shape.
    for (auto &cell : mesh.shape.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = cell->vertex_index(v);
        if (!vertex_touched[vertex_index]) {
          ss.str("");
          ss.clear();
          ss << vertex_index;
          index = ss.str();
          full = base;
          full += index;
          arg = geo_prm.get(full);
          string_to_vector_of_double(arg, vec_p);
          _VectorToPoint(vec_p, p);
          for (unsigned int d = 0; d < dim; ++d)
            cell->vertex(v)[d] = p[d];
          vertex_touched[vertex_index] = true;
        }
      }
    }

    if (mesh.data.refine_after_restart)
      mesh.shape.refine_global(1);
  }

  template<int dim>
  void Mesh<dim>::MeshSetup::CreateShape(Mesh<dim> &mesh) {

    // Check if domain is initiated,
    AssertThrow(mesh.domain->n_cells() > 0, ExcMessage(
        "domain.n_cells() > 0 is not satisified, i.e. the domain is empty, one cannot extract shape form an empty domain!"));

    // Here we store the map between the surface to volume mesh, this is used to have a consistant
    // boundary id between the domain mesh and shape mesh.
    std::map<typename ShapeTriaType<dim>::cell_iterator,
        typename DomainTriaType<dim>::face_iterator> surface_to_volume_mesh;

    // extract boundary mesh
    surface_to_volume_mesh = GridGenerator::extract_boundary_mesh(*mesh.domain, mesh.shape);

    // This loop will set the boundary id for the domain.
    for (auto cell : mesh.domain->active_cell_iterators()) {
      if (cell->at_boundary()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          if (cell->face(f)->at_boundary()) {
            cell->face(f)->set_boundary_id(cell->face(f)->boundary_id());
          } // end of cell face at boundary condition
        } // end of loop over the face
      } // end of cell at boundary condition
    } // end of loop over cells

    // the shape cells are set material id based on the boundary id of the parent domain through the surface_to_volume_mesh map.
    for (auto shape_cell : mesh.shape.active_cell_iterators())
      shape_cell->set_material_id(surface_to_volume_mesh[shape_cell]->boundary_id());

    if (mesh.data.restart_from_geo)
      UpdateShapeFromGeo(mesh);


    // setup boundary id data to be used by the output function
    Vector<double> boundary_ids_shape(mesh.shape.n_active_cells());

    unsigned int cell_id = 0;
    for (auto shape_cell : mesh.shape.active_cell_iterators()) {
      boundary_ids_shape[cell_id] = shape_cell->material_id();
      ++cell_id;
    }


    // pushing data to output vector
    std::vector<std::string> data_names(1, "boundary_id");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(DataComponentInterpretation::component_is_scalar);
    mesh.shape_output.template PushDataName<Vector<double>>(boundary_ids_shape,
                                                            data_names,
                                                            data_component_interpretation,
                                                            &mesh.shape_dof_handler);


  }

/**
 * Function to create Base Triangulation.
 */
  template<int dim>
  void Mesh<dim>::MeshSetup::CreateHyperRectangleBase(Mesh<dim> &mesh) {

    Point<dim> p1;
    Point<dim> p2;
    std::vector<unsigned int> subdivisions;

    switch (dim) {
      case 2:
        p1 = Point<dim>(mesh.data.base_point1[0], mesh.data.base_point1[1]);
        p2 = Point<dim>(mesh.data.base_point2[0], mesh.data.base_point2[1]);
        subdivisions.push_back(mesh.data.base_subdivision[0]);
        subdivisions.push_back(mesh.data.base_subdivision[1]);
        break;
      case 3:
        p1 = Point<dim>(mesh.data.base_point1[0], mesh.data.base_point1[1], mesh.data.base_point1[2]);
        p2 = Point<dim>(mesh.data.base_point2[0], mesh.data.base_point2[1], mesh.data.base_point2[2]);
        subdivisions.push_back(mesh.data.base_subdivision[0]);
        subdivisions.push_back(mesh.data.base_subdivision[1]);
        subdivisions.push_back(mesh.data.base_subdivision[2]);
        break;
    }



    mesh.base->clear();
    GridGenerator::subdivided_hyper_rectangle(*mesh.base, subdivisions, p1, p2,
                                              true);

    mesh.base->refine_global(mesh.data.base_refinements);

    /**
     * min_face_length_base computation. This will be used by the mesh
     * tracking function. This is to decide if we have to slpit the shape mesh.
     */
    double min_dist = (p2[0] - p1[0])/subdivisions[0];
    if ((p2[1] - p1[1])/subdivisions[1] < min_dist)
      min_dist = (p2[1] - p1[1])/subdivisions[1];
    if (dim == 3)
      if ((p2[2] - p1[2])/subdivisions[2] < min_dist)
        min_dist = (p2[2] - p1[2])/subdivisions[2] ;

    mesh.mesh_tracking->min_face_length_base = min_dist / pow(2.0, mesh.data.base_refinements);

    mesh.base_dof_handler.reinit(*mesh.base);
    mesh.base_dof_handler.distribute_dofs(FE_Q<dim>(1));

  } // end of CreateHyperRectangleBase function

  template<int dim>
  void Mesh<dim>::MeshSetup::_VectorToPoint(const std::vector<double> &vec, Point<dim> &p) {
    for (unsigned int i = 0; i < dim; ++i)
      p(i) = vec[i];
  }

} // end of StructuralOptimization namespace

template
class StructuralOptimization::Mesh<2>;

template
class StructuralOptimization::Mesh<3>;
