// C++ headers

// Deal.II headers

// Project headers
#include <ResponseHandler.h>

namespace StructuralOptimization {

  using namespace dealii;

  template<int dim>
  ResponseHandler<dim>::ResponseHandler(Parameter &par_, EddBVP_<dim> &bvp_)
      :
      mpi_communicator(par_.mpi_communicator),
      n_mpi_processes(par_.n_mpi_processes),
      this_mpi_process(par_.this_mpi_process),
      pcout(par_.pcout),
      compute_timer(par_.compute_timer),
      _par(par_),
      _bvp(bvp_),
      _regularization(par_),
      _shape_output(par_.data, par_.mpi_communicator, par_.compute_timer, "responses", bvp_._tria_shape) {
    for (unsigned int id = 0; id < par_.data.responses.size(); ++id) {
      _AddResponse(id, par_.data.responses[id]);
    }
  }


  template<int dim>
  void ResponseHandler<dim>::_ComputeValues() {

    TimerOutput::Scope t(compute_timer, "ResponseHandler<dim>::_ComputeValues");

    _values.clear();

    for (auto &[id, response] : _responses) {
      _values[id] = response->GetFunction();
    }
  }

  template<int dim>
  void ResponseHandler<dim>::RunAdjointBVP() {
    PROJ_MPI_BARRIER
    // Run Adjoint BVP for shape
    for (auto &[id, response] : _responses) {
      response->RunAdjointBVP();
      // TODO: commented below code when implementing compliant mechanism.
//      if(_shape_gradients.find(id) != _shape_gradients.end()) {
//        response->RunAdjointBVP();
//      }
    }
    PROJ_MPI_BARRIER
  }

  template<int dim>
  void ResponseHandler<dim>::GetAdjointOutputData(DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>> &output){
//    std::cout << __FILE__ << ":" << __LINE__ << "|P:" << this_mpi_process<< std::endl;
    PROJ_MPI_BARRIER
    // Run Adjoint BVP for shape
    for (auto &[id, response] : _responses) {
      response->GetAdjointOutputData(output);
      // TODO: commented below code when implementing compliant mechanism.
//      if(_shape_gradients.find(id) != _shape_gradients.end()) {
//        response->GetAdjointOutputData(output);
//      }
    }
    PROJ_MPI_BARRIER
  }

  template<int dim>
  std::map<unsigned int, double> ResponseHandler<dim>::GetValues() {
    _ComputeValues();
    PROJ_MPI_BARRIER
    return _values;
  }

  template<int dim>
  std::map<unsigned int, Vector<double>> ResponseHandler<dim>::GetRawShapeGradients() {
    _ComputeShapeGradients();
    PROJ_MPI_BARRIER
    return _shape_gradients;
  }

  template<int dim>
  std::map<unsigned int, std::map<CellId, double>> ResponseHandler<dim>::GetRawDensityGradients() {
    _ComputeDensityGradients();
    PROJ_MPI_BARRIER
    return _density_gradients;
  }

  template<int dim>
  Vector<double> ResponseHandler<dim>::GetVertexNormals() {
    return _vertex_normals;
  }


  template<int dim>
  unsigned int ResponseHandler<dim>::GetNResponses() {
    return _responses.size();
  }


  template<int dim>
  void ResponseHandler<dim>::NormalizeGradients(std::map<unsigned int, Vector<double>> &gradients,
                                                std::string gradient_normalization) {
    if (gradient_normalization == "")
      gradient_normalization = _par.data.gradient_normalization;

    if (gradient_normalization == "l2_norm")
      _NormalizeGradientsByL2Norm(gradients);
    else if (gradient_normalization == "magnitude")
      _NormalizeGradientsByMagnitude(gradients);
  }

  template<int dim>
  void ResponseHandler<dim>::_NormalizeGradientsByL2Norm(std::map<unsigned int, Vector<double>> &gradients) {
    for (auto &gradient : gradients) {
      double norm = gradient.second.l2_norm();
      gradient.second /= norm;
    }
  }


  template<int dim>
  void ResponseHandler<dim>::_NormalizeGradientsByMagnitude(std::map<unsigned int, Vector<double>> &gradients) {
    for (auto &gradient : gradients) {

      double max_mag = 0.0;

      for (unsigned int i = 0; i < gradient.second.size(); i += dim) {
        double mag = 0;
        if (dim == 2)
          mag = gradient.second[i] * gradient.second[i] + gradient.second[i + 1] * gradient.second[i + 1];
        else //dim ==3
          mag = gradient.second[i] * gradient.second[i] + gradient.second[i + 1] * gradient.second[i + 1] +
              gradient.second[i + 2] * gradient.second[i + 2];
        if (max_mag < mag)
          max_mag = mag;
      }

      for (double & i : gradient.second)
        i = i / std::sqrt(max_mag);
    }
  }


  template<int dim>
  void ResponseHandler<dim>::_ComputeCurrentVertexNormal(
      std::vector<Tensor<1, dim>> &adjacent_shape_cell_normals, Tensor<1, dim> &vertex_normal) {

    vertex_normal = 0;

    for (const auto &shape_cell_normal : adjacent_shape_cell_normals) {
      for (int i = 0; i < dim; ++i) {
        vertex_normal[i] += shape_cell_normal[i];
      }
    }
    vertex_normal /= vertex_normal.norm();
  }


  template<>
  void ResponseHandler<2>::_SetupShapeCellNormals(FEValues<1, 2> &fe_values_shape,
                                                  std::map<typename DoFHandler<1, 2>::active_cell_iterator, Tensor<1, 2>> &shape_cell_normals) {

    Tensor<1, 2> t_normal;
    Point<2> normal, p;

    for (const auto &shape_cell : this->_bvp._dof_handler_shape.active_cell_iterators()) {

      fe_values_shape.reinit(shape_cell);
      t_normal = fe_values_shape.normal_vector(0);
      for (unsigned int d = 0; d < 2; ++d) {
        normal[d] = t_normal[d];
      }

      p = normal;
      // 0.1 factor added to correctly consider the cases with very sharp corners
      p *= _bvp.min_face_length_base * 0.1;
      p += shape_cell->center();

      if (GeometryAlgorithms::PointInShape(p, this->_bvp._tria_shape,
                                           this->_bvp.min_face_length_base))
        normal *= -1;

      shape_cell_normals[shape_cell] = normal;
    }
  }


  template<>
  void ResponseHandler<3>::_SetupShapeCellNormals(FEValues<2, 3> &fe_values_shape,
                                                  std::map<typename DoFHandler<2, 3>::active_cell_iterator, Tensor<1, 3>> &shape_cell_normals) {

    (void) fe_values_shape;
    Point<3> p, normal, base, edge_a, edge_b;
    Tensor<1, 3> normal_t;

    /* In the original code two shape cell normals are computed based on two subtriangles of the quad element.
     * Since we assume that the elements are flat, now only one shape cell normal is computed,
     * based on a randomly chosen three vertices of the quad element. If this approach is invalid,
     * i.e. it will throw an error the code should be brought back to its original form as in
     * postprocess class, shape_sensitivity analysis <3> function in the original code from Stefan Riehl.
     * */

    for (const auto &shape_cell : this->_bvp._dof_handler_shape.active_cell_iterators()) {

      base = shape_cell->vertex(0);

      edge_a = shape_cell->vertex(1);
      edge_b = shape_cell->vertex(3);

      edge_a -= base;
      edge_b -= base;

      //cross_product(normal, edge_a, edge_b);
      normal_t = cross_product_3d(edge_a, edge_b);
      for (unsigned int d = 0; d < 3; ++d) {
        normal[d] = normal_t[d];
      }
      normal /= normal.norm();

      p = base + 0.33 * (edge_a + edge_b);
      // 0.1 factor added to correctly consider the cases with very sharp corners
      p += normal * _bvp.min_face_length_base * 0.1;

      if (GeometryAlgorithms::PointInShape(p, this->_bvp._tria_shape,
                                           this->_bvp.min_face_length_base))
        normal *= -1;

      shape_cell_normals[shape_cell] = normal;
    }
  }


  template<>
  void ResponseHandler<2>::_ComputeShapeGradients() {

    _shape_gradients.clear();
    unsigned int n_responses = _responses.size();

    for (unsigned int response = 0; response < n_responses; ++response)
      _responses[response]->ParseCommonData();

    // Vector of gradient densities corresponds to gradient densities of all responses in the problem
    std::vector<double> shape_cell_gradient_densities(n_responses);
    std::vector<double> base_cell_gradient_densities(n_responses);

    Vector<double> current_function_gradient(_bvp._dof_handler_shape.n_dofs());
    _vertex_normals.reinit(_bvp._dof_handler_shape.n_dofs());
    _mpilocal_vertex_normals.reinit(_bvp._dof_handler_shape.n_dofs());

    std::vector<Vector<double>> _mpilocal_function_gradients(n_responses);
    for (unsigned int response = 0; response < n_responses; ++response)
      _mpilocal_function_gradients[response].reinit(_bvp._dof_handler_shape.n_dofs());

    const QGauss<2> void_quadrature(_par.data.poly_degree + 1);
    const QGauss<2> cut_cell_quadrature(_par.data.cut_cell_quadrature);
    const QGauss<1> shape_quadrature(_par.data.poly_degree + 1);

    hp::QCollection<2> q_collection_boundary;
    q_collection_boundary.push_back(void_quadrature);
    q_collection_boundary.push_back(cut_cell_quadrature);

    FESystem<2> _fe_void(FE_Nothing<2>(), 1);

    FESystem<2> _fe_solid(FE_Q<2>(_par.data.poly_degree), 1);

    FESystem<1, 2> _fe_shape(FE_Q<1, 2>(_par.data.poly_degree), 2);

    hp::FECollection<2> fe_collection;

    fe_collection.push_back(_fe_void);
    fe_collection.push_back(_fe_solid);

    // FEValues for boundary
    hp::FEValues<2> fe_values_hp_boundary(fe_collection, q_collection_boundary,
                                          update_values | update_quadrature_points | update_JxW_values |
                                          update_gradients);

    FEValues<1, 2> fe_values_shape(_fe_shape, shape_quadrature,
                                   update_values | update_quadrature_points | update_JxW_values |
                                   update_normal_vectors);

    //const FEValuesExtractors::Vector displacement(0);
    //std::vector<Tensor<2,2> > local_displacement_gradients(cut_cell_quadrature.size());

    unsigned int dof_index, n_vertices, id;
    double shape_cell_measure;
    Tensor<1, 2> vertex_normal;
    Point<2> vertex;
    Point<2> v1, v2, segment_p1, segment_p2;
    Point<2> dummy_v1, dummy_v2, dummy_v3;

    std::vector<typename DoFHandler<2>::active_cell_iterator> adjacent_cells;
    std::vector<typename DoFHandler<1, 2>::active_cell_iterator> adjacent_cells_shape;

    std::vector<Point<2> > shape_polygon = _bvp._tria_shape.get_vertices();
    std::vector<std::vector<Point<2>>> dummy_subtriangulation;

    std::map<typename DoFHandler<1, 2>::active_cell_iterator, Tensor<1, 2>> shape_cell_normals;
    std::vector<Tensor<1, 2>> adjacent_shape_cell_normals;

    typename DoFHandler<2>::active_cell_iterator base_cell = _bvp._smoothing_dof_handler.begin_active();

    // setup cell normal vectors
    _SetupShapeCellNormals(fe_values_shape, shape_cell_normals);

    // integrate sensitivity_density alongside shape boundary
    n_vertices = _bvp._tria_shape.n_vertices();

    for (unsigned int v = 0; v < n_vertices; ++v) {
      dof_index = 0;
      adjacent_cells_shape = GridTools::find_cells_adjacent_to_vertex(_bvp._dof_handler_shape, v);
      adjacent_shape_cell_normals.clear();

      if (adjacent_cells_shape.empty())
        std::cout << "Could not establish adjacent_cells for shape vertex.." << std::endl;

      //Extract shape cell normals only for the adjacent shape cells
      for (const auto &adj_cell : adjacent_cells_shape)
        adjacent_shape_cell_normals.push_back(shape_cell_normals[adj_cell]);

      _ComputeCurrentVertexNormal(adjacent_shape_cell_normals, vertex_normal);

      bool flag = true;

      for (const auto &adj_cell : adjacent_cells_shape) {

        std::fill(shape_cell_gradient_densities.begin(), shape_cell_gradient_densities.end(), 0);

        id = adj_cell->material_id();

        if (flag) {
          if (adj_cell->vertex_index(0) == v) {
            vertex = adj_cell->vertex(0);
            dof_index = adj_cell->vertex_dof_index(0, 0);
          } else if (adj_cell->vertex_index(1) == v) {
            vertex = adj_cell->vertex(1);
            dof_index = adj_cell->vertex_dof_index(1, 0);
          } else
            std::cout << "Could not establish dof_index.." << std::endl;
        }
        flag = false;

        v1 = adj_cell->vertex(0);
        v2 = adj_cell->vertex(1);

        shape_cell_measure = adj_cell->measure();

        for (unsigned int base_c = 0; base_c < _bvp.boundary_cell_assignment[id].size(); ++base_c) {

          base_cell = _bvp.boundary_cell_assignment[id][base_c];
          if (!base_cell->is_locally_owned())
            continue;

          fe_values_hp_boundary.reinit(base_cell);

          if (EddTools::BaseCellIsCutByShapeCell(base_cell, v1, v2)) {

            // determine segment_p1 and segment_p2
            EddTools::DetermineCellIntersectionSegment(base_cell, v1, v2, segment_p1, segment_p2);

            // integrate cell contribution from segment_p1 to segment_p2
            std::fill(base_cell_gradient_densities.begin(), base_cell_gradient_densities.end(), 0);

            for (unsigned int response = 0; response < n_responses; ++response) {

              _responses[response]->IntegrateBaseCellShapeGradient(base_cell, segment_p1, segment_p2,
                                                              dummy_subtriangulation, dummy_v1, dummy_v2,
                                                              dummy_v3,
                                                              vertex, shape_cell_measure,
                                                              base_cell_gradient_densities[response]);

              // add cell contributions for current shape vertex
              shape_cell_gradient_densities[response] += base_cell_gradient_densities[response];
            }
          }
        }

        for (unsigned int response = 0; response < n_responses; ++response)
          for (unsigned int d = 0; d < 2; ++d)
            _mpilocal_function_gradients[response][dof_index + d] +=
                shape_cell_gradient_densities[response] * vertex_normal[d];

      }
      for (unsigned int d = 0; d < 2; ++d)
        _mpilocal_vertex_normals[dof_index + d] = vertex_normal[d];
    }

    // Sum up local to processor contributions
    for (unsigned int response = 0; response < n_responses; ++response) {

      current_function_gradient.reinit(_bvp._dof_handler_shape.n_dofs());

      for (unsigned int v = 0; v < n_vertices * 2; ++v) {
        MPI_Allreduce(&_mpilocal_function_gradients[response][v], &current_function_gradient[v],
                      1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      }
      _shape_gradients[response] = current_function_gradient;

    }

    for (unsigned int v = 0; v < n_vertices * 2; ++v) {
      MPI_Allreduce(&_mpilocal_vertex_normals[v], &_vertex_normals[v],
                    1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    for (unsigned int response = 0; response < n_responses; ++response)
      _bvp.shape_constraints.distribute(_shape_gradients[response]);


  }


  template<>
  void ResponseHandler<3>::_ComputeShapeGradients() {

    _shape_gradients.clear();
    unsigned int n_responses = _responses.size();

    for (unsigned int response = 0; response < n_responses; ++response)
      _responses[response]->ParseCommonData();

    // Vector of gradient densities correponds to gradient densitities of all responses in the problem
    std::vector<double> shape_cell_gradient_densities(n_responses);
    std::vector<double> base_cell_gradient_densities(n_responses);

    Vector<double> current_function_gradient(_bvp._dof_handler_shape.n_dofs());
    _vertex_normals.reinit(_bvp._dof_handler_shape.n_dofs());
    _mpilocal_vertex_normals.reinit(_bvp._dof_handler_shape.n_dofs());

    std::vector<Vector<double>> _mpilocal_function_gradients(n_responses);
    for (unsigned int response = 0; response < n_responses; ++response)
      _mpilocal_function_gradients[response].reinit(_bvp._dof_handler_shape.n_dofs());

    const QGauss<3> void_quadrature(_par.data.poly_degree + 1);
    const QGauss<3> cut_cell_quadrature(_par.data.cut_cell_quadrature);
    const QGauss<2> shape_quadrature(_par.data.poly_degree + 1);

    hp::QCollection<3> q_collection_boundary;
    q_collection_boundary.push_back(void_quadrature);
    q_collection_boundary.push_back(cut_cell_quadrature);

    FESystem<3> _fe_void(FE_Nothing<3>(), 1);

    FESystem<3> _fe_solid(FE_Q<3>(_par.data.poly_degree), 1);

    FESystem<2, 3> _fe_shape(FE_Q<2, 3>(1), 3);

    hp::FECollection<3> fe_collection;

    fe_collection.push_back(_fe_void);
    fe_collection.push_back(_fe_solid);

    // FEValues for boundary
    hp::FEValues<3> fe_values_hp_boundary(fe_collection, q_collection_boundary,
                                          update_values | update_quadrature_points | update_JxW_values |
                                          update_gradients);

    FEValues<2, 3> fe_values_shape(_fe_shape, shape_quadrature,
                                   update_values | update_quadrature_points | update_JxW_values |
                                   update_normal_vectors);

    Tensor<1, 3> vertex_normal;
    Point<3> vertex;
    Point<3> v1, v2, v3;
    Point<3> dummy_p1, dummy_p2;
    unsigned int dof_index, cell_vertex_index, id, n_vertices;
    double dummy_shape_cell_measure;

    n_vertices = _bvp._tria_shape.n_vertices();

    std::vector<typename DoFHandler<3>::active_cell_iterator> adjacent_cells;
    std::vector<typename DoFHandler<2, 3>::active_cell_iterator> adjacent_cells_shape;

    std::vector<Point<3>> shape_polygon = _bvp._tria_shape.get_vertices();
    std::vector<std::vector<Point<3>>> subtriangulation;
    std::vector<Point<3>> tmp_intersection_vertices;

    std::map<typename DoFHandler<2, 3>::active_cell_iterator, Tensor<1, 3>> shape_cell_normals;
    std::vector<Tensor<1, 3>> adjacent_shape_cell_normals;

    typename DoFHandler<3>::active_cell_iterator base_cell = _bvp._smoothing_dof_handler.begin_active();

    // setup cell normal vectors
    _SetupShapeCellNormals(fe_values_shape, shape_cell_normals);

    for (unsigned int v = 0; v < n_vertices; ++v) {
      dof_index = 0;
      adjacent_cells_shape = GridTools::find_cells_adjacent_to_vertex(_bvp._dof_handler_shape, v);
      adjacent_shape_cell_normals.clear();

      if (adjacent_cells_shape.empty())
        std::cout << "Could not establish adjacent_cells for shape vertex.." << std::endl;

      //Extract shape cell normals only for the adjacent shape cells
      for (const auto &adj_cell : adjacent_cells_shape)
        adjacent_shape_cell_normals.push_back(shape_cell_normals[adj_cell]);

      _ComputeCurrentVertexNormal(adjacent_shape_cell_normals, vertex_normal);

      for (auto &sha_c : adjacent_cells_shape) {
        cell_vertex_index = 0;
        if (sha_c->vertex_index(0) == v) {
          cell_vertex_index = 0;
          dof_index = sha_c->vertex_dof_index(0, 0);
          vertex = sha_c->vertex(0);
        } else if (sha_c->vertex_index(1) == v) {
          cell_vertex_index = 1;
          dof_index = sha_c->vertex_dof_index(1, 0);
          vertex = sha_c->vertex(1);
        } else if (sha_c->vertex_index(2) == v) {
          cell_vertex_index = 2;
          dof_index = sha_c->vertex_dof_index(2, 0);
          vertex = sha_c->vertex(2);
        } else if (sha_c->vertex_index(3) == v) {
          cell_vertex_index = 3;
          dof_index = sha_c->vertex_dof_index(3, 0);
          vertex = sha_c->vertex(3);
        } else {
          continue;
        }

        for (unsigned int tria = 0; tria < 2; ++tria) {
          id = sha_c->material_id();

          std::fill(shape_cell_gradient_densities.begin(), shape_cell_gradient_densities.end(), 0);

          if (tria == 0 && cell_vertex_index == 2)
            continue;
          else if (tria == 1 && cell_vertex_index == 1)
            continue;

          v1 = sha_c->vertex(0);

          if (tria == 0) {
            v2 = sha_c->vertex(1);
            v3 = sha_c->vertex(3);
          } else if (tria == 1) {
            v2 = sha_c->vertex(3);
            v3 = sha_c->vertex(2);
          }

          for (unsigned int base_c = 0; base_c < _bvp.boundary_cell_assignment[id].size(); ++base_c) {
            base_cell = _bvp.boundary_cell_assignment[id][base_c];
            if (!base_cell->is_locally_owned())
              continue;
            fe_values_hp_boundary.reinit(base_cell);

            if (EddTools::BaseCellIsCutByShapeTria(base_cell, v1, v2, v3, tmp_intersection_vertices) &&
                tmp_intersection_vertices.size() > 2) {
              // establish subtriangulation
              EddTools::TriangulateIntersectionVertices(tmp_intersection_vertices, subtriangulation);

              if (subtriangulation.empty())
                std::cout << "subtriangulation.size(): " << subtriangulation.size() << std::endl;

              // integrate volume sensitivity
              std::fill(base_cell_gradient_densities.begin(), base_cell_gradient_densities.end(), 0);

              for (unsigned int response = 0; response < n_responses; ++response) {

                _responses[response]->IntegrateBaseCellShapeGradient(base_cell, dummy_p1, dummy_p2,
                                                                subtriangulation, v1, v2, v3,
                                                                vertex, dummy_shape_cell_measure,
                                                                base_cell_gradient_densities[response]);

                // add cell contributions for current shape vertex
                shape_cell_gradient_densities[response] += base_cell_gradient_densities[response];
              }
            }
          }

          for (unsigned int response = 0; response < n_responses; ++response)
            for (unsigned int d = 0; d < 3; ++d)
              _mpilocal_function_gradients[response][dof_index + d] +=
                  shape_cell_gradient_densities[response] * vertex_normal[d];
        }
      }
      for (unsigned int d = 0; d < 3; ++d) {
        _mpilocal_vertex_normals[dof_index + d] = vertex_normal[d];
      }
    }

    // Sum up local to processor contributions
    for (unsigned int response = 0; response < n_responses; ++response) {

      current_function_gradient.reinit(_bvp._dof_handler_shape.n_dofs());

      for (unsigned int v = 0; v < n_vertices * 3; ++v) {
        MPI_Allreduce(&_mpilocal_function_gradients[response][v], &current_function_gradient[v],
                      1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      }
      _shape_gradients[response] = current_function_gradient;
    }

    for (unsigned int v = 0; v < n_vertices * 3; ++v) {
      MPI_Allreduce(&_mpilocal_vertex_normals[v], &_vertex_normals[v],
                    1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    for (unsigned int response = 0; response < n_responses; ++response)
      _bvp.shape_constraints.distribute(_shape_gradients[response]);
  }

  template<int dim>
  void ResponseHandler<dim>::_ComputeDensityGradients() {
    // Store the sensitivities in the map _density_gradients: response id -> map of gradient values
    _density_gradients.clear();
    double base_cell_gradient_density;

    // Iterate over base cells
    for (const auto &cell : _bvp._dof_handler_base.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;
      for (unsigned int response = 0; response < _responses_names.size(); ++response) {
        if (cell->material_id() == e_inside_cell) {
          _responses[response]->ComputeCellDensityGradient(cell, base_cell_gradient_density);
          _density_gradients[response][cell->id()] = base_cell_gradient_density;
        }
      }
    }
  }

  template<int dim>
  void ResponseHandler<dim>::_AddResponse(const unsigned int &id, const std::string &response_name) {
    if (response_name == "compliance")
      _responses[id] = std::make_unique<Compliance<dim>>(_par, _bvp);
    else if (response_name == "volume")
      _responses[id] = std::make_unique<Volume<dim>>(_par, _bvp);
    else
        throw std::runtime_error("Wrong response name.");
    _responses_names[id] = response_name;
  }


  template<int dim>
  void ResponseHandler<dim>::_OutputResults() {
    TimerOutput::Scope t(this->compute_timer,
                         "ResponseHandler<dim>::_OutputResults");

    // pushing data to output vector
    for (unsigned int id = 0; id < _responses_names.size(); ++id) {
      std::vector<std::string> grad_names(dim, _responses_names[id]);
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          grad_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
      _shape_output.template PushDataName<Vector<double>>(_shape_gradients[id], grad_names,
                                                          grad_component_interpretation,
                                                          &_bvp._dof_handler_shape);
    }

    _shape_output.WriteDataOutput();
  }


} // End of StructuralOptimization namespace


template
class StructuralOptimization::ResponseHandler<2>;

template
class StructuralOptimization::ResponseHandler<3>;