// C++ headers

// Deal.II headers

// Project headers
#include <Optimizer_.h>

namespace StructuralOptimization {

  template<int dim>
  Optimizer_<dim>::Optimizer_(Parameter &par_, Mesh<dim> &mesh_, EddBVP_<dim> &bvp_)
      :mpi_communicator(par_.mpi_communicator),
       this_mpi_process(par_.this_mpi_process),
       pcout(par_.pcout),
       compute_timer(par_.compute_timer),
       _data(par_.data),
       _mesh(mesh_),
       _bvp(bvp_),
       _regularization(par_),
       _response_handler(std::make_unique<ResponseHandler<dim>>(par_, bvp_)),
       _shape_output(std::make_unique<DataOutput<dim - 1, ShapeTriaType<dim>, DoFHandler<dim - 1, dim>>>
                         (par_.data, par_.mpi_communicator, par_.compute_timer, "shape-opt", mesh_.shape)),
       _base_output(std::make_unique<DataOutput<dim, BaseTriaType<dim>, DoFHandler<dim>>>
                        (par_.data, par_.mpi_communicator, par_.compute_timer, "base-opt", *mesh_.base))
      {

      }


  template<int dim>
  void Optimizer_<dim>:: _SetupShape() {

    _bvp._dof_handler_shape.clear();
    _bvp._dof_handler_shape.reinit(_mesh.shape);
    _bvp._dof_handler_shape.distribute_dofs(FESystem<dim - 1, dim>(FE_Q<dim - 1, dim>(_data.poly_degree), 2));

    _bvp.shape_constraints.clear();

    DoFTools::make_hanging_node_constraints(_bvp._dof_handler_shape, _bvp.shape_constraints);

    unsigned int dof_index;
    double comp = 0;

    for (const auto &shape_cell : _bvp._dof_handler_shape.active_cell_iterators()) {
      for (unsigned int id = 0; id < _data.non_design_id.size(); ++id) {

        if (shape_cell->material_id() == (unsigned int) _data.non_design_id[id]) {
          comp = _data.non_design_comp[id];

          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
            dof_index = shape_cell->vertex_dof_index(v, comp);
            _bvp.shape_constraints.add_line(dof_index);
            _bvp.shape_constraints.set_inhomogeneity(dof_index, 0);
          }
        }
      }
    }

    _bvp.shape_constraints.close();
  }

  template<int dim>
  void Optimizer_<dim>:: _WriteShapeOutput(const unsigned int &iter) {
    // Write shape and base output
    _shape_output->WriteDataOutput(iter);
  }

  template<int dim>
  void Optimizer_<dim>:: _WriteBaseOutput(const unsigned int &iter) {
    _response_handler->GetAdjointOutputData(*_base_output);
    PROJ_MPI_BARRIER

    _bvp.GetAllOutputData(*_base_output);
    PROJ_MPI_BARRIER

    _base_output->WriteDataOutput(iter);
  }


  template<int dim>
  void Optimizer_<dim>::_ExecuteRefinement() {
    _mesh.RefineShape();

    // Everytime shape is refined update desired narrow as well. This prevents the error segment not found in mesh tracking.
    _data.narrow = _data.narrow * 0.5;

    _bvp._dof_handler_shape.clear();
    _mesh.ResetBase(_bvp._dof_handler_base);
    _mesh.mesh_tracking->RunTracking();

  }

  template <int dim>
  void Optimizer_<dim>::_TrackDensities() {

    // Reduction of the pseudo densities
    std::vector<std::map<CellId, double>> tmp_vec_dens = Utilities::MPI::all_gather(mpi_communicator, _pseudo_densities);
    std::map<CellId, double> _pseudo_densities_reduced;
    for(const auto& dens_map : tmp_vec_dens)
      for (const auto&[cellid, val] : dens_map)
        _pseudo_densities_reduced[cellid] = val;

    // Reduction of the sensitivities
    std::vector<std::map<CellId, double>> tmp_vec_sens = Utilities::MPI::all_gather(mpi_communicator, _density_descent_direction);
    std::map<CellId, double> _density_descent_direction_reduced;
    for(const auto& sens_map : tmp_vec_sens)
      for (const auto&[cellid, val] : sens_map)
        _density_descent_direction_reduced[cellid] = val;

    // create neighbour list
    if (_neighbour_list.empty()) { // Assemble the list only once
      std::map<CellId, Point<dim>> cellId_center_proc;
      for (const auto &cell : _bvp._dof_handler_base.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;
        cellId_center_proc[cell->id()] = cell->center();
      }

      std::vector<std::map<CellId, Point<dim>>> temp_center = Utilities::MPI::all_gather(
          mpi_communicator, cellId_center_proc);

      std::map<CellId, Point<dim>> cellId_center;

      for(const auto& center_map : temp_center)
        for (const auto&[cellid, val] : center_map)
          if(cellId_center.find(cellid) == cellId_center.end())
            cellId_center[cellid] = val;

      for (const auto &cell : _bvp._dof_handler_base.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;

        for (const auto &[cellid, val] : cellId_center) {
          if(cell->id() == cellid)
            continue;

          Point<dim> center = cell->center();
          double distance = center.distance(val);
          if (distance < _data.initial_filter_radius)
            _neighbour_list[cell->id()][cellid] = _data.initial_filter_radius - distance;
        }
      }
    }
    // end of neighbour list creation

    std::map<CellId, bool> new_variables;

    // Track the new design variables and assign a filtered pseudo density value to them
    for (const auto& cell : this->_bvp._dof_handler_base.active_cell_iterators()) {

      new_variables[cell->id()] = false;

      if (!cell->is_locally_owned())
        continue;
      if (cell->material_id() != e_inside_cell)
        continue;
      if (_pseudo_densities[cell->id()] > 0.0 && _pseudo_densities[cell->id()] <= 1.0)
        continue;

      double weight_count = 0;

      _pseudo_densities[cell->id()] = 0.0;
      new_variables[cell->id()] = true;

      for (auto &[neighbour_id, dist] : _neighbour_list[cell->id()]) {
        if (_pseudo_densities_reduced[neighbour_id] < 1e-20)
          continue;

        // Consider boundary cells, but add density of 1.0 instead of 1.001
        if(_pseudo_densities_reduced[neighbour_id] > 1.0)
          _pseudo_densities[cell->id()] += 1.0 * dist;
        else
          _pseudo_densities[cell->id()] += _pseudo_densities_reduced[neighbour_id] * dist;
        weight_count += dist;
      }

      if (weight_count > 1e-8)
        _pseudo_densities[cell->id()] /= weight_count;
      else
        _pseudo_densities[cell->id()] = 1.0;

    }

    // The same for sensitivity
    for (const auto& cell : _bvp._dof_handler_base.active_cell_iterators()) {

      if (!cell->is_locally_owned())
        continue;
      if(!new_variables[cell->id()])
        continue;

      double weight_count = 0;
      _density_descent_direction[cell->id()] = 0.0;

      for (auto &[neighbour_id, dist] : _neighbour_list[cell->id()]) {
        if (_density_descent_direction_reduced.find(neighbour_id) == _density_descent_direction_reduced.end())
          continue;

        _density_descent_direction[cell->id()] += _density_descent_direction_reduced[neighbour_id] * dist;
        weight_count += dist;
      }

      if (weight_count > 1e-8)
        _density_descent_direction[cell->id()] /= (weight_count);
      else
        _density_descent_direction[cell->id()] = 0.0;
    }

    for (const auto& cell : _bvp._dof_handler_base.active_cell_iterators()) {
      if (cell->material_id() != e_inside_cell)
        _density_descent_direction.erase(cell->id());
    }
  }

  template<int dim>
  void Optimizer_<dim>::GenerateShapeHoles(){
    TimerOutput::Scope t(compute_timer,
                         "Optimizer_<dim>::GenerateShapeHoles");
    if(dim ==3)
      throw std::runtime_error ("will not work in 3D.");

    const std::vector<std::vector<Point<dim>>> vec_clusters = _SegregateCellsToIslands();
    const std::vector<std::vector<Point<dim>>> vec_polygons = _ConvertClusterToPolygons(vec_clusters, _mesh.mesh_tracking->min_face_length_base);
    _mesh.MergePolygonToShapeMesh(vec_polygons);
    _SetupShape();

  }

  template<int dim>
  void Optimizer_<dim>::MergeOverlappingHoles(){
    TimerOutput::Scope t(compute_timer,
                         "Optimizer_<dim>::MergeOverlappingHoles");
    if(dim ==3)
      throw std::runtime_error ("will not work in 3D.");

    std::map<unsigned int, std::vector<typename ShapeTriaType<dim>::active_cell_iterator>> map_matid_celliter;

    for(auto &shape_cell : _bvp._dof_handler_shape.active_cell_iterators()){
      if(shape_cell->material_id() < 100)
        map_matid_celliter[99].push_back(shape_cell);
      else
        map_matid_celliter[shape_cell->material_id()].push_back(shape_cell);
    }

    std::map<unsigned int, std::vector<unsigned int>> intersection_matrix;
    typename std::map<unsigned int, std::vector<typename ShapeTriaType<dim>::active_cell_iterator>>::iterator it1 = map_matid_celliter.begin(),
        it2 = map_matid_celliter.begin();

    for ( ; it1 != map_matid_celliter.end(); ++it1 ){

      it2 = it1;
      ++it2;
      if(it2 == map_matid_celliter.end())
        continue;
      for ( ; it2 != map_matid_celliter.end(); ++it2 ){
        bool exit_loop = false;
        for(auto &cell1 : it1->second){
          Point<dim> c1p1(cell1->vertex(0)), c1p2(cell1->vertex(1));
          for(auto &cell2 : it2->second){
            Point<dim> c2p1(cell2->vertex(0)), c2p2(cell2->vertex(1));
            if(GeometryAlgorithms::DoLinesIntersect(c1p1, c1p2, c2p1, c2p2)){
              intersection_matrix[it1->first].push_back(it2->first);
              exit_loop = true;
              break;
            }
          } // loop over it2->second
          if(exit_loop)
            break;
        } // loop over it1->second
      }
    }


    if(intersection_matrix.empty())
      return;

    std::vector<std::vector<unsigned int>> polygons_to_merge;
    for(const auto &[parent, children] : intersection_matrix){
      bool inserted_something = false;
      for(unsigned int it=0; it < polygons_to_merge.size(); ++it){
        if( std::find(polygons_to_merge[it].begin(), polygons_to_merge[it].end(), parent) != polygons_to_merge[it].end()){
          polygons_to_merge[it].insert(polygons_to_merge[it].end(), children.begin(), children.end());
          inserted_something = true;
          break;
        }
      }
      if(!inserted_something){
        polygons_to_merge.push_back({parent});
        polygons_to_merge.rbegin()->insert(polygons_to_merge.rbegin()->end(), children.begin(), children.end());
      }
    }

    // removing duplicates
    for(auto &vec : polygons_to_merge){
      std::sort(vec.begin(), vec.end());
      vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    }

    if(polygons_to_merge.empty())
      return;

    // collects all the points to merge.
    const unsigned int n_vertices = _mesh.shape.n_used_vertices();
    unsigned int vertex_index =0;
    std::vector<unsigned int> matid_to_merge;
    std::vector<std::vector<Point<dim>>> intersecting_points;
    for(const auto& vec_id : polygons_to_merge){
      intersecting_points.push_back({});
      for(const auto& id : vec_id){
        matid_to_merge.push_back(id);
        std::vector<bool> vertex_touched(n_vertices, false);
        for(auto& shape_cell : map_matid_celliter[id]){
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
            vertex_index = shape_cell->vertex_index(v);
            if (!vertex_touched[vertex_index]) {
              vertex_touched[vertex_index] = true;
              intersecting_points.rbegin()->push_back(shape_cell->vertex(v));
            } // if !vertex_touched
          } // end of loop over verticex per cell
        }
      }
    }

    std::vector<std::vector<Point<dim>>> after_points = _ConvertClusterToPolygons(intersecting_points, 0.0);

    _mesh.RemoveAndMergePolygonToShapeMesh(after_points, matid_to_merge);
    _SetupShape();
  }

  template<int dim>
  bool Optimizer_<dim>::ProjectShapeToTopologyFeature(){

    TimerOutput::Scope t(compute_timer,
                         "AlMoMOptimizer<dim>::ProjectShapeToTopologyFeature");

    // In case we find a non local base cell for projection
    std::vector<std::map<CellId, double>> tmp_vec_dens = Utilities::MPI::all_gather(mpi_communicator, _pseudo_densities);
    std::map<CellId, double> pseudo_densities_all_proc;
    for(const auto& dens_map : tmp_vec_dens)
      for (const auto&[cellid, val] : dens_map)
        pseudo_densities_all_proc[cellid] = val;

    // Some help variables
    const unsigned int n_vertices = _mesh.shape.n_used_vertices();
    std::vector<bool> vertex_touched(n_vertices, false);
    unsigned int vertex_index =0, dof_index=0;

    Vector<double> min_face_legth_vertex_normal(_response_handler->GetVertexNormals());
    Tensor<1,dim> vertex_grad;

    Vector<double> new_shape_pos(_bvp._dof_handler_shape.n_dofs());
    Vector<double> initial_shape_pos(_bvp._dof_handler_shape.n_dofs());
    std::vector<bool> should_move_vertex(n_vertices, true);

    // In below loop the following tasks are performed:
    // 1. based on material_id, if material_id == non_design_id the vertex is not considered.
    // 2. initialize new_shape_pos and initial_shape_pos
    // 3. initialize min_face_length_vertex_normal, which is an update vector. The direction is vertex normal vector and magniture is min_face_length_base
    // 4. check if neighbour in update dir is solid cell, if solid then the vertex is not considered.
    for (const auto &shape_cell : _bvp._dof_handler_shape.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
        vertex_index = shape_cell->vertex_index(v);

        if(std::find(_data.non_design_id.begin(), _data.non_design_id.end(), (int)shape_cell->material_id()) != _data.non_design_id.end()) {
          should_move_vertex[vertex_index] = false;
          vertex_touched[vertex_index] = true;
          continue;
        }

        if (!vertex_touched[vertex_index]) {
          vertex_touched[vertex_index] = true;
          for (unsigned int d = 0; d < dim; ++d) {
            dof_index = shape_cell->vertex_dof_index(v, d);
            vertex_grad[d] = min_face_legth_vertex_normal[dof_index];
            new_shape_pos[dof_index] = shape_cell->vertex(v)[d];
            initial_shape_pos[dof_index] = shape_cell->vertex(v)[d];
          }
          vertex_grad /= vertex_grad.norm();
          vertex_grad *= (-_mesh.mesh_tracking->min_face_length_base);// / dealii::numbers::SQRT2);
          for (unsigned int d = 0; d < dim; ++d) {
            dof_index = shape_cell->vertex_dof_index(v, d);
            min_face_legth_vertex_normal[dof_index] = vertex_grad[d];
          }

          try {
            Point<dim> updated_point = shape_cell->vertex(v) + vertex_grad;
            auto base_cell = GridTools::find_active_cell_around_point(_bvp._dof_handler_base, updated_point);

            //if(base_cell != _bvp._dof_handler_base.end()) {
              should_move_vertex[vertex_index] =
                  pseudo_densities_all_proc[base_cell->id()] < _data.quasi_void_threshold;
            //} else {
              //should_move_vertex[vertex_index] = false;
            //}
          } // end of try block
          catch (ExceptionBase &) {
            should_move_vertex[vertex_index] = false; // in this proc cell was not found continue moving
          }  // end of catch block
/*
          Point<dim> updated_point = shape_cell->vertex(v) + vertex_grad;
          auto base_cell = GridTools::find_active_cell_around_point(_bvp._dof_handler_base, updated_point);

          if(base_cell != _bvp._dof_handler_base.end()) {
            should_move_vertex[vertex_index] =
                pseudo_densities_all_proc[base_cell->id()] < _data.quasi_void_threshold;
          } else {
            should_move_vertex[vertex_index] = false;
          }
*/
        } // if vertex touched condition.
      } // end of loop over shape vertex

    } // end of loop over shape cells

    // MPI all gather.
    std::vector<std::vector<bool>> all_should_move_vertex = Utilities::MPI::all_gather(mpi_communicator, should_move_vertex);
    // even if one processor says we should move then we continue to move.
    // refill the should_move_vertex by checking across all proc.
    for(unsigned int vi =0; vi < should_move_vertex.size(); ++vi){
      std::vector<bool> vi_val_all_proc(Utilities::MPI::n_mpi_processes(mpi_communicator));
      for(unsigned int pi=0; pi < Utilities::MPI::n_mpi_processes(mpi_communicator); ++pi)
        vi_val_all_proc[pi] = all_should_move_vertex[pi][vi];
      if(std::find(vi_val_all_proc.begin(), vi_val_all_proc.end(), false) != vi_val_all_proc.end()) //all values are false
        should_move_vertex[vi] = false;
      else {
        should_move_vertex[vi] = true;
      }
    }

    // if non of the vertices should be moved, then we exit the function
    if(std::find(should_move_vertex.begin(), should_move_vertex.end(), true) == should_move_vertex.end())
      return false;

    unsigned int iter_counter=0;
    while(true) { // this loop is broken only when non of the vertices are allowed to move.

      // Move all vertex that should be moved, in the direction of design update by min_face_length_base
      // Then check in which base_cell the shape vertex lie using
      // GridTools::find_active_cell_around_point(), this leads to 3 scenario:
      // 1. exception of type GridTools::ExcPointNotFound -> the current proc does not have the cell. So should_move_vertex is set to false in this proc.
      // 2. cell found, density is less than threshold -> the vertex should continue march. So should_move_vertex is set to true
      // 3. cell found, density is GREATER than threshold -> the vertex has found its destination. should_move_vertex is set to FALSE.
      PROJ_MPI_BARRIER
      std::fill(vertex_touched.begin(), vertex_touched.end(), false);
      for (const auto &shape_cell : _bvp._dof_handler_shape.active_cell_iterators()) {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
          vertex_index = shape_cell->vertex_index(v);
          if(!should_move_vertex[vertex_index])
            continue;
          Point<dim> updated_point;
          if (!vertex_touched[vertex_index]) {
            vertex_touched[vertex_index] = true;
            for (unsigned int d = 0; d < dim; ++d) {
              dof_index = shape_cell->vertex_dof_index(v, d);
              new_shape_pos[dof_index] += min_face_legth_vertex_normal[dof_index];
              updated_point[d] = new_shape_pos[dof_index];
            }
/*
            auto base_cell = GridTools::find_active_cell_around_point(_bvp._dof_handler_base, updated_point);
            if(base_cell != _bvp._dof_handler_base.end())
              should_move_vertex[vertex_index] = pseudo_densities_all_proc[base_cell->id()] < _data.quasi_void_threshold;
            else
              should_move_vertex[vertex_index] = false;
*/

            try {
              auto base_cell = GridTools::find_active_cell_around_point(_bvp._dof_handler_base, updated_point);

              should_move_vertex[vertex_index] = pseudo_densities_all_proc[base_cell->id()] < _data.quasi_void_threshold;
            }
            catch (dealii::ExceptionBase &) {
              should_move_vertex[vertex_index] = false; // in this proc cell was not found continue moving
            }

          } // if vertex touched condition.
        } // end of loop over shape vertex
      } // end of loop over shape cells

      all_should_move_vertex.clear();
      PROJ_MPI_BARRIER
      all_should_move_vertex = Utilities::MPI::all_gather(mpi_communicator, should_move_vertex);

      // even if one processor says we should move then we continue to move.
      // refill the should_move_vertex by checking across all proc.
      for(unsigned int vi =0; vi < should_move_vertex.size(); ++vi){
        std::vector<bool> vi_val_all_proc(Utilities::MPI::n_mpi_processes(mpi_communicator));
        for(unsigned int pi=0; pi < Utilities::MPI::n_mpi_processes(mpi_communicator); ++pi)
          vi_val_all_proc[pi] = all_should_move_vertex[pi][vi];
        if(std::find(vi_val_all_proc.begin(), vi_val_all_proc.end(), false) != vi_val_all_proc.end()) //all values are false
          should_move_vertex[vi] = false;
        else
          should_move_vertex[vi] = true;
      }
      PROJ_MPI_BARRIER

      if(std::find(should_move_vertex.begin(), should_move_vertex.end(), true) == should_move_vertex.end()) {
        break;
      }

      if(iter_counter > 20) {  // just a safety check, to avoid infinite loop
        break;
      }

    } // end of while loop

    // Finally move shape
    Vector<double> merge_shape_update(new_shape_pos), before_shape;
    merge_shape_update -= initial_shape_pos;

    _regularization.RunSimpleTractionMethod(_bvp._dof_handler_shape, _bvp.shape_constraints,
                                                  merge_shape_update, _response_handler->GetVertexNormals());

    // Scale down the moving step to avoid unstable behavior.
    merge_shape_update *= _data.open_void_elim_factor;

/*
    std::vector<std::string> names(dim, "s-d_projection");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        proj_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    _shape_output->template PushDataName<Vector<double>>(merge_shape_update, names,
                                                         proj_component_interpretation,
                                                         &_bvp._dof_handler_shape);
*/
    _mesh.MoveShape(_bvp._dof_handler_shape, merge_shape_update, before_shape);

    PROJ_MPI_BARRIER
    _mesh.ResetBase(_bvp._dof_handler_base);
    _mesh.mesh_tracking->RunTracking();
    _TrackDensities();
    _bvp.SetPseudoDensities(_pseudo_densities);
    _bvp.Run();
    PROJ_MPI_BARRIER

    return true;

  }

  template<int dim>
  std::vector<std::vector<Point<dim>>> Optimizer_<dim>::_SegregateCellsToIslands() {

    std::vector<std::vector<std::pair<Point<dim>, CellId>>> clusters_proc;

    for(const auto& cell : this->_mesh.base->active_cell_iterators()){
      if(!cell->is_locally_owned())
        continue;

      if (cell->material_id() != e_inside_cell )
        continue;

      if(_pseudo_densities[cell->id()] > 0.5)
        continue;

      // check if cell->center() is diagonal distance away from any point.
      // if it is within the distance, then it belongs to the cluster.
      std::vector<unsigned int> cell_belong_to_cluster;
      unsigned int cluster_counter = 0;
      for (auto &clust : clusters_proc) {
        for (const auto &point_cellid : clust) {
          if (point_cellid.first.distance(cell->center()) <
              (dealii::numbers::SQRT2 * this->_mesh.mesh_tracking->min_face_length_base)) {
            if(std::find(cell_belong_to_cluster.begin(), cell_belong_to_cluster.end(), cluster_counter) == cell_belong_to_cluster.end())
              cell_belong_to_cluster.push_back(cluster_counter);
          }
        } // end of loop over points in a cluster
        ++cluster_counter;
      } // end of loop over clusters

      // if cell does not belong to any cluster, it forms a new cluster.
      if (cell_belong_to_cluster.empty()) {
        clusters_proc.push_back({std::make_pair(cell->center(), cell->id())});
      }

      // if size is 1, push the center to the corresponding cluster.
      if(cell_belong_to_cluster.size() == 1){
        clusters_proc[cell_belong_to_cluster[0]].push_back(std::make_pair(cell->center(), cell->id()));
      }

      // if cell belongs to more than one cluster, then it should be combined
      if (cell_belong_to_cluster.size() > 1) {
        std::sort(cell_belong_to_cluster.begin(), cell_belong_to_cluster.end(), std::greater<unsigned int>());
        clusters_proc[cell_belong_to_cluster[cell_belong_to_cluster.size()-1]].push_back(std::make_pair(cell->center(), cell->id()));
        for (unsigned int it = 0; it < (cell_belong_to_cluster.size()-1); ++it) {
          clusters_proc[cell_belong_to_cluster[cell_belong_to_cluster.size()-1]].insert(clusters_proc[cell_belong_to_cluster[cell_belong_to_cluster.size()-1]].end(),
                                                                                        clusters_proc[cell_belong_to_cluster[it]].begin(),
                                                                                        clusters_proc[cell_belong_to_cluster[it]].end());
          clusters_proc.erase(clusters_proc.begin() + cell_belong_to_cluster[it]);
        }
      }

    } // end of loop over cells.

    // get clusters from all processors. Check if clusters from different proc can be combined.
    PROJ_MPI_BARRIER
    std::vector<std::vector<std::pair<Point<dim>, CellId>>> final_clusters;
    if(Utilities::MPI::n_mpi_processes(mpi_communicator) > 1) {
      std::vector<std::vector<std::vector<std::pair<Point<dim>, CellId>>>> all_clusters = Utilities::MPI::all_gather(mpi_communicator,
                                                                                                                     clusters_proc);
      final_clusters = all_clusters[0];
      for(unsigned int proc = 1; proc < all_clusters.size(); ++proc){
        for(const auto& cluster : all_clusters[proc]){
          std::vector<unsigned int> i_belong_to_following_clusters;
          for(const auto& point_cellid : cluster){
            for(unsigned int f_index = 0; f_index < final_clusters.size(); ++f_index){
              for(const auto& f_point_cellid : final_clusters[f_index]){
                if(f_point_cellid.first.distance(point_cellid.first) <= (dealii::numbers::SQRT2 * _mesh.mesh_tracking->min_face_length_base * 1.00001) ){
                  if(std::find(i_belong_to_following_clusters.begin(), i_belong_to_following_clusters.end(), f_index) == i_belong_to_following_clusters.end())
                    i_belong_to_following_clusters.push_back(f_index);
                  break;
                }
              } // end of loop over points in f_cluster
            } // end of loop over cluster in final_clusters
            // combine cluster based on i_belong_to_following_clusters
          } // end of loop over points in cluster
          if(i_belong_to_following_clusters.empty()){
            final_clusters.push_back(cluster);
          }else if(i_belong_to_following_clusters.size() == 1){
            final_clusters[i_belong_to_following_clusters[0]].insert(
                final_clusters[i_belong_to_following_clusters[0]].end(),
                cluster.begin(),
                cluster.end());
          }else {
            final_clusters[i_belong_to_following_clusters[0]].insert(
                final_clusters[i_belong_to_following_clusters[0]].end(),
                cluster.begin(),
                cluster.end());
            for (unsigned int i_belong_index = 1; i_belong_index < i_belong_to_following_clusters.size(); ++i_belong_index) {
              final_clusters[i_belong_to_following_clusters[0]].insert(
                  final_clusters[i_belong_to_following_clusters[0]].end(),
                  final_clusters[i_belong_to_following_clusters[i_belong_index]].begin(),
                  final_clusters[i_belong_to_following_clusters[i_belong_index]].end());
              final_clusters.erase(final_clusters.begin() + i_belong_to_following_clusters[i_belong_index]);
            }
          } // if condition
        } // end of loop over cluster in a proc
      } // end of loop over all_clusters
    }
    else{
      final_clusters = clusters_proc;
    }

    // to remove clusters touching boundary
    for(const auto& shape_cell : _mesh.shape.active_cell_iterators()) {
      // check if boundary cell->center() is diagonal distance away from any point.
      // if it is within the distance, then the cluster touches the boundary and the boundary should be removed.
      std::vector<unsigned int> boundary_cell_touch_cluster;
      unsigned int cluster_counter = 0;
      for (auto &clust : final_clusters) {
        for (const auto &point_cellid : clust) {
          if (point_cellid.first.distance(shape_cell->center()) <
              (dealii::numbers::SQRT2 * _mesh.mesh_tracking->min_face_length_base)) {
            if(std::find(boundary_cell_touch_cluster.begin(), boundary_cell_touch_cluster.end(), cluster_counter) == boundary_cell_touch_cluster.end())
              boundary_cell_touch_cluster.push_back(cluster_counter);
          }
        } // end of loop over points in a cluster
        ++cluster_counter;
      } // end of loop over clusters

      // if boundary cell touch a cluster, then the cluster should be removed from the list of clusters
      std::sort(boundary_cell_touch_cluster.begin(), boundary_cell_touch_cluster.end(), std::greater<unsigned int>());
      for(const auto& index : boundary_cell_touch_cluster){
        final_clusters.erase(final_clusters.begin() + index);
      }
    }

    double area_of_a_cell = _mesh.mesh_tracking->min_face_length_base * _mesh.mesh_tracking->min_face_length_base;
    double filter_area = dealii::numbers::PI * std::pow(_data.initial_filter_radius, 2);
    unsigned int min_num_cells = 16 > std::ceil(_data.hole_area_threshold/area_of_a_cell) ? 16 : std::ceil(filter_area/area_of_a_cell);

    // removing cluster smaller than 8 elements
    auto iter = std::remove_if(final_clusters.begin(),
                               final_clusters.end(),
                               [&](std::vector<std::pair<Point<dim>, CellId>> vec){
                                 return vec.size()<min_num_cells;});
    final_clusters.erase(iter, final_clusters.end());

    std::vector<std::map<CellId, double>> tmp_vec_dens = Utilities::MPI::all_gather(mpi_communicator, _pseudo_densities);
    std::map<CellId, double> _pseudo_densities_reduced;
    for(const auto& dens_map : tmp_vec_dens)
      for (const auto&[cellid, val] : dens_map)
          _pseudo_densities_reduced[cellid] = val;


    //get the average density in each cluster.
    std::vector<double> avg_density_of_clusters;
    for (const auto &f_clust : final_clusters) {
      double sum_density = 0.0;
      for (const auto &f_point_cellid : f_clust) {
        sum_density += _pseudo_densities_reduced[f_point_cellid.second];
      }
      avg_density_of_clusters.push_back(sum_density / f_clust.size());
    }

    //remove cluster with avg density less than quasi_void_threshold
    for (unsigned int i = 0; i < avg_density_of_clusters.size(); ++i) {
      if (avg_density_of_clusters[i] > 0.25)
        final_clusters.erase(final_clusters.begin() + i);
    }

    // strip the cellid and prepare vector to return.
    std::vector<std::vector<Point<dim>>> result;
    for(const auto& f_clust : final_clusters){
      std::vector<Point<dim>> r_clust;
      for(const auto& f_point_cellid : f_clust){
        r_clust.push_back(f_point_cellid.first);
      }
      result.push_back(r_clust);
    }

    return result;
  }

  template<int dim>
  std::vector<std::vector<Point<dim>>> Optimizer_<dim>::_ConvertClusterToPolygons(const std::vector<std::vector<Point<dim>>>& vec_clusters, double polygon_expansion_length) {

    std::vector<std::vector<Point<dim>>> vec_polygons;
    for(const auto& cluster : vec_clusters){
      std::vector<Point<dim>> polygon = _ConvexHullGeneration(cluster, polygon_expansion_length, _mesh.avg_shape_el_size);
      vec_polygons.push_back(polygon);
    }

    return vec_polygons;
  }

  template<int dim>
  std::vector<Point<dim>> Optimizer_<dim>::_ConvexHullGeneration(const std::vector<Point<dim>> &cluster, double hull_expansion_length, double segment_size){
    std::vector<Point<dim>> hull;

    unsigned int n = cluster.size();
    // a[i].second -> y-coordinate of the ith point
    if (n < 3)
    {
      throw std::runtime_error ("Convex hull not possible");
    }

    // Finding the point with minimum and
    // maximum x-coordinate
    int min_x = 0, max_x = 0;
    for (unsigned int i=1; i<n; i++)
    {
      if (cluster[i][0] < cluster[min_x][0])
        min_x = i;
      if (cluster[i][0] > cluster[max_x][0])
        max_x = i;
    }
    // Recursively find convex hull points on
    // one side of line joining a[min_x] and
    // a[max_x]
    GeometryAlgorithms::QuickHull(hull, cluster, n, cluster[min_x], cluster[max_x], 1);


    // Recursively find convex hull points on
    // other side of line joining a[min_x] and
    // a[max_x]
    GeometryAlgorithms::QuickHull(hull, cluster, n, cluster[min_x], cluster[max_x], -1);

    // remove duplicate points, algorithm generates duplicate points which have to removed.
    auto end = hull.end();
    for (auto it = hull.begin(); it != end; ++it) {
      end = std::remove(it + 1, end, *it);
    }
    hull.erase(end, hull.end());

    std::vector<Point<dim>> polygon = GeometryAlgorithms::GeneratePolygon(hull,
                                                                          segment_size,
                                                                          hull_expansion_length);

    return polygon;
  }

  template<int dim>
  void Optimizer_<dim>::_MapL2Normalize(std::map<dealii::CellId, double> &map) {

    std::vector<std::map<CellId, double>> vec_map = Utilities::MPI::all_gather(mpi_communicator, map);
    std::map<CellId, double> map_reduced;
    for(const auto& map_proc : vec_map)
      for (const auto&[cellid, val] : map_proc)
        map_reduced[cellid] = val;

    Vector<double> vec(map_reduced.size());
    unsigned int vec_i = 0;
    for(const auto &[cellid, val] : map_reduced) {
      vec[vec_i] = val;
      ++vec_i;
    }
    double vec_l2norm = vec.l2_norm();

    for(auto &[cellid, val] : map)
      val /= vec_l2norm;
  }

  template<int dim>
  Vector<double> Optimizer_<dim>::GenerateVectorFromMap(std::map<CellId, double> data) {

    Vector<double> result(_mesh.base->n_active_cells());
    int i = 0;
    for (auto cell : _mesh.base->active_cell_iterators()) {
      if(!cell->is_locally_owned()) {
        ++i;
        continue;
      }
      result[i] = data[cell->id()];
      ++i;
    }

    return result;

  }

  template<int dim>
  void Optimizer_<dim>::_ComputeDensityStatistics() {

    double n_design_cells_proc=0,
        n_solid_cells_proc=0,
        n_void_cells_proc=0,
        n_grey_cells_proc=0;

    for(const auto &[cellid, dens] : _pseudo_densities){
      if(dens > 1e-4){
        ++n_design_cells_proc;
        if(dens < _data.quasi_void_threshold)
          ++n_void_cells_proc;
        else if(dens > _data.quasi_solid_threshold)
          ++n_solid_cells_proc;
        else
          ++n_grey_cells_proc;
      }
    }

    n_design_cells = Utilities::MPI::sum(n_design_cells_proc, mpi_communicator);
    n_solid_cells = Utilities::MPI::sum(n_solid_cells_proc, mpi_communicator);
    n_void_cells = Utilities::MPI::sum(n_void_cells_proc, mpi_communicator);
    n_grey_cells = Utilities::MPI::sum(n_grey_cells_proc, mpi_communicator);

    if(this->_data.verbose)
      if(this_mpi_process ==0)
        std::cout << "n_design_cells= " << n_design_cells
                  << " | n_solid_cells= " << n_solid_cells << " (" << ((100.0*n_solid_cells)/n_design_cells) << "%)"
                  << " | n_void_cells= " << n_void_cells << " (" << ((100.0*n_void_cells)/n_design_cells) << "%)"
                  << " | n_grey_cells= " << n_grey_cells << " (" << ((100.0*n_grey_cells)/n_design_cells) << "%)" << std::endl;

    PROJ_MPI_BARRIER

  }

}// End of StructuralOptimization namespace


template
class StructuralOptimization::Optimizer_<2>;

template
class StructuralOptimization::Optimizer_<3>;
