// C++ headers

// Deal.II headers

// Project headers
#include <Volume.h>

namespace StructuralOptimization {

  template<int dim>
  Volume<dim>::Volume(Parameter &par_, EddBVP_<dim> &bvp_)
      :
      Response_<dim>(par_, bvp_),
      mpi_communicator(par_.mpi_communicator),
      n_mpi_processes(par_.n_mpi_processes),
      this_mpi_process(par_.this_mpi_process),
      pcout(par_.pcout),
      compute_timer(par_.compute_timer) {
  }


  template<int dim>
  Volume<dim>::~Volume() = default;


  template<int dim>
  double Volume<dim>::GetFunction() {

    _mpilocal_function_value = 0;

    // For detailed explanation of assembly process, refer to EddElasticityLin.cc
    QGauss<dim> quadrature_formula_inside(this->_data.poly_degree + 1);
    QGauss<dim> quadrature_formula_boundary(this->_data.cut_cell_quadrature);

    hp::QCollection<dim> q_collection_boundary, q_collection_inside;

    q_collection_inside.push_back(quadrature_formula_inside); // void cells
    q_collection_inside.push_back(quadrature_formula_inside); // solid cells

    q_collection_boundary.push_back(quadrature_formula_inside); // void cells
    q_collection_boundary.push_back(quadrature_formula_boundary); // solid cells



    FESystem<dim> _fe_void_disp_field(FE_Nothing<dim>(), dim);
    FESystem<dim> _fe_void_disp_electric_field(FE_Nothing<dim>(), dim, // displacement
                                               FE_Nothing<dim>(), 1); // potential

    FESystem<dim> _fe_solid_disp_field(FE_Q<dim>(this->_data.poly_degree), dim);
    FESystem<dim> _fe_solid_disp_electric_field(FE_Q<dim>(this->_data.poly_degree), dim, // displacement
                                                FE_Q<dim>(this->_data.poly_degree), 1); // potential

    FESystem<dim> _fe_solid_bimorph_elast_field(FE_Q<dim>(this->_data.poly_degree), dim, // displacement
                                                FE_Nothing<dim>(), 1); // potential

    hp::FECollection<dim> fe_collection;

    // From this point remember in all collections the first item is void and the 2nd item is solid.

    if (this->_data.bvp_type == "elasticityLin" || this->_data.bvp_type == "elasticityNonLin" ||
        this->_data.bvp_type == "normalModes") {
      fe_collection.push_back(_fe_void_disp_field);
      fe_collection.push_back(_fe_solid_disp_field);
    } else if (this->_data.bvp_type == "electroElasticityLin") {
      fe_collection.push_back(_fe_void_disp_electric_field);
      fe_collection.push_back(_fe_solid_disp_electric_field);
    } else if (this->_data.bvp_type == "bimorphPiezoStatic" || this->_data.bvp_type == "bimorphPiezoNormalModes") {
      q_collection_inside.push_back(quadrature_formula_inside); // solid cells
      q_collection_boundary.push_back(quadrature_formula_boundary); // solid cells

      fe_collection.push_back(_fe_void_disp_electric_field);
      fe_collection.push_back(_fe_solid_bimorph_elast_field);
      fe_collection.push_back(_fe_solid_disp_electric_field);
    } else
      throw std::runtime_error("Wrong bvp_type to compute volume constraint.");

    // FEValues for inside
    hp::FEValues<dim> fe_values_hp_inside(fe_collection, q_collection_inside,
                                          update_values | update_gradients | update_quadrature_points |
                                          update_JxW_values);

    // FEValues for boundary
    hp::FEValues<dim> fe_values_hp_boundary(fe_collection, q_collection_boundary,
                                            update_values | update_gradients | update_quadrature_points |
                                            update_JxW_values);

    for (const auto &cell : this->_bvp._dof_handler_base.active_cell_iterators()) {
      // if cell belong to some other domain, do nothing
      if (!cell->is_locally_owned())
        continue;

      if (cell->material_id() == e_outside_cell)
        continue;

      fe_values_hp_inside.reinit(cell);
      fe_values_hp_boundary.reinit(cell);

      // Check if the cell is inside the domain, so after this condition we do not have to worry about the outside cells.
      if (EddTools::CellIsInside<dim>(cell)) {
        // If the cell is inner, append whole cell volume
        if (cell->material_id() != e_boundary_cell) {
          if(this->_data.problem_type == "couple" && !first_call)
            _mpilocal_function_value += cell->measure() * this->_bvp.GetPseudoDensity(cell->id());
          else
            _mpilocal_function_value += cell->measure();
        }

          // Else the cell is a boundary cell
        else {
          const FEValues<dim> &fe_values = fe_values_hp_boundary.get_present_fe_values();

          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            Point<dim> q_point = fe_values.quadrature_point(q);

            if (GeometryAlgorithms::PointInShape(q_point, this->_bvp._tria_shape,
                                                 this->_bvp.min_face_length_base)) {
              _mpilocal_function_value += cell->measure() * 1. / fe_values.n_quadrature_points;
            }
          }
        }
      }
    }
    // Sum up local to processor contributions
    MPI_Allreduce(&_mpilocal_function_value, &_function_value,
                  1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    first_call = false;

    return _function_value;
  }


  template<>
  void Volume<2>::IntegrateBaseCellShapeGradient(typename DoFHandler<2>::active_cell_iterator &cell,
                                                 Point<2> &segment_p1, Point<2> &segment_p2,
                                                 std::vector<std::vector<Point<2>>> &subtriangulation,
                                                 Point<2> &v1, Point<2> &v2, Point<2> &v3,
                                                 Point<2> &anchor,
                                                 double &shape_cell_measure,
                                                 double &base_cell_gradient_density) {

    // dim=2 --> ignore subtriangulation, v1, v2, v3 input
    (void) subtriangulation, (void) v1, (void) v2, (void) v3;

    base_cell_gradient_density = 0;

    Point<2> q_point;
    std::vector<Point<2> > q_points;

    double frac, N_anchor;
    double segment_jacobian, segment_gauss_weight, vol_density;

    const unsigned int n_q_points = 2;
    const double frac_1 = 0.2113249;
    const double frac_2 = 0.7886751;

    q_points.clear();

    // determine location of quadrature points within hda cell
    for (unsigned int q = 0; q < n_q_points; ++q) {
      frac = frac_1;
      if (q == 1)
        frac = frac_2;

      q_point = (1 - frac) * segment_p1;
      q_point += frac * segment_p2;

      if (!cell->point_inside(q_point))
        std::cout << "Integration point not within cell domain.." << std::endl;

      q_points.push_back(q_point);
    }


    segment_jacobian = segment_p1.distance(segment_p2);
    segment_gauss_weight = .5;

    vol_density = 1;

    // integrate quadrature point contributions
    for (unsigned int q = 0; q < n_q_points; ++q) {
      q_point = q_points[q];
      N_anchor = 1 - (anchor.distance(q_point) / shape_cell_measure);
      base_cell_gradient_density += N_anchor * vol_density * segment_gauss_weight * segment_jacobian;
    }
  }


  template<>
  void Volume<3>::IntegrateBaseCellShapeGradient(typename DoFHandler<3>::active_cell_iterator &cell,
                                                 Point<3> &segment_p1, Point<3> &segment_p2,
                                                 std::vector<std::vector<Point<3>>> &subtriangulation,
                                                 Point<3> &v1, Point<3> &v2, Point<3> &v3,
                                                 Point<3> &anchor,
                                                 double &shape_cell_measure,
                                                 double &base_cell_gradient_density) {

    // dim=3 --> ignore segment_p1, segment_p2 and shape_cell_measure input
    (void) cell, (void) segment_p1, (void) segment_p2, (void) shape_cell_measure;

    base_cell_gradient_density = 0;

    Tensor<1, 3> edge_a, edge_b, edge_c;
    Point<3> q_point;
    Tensor<1, 3> b0, b1, b2;
    double len_a, len_b, len_c;
    double s, tmp_area;
    double d00, d01, d11, d20, d21, denom;
    double u, v, w, N_anchor;
    unsigned int anchor_bary;
    std::vector<Point<3>> tmp_tria;
    std::vector<Point<3>> q_points;

    unsigned int n_q_points = 3;
    double unit_tria_area = 0.5;
    double gauss_weight = 1.0 / 3.0;
    double vol_density = 1.0;
    const double tol = 1.e-8;

    b0 = v2 - v1;
    b1 = v3 - v1;

    anchor_bary = 0;
    if (anchor.distance(v1) < tol)
      anchor_bary = 1;
    else if (anchor.distance(v2) < tol)
      anchor_bary = 2;
    else if (anchor.distance(v3) < tol)
      anchor_bary = 3;

    if (anchor_bary == 0)
      std::cout << "Could not establish anchor bary coordinate." << std::endl;

    for (unsigned int tria = 0; tria < subtriangulation.size(); ++tria) {
      // get current tria
      tmp_tria = subtriangulation[tria];

      edge_a = tmp_tria[1] - tmp_tria[0];
      edge_b = tmp_tria[2] - tmp_tria[0];
      edge_c = tmp_tria[2] - tmp_tria[1];

      len_a = edge_a.norm();
      len_b = edge_b.norm();
      len_c = edge_c.norm();

      // evaluate area of current tria
      s = 0.5 * (len_a + len_b + len_c);
      tmp_area = std::pow(s * (s - len_a) * (s - len_b) * (s - len_c), 0.5);

      // determine quadrature point locations
      q_points.clear();
      for (unsigned int q = 0; q < n_q_points; ++q) {
        if (q == 0)
          q_point = 0.5 * (tmp_tria[0] + tmp_tria[1]);
        else if (q == 1)
          q_point = 0.5 * (tmp_tria[0] + tmp_tria[2]);
        else if (q == 2)
          q_point = 0.5 * (tmp_tria[1] + tmp_tria[2]);

        q_points.push_back(q_point);
      }

      // loop over quadrature points
      for (unsigned int q = 0; q < n_q_points; ++q) {
        q_point = q_points[q];
        b2 = q_point - v1;

        // barycentric coordinates (u,v,w) of q_point with respect to triangle (v1, v2, v3)
        d00 = b0 * b0;
        d01 = b0 * b1;
        d11 = b1 * b1;
        d20 = b2 * b0;
        d21 = b2 * b1;
        denom = d00 * d11 - d01 * d01;
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;

        N_anchor = 0;
        if (anchor_bary == 1)
          N_anchor = u;
        else if (anchor_bary == 2)
          N_anchor = v;
        else if (anchor_bary == 3)
          N_anchor = w;

        base_cell_gradient_density += N_anchor * vol_density * gauss_weight * unit_tria_area * 2.0 * tmp_area;
      }
    }
  }

  template<int dim>
  void Volume<dim>::ComputeCellDensityGradient(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                               double &base_cell_gradient_density) {

    base_cell_gradient_density = cell->measure();

  }

  template<int dim>
  void Volume<dim>::ParseCommonData() {

  }


} // End of StructuralOptimization namespace


template
class StructuralOptimization::Volume<2>;

template
class StructuralOptimization::Volume<3>;
