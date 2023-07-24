// C++ headers

// Deal.II headers

// Project headers
#include <Compliance.h>

namespace StructuralOptimization {

  template<int dim>
  Compliance<dim>::Compliance(Parameter &par_, EddBVP_<dim> &bvp_)
      :
      Response_<dim>(par_, bvp_),
      mpi_communicator(par_.mpi_communicator),
      n_mpi_processes(par_.n_mpi_processes),
      this_mpi_process(par_.this_mpi_process),
      pcout(par_.pcout),
      compute_timer(par_.compute_timer) {

  }


  template<int dim>
  Compliance<dim>::~Compliance() = default;


  template<int dim>
  double Compliance<dim>::GetFunction() {

#ifdef STOP_USE_TRILINOS
    Vector<double> global_solution(this->_bvp._solution);
    Vector<double> global_rhs(this->_bvp._system_rhs);
    _function_value = global_solution * global_rhs;
#else
    _function_value = this->_bvp._solution * this->_bvp._system_rhs;
#endif

    return _function_value;
  }


  template<>
  void Compliance<2>::IntegrateBaseCellShapeGradient(typename DoFHandler<2>::active_cell_iterator &cell,
                                                     Point<2> &segment_p1, Point<2> &segment_p2,
                                                     std::vector<std::vector<Point<2>>> &subtriangulation,
                                                     Point<2> &v1, Point<2> &v2, Point<2> &v3,
                                                     Point<2> &anchor,
                                                     double &shape_cell_measure,
                                                     double &base_cell_gradient_density) {

    // dim=2 --> ignore subtriangulation, v1, v2, v3 input
    (void) subtriangulation, (void) v1, (void) v2, (void) v3;

    base_cell_gradient_density = 0;

    unsigned int dofs_per_cell, dof_index;
    Point<2> q_point, unit_q_point;
    std::vector<Point<2> > q_points, unit_q_points;

    double stress_11, stress_12, stress_22;
    double strain_11, strain_12, strain_22;
    double tmp_cmp_density;
    std::vector<double> cmp_density;
    std::vector<unsigned int> local_dof_indices;

    double frac, d_ext, N_anchor, N_i;
    double segment_jacobian, segment_gauss_weight;

    const unsigned int n_q_points = 2;
    const double frac_1 = 0.2113249;
    const double frac_2 = 0.7886751;

    FESystem<2> _fe_smoothing(FE_Q<2>(this->_data.poly_degree), 1);

    // Get the stress and strain vectors from the BVP, use enum
    std::map<unsigned int, Vector<double>> postprocessing_data;

    for (unsigned int i = e_sym_grad_u_11; i <= e_sym_grad_u_22; ++i) {
      postprocessing_data[i] = _bvp.GetPostprocessingData(i);
    }
    for (unsigned int i = e_cauchy_s_11; i <= e_cauchy_s_22; ++i) {
      postprocessing_data[i] = _bvp.GetPostprocessingData(i);
    }

    q_points.clear();
    unit_q_points.clear();
    cmp_density.clear();

    // determine location of quadrature points within base cell
    for (unsigned int q = 0; q < n_q_points; ++q) {
      frac = frac_1;
      if (q == 1)
        frac = frac_2;

      q_point = (1 - frac) * segment_p1;
      q_point += frac * segment_p2;

      if (!cell->point_inside(q_point))
        std::cout << "Integration point not within cell domain.." << std::endl;;

      for (unsigned int d = 0; d < 2; ++d) {
        d_ext = 2 * std::abs((cell->center()[d] - cell->vertex(0)[d]));
        unit_q_point[d] = (q_point[d] - cell->vertex(0)[d]) / d_ext;
      }

      q_points.push_back(q_point);
      unit_q_points.push_back(unit_q_point);
    }

    // determine cmp density at each quadrature point
    dofs_per_cell = cell->get_fe().dofs_per_cell;
    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int q = 0; q < n_q_points; ++q) {
      unit_q_point = unit_q_points[q];

      stress_11 = 0;
      stress_12 = 0;
      stress_22 = 0;
      strain_11 = 0;
      strain_12 = 0;
      strain_22 = 0;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {

        dof_index = local_dof_indices[i];

        {
          N_i = _fe_smoothing.shape_value(i, unit_q_point);

          stress_11 += N_i * postprocessing_data[e_cauchy_s_11][dof_index];
          stress_12 += N_i * postprocessing_data[e_cauchy_s_12][dof_index];
          stress_22 += N_i * postprocessing_data[e_cauchy_s_22][dof_index];

          strain_11 += N_i * postprocessing_data[e_sym_grad_u_11][dof_index];
          strain_12 += N_i * postprocessing_data[e_sym_grad_u_12][dof_index];
          strain_22 += N_i * postprocessing_data[e_sym_grad_u_22][dof_index];
        }
      }

      tmp_cmp_density = stress_11 * strain_11 + 2 * stress_12 * strain_12 + stress_22 * strain_22;

      tmp_cmp_density *= -1;

      cmp_density.push_back(tmp_cmp_density);

    }

    segment_jacobian = segment_p1.distance(segment_p2);
    segment_gauss_weight = .5;

    // integrate quadrature point contributions
    for (unsigned int q = 0; q < n_q_points; ++q) {
      q_point = q_points[q];
      N_anchor = 1 - (anchor.distance(q_point) / shape_cell_measure);
      base_cell_gradient_density += N_anchor * cmp_density[q] * segment_gauss_weight * segment_jacobian;
    }
  }

  template<>
  void Compliance<3>::IntegrateBaseCellShapeGradient(typename DoFHandler<3>::active_cell_iterator &cell,
                                                     Point<3> &segment_p1, Point<3> &segment_p2,
                                                     std::vector<std::vector<Point<3>>> &subtriangulation,
                                                     Point<3> &v1, Point<3> &v2, Point<3> &v3,
                                                     Point<3> &anchor,
                                                     double &shape_cell_measure,
                                                     double &base_cell_gradient_density) {
    (void) segment_p1, (void) segment_p2, (void) shape_cell_measure;

    base_cell_gradient_density = 0;

    Tensor<1, 3> edge_a, edge_b, edge_c;
    Point<3> q_point, unit_q_point;
    Tensor<1, 3> b0, b1, b2;
    double len_a, len_b, len_c;
    double s, tmp_area, tmp_cmp_density, d_ext;
    double d00, d01, d11, d20, d21, denom;
    double u, v, w, N_anchor, N_i;
    double stress_11, stress_12, stress_22, stress_13, stress_23, stress_33;
    double strain_11, strain_12, strain_22, strain_13, strain_23, strain_33;
    unsigned int anchor_bary;
    unsigned int dof_index;
    std::vector<Point<3>> tmp_tria;
    std::vector<Point<3>> q_points, unit_q_points;
    std::vector<double> cmp_density;

    unsigned int dofs_per_cell;
    std::vector<unsigned int> local_dof_indices;
    unsigned int n_q_points = 3;
    double unit_tria_area = 0.5;
    double gauss_weight = 1.0 / 3.0;
    const double tol = 1.e-8;

    FESystem<3> _fe_smoothing(FE_Q<3>(this->_data.poly_degree), 1);

    // Get the stress and strain vectors from the BVP, use enum
    std::map<unsigned int, Vector<double>> postprocessing_data;

    for (unsigned int i = e_sym_grad_u_11; i <= e_sym_grad_u_33; ++i) {
      postprocessing_data[i] = _bvp.GetPostprocessingData(i);
    }
    for (unsigned int i = e_cauchy_s_11; i <= e_cauchy_s_33; ++i) {
      postprocessing_data[i] = _bvp.GetPostprocessingData(i);
    }

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
      std::cout << "could not establish anchor bary coord.." << std::endl;


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
      unit_q_points.clear();
      for (unsigned int q = 0; q < n_q_points; ++q) {
        if (q == 0)
          q_point = 0.5 * (tmp_tria[0] + tmp_tria[1]);
        else if (q == 1)
          q_point = 0.5 * (tmp_tria[0] + tmp_tria[2]);
        else if (q == 2)
          q_point = 0.5 * (tmp_tria[1] + tmp_tria[2]);

        for (unsigned int d = 0; d < 3; ++d) {
          d_ext = 2 * std::abs((cell->center()[d] - cell->vertex(0)[d]));
          unit_q_point[d] = (q_point[d] - cell->vertex(0)[d]) / d_ext;
        }

        q_points.push_back(q_point);
        unit_q_points.push_back(unit_q_point);
      }

      // determine cmp density at each quadrature point
      dofs_per_cell = cell->get_fe().dofs_per_cell;
      local_dof_indices.clear();
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      cmp_density.clear();
      for (unsigned int q = 0; q < n_q_points; ++q) {
        unit_q_point = unit_q_points[q];

        stress_11 = 0;
        stress_12 = 0;
        stress_22 = 0;
        stress_13 = 0;
        stress_23 = 0;
        stress_33 = 0;
        strain_11 = 0;
        strain_12 = 0;
        strain_22 = 0;
        strain_13 = 0;
        strain_23 = 0;
        strain_33 = 0;

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          dof_index = local_dof_indices[i];

          {
            N_i = _fe_smoothing.shape_value(i, unit_q_point);

            stress_11 += N_i * postprocessing_data[e_cauchy_s_11][dof_index];
            stress_12 += N_i * postprocessing_data[e_cauchy_s_12][dof_index];
            stress_22 += N_i * postprocessing_data[e_cauchy_s_22][dof_index];
            stress_13 += N_i * postprocessing_data[e_cauchy_s_13][dof_index];
            stress_23 += N_i * postprocessing_data[e_cauchy_s_23][dof_index];
            stress_33 += N_i * postprocessing_data[e_cauchy_s_33][dof_index];

            strain_11 += N_i * postprocessing_data[e_sym_grad_u_11][dof_index];
            strain_12 += N_i * postprocessing_data[e_sym_grad_u_12][dof_index];
            strain_22 += N_i * postprocessing_data[e_sym_grad_u_22][dof_index];
            strain_13 += N_i * postprocessing_data[e_sym_grad_u_13][dof_index];
            strain_23 += N_i * postprocessing_data[e_sym_grad_u_23][dof_index];
            strain_33 += N_i * postprocessing_data[e_sym_grad_u_33][dof_index];
          }
        }

        tmp_cmp_density =
            stress_11 * strain_11 + 2 * stress_12 * strain_12 + stress_22 * strain_22 + 2 * stress_13 * strain_13 +
            2 * stress_23 * strain_23 + stress_33 * strain_33;
        tmp_cmp_density *= -1;

        cmp_density.push_back(tmp_cmp_density);
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

        base_cell_gradient_density +=
            N_anchor * cmp_density[q] * gauss_weight * unit_tria_area * 2.0 * tmp_area;
      }
    }
  }

  template<int dim>
  void Compliance<dim>::ComputeCellDensityGradient(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                   double &base_cell_gradient_density) {

    // Get local solution
    const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
    Vector<double> cell_solution(dofs_per_cell);
    cell->get_dof_values(this->_bvp._solution, cell_solution);
    // Get local stiffness matrix
    FullMatrix<double> cell_stiffness_matrix = this->_bvp.GetCellStiffnessMatrix(cell->id());
    // Get penalty factor
    double penalty = this->_bvp.penalty;

    // Get the design variable
    double pseudo_density = this->_bvp.GetPseudoDensity(cell->id());
    // Compute the sensitivity
    cell_stiffness_matrix *= penalty * pow(pseudo_density, penalty-1.0);
    Vector<double> tmp(dofs_per_cell);

    if(!cell_stiffness_matrix.empty())
      cell_stiffness_matrix.vmult(tmp, cell_solution);

    base_cell_gradient_density = - (cell_solution * tmp);
  }

  template<int dim>
  void Compliance<dim>::ParseCommonData() {

  }

} // End of StructuralOptimization namespace


template
class StructuralOptimization::Compliance<2>;

template
class StructuralOptimization::Compliance<3>;
