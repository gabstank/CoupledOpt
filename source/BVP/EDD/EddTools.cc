// C++ headers

// Deal.II headers

// Project headers
#include <EddTools.h>

namespace StructuralOptimization {

  namespace EddTools {

    //----------------------------------------------2D methods-----------------------------------------------------

    bool BaseCellIsCutByShapeCell(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                              Point<2> &sha_v1, Point<2> &sha_v2) {
      Assert(base_cell->is_locally_owned(), ExcMessage("Cell is not locally owned"));

      bool tmp = false, one_is_left = false, one_is_right = false;
      unsigned int counter = 0;
      std::vector<unsigned int> marked_vertices;
      double x_min, x_max, y_min, y_max, cross_product;
      Point<2> vertex, segment, tmp_segment, v1_dist, v2_dist, segment_dist;


      x_min = sha_v1[0];
      x_max = sha_v1[0];
      y_min = sha_v1[1];
      y_max = sha_v1[1];

      if (sha_v2[0] < x_min)
        x_min = sha_v2[0];
      if (sha_v2[0] > x_max)
        x_max = sha_v2[0];
      if (sha_v2[1] < y_min)
        y_min = sha_v2[1];
      if (sha_v2[1] > y_max)
        y_max = sha_v2[1];

      marked_vertices.clear();
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v) {
        vertex = base_cell->vertex(v);

        if (vertex[0] < x_max && vertex[0] > x_min) {
          ++counter;
          marked_vertices.push_back(v);
        }

        if (vertex[1] < y_max && vertex[1] > y_min) {
          ++counter;
          marked_vertices.push_back(v);
        }
      }

      segment = sha_v2;
      segment -= sha_v1;

      if (counter > 1) {
        //check if at least one point is on left side and one point is on right side of shape_segment
        for (unsigned int i = 0; i < marked_vertices.size(); ++i) {
          vertex = base_cell->vertex(marked_vertices[i]);

          tmp_segment = vertex;
          tmp_segment -= sha_v1;

          cross_product = segment[0] * tmp_segment[1] - segment[1] * tmp_segment[0];

          if (cross_product > 0) {
            one_is_right = true;
          } else {
            one_is_left = true;
          }
        }
      }

      // this is for the special case that one shape_vertex lies at the boundary of a hold_all cell
      segment_dist = segment;
      segment_dist *= 1.e-8;

      v1_dist = sha_v1;
      v1_dist += segment_dist;

      v2_dist = sha_v2;
      v2_dist -= segment_dist;

      if ((one_is_left && one_is_right) || base_cell->point_inside(v1_dist) || base_cell->point_inside(v2_dist))
        tmp = true;

      return tmp;

    }

    // _return_segment_intersection [dim=2]
    // http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    Point<2> ReturnSegmentIntersection(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                                   unsigned int &f, Point<2> &sha_v1, Point<2> &sha_dir) {
      Assert(base_cell->is_locally_owned(), ExcMessage("Cell is not locally owned"));

      Point<2> p, q, r, s;
      Point<2> q_p, s_tmp, intersect;

      q = sha_v1;
      s = sha_dir;

      p = base_cell->face(f)->vertex(0);
      r = base_cell->face(f)->vertex(1);
      r -= p;

      double t = -1;
      double r_s = r[0] * s[1] - r[1] * s[0];

      q_p = q;
      q_p -= p;
      s_tmp = s;
      s_tmp /= r_s;
      t = q_p[0] * s_tmp[1] - q_p[1] * s_tmp[0];

      intersect = r;
      intersect *= t;
      intersect += p;

      return intersect;
    }

    bool ShapeCellIsCutByBaseFace(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                              unsigned int &f, Point<2> &sgm_p1, Point<2> &sgm_dir) {
      Assert(base_cell->is_locally_owned(), ExcMessage("Cell is not locally owned"));
      bool tmp = false;

      double cross_product1, cross_product2;
      Point<2> face_v1, face_v2, vector1, vector2;

      face_v1 = base_cell->face(f)->vertex(0);
      face_v2 = base_cell->face(f)->vertex(1);

      vector1 = face_v1;
      vector1 -= sgm_p1;

      vector2 = face_v2;
      vector2 -= sgm_p1;

      cross_product1 = sgm_dir[0] * vector1[1] - sgm_dir[1] * vector1[0];
      cross_product2 = sgm_dir[0] * vector2[1] - sgm_dir[1] * vector2[0];

      if (cross_product1 * cross_product2 < 0)
        tmp = true;

      return tmp;
    }

    bool BaseFaceIsCutByShapeCell(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                              unsigned int &f, Point<2> &sha_v1, Point<2> &sha_v2) {
      Assert(base_cell->is_locally_owned(), ExcMessage("Cell is not locally owned"));

      bool tmp = false;

      double cross_product1, cross_product2;
      Point<2> face_v1, face_v2, face_vector;
      Point<2> vector1, vector2;

      face_v1 = base_cell->face(f)->vertex(0);
      face_v2 = base_cell->face(f)->vertex(1);

      face_vector = face_v2;
      face_vector -= face_v1;

      vector1 = sha_v1;
      vector1 -= face_v1;

      vector2 = sha_v2;
      vector2 -= face_v1;

      cross_product1 = face_vector[0] * vector1[1] - face_vector[1] * vector1[0];
      cross_product2 = face_vector[0] * vector2[1] - face_vector[1] * vector2[0];

      if (cross_product1 * cross_product2 < 0)
        tmp = true;

      return tmp;
    }

    void DetermineCellIntersectionSegment(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                                      Point<2> &sha_v1, Point<2> &sha_v2,
                                                      Point<2> &sgm_p1, Point<2> &sgm_p2) {
      Assert(base_cell->is_locally_owned(), ExcMessage("Cell is not locally owned"));


      bool v1_in_base = base_cell->point_inside(sha_v1);
      bool v2_in_base = base_cell->point_inside(sha_v2);

      if(v1_in_base && v2_in_base){
        sgm_p1 = sha_v1;
        sgm_p2 = sha_v2;
        return;
      }

      Point<2> segment_vector, intersect;
      bool is_corner_cell = false;
      bool segment_p1_is_set = false;
      bool segment_p2_is_set = false;

      if (v1_in_base || v2_in_base)
        is_corner_cell = true;

      segment_vector = sha_v2;
      segment_vector -= sha_v1;

      if (base_cell->point_inside(sha_v1) || base_cell->point_inside(sha_v2))
        is_corner_cell = true;

      if (is_corner_cell) {
        // determine segment_p1
        if (base_cell->point_inside(sha_v1))
          sgm_p1 = sha_v1;
        else
          sgm_p1 = sha_v2;

        // determine segment_p2
        for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
          if (ShapeCellIsCutByBaseFace(base_cell, f, sgm_p1, segment_vector)) {
            if (BaseFaceIsCutByShapeCell(base_cell, f, sha_v1, sha_v2)) {
              intersect = ReturnSegmentIntersection(base_cell, f, sha_v1, segment_vector);
              sgm_p2 = intersect;
              segment_p2_is_set = true;
            }
          }
        }

        // check for segment_p2
        if (!segment_p2_is_set) {
          std::cout << "did not find segment_p2.. at cell_center: " << base_cell->center() << std::endl;
          std::cout << "v1: " << sha_v1 << " -- v2: " << sha_v2 << std::endl;
          // safeguard: zero contribution
          sgm_p2 = sgm_p1;
        }
      } else if (!is_corner_cell) {
        // determine segment_p1 and segment_p2
        for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
          if (ShapeCellIsCutByBaseFace(base_cell, f, sha_v1, segment_vector)) {
            if (BaseFaceIsCutByShapeCell(base_cell, f, sha_v1, sha_v2)) {
              intersect = ReturnSegmentIntersection(base_cell, f, sha_v1, segment_vector);

              if (!segment_p1_is_set) {
                sgm_p1 = intersect;
                segment_p1_is_set = true;
              } else
                sgm_p2 = intersect;
            }
          }
        }
      }
    }

//-------------------------------------------------3D methods--------------------------------------------------------

    bool _BaseFaceIsCutByShapeTriaEdge(typename DoFHandler<3>::active_cell_iterator &base_cell,
                                                   unsigned int &f,
                                                   Point<3> start, Point<3> dir,
                                                   std::vector<Point<3> > &cell_intersections) {

      bool tmp = false;
      bool is_duplicate = false;

      double det, inv_det;
      double u, v, t;
      const double tol = 1.e-6;

      Point<3> base_base, base_edge_a, base_edge_b;
      Point<3> v_p, v_q, v_t;
      Tensor<1, 3> t_v_p, t_v_q;
      Point<3> intersect;


      for (unsigned int face_tria = 0; face_tria < 4; ++face_tria) {

        if (face_tria == 0) {
          base_base = base_cell->face(f)->vertex(0);
          base_edge_a = base_cell->face(f)->vertex(1);
          base_edge_b = base_cell->face(f)->vertex(3);
        } else if (face_tria == 1) {
          base_base = base_cell->face(f)->vertex(0);
          base_edge_a = base_cell->face(f)->vertex(3);
          base_edge_b = base_cell->face(f)->vertex(2);
        } else if (face_tria == 2) {
          base_base = base_cell->face(f)->vertex(1);
          base_edge_a = base_cell->face(f)->vertex(0);
          base_edge_b = base_cell->face(f)->vertex(2);
        } else if (face_tria == 3) {
          base_base = base_cell->face(f)->vertex(1);
          base_edge_a = base_cell->face(f)->vertex(2);
          base_edge_b = base_cell->face(f)->vertex(3);
        }

        base_edge_a -= base_base;
        base_edge_b -= base_base;

        t_v_p = cross_product_3d(dir, base_edge_b);
        for (unsigned int d = 0; d < 3; ++d) {
          v_p[d] = t_v_p[d];
        }
        det = base_edge_a * v_p;

        if (std::abs(det) < tol)
          continue;

        inv_det = 1 / det;

        v_t = start;
        v_t -= base_base;

        u = v_p * v_t * inv_det;
        if (u < tol || u > 1 + tol)
          continue;

        t_v_q = cross_product_3d(v_t, base_edge_a);
        for (unsigned int d = 0; d < 3; ++d) {
          v_q[d] = t_v_q[d];
        }

        v = dir * v_q * inv_det;
        if (v < tol || u + v > 1 + tol)
          continue;

        t = base_edge_b * v_q * inv_det;
        if (t > -tol && t < 1 + tol) {
          tmp = true;
          intersect = dir;
          intersect *= t;
          intersect += start;

          if (cell_intersections.size() == 0) {
            cell_intersections.push_back(intersect);
          } else {
            for (unsigned int s = 0; s < cell_intersections.size(); ++s) {
              if (intersect.distance(cell_intersections[s]) < tol)
                is_duplicate = true;
            }
            if (!is_duplicate)
              cell_intersections.push_back(intersect);
          }
        }
      }

      return tmp;
    }

    bool _ShapeTriaIsCutByBaseEdge(Point<3> &tria_base, Point<3> &tria_edge_a, Point<3> &tria_edge_b,
                                               Point<3> &start, Point<3> &dir,
                                               std::vector<Point<3> > &cell_intersections) {

      bool tmp = false;
      bool is_duplicate = false;

      double det, inv_det;
      double u, v, t;
      const double tol = 1.e-6;

      Point<3> v_p, v_q, v_t;
      Tensor<1, 3> t_v_p, t_v_q;
      Point<3> intersect;

      t_v_p = cross_product_3d(dir, tria_edge_b);
      for (unsigned int d = 0; d < 3; ++d) {
        v_p[d] = t_v_p[d];
      }
      det = tria_edge_a * v_p;

      if (std::abs(det) < tol)
        return false;

      inv_det = 1 / det;

      v_t = start;
      v_t -= tria_base;

      u = v_p * v_t * inv_det;
      if (u < tol || u > 1 + tol)
        return false;

      t_v_q = cross_product_3d(v_t, tria_edge_a);
      for (unsigned int d = 0; d < 3; ++d) {
        v_q[d] = t_v_q[d];
      }

      v = dir * v_q * inv_det;
      if (v < tol || u + v > 1 + tol)
        return false;

      t = tria_edge_b * v_q * inv_det;
      if (t > -tol && t < 1 + tol) {
        tmp = true;
        intersect = dir;
        intersect *= t;
        intersect += start;

        if (cell_intersections.size() == 0) {
          cell_intersections.push_back(intersect);
        } else {
          for (unsigned int s = 0; s < cell_intersections.size(); ++s) {
            if (intersect.distance(cell_intersections[s]) < tol)
              is_duplicate = true;
          }
          if (!is_duplicate)
            cell_intersections.push_back(intersect);
        }
      }
      return tmp;
    }

    bool BaseCellIsCutByShapeTria(typename DoFHandler<3>::active_cell_iterator &base_cell,
                                              Point<3> &sha_v1, Point<3> &sha_v2, Point<3> &sha_v3,
                                              std::vector<Point<3>> &cell_intersections) {

      bool tmp = false;

      Point<3> base, edge_dir;
      Point<3> start, dir;
      Point<3> edge_a, edge_b, edge_c, edge;

      cell_intersections.clear();

      edge_a = sha_v2;
      edge_a -= sha_v1;

      edge_b = sha_v3;
      edge_b -= sha_v1;

      edge_c = sha_v2;
      edge_c -= sha_v3;

      // test if base_cell contains shape_tria vertex
      if (base_cell->point_inside(sha_v1)) {
        cell_intersections.push_back(sha_v1);
        tmp = true;
      }
      if (base_cell->point_inside(sha_v2)) {
        cell_intersections.push_back(sha_v2);
        tmp = true;
      }
      if (base_cell->point_inside(sha_v3)) {
        cell_intersections.push_back(sha_v3);
        tmp = true;
      }


      // test if shape_tria edges intersect base_cell_faces
      for (unsigned int shape_edge = 0; shape_edge < 3; ++shape_edge) {
        if (shape_edge == 0) {
          base = sha_v1;
          edge_dir = edge_a;
        } else if (shape_edge == 1) {
          base = sha_v1;
          edge_dir = edge_b;
        } else if (shape_edge > 1) {
          base = sha_v3;
          edge_dir = edge_c;
        }

        for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f) {
          if (_BaseFaceIsCutByShapeTriaEdge(base_cell, f, base, edge_dir, cell_intersections))
            tmp = true;
        }
      }


      // test if base_cell_edges intersect shape_tria
      base = sha_v1;
      for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f) {

        for (unsigned int e = 0; e < 4; ++e) {
          if (e == 0) {
            start = base_cell->face(f)->vertex(0);
            dir = base_cell->face(f)->vertex(1);
          } else if (e == 1) {
            start = base_cell->face(f)->vertex(0);
            dir = base_cell->face(f)->vertex(2);
          } else if (e == 2) {
            start = base_cell->face(f)->vertex(1);
            dir = base_cell->face(f)->vertex(3);
          } else if (e > 2) {
            start = base_cell->face(f)->vertex(2);
            dir = base_cell->face(f)->vertex(3);
          }

          dir -= start;

          if (_ShapeTriaIsCutByBaseEdge(base, edge_a, edge_b, start, dir, cell_intersections)) {
            tmp = true;
          }
        }
      }
      return tmp;
    }

    void TriangulateIntersectionVertices(std::vector<Point<3> > &intersection_vertices,
                                                     std::vector<std::vector<Point<3> > > &subtriangulation) {

      subtriangulation.clear();

      const unsigned int n_vertices = intersection_vertices.size();

      Point<3> dir;
      Tensor<1, 3> tria_normal, diag_normal;
      Point<3> tria_a, tria_b;
      Point<3> diag_p, check;
      Point<3> base;

      bool all_positive = false, all_negative = false;
      double val;

      std::vector<Point<3> > check_vec, remaining, edge_vertices;
      std::vector<std::vector<Point<3> > > tmp_triangulation;
      std::vector<Point<3> > intersection_triangle;


      // n=3 intersection vertices
      if (intersection_vertices.size() == 3) {
        intersection_triangle.clear();

        for (unsigned int v = 0; v < n_vertices; ++v) {
          intersection_triangle.push_back(intersection_vertices[v]);

        }

        // return value: subtriangulation
        subtriangulation.push_back(intersection_triangle);
      }

      // n>3 intersection vertices
      if (intersection_vertices.size() > 3) {
        base = intersection_vertices[0];

        tria_a = intersection_vertices[1];
        tria_b = intersection_vertices[2];
        tria_a -= base;
        tria_b -= base;

        tria_normal = cross_product_3d(tria_a, tria_b);

        remaining.clear();
        edge_vertices.clear();

        edge_vertices.push_back(base);

        for (unsigned int diag = 0; diag < n_vertices - 1; ++diag) {
          diag_p = intersection_vertices[diag + 1];
          remaining.push_back(diag_p);

          check_vec.clear();

          for (unsigned int c = 0; c < n_vertices - 2; ++c) {
            if (c < diag) {
              check_vec.push_back(intersection_vertices[c + 1]);
            } else if (c >= diag) {
              check_vec.push_back(intersection_vertices[c + 2]);
            }
          }

          dir = diag_p;
          dir -= base;

          diag_normal = cross_product_3d(tria_normal, dir);

          all_positive = true;
          all_negative = true;

          for (unsigned int v = 0; v < n_vertices - 2; ++v) {
            check = check_vec[v];
            val = (check - base) * diag_normal;

            if (val > 0)
              all_negative = false;

            if (val < 0)
              all_positive = false;
          }

          if (all_positive || all_negative) {
            // edge
            edge_vertices.push_back(diag_p);
          }
        }

        if (edge_vertices.size() != 3) {
          std::cout << "could not establish edge triangle.." << std::endl;
          for (unsigned int i = 0; i < intersection_vertices.size(); ++i)
            std::cout << "Vertices: " << i << " : " << intersection_vertices[i] << std::endl;
        }

        if (edge_vertices.size() == 3) {
          // edge tria
          TriangulateIntersectionVertices(edge_vertices, tmp_triangulation);
          for (unsigned int i = 0; i < tmp_triangulation.size(); ++i) {
            subtriangulation.push_back(tmp_triangulation[i]);
          }

          // remaining
          TriangulateIntersectionVertices(remaining, tmp_triangulation);
          for (unsigned int i = 0; i < tmp_triangulation.size(); ++i) {
            subtriangulation.push_back(tmp_triangulation[i]);
          }
        }
      }
    }

  } // end of EddTools namespace

} // end of StructuralOptimization namespace