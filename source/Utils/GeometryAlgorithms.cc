#include <GeometryAlgorithms.h>

namespace StructuralOptimization {

  namespace GeometryAlgorithms {

// _point_in_bbox [dim=2]
    bool PointInBbox(Point<2> &p, std::vector<Point<2> > &polygon) {

      // bounding box check
      double minX, maxX, minY, maxY;
      minX = polygon[0][0];
      maxX = polygon[0][0];
      minY = polygon[0][1];
      maxY = polygon[0][1];

      for (unsigned int i = 0; i < polygon.size(); ++i) {

        if (polygon[i][0] < minX)
          minX = polygon[i][0];
        if (polygon[i][0] > maxX)
          maxX = polygon[i][0];

        if (polygon[i][1] < minY)
          minY = polygon[i][1];
        if (polygon[i][1] > maxY)
          maxY = polygon[i][1];

      }

      if (p[0] < minX || p[0] > maxX || p[1] < minY || p[1] > maxY)
        return false;
      else
        return true;

    }

// _point_in_bbox [dim=3]
    bool PointInBbox(Point<3> &p, std::vector<Point<3> > &polyeder) {

      // bounding box check
      double minX, maxX, minY, maxY, minZ, maxZ;
      minX = polyeder[0][0];
      maxX = polyeder[0][0];
      minY = polyeder[0][1];
      maxY = polyeder[0][1];
      minZ = polyeder[0][2];
      maxZ = polyeder[0][2];

      for (unsigned int i = 0; i < polyeder.size(); ++i) {

        if (polyeder[i][0] < minX)
          minX = polyeder[i][0];
        if (polyeder[i][0] > maxX)
          maxX = polyeder[i][0];

        if (polyeder[i][1] < minY)
          minY = polyeder[i][1];
        if (polyeder[i][1] > maxY)
          maxY = polyeder[i][1];

        if (polyeder[i][2] < minZ)
          minZ = polyeder[i][2];
        if (polyeder[i][2] > maxZ)
          maxZ = polyeder[i][2];

      }

      if (p[0] < minX || p[0] > maxX || p[1] < minY || p[1] > maxY || p[2] < minZ || p[2] > maxZ)
        return false;
      else
        return true;

    }

// _point_in_polygon [dim=2]
// http://stackoverflow.com/questions/217578/point-in-polygon-aka-hit-test
    bool PointInPolygon(Point<2> &p, ShapeTriaType<2> &tria_shape, double min_face_length_base) {

      std::vector<Point<2> > polygon = tria_shape.get_vertices();

      bool inside = false;

      // bounding box check
      if (!GeometryAlgorithms::PointInBbox(p, polygon))
        return false;


      // ray trace

      double minX = polygon[0][0];
      for (unsigned int i = 0; i < polygon.size(); ++i) {
        if (polygon[i][0] < minX)
          minX = polygon[i][0];
      }

      Point<2> ray_start, ray_end;
      ray_end = p;
      ray_start[0] = minX - min_face_length_base;
      ray_start[1] = p[1] + min_face_length_base;

      Point<2> face_begin, face_end;
      unsigned int n_intersections = 0;
      double a1, a2, b1, b2, c1, c2, d1, d2;

      for (auto cell: tria_shape.active_cell_iterators()) {

        face_begin = cell->vertex(0);
        face_end = cell->vertex(1);

        a1 = ray_end[1] - ray_start[1];
        b1 = ray_start[0] - ray_end[0];
        c1 = (ray_end[0] * ray_start[1]) - (ray_start[0] * ray_end[1]);
        d1 = (a1 * face_begin[0]) + (b1 * face_begin[1]) + c1;
        d2 = (a1 * face_end[0]) + (b1 * face_end[1]) + c1;

        if (d1 > 0 && d2 > 0)
          continue;
        if (d1 < 0 && d2 < 0)
          continue;

        a2 = face_end[1] - face_begin[1];
        b2 = face_begin[0] - face_end[0];
        c2 = (face_end[0] * face_begin[1]) - (face_begin[0] * face_end[1]);
        d1 = (a2 * ray_start[0]) + (b2 * ray_start[1]) + c2;
        d2 = (a2 * ray_end[0]) + (b2 * ray_end[1]) + c2;

        if (d1 > 0 && d2 > 0)
          continue;
        if (d1 < 0 && d2 < 0)
          continue;

        // if we arrive here: either intersection or collinear (do we need special treatment for collinear?)
        n_intersections++;
      }

      if ((n_intersections % 2) == 1)
        inside = true;

      return inside;

    }

// _point_in_polyeder [dim=3]
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    bool PointInPolyeder(Point<3> &p, ShapeTriaType<3> &tria_shape, double min_face_length_base) {

      std::vector<Point<3> > polyeder = tria_shape.get_vertices();
      bool inside = false;

      // bounding box check
      if (!GeometryAlgorithms::PointInBbox(p, polyeder))
        return false;

      double minX, minZ;
      minX = polyeder[0][0];
      minZ = polyeder[0][2];
      for (unsigned int i = 0; i < polyeder.size(); ++i) {

        if (polyeder[i][0] < minX)
          minX = polyeder[i][0];

        if (polyeder[i][2] < minZ)
          minZ = polyeder[i][2];

      }

      Point<3> ray_start, ray_end, ray_dir;
      ray_start = p;
      ray_end[0] = minX - min_face_length_base;
      ray_end[1] = p[1] + min_face_length_base;
      ray_end[2] = minZ - min_face_length_base;
      ray_dir = ray_end;
      ray_dir -= ray_start;

      Point<3> edge_a, edge_b, base, v_t, cell_dir;
      Tensor<1, 3> v_p, v_q;
      unsigned int n_intersections = 0;
      double det, inv_det, u, v, t;
      const double tol = 1.e-6;
      std::vector<double> t_list, u_list, v_list;

      //std::cout << "check intersect for point: " << p << " to " << ray_end << std::endl;

      for (auto cell : tria_shape.active_cell_iterators()) {
        base = cell->vertex(0);

        for (unsigned int tria = 0; tria < 2; ++tria) {

          if (tria == 0) {
            edge_a = cell->vertex(1);
            edge_b = cell->vertex(3);
          } else {
            edge_a = cell->vertex(3);
            edge_b = cell->vertex(2);
          }

          edge_a -= base;
          edge_b -= base;

          //cross_product(v_p,ray_dir,edge_b);
          v_p = cross_product_3d(ray_dir, edge_b);
          det = edge_a * v_p;

          if (std::abs(det) < tol)
            continue;

          inv_det = 1 / det;

          v_t = ray_start;
          v_t -= base;

          u = v_p * v_t * inv_det;
          if (u < 0 || u > 1)
            continue;

          //cross_product(v_q,v_t,edge_a);
          v_q = cross_product_3d(v_t, edge_a);

          v = ray_dir * v_q * inv_det;
          if (v < 0 || u + v > 1)
            continue;

          t = edge_b * v_q * inv_det;
          if (t > tol) {
            t_list.push_back(t);
            u_list.push_back(u);
            v_list.push_back(v);
            n_intersections++;
          }

        }

      }

      if (n_intersections > 1) {
        //std::cout << "n_inter: " << n_intersections << std::endl;
        for (unsigned int comp_i = 0; comp_i < t_list.size() - 1; ++comp_i) {
          for (unsigned int comp_j = comp_i + 1; comp_j < t_list.size(); ++comp_j) {

            if (std::abs(t_list[comp_i] - t_list[comp_j]) < tol) {
              n_intersections--;
              //std::cout << "double count.." << std::endl;
            }

          }
        }
      }

      if ((n_intersections % 2) == 1)
        inside = true;

      return inside;

    }

// point_in_shape [dim=2]
    bool PointInShape(Point<2> &p, ShapeTriaType<2> &tria_shape, double min_face_length_base) {
      return PointInPolygon(p, tria_shape, min_face_length_base);
    }

// point_in_shape [dim=3]
    bool PointInShape(Point<3> &p, ShapeTriaType<3> &tria_shape, double min_face_length_base) {
      return PointInPolyeder(p, tria_shape, min_face_length_base);
    }


// below 3 functions are to find if 2 line segments intersect
// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
    bool DoLinesIntersect(Point<2> &p1, Point<2> &q1, Point<2> &p2, Point<2> &q2) {
      // Find the four orientations needed for general and
      // special cases
      int o1 = Orientation(p1, q1, p2);
      int o2 = Orientation(p1, q1, q2);
      int o3 = Orientation(p2, q2, p1);
      int o4 = Orientation(p2, q2, q1);

      // General case
      if (o1 != o2 && o3 != o4)
        return true;

      // Special Cases
      // p1, q1 and p2 are colinear and p2 lies on segment p1q1
      if (o1 == 0 && OnSegment(p1, p2, q1)) return true;

      // p1, q1 and q2 are colinear and q2 lies on segment p1q1
      if (o2 == 0 && OnSegment(p1, q2, q1)) return true;

      // p2, q2 and p1 are colinear and p1 lies on segment p2q2
      if (o3 == 0 && OnSegment(p2, p1, q2)) return true;

      // p2, q2 and q1 are colinear and q1 lies on segment p2q2
      if (o4 == 0 && OnSegment(p2, q1, q2)) return true;

      return false; // Doesn't fall in any of the above cases
    }


    bool DoLinesIntersect(Point<3> &p1, Point<3> &q1, Point<3> &p2, Point<3> &q2){
      (void) p1;
      (void) q1;
      (void) p2;
      (void) q2;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      return false;
    }

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
    int Orientation(Point<2> &p, Point<2> &q, Point<2> &r) {
      // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
      // for details of below formula.
      double val = (q[1] - p[1]) * (r[0] - q[0]) -
                   (q[0] - p[0]) * (r[1] - q[1]);

      if (std::abs(val) < 1e-8) return 0;  // colinear

      return (val > 0) ? 1 : 2; // clock or counterclock wise
    }

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
    bool OnSegment(Point<2> &p, Point<2> &q, Point<2> &r) {
      if (q[0] <= std::max(p[0], r[0]) && q[0] >= std::min(p[0], r[0]) &&
          q[1] <= std::max(p[1], r[1]) && q[1] >= std::min(p[1], r[1]))
        return true;

      return false;
    }
// Return true if shape cut itself.

    bool DoShapeCutItself(DoFHandler<1, 2> &dof_handler_shape, unsigned int n_mpi_processes,
                          unsigned int this_mpi_process) {

      // division of work among process.
      typedef typename DoFHandler<1, 2>::active_cell_iterator active_cell_iterator;
      std::vector<std::pair<active_cell_iterator, active_cell_iterator> > proc_ranges =
          Threads::split_range<active_cell_iterator>(dof_handler_shape.begin_active(),
                                                     dof_handler_shape.end(),
                                                     n_mpi_processes);


      for (active_cell_iterator first_cell = proc_ranges[this_mpi_process].first;
           first_cell < proc_ranges[this_mpi_process].second; ++first_cell) {
        Point<2> p1 = first_cell->vertex(0);
        Point<2> q1 = first_cell->vertex(1);
        for (auto second_cell : dof_handler_shape.active_cell_iterators()) {
          Point<2> p2 = second_cell->vertex(0);
          Point<2> q2 = second_cell->vertex(1);

          Tensor<1, 2> p1_p2 = p1 - p2;
          Tensor<1, 2> p1_q2 = p1 - q2;
          Tensor<1, 2> q1_p2 = q1 - p2;
          Tensor<1, 2> q1_q2 = q1 - q2;
          if (p1_p2.norm_square() < 1e-6 || p1_q2.norm_square() < 1e-6 || q1_p2.norm_square() < 1e-6 ||
              q1_q2.norm_square() < 1e-6)
            continue;
          // Check if cells are neighbours, then continue

          if (DoLinesIntersect(p1, q1, p2, q2))
            return true;

        }
      }
      return false;

    }

    bool DoShapeCutItself(DoFHandler<2, 3> &dof_handler_shape, unsigned int n_mpi_processes,
                          unsigned int this_mpi_process) {
      // division of work among process.

      if (dof_handler_shape.n_dofs() > 0) {
        COUT_DEBUG << "This is not implemented." << std::endl;
        return false;
      } else { // this is the actual implementation.
        // division of work among process.
        typedef typename DoFHandler<2, 3>::active_cell_iterator active_cell_iterator;
        std::vector<std::pair<active_cell_iterator, active_cell_iterator> > proc_ranges =
            Threads::split_range<active_cell_iterator>(dof_handler_shape.begin_active(),
                                                       dof_handler_shape.end(),
                                                       n_mpi_processes);


        for (active_cell_iterator first_cell = proc_ranges[this_mpi_process].first;
             first_cell < proc_ranges[this_mpi_process].second; ++first_cell) {
          return false;
        }

      }

      return false;
    }

    int FindSide(Point<2> p1, Point<2> p2, Point<2> p)
    {
      double val = (p[1] - p1[1]) * (p2[0] - p1[0]) -
                   (p2[1] - p1[1]) * (p[0] - p1[0]);

      if (val > 0)
        return 1;
      if (val < 0)
        return -1;
      return 0;
    }


    int FindSide(Point<3> p1, Point<3> p2, Point<3> p){
      (void) p1; (void) p2; (void) p;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      return 0;
    }

    double LineDist(Point<2> p1, Point<2> p2, Point<2> p)
    {
      return abs ((p[1] - p1[1]) * (p2[0] - p1[0]) -
                  (p2[1] - p1[1]) * (p[0] - p1[0]));
    }

    double LineDist(Point<3> p1, Point<3> p2, Point<3> p){
      (void) p1; (void) p2; (void) p;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      return 0.0;
    }

    Point<2> GetCentroid(const std::vector<Point<2>>& points){
      typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> point_t;
      typedef boost::geometry::model::multi_point<point_t> mpoint_t;

      mpoint_t mpt1;

      for(const auto& p : points)
        boost::geometry::append(mpt1, point_t(p[0], p[1]));

      point_t c;
      boost::geometry::centroid(mpt1, c);

      Point<2> result(c.get<0>(), c.get<1>());

      return result;
    }

    Point<3> GetCentroid(const std::vector<Point<3>>& points){
      (void) points;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      Point<3> result;
      return result;
    }

    double GetPolygonArea(const std::vector<Point<2>>& points){

      typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> point_t;

      typedef boost::geometry::model::polygon<point_t, false> polygon_t;

      polygon_t poly;

      // get the points in proper order.
      std::vector<std::size_t> permutation = GetPermutationForClosedPolygon(points);

      for(const auto& perm : permutation )
        boost::geometry::append(poly.outer(), point_t(points[perm][0], points[perm][1] ));

      double area = boost::geometry::area(poly);

      return area;

    }

    double GetPolygonArea(const std::vector<Point<3>>& points){
      (void) points;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      double result=0;
      return result;
    }

    void QuickHull(std::vector<Point<2>> &hull, const std::vector<Point<2>> &cluster, int n, Point<2> p1, Point<2> p2, int side)
    {
      int ind = -1;
      double max_dist = 0;

      // finding the point with maximum distance
      // from L and also on the specified side of L.
      for (int i=0; i<n; i++)
      {
        double temp = LineDist(p1, p2, cluster[i]);
        if (FindSide(p1, p2, cluster[i]) == side && temp > max_dist)
        {
          ind = i;
          max_dist = temp;
        }
      }

      // If no point is found, add the end points
      // of L to the convex hull.
      if (ind == -1)
      {
        hull.push_back(p1);
        hull.push_back(p2);
        return;
      }

      // Recur for the two parts divided by a[ind]
      QuickHull(hull, cluster, n, cluster[ind], p1, -FindSide(cluster[ind], p1, p2));
      QuickHull(hull, cluster, n, cluster[ind], p2, -FindSide(cluster[ind], p2, p1));
    }

    void QuickHull(std::vector<Point<3>> &hull, const std::vector<Point<3>> &cluster, int n, Point<3> p1, Point<3> p2, int side){
      (void) hull; (void) cluster; (void) n; (void) p1; (void) p2; (void) side;
      if(true)
        throw std::runtime_error( "Will not work in 3D" );
    }

    Vector<double> GetSquaredPolar(const Point<2>& point, const Point<2>& centre){
      Vector<double> polar(2);
      polar[0] = std::atan2(point[1]-centre[1], point[0]-centre[0]);
      polar[1] = std::pow( point[0]-centre[0], 2) + std::pow( point[1]-centre[1], 2);
      return polar;
    }

    Vector<double> GetSquaredPolar(const Point<3>& point, const Point<3>& center){
      (void) point; (void) center;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      Vector<double> result;
      return result;
    }

    std::vector<std::size_t> GetPermutationForClosedPolygon(const std::vector<Point<2>> &points){
      Point<2> centroid = GetCentroid(points);

      // get polar coordinates of the points
      std::vector<Vector<double>> polar_points;
      for(const auto& p : points)
        polar_points.push_back(GetSquaredPolar(p, centroid));

      //get sort permutation based on polar coord.
      std::vector<std::size_t> permutation(polar_points.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::stable_sort(permutation.begin(), permutation.end(),
                       [&](std::size_t i, std::size_t j)
                       {
                         if(polar_points[i][0] < polar_points[j][0])
                           return true;
                         if(polar_points[i][0] > polar_points[j][0])
                           return false;
                         if(polar_points[i][1] < polar_points[j][1])
                           return true;
                         if(polar_points[i][1] > polar_points[j][1])
                           return false;
                         return true;
                       });
      return permutation;

    }

    std::vector<std::size_t> GetPermutationForClosedPolygon(const std::vector<Point<3>> &points){
      (void) points;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      std::vector<std::size_t> result;
      return result;
    }

    std::vector<Point<2>> GeneratePolygon(const std::vector<Point<2>> &points, const double& min_shape_length, const double& min_base_face_length){


      std::vector<std::size_t> permutation = GetPermutationForClosedPolygon(points);

      std::vector<Point<2>> coarse_polygon;
      for(const auto& perm : permutation )
        coarse_polygon.push_back(points[perm]);

      // add points in between if edge length is smaller than min_shape_length
      std::vector<Point<2>> refined_polygon;
      refined_polygon.push_back(coarse_polygon[0]);
      for(unsigned int it = 1; it <= coarse_polygon.size(); ++it){
        if(it < coarse_polygon.size()) {
          double distance = coarse_polygon[it].distance(coarse_polygon[it - 1]);
          if (distance > min_shape_length) {
            unsigned int n_interval = std::ceil(distance / min_shape_length);
            Tensor<1, 2> direction;
            direction = coarse_polygon[it] - coarse_polygon[it - 1];
            direction /= direction.norm();
            Tensor<1, 2> step = (distance / (double) n_interval) * direction;
            for (unsigned int i = 1; i < n_interval; ++i) {
              Point<2> pt(coarse_polygon[it - 1] + (i * step));
              refined_polygon.push_back(pt);
            }
          }
          refined_polygon.push_back(coarse_polygon[it]);
        }
        else{
          double distance = coarse_polygon[0].distance(coarse_polygon[it - 1]);
          if (distance > min_shape_length) {
            unsigned int n_interval = std::ceil(distance / min_shape_length);
            Tensor<1, 2> direction;
            direction = coarse_polygon[0] - coarse_polygon[it - 1];
            direction /= direction.norm();
            Tensor<1, 2> step = (distance / (double) n_interval) * direction;
            for (unsigned int i = 1; i < n_interval; ++i) {
              Point<2> pt(coarse_polygon[it - 1] + (i * step));
              refined_polygon.push_back(pt);
            }
          }
        }
      }


      // Move the nodes in radial direction by min_base_length so that the polygon
      // is built on the base cells with higher density.
      Point<2> centroid = GetCentroid(refined_polygon);
      for (auto &point : refined_polygon) {
        Tensor<1, 2> direction;
        direction = point - centroid;
        direction /= direction.norm();
        direction *= min_base_face_length;
        point = point + 0.5 * direction;
      }


      return refined_polygon;
    }


    std::vector<Point<3>> GeneratePolygon(const std::vector<Point<3>> &points, const double& min_shape_length, const double& min_base_face_length){
      (void) points; (void) min_shape_length; (void) min_base_face_length;
      std::string message = "Will not work in 3D " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " | " + std::string(__func__ );
      if(true)
        throw std::runtime_error( message );
      std::vector<Point<3>> result;
      return result;
    }


  } // end of namespace GeometryAlgorithms

} // end of namespace StructuralOptimization
