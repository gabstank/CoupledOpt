#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/io/svg/write_svg.hpp>

#include <Utilities.h>

namespace StructuralOptimization {

namespace GeometryAlgorithms {

  /**
   * @brief Function to check if Point p is inside a box formed by points in vector polygon.
   *
   * @param p
   * @param polygon
   * @return true
   * @return false
   */
  bool PointInBbox(Point<2> &p, std::vector<Point<2> > &polygon);

  /**
   * @brief Function to check if Point p is inside a cube formed by points in vector polyeder.
   *
   * @param p
   * @param polyeder
   * @return true
   * @return false
   */
  bool PointInBbox(Point<3> &p, std::vector<Point<3> > &polyeder);

  /**
   * @brief Function to check if the point p is inside the tria_shape for 2D.
   *
   * @param p
   * @param tria_shape
   * @param min_face_length_base
   * @return true
   * @return false
   */
  bool PointInShape(Point<2> &p, ShapeTriaType<2> &tria_shape, double min_face_length_base);

  /**
   * @brief Function to check if the point p is inside the tria_shape for 3D.
   *
   * @param p
   * @param tria_shape
   * @param min_face_length_base
   * @return true
   * @return false
   */
  bool PointInShape(Point<3> &p, ShapeTriaType<3> &tria_shape, double min_face_length_base);

  /**
   * @brief Function to check if Point p is inside a polygon formed by tria_shape.
   *
   * @param p
   * @param tria_shape
   * @param min_face_length_base
   * @return true
   * @return false
   */
  bool PointInPolygon(Point<2> &p, ShapeTriaType<2> &tria_shape, double min_face_length_base);

  /**
   * @brief Function to check if Point p is inside a polyeder formed by tria_shape.
   *
   * @param p
   * @param tria_shape
   * @param min_face_length_base
   * @return true
   * @return false
   */
  bool PointInPolyeder(Point<3> &p, ShapeTriaType<3> &tria_shape, double min_face_length_base);

  /**
   * @brief Function to check if line segment 'p1q1'and 'p2q2' intersect.
   *  https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
   *
   * @param p1
   * @param q1
   * @param p2
   * @param q2
   * @return true
   * @return false
   */
  bool DoLinesIntersect(Point<2> &p1, Point<2> &q1, Point<2> &p2, Point<2> &q2);

  bool DoLinesIntersect(Point<3> &p1, Point<3> &q1, Point<3> &p2, Point<3> &q2);


  /**
   * @brief  To find orientation of ordered triplet (p, q, r).
   * The function returns following values
   * 0 --> p, q and r are colinear
   * 1 --> Clockwise
   * 2 --> Counterclockwise
   *
   * @param p
   * @param q
   * @param r
   * @return int
   */
  int Orientation(Point<2> &p, Point<2> &q, Point<2> &r);

  /**
   * @brief Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
   *
   * @param p
   * @param q
   * @param r
   * @return true
   * @return false
   */
  bool OnSegment(Point<2> &p, Point<2> &q, Point<2> &r);

  /**
   * @brief Function to check if the shape is self intersecting.
   *
   * @param dof_handler_shape
   * @param n_mpi_processes
   * @param this_mpi_process
   * @return true
   * @return false
   */
  bool DoShapeCutItself(DoFHandler<1, 2> &dof_handler_shape, unsigned int n_mpi_processes,
                        unsigned int this_mpi_process);

  /**
   * @brief Function to check if the shape is self intersecting.
   *
   * @param dof_handler_shape
   * @param n_mpi_processes
   * @param this_mpi_process
   * @return true
   * @return false
   */
  bool DoShapeCutItself(DoFHandler<2, 3> &dof_handler_shape, unsigned int n_mpi_processes,
                        unsigned int this_mpi_process);

  // Returns the side of point p with respect to line
// joining points p1 and p2.
  /**
   * Returns the side of point p with respect to line joining points p1 and p2.
   * This function is used to find the convex hull, according to
   * https://www.geeksforgeeks.org/quickhull-algorithm-convex-hull/
   * @param p1
   * @param p2
   * @param p
   * @return
   */
  int FindSide(Point<2> p1, Point<2> p2, Point<2> p);

  /**
   * Dummy function, will not work in 3D. will just throw an error if called
   * @param p1
   * @param p2
   * @param p
   * @return
   */
  int FindSide(Point<3> p1, Point<3> p2, Point<3> p);

  /**
   * returns a value proportional to the distance
   * between the point p and the line joining the
   * points p1 and p2
   * @param p1
   * @param p2
   * @param p
   * @return
   */
  double LineDist(Point<2> p1, Point<2> p2, Point<2> p);

  /**
   * Dummy function, will not work in 3D. will just throw an error if called
   * @param p1
   * @param p2
   * @param p
   * @return
   */
  double LineDist(Point<3> p1, Point<3> p2, Point<3> p);


  /**
   * End points of line L are p1 and p2. side can have value
   * 1 or -1 specifying each of the parts made by the line L
   * https://www.geeksforgeeks.org/quickhull-algorithm-convex-hull/
   * @param hull
   * @param cluster
   * @param n
   * @param p1
   * @param p2
   * @param side
   */
  void QuickHull(std::vector<Point<2>> &hull, const std::vector<Point<2>> &cluster, int n, Point<2> p1, Point<2> p2, int side);

  void QuickHull(std::vector<Point<3>> &hull, const std::vector<Point<3>> &cluster, int n, Point<3> p1, Point<3> p2, int side);

  /**
   * Given a vector of points this function returns the centroid of the set of points
   * @param points
   * @return
   */
  Point<2> GetCentroid(const std::vector<Point<2>>& points);

  Point<3> GetCentroid(const std::vector<Point<3>>& points);

  /**
   * Given a vector of points that form a polygon, this function returns the area of polygon
   * @param points
   * @return
   */
  double GetPolygonArea(const std::vector<Point<2>>& points);

  double GetPolygonArea(const std::vector<Point<3>>& points);

  /**
   * Function to get the polar coordinate of point wrt centre.
   * Will return angle in first index and the sq radius in 2nd index of vector.
   * @param point
   * @param center
   * @return
   */
  Vector<double> GetSquaredPolar(const Point<2>& point, const Point<2>& centre);

  Vector<double> GetSquaredPolar(const Point<3>& point, const Point<3>& centre);

  /**
   * Given a vector of unordered points, a polygon is generated.
   * Based on the post given in.
   * https://stackoverflow.com/questions/59287928/algorithm-to-create-a-polygon-from-points
   * Further after the polygon is formed extra vertices are added on edges greater than min_shape_length
   * Finally the nodes are moved out radially by distance of min_face_length_base, so that the polygon
   * is formed in the cells with higher density.
   * @param points
   * @param min_shape_length
   * @param min_base_face_length
   * @return
   */
  std::vector<Point<2>> GeneratePolygon(const std::vector<Point<2>> &points, const double& min_shape_length, const double& min_base_face_length);

  std::vector<Point<3>> GeneratePolygon(const std::vector<Point<3>> &points, const double& min_shape_length, const double& min_base_face_length);

  /**
   * Given points which form a regular polygon but not in order, this function return the permutation to reorder points
   * which can be used to oder the points.
   * @param points
   * @return
   */
  std::vector<std::size_t> GetPermutationForClosedPolygon(const std::vector<Point<2>> &points);

  std::vector<std::size_t> GetPermutationForClosedPolygon(const std::vector<Point<3>> &points);

}// end of namespace GeometryAlgorithms

}// end of namespace StructuralOptimization
