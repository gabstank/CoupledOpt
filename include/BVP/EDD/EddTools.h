#pragma once

#include <EddBVP_.h>

namespace StructuralOptimization {

  namespace EddTools {

    /**
     * @brief Function to tell if the given cell is outside the shape.
     * i.e. the cell is not inside and boundary.
     *
     * @param cell
     * @return true
     * @return false
     */
    template <int dim>
    inline bool
    CellIsOutside(const typename DoFHandler<dim>::cell_iterator &cell) {
      return (!(cell->material_id() == e_boundary_cell) && !(cell->material_id() == e_inside_cell)
              && !(cell->material_id() == e_boundary_electrode_cell));
    }

    /**
     * @brief Function to tell if the cell is inside the shape.
     * i.e. the cell is either inside or boundary.
     *
     * @param cell
     * @return true
     * @return false
     */
    template <int dim>
    inline bool
    CellIsInside(const typename DoFHandler<dim>::cell_iterator &cell) {
      return (cell->material_id() == e_boundary_cell || cell->material_id() == e_inside_cell
              || cell->material_id() == e_boundary_electrode_cell);
    }

    /**
     * @brief Function to tell if base_cell is cut by segment formed from sha_v1 and sha_v2.
     *
     * @param base_cell
     * @param sha_v1
     * @param sha_v2
     * @return true
     * @return false
     */
    bool BaseCellIsCutByShapeCell(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                  Point<2> &sha_v1, Point<2> &sha_v2);

    /**
     * @brief Function to find the segment of line formed from sha_v1 - sha_v2 that intersects with the base_cell,
     * the interesected segment is made of sgm_p1 - sgm_p2
     *
     * @param base_cell
     * @param sha_v1
     * @param sha_v2
     * @param sgm_p1
     * @param sgm_p2
     */
    void DetermineCellIntersectionSegment(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                          Point<2> &sha_v1, Point<2> &sha_v2,
                                          Point<2> &sgm_p1, Point<2> &sgm_p2);

    /**
     * @brief Function to get the intersection of segment formed by sha_v1 and sha_dir with the face f of base_cell.
     * Only used for dim==2.
     * Reference: http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
     * @param base_cell
     * @param f
     * @param sha_v1
     * @param sha_dir
     * @return Point<dim>
     */
    Point<2> ReturnSegmentIntersection(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                         unsigned int &f, Point<2> &sha_v1, Point<2> &sha_dir);

    /**
     * @brief Function to know if the shape cell segment formed by sgm_p1 and sgm_dir is cutting face f of base_cell.
     * Only used for dim==2
     *
     * @param base_cell
     * @param f
     * @param sgm_p1
     * @param sgm_dir
     * @return true
     * @return false
     */
    bool ShapeCellIsCutByBaseFace(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                  unsigned int &f, Point<2> &sgm_p1, Point<2> &sgm_dir);

    /**
     * @brief Function to know if the line formed by sha_v1 and sha_v2 is cutting the face f of base_cell
     * Only used for dim==2
     *
     * @param base_cell
     * @param f
     * @param sha_v1
     * @param sha_v2
     * @return true
     * @return false
     */
    bool BaseFaceIsCutByShapeCell(typename DoFHandler<2>::active_cell_iterator &base_cell,
                                  unsigned int &f, Point<2> &sha_v1, Point<2> &sha_v2);

    /**
     * @brief Function to know if the tria surface formed by sha_v1, sha_v2 and sha_v3 cuts the base_cell.
     * The intersection points are pushed to the vector cell_intersections.
     * Only used for dim==3
     * @param base_cell
     * @param sha_v1
     * @param sha_v2
     * @param sha_v3
     * @param cell_intersections
     * @return true
     * @return false
     */
    bool BaseCellIsCutByShapeTria(typename DoFHandler<3>::active_cell_iterator &base_cell,
                                  Point<3> &sha_v1, Point<3> &sha_v2, Point<3> &sha_v3,
                                  std::vector<Point<3> > &cell_intersections);

    /**
     * @brief Function to tria edge formed by start and dir is cutting the face f of the base_cell.
     * Only used for dim==3
     * @param base_cell
     * @param f
     * @param start
     * @param dir
     * @param cell_intersections
     * @return true
     * @return false
     */
    bool _BaseFaceIsCutByShapeTriaEdge(typename DoFHandler<3>::active_cell_iterator &base_cell,
                                       unsigned int &f,
                                       Point<3> start, Point<3> dir,
                                       std::vector<Point<3> > &cell_intersections);

    /**
     * @brief Function to if base edge formed by start and dir is cutting shape tria formed by tria_base, tria_edge_a and tria_edge_b.
     * Only used for dim==3
     * @param tria_base
     * @param tria_edge_a
     * @param tria_edge_b
     * @param start
     * @param dir
     * @param cell_intersections
     * @return true
     * @return false
     */
    bool _ShapeTriaIsCutByBaseEdge(Point<3> &tria_base, Point<3> &tria_edge_a, Point<3> &tria_edge_b,
                                   Point<3> &start, Point<3> &dir,
                                   std::vector<Point<3> > &cell_intersections);

    /**
     * @brief Function to form subtriangulation from the set of intersection_vertices.
     *
     * @param intersection_vertices
     * @param subtriangulation
     */
    void TriangulateIntersectionVertices(std::vector<Point<3> > &intersection_vertices,
                                         std::vector<std::vector<Point<3> >> &subtriangulation);

  } // end of EddTools namespace

} // end of StructuralOptimization namespace
