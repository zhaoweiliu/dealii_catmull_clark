//
//  Catmull_Clark_DoFs_Implementation.cpp
//  step-4
//
//  Created by zhaowei Liu on 09/12/2019.
//

#include "Catmull_Clark_Data.hpp"

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
const Quadrature<dim>
CatmullClark<dim, spacedim>::get_adaptive_quadrature(int L, Quadrature<2> qpts) const
{
  std::vector<Point<dim>> aqpts;
  std::vector<double>     wts;
  std::vector<Point<dim>> gauss_pts = qpts.get_points();
  std::vector<double>     gauss_wts = qpts.get_weights();
  for (int i = 1; i < L + 1; ++i)
    {
      for (unsigned int iq = 0; iq < gauss_pts.size(); ++iq)
        {
          aqpts.push_back({gauss_pts[iq][0] * pow(0.5, i),
                           gauss_pts[iq][1] * pow(0.5, i) + pow(0.5, i)});
          wts.push_back(gauss_wts[iq] * pow(0.25, i));
          aqpts.push_back({gauss_pts[iq][0] * pow(0.5, i) + pow(0.5, i),
                           gauss_pts[iq][1] * pow(0.5, i) + pow(0.5, i)});
          wts.push_back(gauss_wts[iq] * pow(0.25, i));
          aqpts.push_back({gauss_pts[iq][0] * pow(0.5, i) + pow(0.5, i),
                           gauss_pts[iq][1] * pow(0.5, i)});
          wts.push_back(gauss_wts[iq] * pow(0.25, i));
        }
    }

  const QGauss<dim>       mqpt(2);
  std::vector<Point<dim>> gpts = mqpt.get_points();
  for (unsigned int iq = 0; iq < mqpt.size(); ++iq)
    {
      aqpts.push_back({gpts[iq][0] * pow(0.5, L), gpts[iq][1] * pow(0.5, L)});
      wts.push_back(mqpt.get_weights()[iq] * pow(0.25, L));
    }
  return {aqpts, wts};
}



template <int dim, int spacedim>
const Quadrature<dim> CatmullClark<dim, spacedim>::edge_cell_boundary_quadrature(const unsigned int v0_id) const
{
  std::vector<Point<dim>>     qpts_2d;
  std::vector<double>         wts_2d;
  const QGauss<dim - 1>       quadrature_1d(3);
  std::vector<Point<dim - 1>> gauss_pts = quadrature_1d.get_points();
  std::vector<double>         gauss_wts = quadrature_1d.get_weights();
  for (unsigned int iq = 0; iq < gauss_pts.size(); ++iq)
    {
        switch (v0_id) {
            case 0:
                qpts_2d.push_back({gauss_pts[iq][0], 0.});
                break;
            case 1:
                qpts_2d.push_back({1.0, gauss_pts[iq][0]});
                break;
            case 2:
                qpts_2d.push_back({0., gauss_pts[iq][0]});
            break;
            case 3:
                qpts_2d.push_back({gauss_pts[iq][0], 1.});
            break;
            default:
                break;
        }
      wts_2d.push_back(gauss_wts[iq]);
    }
  return {qpts_2d, wts_2d};
}


template <int dim, int spacedim>
const Quadrature<dim> CatmullClark<dim, spacedim>::corner_cell_boundary_quadrature(const unsigned int v0_id) const
{
  std::vector<Point<dim>>     qpts_2d;
  std::vector<double>         wts_2d;
  const QGauss<dim - 1>       quadrature_1d(3);
  std::vector<Point<dim - 1>> gauss_pts = quadrature_1d.get_points();
  std::vector<double>         gauss_wts = quadrature_1d.get_weights();
  for (unsigned int iq = 0; iq < gauss_pts.size(); ++iq)
    {
        switch (v0_id) {
            case 0:
                qpts_2d.push_back({gauss_pts[iq][0], 0.});
                qpts_2d.push_back({0., gauss_pts[iq][0]});
                break;
            case 1:
                qpts_2d.push_back({gauss_pts[iq][0], 0.});
                qpts_2d.push_back({1., gauss_pts[iq][0]});
                break;
            case 2:
                qpts_2d.push_back({gauss_pts[iq][0], 1.});
                qpts_2d.push_back({0., gauss_pts[iq][0]});
                break;
            case 3:
                qpts_2d.push_back({gauss_pts[iq][0], 1.});
                qpts_2d.push_back({1., gauss_pts[iq][0]});
                break;
            default:
                break;
        }
        wts_2d.push_back(gauss_wts[iq]);
        wts_2d.push_back(gauss_wts[iq]);
    }
  return {qpts_2d, wts_2d};
}

template <int dim, int spacedim>
const Quadrature<dim> CatmullClark<dim, spacedim>::empty_boundary_quadrature() const
{
    return Quadrature<dim>({Point<dim>()},{0});
}



template <int dim, int spacedim>
void CatmullClark<dim, spacedim>::set_hp_objects(
  hp::DoFHandler<dim, spacedim> &dof_handler,
  const unsigned int             multiplicity)
{
  n_element = multiplicity;
  cell_patch_vector = cell_patches(dof_handler);
  new_order_for_cells(dof_handler);
  set_FECollection(dof_handler, n_element);
}



template <int dim, int spacedim>
void CatmullClark<dim, spacedim>::set_FECollection(
  hp::DoFHandler<dim, spacedim> &dof_handler,
  const unsigned int             n_element)
{
  std::map<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>>
               map_valence_to_fe_indices;
  unsigned int i_fe = 0;
  QGauss<dim>  qpts(2);
  auto         qpts_irreg = get_adaptive_quadrature(5, qpts);
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
       ++cell)
    {
      unsigned int valence;
      switch (unsigned int ncell_in_patch =
                cell_patch_vector[cell->active_cell_index()].size())
        {
          case 4:
            valence = 1;
            break;
          case 6:
            valence = 2;
            break;
          default:
            valence = ncell_in_patch - 5;
            break;
        }
      std::array<unsigned int, 4> verts_id;
      if (valence == 1)
        {
          std::vector<unsigned int> edges_on_boundary(0);
          for (unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie)
            {
              if (cell->at_boundary(ie))
                {
                  edges_on_boundary.push_back(ie);
                }
            }
          verts_id = verts_id_on_boundary(edges_on_boundary);
        }
      else
        {
          if (valence == 2)
            {
              for (unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell;
                   ++ie)
                {
                  if (cell->at_boundary(ie))
                    {
                      verts_id = rotated_vertices(ie);
                    }
                }
            }
          else
            {
              if (valence == 4)
                {
                  verts_id = {0, 1, 2, 3};
                }
              else
                {
                  std::map<unsigned int,
                           typename hp::DoFHandler<dim, spacedim>::
                             active_cell_iterator>
                    cells = ordered_cell_patch_vector[cell->active_cell_index()];
                  int ex_vertex_index;
                  for (unsigned int iv = 0; iv < 4; ++iv)
                    {
                      unsigned int n = 0;
                      for (unsigned int icell = 1; icell < 3; ++icell)
                        {
                          for (unsigned int jv = 0; jv < 4; ++jv)
                            {
                              if (cell->vertex_index(iv) ==
                                  cells[icell]->vertex_index(jv))
                                {
                                  n += 1;
                                }
                            }
                        }
                      if (n == 2)
                        {
                          ex_vertex_index = iv;
                        }
                    }
                  verts_id[0]     = ex_vertex_index;
                  auto next_verts = next_vertices(ex_vertex_index);
                  verts_id[1]     = next_verts[0];
                  verts_id[2]     = next_verts[1];
                  verts_id[3]     = next_verts[2];
                }
            }
        }

      std::map<unsigned int,
               std::vector<std::pair<unsigned int, unsigned int>>>::iterator
        iter_fe_valence;
      iter_fe_valence = map_valence_to_fe_indices.find(valence);
      bool exsit      = false;
      if (iter_fe_valence != map_valence_to_fe_indices.end())
        {
          for (unsigned int ip = 0; ip < iter_fe_valence->second.size(); ++ip)
            {
              if (verts_id[0] == iter_fe_valence->second[ip].first)
                {
                  cell->set_active_fe_index(iter_fe_valence->second[ip].second);
                  exsit = true;
                }
            }
          if (exsit == false)
            {
              FE_Catmull_Clark<dim, spacedim> fe(valence, verts_id, this->reference_ptr());
              if (n_element == 1)
                {fe_collection.push_back(fe);}
              else
                {
                 fe_collection.push_back(FESystem<dim, spacedim>(fe, n_element));
                }
              if (valence == 1 || valence == 2 || valence == 4)
                {
                  q_collection.push_back(qpts);
                }
              else
                {
                  q_collection.push_back(qpts_irreg);
                }

              switch (valence)
                {
                  case (1):
                    q_boundary_collection.push_back(
                      corner_cell_boundary_quadrature(verts_id[0]));
                    break;
                  case (2):
                    q_boundary_collection.push_back(
                      edge_cell_boundary_quadrature(verts_id[0]));
                    break;
                  default:
                    q_boundary_collection.push_back(
                      empty_boundary_quadrature());
                    break;
                }

              iter_fe_valence->second.push_back({verts_id[0], i_fe});
              cell->set_active_fe_index(i_fe);
              ++i_fe;
            }
        }
      else
        {
          map_valence_to_fe_indices.insert(
            std::pair<unsigned int,
                      std::vector<std::pair<unsigned int, unsigned int>>>(
              valence, {{verts_id[0], i_fe}}));
          FE_Catmull_Clark<dim, spacedim> fe(valence, verts_id, this->reference_ptr());
          if (n_element == 1)
            {fe_collection.push_back(fe);}
          else{
          fe_collection.push_back(FESystem<dim, spacedim>(fe, n_element));
            }
          if (valence == 1 || valence == 2 || valence == 4)
            {
              q_collection.push_back(qpts);
            }
          else
            {
              q_collection.push_back(qpts_irreg);
            }
          switch (valence)
            {
              case (1):
                q_boundary_collection.push_back(
                  corner_cell_boundary_quadrature(verts_id[0]));
                break;
              case (2):
                q_boundary_collection.push_back(
                  edge_cell_boundary_quadrature(verts_id[0]));
                break;
              default:
                q_boundary_collection.push_back(empty_boundary_quadrature());
                break;
            }
          cell->set_active_fe_index(i_fe);
          ++i_fe;
        }
    }
  indices_mapping_valence_to_fe = map_valence_to_fe_indices;
}



template <int dim, int spacedim>
void CatmullClark<dim, spacedim>::set_MappingCollection(
  const hp::DoFHandler<dim, spacedim> &dof_handler,
  Vector<double> &                     vec_values,
  const unsigned int                   n_element)
{
  const ComponentMask mask(spacedim, true);

  const auto &vertices = dof_handler.get_triangulation().get_vertices();
  vec_values.reinit(vertices.size() * spacedim);

  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
     ++cell)
  {
    std::vector<types::global_dof_index> cell_dof_indices(
      cell->get_fe().dofs_per_cell, 0);
    std::vector<types::global_dof_index> non_local_dof_indices(
      cell->get_fe().non_local_dofs_per_cell, 0);
    cell->get_dof_indices(cell_dof_indices);
    for (unsigned int iv = 0; iv < 4; ++iv)
      {
        unsigned i_first_dof = iv * n_element;
        indices_mapping.insert(
          {cell->vertex_index(iv), cell_dof_indices[i_first_dof]});
      }
  }

  for (unsigned int v_id = 0; v_id < indices_mapping.size(); ++v_id)
    {
      for (unsigned int j = 0; j < spacedim; ++j)
        {
          unsigned int first_dof_id    = indices_mapping.find(v_id)->second;
          vec_values[first_dof_id + j] = vertices[v_id][j];
        }
    }

  unsigned int fe_id = 0;
  for (unsigned int iv = 0; iv < indices_mapping_valence_to_fe.size(); ++iv)
    {
      for (unsigned int i = 0; i < indices_mapping_valence_to_fe[iv].size();
           ++i, ++fe_id)
        {
          MappingFEField_hp<dim, spacedim, Vector<double>, hp::DoFHandler<2, 3>>
            mapping(dof_handler, vec_values, fe_id, mask);
          mapping_collection.push_back(mapping);
        }
    }
}



template <int dim, int spacedim>
std::vector<
  std::set<typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>>
CatmullClark<dim, spacedim>::cell_patches(
  const hp::DoFHandler<dim, spacedim> &dof_handler)
{
  std::multimap<unsigned int,
                typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
    vertex_to_cell_map;
  for (typename hp::DoFHandler<dim, spacedim>::active_cell_iterator cell =
         dof_handler.begin_active();
       cell != dof_handler.end();
       ++cell)
    for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
      vertex_to_cell_map.insert({cell->vertex_index(v), cell});

  std::vector<
    std::set<typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>>
    vector_of_sets;

  for (typename hp::DoFHandler<dim, spacedim>::active_cell_iterator cell =
         dof_handler.begin_active();
       cell != dof_handler.end();
       ++cell)
    {
      std::set<typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
        cells_in_patch;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          typename std::multimap<unsigned int,
                                 typename hp::DoFHandler<dim, spacedim>::
                                   active_cell_iterator>::iterator map_it;
          for (map_it = vertex_to_cell_map.begin();
               map_it != vertex_to_cell_map.end();
               map_it++)
            {
              if (map_it->first == cell->vertex_index(v))
                {
                  cells_in_patch.insert(map_it->second);
                }
            }
        }
      vector_of_sets.push_back(cells_in_patch);
    }
  return vector_of_sets;
}



template <int dim, int spacedim>
std::map<unsigned int,
         typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
CatmullClark<dim, spacedim>::ordering_cells_in_patch(
  typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell,
  std::set<typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
    cells_in_patch)
{
  std::map<unsigned int,
           typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
    cell_patch_map;
    if (cell->at_boundary() == false)
    {
      if (cells_in_patch.size() == 9)
        {
          /*
           *-----*-----*-----*
           |     |     |     |
           |  5  |  3  |  6  |
           |     |     |     |
           *-----*-----*-----*
           |     |     |     |
           |  1  |  0  |  2  |
           |     |     |     |
           *-----*-----*-----*
           |     |     |     |
           |  7  |  4  |  8  |
           |     |     |     |
           *-----*-----*-----*
           */

          /*
           *       2
           *    0-----1
           *    |     |
           *  0 |     | 1
           *    |     |
           *    2-----3
           *       3
           */
          typename std::set<typename hp::DoFHandler<dim, spacedim>::
                              active_cell_iterator>::iterator it_cell;
          for (it_cell = cells_in_patch.begin();
               it_cell != cells_in_patch.end();
               ++it_cell)
            {
              typename hp::DoFHandler<dim, spacedim>::active_cell_iterator
                icell = *it_cell;
                if (icell->active_cell_index() == cell->active_cell_index())
                {
                  cell_patch_map.insert({0, icell});
                }
              else
                {
                  bool neighbour_cell = false;
                  for (unsigned int ie = 0;
                       ie < GeometryInfo<2>::faces_per_cell;
                       ++ie)
                    {
                      if (int(icell->active_cell_index()) ==
                          cell->neighbor_index(ie))
                        {
                          cell_patch_map.insert({ie + 1, icell});
                          neighbour_cell = true;
                        }
                    }
                  if (!neighbour_cell)
                    {
                      for (unsigned int iu = 0;
                           iu < GeometryInfo<2>::vertices_per_cell;
                           ++iu)
                        {
                          for (unsigned int iv = 0;
                               iv < GeometryInfo<2>::vertices_per_cell;
                               ++iv)
                            {
                              if (icell->vertex_index(iv) ==
                                  cell->vertex_index(iu))
                                {
                                  cell_patch_map.insert({5 + iu, icell});
                                }
                            }
                        }
                    }
                }
            }
        }
      else
        {
          std::vector<int> valence(GeometryInfo<2>::vertices_per_cell, 0);
          std::multimap<
            unsigned int,
            typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
            vertex_cells_map;
          for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
            {
              typename std::set<typename hp::DoFHandler<dim, spacedim>::
                                  active_cell_iterator>::iterator it_cell;
              for (it_cell = cells_in_patch.begin();
                   it_cell != cells_in_patch.end();
                   ++it_cell)
                {
                  typename hp::DoFHandler<dim, spacedim>::active_cell_iterator
                    icell = *it_cell;
                  for (unsigned int iv = 0;
                       iv < GeometryInfo<2>::vertices_per_cell;
                       ++iv)
                    {
                        if (icell->vertex_index(iv) == cell->vertex_index(v))
                        {
                          valence[v] += 1;
                          vertex_cells_map.insert({v, icell});
                        }
                    }
                }
            }
          int                         n_extraordinary = 0;
          std::array<unsigned int, 4> face_loop;
          int                         exv;
          int                         val;
          for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
            {
              if (valence[v] != 4)
                {
                  n_extraordinary += 1;
                  exv       = v;
                  face_loop = vertex_face_loop(v);
                  val       = valence[v];
                }
            }
          std::array<unsigned int, 3> vertices_regular = next_vertices(exv);
          if (n_extraordinary > 1)
            {
              throw std::runtime_error(
                "can not analyse cell with more than one extraordinary vertices, please refine your triangulation.");
            }
          /*                  *
                            / |
                          /   |
                        /     |
           *-----*-----* N-2  *-----*
           |     |     |     /     /
           | N+4 | N-1 |  ..  2  /
           |     |     | /     /
           *-----*-----*-----*
           |     |     |     |
           | N+3 |  0  |  1  |
           |     |     |     |
           *-----*-----*-----*
           |     |     |     |
           |  N  | N+1 | N+2 |
           |     |     |     |
           *-----*-----*-----*         */


          /* regenerated face indices as a loop like
           *       3
           *    *-----*
           *    |     |
           *  2 |     | 0
           *    |     |
           *    *-----*
           *       1
           */
          /* regenerated vertex indices as a loop like
           *    3-----0
           *    |     |
           *    |     |
           *    |     |
           *    2-----1
           */
          std::map<unsigned int, unsigned int> face_to_neighbour,
            vertex_to_cell;
          face_to_neighbour.insert({0, 1});
          face_to_neighbour.insert({1, val + 1});
          face_to_neighbour.insert({2, val + 3});
          face_to_neighbour.insert({3, val - 1});
          vertex_to_cell.insert({0, val + 2});
          vertex_to_cell.insert({1, val});
          vertex_to_cell.insert({2, val + 4});
          typename std::set<typename hp::DoFHandler<dim, spacedim>::
                              active_cell_iterator>::iterator it_cell;
          for (it_cell = cells_in_patch.begin();
               it_cell != cells_in_patch.end();
               ++it_cell)
            {
              typename hp::DoFHandler<dim, spacedim>::active_cell_iterator
                icell = *it_cell;
                if (icell->active_cell_index() == cell->active_cell_index())
                {
                  cell_patch_map.insert({0, icell});
                }
              else
                {
                  bool neighbour_cell = false;
                  for (auto it_face = face_to_neighbour.begin();
                       it_face != face_to_neighbour.end();
                       ++it_face)
                    {
                      if (int(icell->active_cell_index()) ==
                          cell->neighbor_index(face_loop[it_face->first]))
                        {
                          cell_patch_map.insert({it_face->second, icell});
                          neighbour_cell = true;
                        }
                    }

                  if (!neighbour_cell)
                    {
                      for (auto it_vert = vertex_to_cell.begin();
                           it_vert != vertex_to_cell.end();
                           ++it_vert)
                        {
                          for (unsigned int iv = 0;
                               iv < GeometryInfo<2>::vertices_per_cell;
                               ++iv)
                            {
                              if (icell->vertex_index(iv) ==
                                  cell->vertex_index(
                                    vertices_regular[it_vert->first]))
                                {
                                  cell_patch_map.insert(
                                    {it_vert->second, icell});
                                }
                            }
                        }
                    }
                }
            }
          auto icell = cell_patch_map.find(1)->second;
          std::vector<
            typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
            cells_with_exv;
          for (auto cell_with_exv = vertex_cells_map.begin();
               cell_with_exv != vertex_cells_map.end();
               ++cell_with_exv)
            {
              if (int(cell_with_exv->first) == exv &&
                  cell_with_exv->second->active_cell_index() !=
                    cell_patch_map.find(0)->second->active_cell_index() &&
                  cell_with_exv->second->active_cell_index() !=
                    cell_patch_map.find(val - 1)->second->active_cell_index())
                {
                  cells_with_exv.push_back(cell_with_exv->second);
                }
            }
          for (int n = 2; n < val - 1; ++n)
            {
              auto this_cell = icell;
              for (auto &cell_with_exv : cells_with_exv)
                {
                  if (cell_with_exv->active_cell_index() !=
                      this_cell->active_cell_index())
                    {
                      for (unsigned int ie = 0;
                           ie < GeometryInfo<2>::faces_per_cell;
                           ++ie)
                        {
                          if (this_cell->neighbor_index(ie) ==
                                int(cell_with_exv->active_cell_index()) &&
                              cell_with_exv->active_cell_index() !=
                                cell_patch_map.find(n - 2)
                                  ->second->active_cell_index())
                            {
                              cell_patch_map.insert({n, cell_with_exv});
                              icell = cell_with_exv;
                            }
                        }
                    }
                }
            }
        }
    }
  else
    {
      if (cells_in_patch.size() == 6)
        {
          /*
           ordering indices of faces and cells, cell 0 is the target cell.
           *-----*-----*-----*
           |     |     |     |
           |  5  |  2  |  4  |
           |     |     |     |
           *-----*-----*-----*
           |     |     |     |
           |  3  |  0  |  1  |
           |     |     |     |
           *-----*-----*-----*     */
          /*       2
           *    0-----1
           *    |     |
           *  0 |     | 1
           *    |     |
           *    2-----3
           *       3
           */
          std::array<unsigned int, 3> local_face_indices;
          std::array<unsigned int, 2> local_vertex_indices;
          int                         n_boundary_cell = 0;
          for (unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie)
            {
                if (cell->at_boundary(ie))
                {
                  local_face_indices   = loop_faces(ie);
                  local_vertex_indices = opposite_vertices(ie);
                  n_boundary_cell += 1;
                }
            }
          Assert(
            n_boundary_cell == 1,
            ExcMessage(
              "This is not an element with one edge on physical boundary."));
          typename std::set<typename hp::DoFHandler<dim, spacedim>::
                              active_cell_iterator>::iterator it_cell;
          for (it_cell = cells_in_patch.begin();
               it_cell != cells_in_patch.end();
               ++it_cell)
            {
              typename hp::DoFHandler<dim, spacedim>::active_cell_iterator
                icell = *it_cell;
                if (icell->active_cell_index() == cell->active_cell_index())
                {
                  cell_patch_map.insert({0, icell});
                }
              else
                {
                  bool neighbour_cell = false;
                  for (unsigned int inei = 0; inei < local_face_indices.size();
                       ++inei)
                    {
                      if (int(icell->active_cell_index()) ==
                          cell->neighbor_index(local_face_indices[inei]))
                        {
                          cell_patch_map.insert({1 + inei, icell});
                          neighbour_cell = true;
                        }
                    }
                  if (!neighbour_cell)
                    {
                      for (unsigned int iu = 0;
                           iu < local_vertex_indices.size();
                           ++iu)
                        {
                          unsigned int id_v =
                            cell->vertex_index(local_vertex_indices[iu]);
                          for (unsigned int iv = 0;
                               iv < GeometryInfo<2>::vertices_per_cell;
                               ++iv)
                            {
                              if (icell->vertex_index(iv) == id_v)
                                {
                                  cell_patch_map.insert({4 + iu, icell});
                                }
                            }
                        }
                    }
                }
            }
        }
      else
        {
          if (cells_in_patch.size() == 4)
            {
              /*
               ordering indices of faces and cells, cell 0 is the target cell.
               *-----*-----*
               |     |     |
               |  2  |  3  |
               |     |     |
               *-----*-----*
               |     |     |
               |  0  |  1  |
               |     |     |
               *-----*-----*     */
              /*       2
               *    0-----1
               *    |     |
               *  0 |     | 1
               *    |     |
               *    2-----3
               *       3
               */

              std::vector<unsigned int> faces_on_boundary(0);
              for (unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell;
                   ++ie)
                {
                    if (cell->at_boundary(ie))
                    {
                      faces_on_boundary.push_back(ie);
                    }
                }
              Assert(
                faces_on_boundary.size() == 2,
                ExcMessage(
                  "This is not an element with two edges on physical boundary."));

              std::array<unsigned int, 2> f_n_b =
                faces_not_on_boundary(faces_on_boundary);

              typename std::set<typename hp::DoFHandler<dim, spacedim>::
                                  active_cell_iterator>::iterator it_cell;
              for (it_cell = cells_in_patch.begin();
                   it_cell != cells_in_patch.end();
                   ++it_cell)
                {
                  typename hp::DoFHandler<dim, spacedim>::active_cell_iterator
                    icell = *it_cell;
                    if (icell->active_cell_index() == cell->active_cell_index())
                    {
                      cell_patch_map.insert({0, icell});
                    }
                  else
                    {
                      bool neighbour_cell = false;
                      for (unsigned int inei = 0; inei < f_n_b.size(); ++inei)
                        {
                          if (int(icell->active_cell_index()) ==
                              cell->neighbor_index(f_n_b[inei]))
                            {
                              cell_patch_map.insert({1 + inei, icell});
                              neighbour_cell = true;
                            }
                        }
                      if (!neighbour_cell)
                        {
                          cell_patch_map.insert({3, icell});
                        }
                    }
                }
            }
          else
            {
              throw std::runtime_error(
                "current code can not deal with boundary extraordinary vertex.");
            }
        }
    }
  return cell_patch_map;
}



template <int dim, int spacedim>
void CatmullClark<dim, spacedim>::new_order_for_cells(
  hp::DoFHandler<dim, spacedim> &dof_handler)
{
  std::vector<std::vector<unsigned int>> dof_indices_order_vector(
    dof_handler.get_triangulation().n_active_cells());
    ordered_cell_patch_vector.resize(dof_handler.get_triangulation().n_active_cells());
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
       ++cell)
    {
        ordered_cell_patch_vector [cell->active_cell_index()] =
        ordering_cells_in_patch(cell,
                                  cell_patch_vector[cell->active_cell_index()]);
    }
}



template <int dim, int spacedim>
const std::vector<unsigned int> CatmullClark<dim, spacedim>::get_neighbour_dofs(
  typename hp::DoFHandler<dim, spacedim>::active_cell_iterator cell_0,
  typename hp::DoFHandler<dim, spacedim>::active_cell_iterator cell_neighbour,
  unsigned int                                                 n_element) const
{
  // left to right, up to down
  std::vector<types::global_dof_index> cell_0_indices(
    cell_0->get_fe().dofs_per_cell, 0);
  std::vector<types::global_dof_index> cell_neighbour_indices(
    cell_neighbour->get_fe().dofs_per_cell, 0);
  cell_0->get_dof_indices(cell_0_indices);
  cell_neighbour->get_dof_indices(cell_neighbour_indices);
  /*
   *       2
   *    0-----1
   *    |     |
   *  0 |     | 1
   *    |     |
   *    2-----3
   *       3
   */
  // find how many common DoFs the two cells share
  int              n_common = 0;
  std::vector<int> ith_vert(0);
  for (int i = 0; i < 4; ++i)
    {
      int ii = i * n_element;
      for (int j = 0; j < 4; ++j)
        {
          int jj = j * n_element;
          if (cell_neighbour_indices[ii] == cell_0_indices[jj])
            {
              n_common += 1;
              ith_vert.push_back(i);
            }
        }
    }
  if (n_common == 2)
    {
      std::vector<unsigned int> vert_i =
        opposite_face_dofs(ith_vert[0], ith_vert[1]);
      std::vector<unsigned int> neighbour_dofs_vec;
      for (unsigned int iel = 0; iel < n_element; ++iel)
        {
          neighbour_dofs_vec.push_back(
            static_cast<unsigned int>(static_cast<int>(
              cell_neighbour_indices[vert_i[0] * n_element + iel])));
        }
      for (unsigned int iel = 0; iel < n_element; ++iel)
        {
          neighbour_dofs_vec.push_back(
            static_cast<unsigned int>(static_cast<int>(
              cell_neighbour_indices[vert_i[1] * n_element + iel])));
        }
      return neighbour_dofs_vec;
    }
  else if (n_common == 1)
    {
      int                       vert_i = opposite_vertex_dofs(ith_vert[0]);
      std::vector<unsigned int> neighbour_dofs_vec;
      for (unsigned int iel = 0; iel < n_element; ++iel)
        {
          neighbour_dofs_vec.push_back(
            static_cast<unsigned int>(static_cast<int>(
              cell_neighbour_indices[vert_i * n_element + iel])));
        }
      return neighbour_dofs_vec;
    }
  else
    {
      throw std::runtime_error("The two cells have no common DoFs.");
    }
}



template <int dim, int spacedim>
const std::array<unsigned int, 4> CatmullClark<dim, spacedim>::vertex_face_loop(
  const unsigned int local_vertex_id) const
{
  switch (local_vertex_id)
    {
      case 0:
        return {2, 1, 3, 0};
      case 1:
        return {1, 3, 0, 2};
      case 2:
        return {0, 2, 1, 3};
      case 3:
        return {3, 0, 2, 1};
      default:
        throw std::runtime_error("vertex_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
unsigned int CatmullClark<dim, spacedim>::opposite_vertex_dofs(unsigned int i) const
{
  switch (i)
    {
      case 0:
        return 3;
      case 1:
        return 2;
      case 2:
        return 1;
      case 3:
        return 0;
      default:
        throw std::runtime_error("vertex_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
std::vector<unsigned int>
const CatmullClark<dim, spacedim>::opposite_face_dofs(unsigned int i, unsigned int j) const
{
  switch (i)
    {
      case 0:
        switch (j)
          {
            case 1:
              return {2, 3};
            case 2:
              return {1, 3};
          }
      case 1:
        return {0, 2};
      case 2:
        return {0, 1};

      default:
        throw std::runtime_error("vertex_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int, 3>
CatmullClark<dim, spacedim>::next_vertices(const unsigned int local_vertex_id) const
{
  switch (local_vertex_id)
    {
      case 0:
        return {1, 3, 2};
      case 1:
        return {3, 2, 0};
      case 2:
        return {0, 1, 3};
      case 3:
        return {2, 0, 1};
      default:
        throw std::runtime_error("vertex_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int, 3>
CatmullClark<dim, spacedim>::loop_faces(const unsigned int local_face_id) const
{
  switch (local_face_id)
    {
      case 0:
        return {3, 1, 2};
      case 1:
        return {2, 0, 3};
      case 2:
        return {0, 3, 1};
      case 3:
        return {1, 2, 0};
      default:
        throw std::runtime_error("face_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int, 2>
CatmullClark<dim, spacedim>::opposite_vertices(const unsigned int local_face_id) const
{
  switch (local_face_id)
    {
      case 0:
        return {3, 1};
      case 1:
        return {0, 2};
      case 2:
        return {2, 3};
      case 3:
        return {1, 0};
      default:
        throw std::runtime_error("face_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int, 2>
CatmullClark<dim, spacedim>::faces_not_on_boundary(
  const std::vector<unsigned int> m) const
{
  /*
   *       2
   *    0-----1
   *    |     |
   *  0 |     | 1
   *    |     |
   *    2-----3
   *       3
   */

  if (m.size() != 2)
    {
      throw std::runtime_error("m must has two entries");
    }

  switch (m[0])
    {
      case 0:
        switch (m[1])
          {
            case 3:
              return {1, 2};
            case 2:
              return {3, 1};
          }
      case 1:
        switch (m[1])
          {
            case 3:
              return {2, 0};
            case 2:
              return {0, 3};
          }

      default:
        throw std::runtime_error("faces_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int, 4>
CatmullClark<dim, spacedim>::verts_id_on_boundary(
  const std::vector<unsigned int> m) const
{
  /*
   *       2
   *    0-----1
   *    |     |
   *  0 |     | 1
   *    |     |
   *    2-----3
   *       3
   */

  if (m.size() != 2)
    {
      throw std::runtime_error("m must has two entries");
    }

  switch (m[0])
  {
//      case 0:
//          switch (m[1])
//          {
//              case 3:
//                  return {2, 3, 1, 0};
//              case 2:
//                  return {0, 2, 3, 1};
//          }
//      case 1:
//          switch (m[1])
//          {
//              case 3:
//                  return {3, 1, 0, 2};
//              case 2:
//                  return {1, 0, 2, 3};
//          }
      case 0:
          switch (m[1])
          {
              case 3:
                  return {2, 0, 1, 3};
              case 2:
                  return {0, 1, 3, 2};
          }
      case 1:
          switch (m[1])
          {
              case 3:
                  return {3, 2, 0, 1};
              case 2:
                  return {1, 3, 2, 0};
          }
          
      default:
          throw std::runtime_error("faces_id_not_valid.");
          break;
    }
}



template <int dim, int spacedim>
unsigned int
CatmullClark<dim, spacedim>::common_face_local_id(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_0,
                                                  typename hp::DoFHandler<dim, spacedim>::active_cell_iterator cell) const
{
  unsigned int face_id;
  for (unsigned i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
    {
      for (unsigned j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j)
        {
          if (cell_0->face_index(i) == cell->face_index(j))
            {
              face_id = j;
            }
        }
    }
  return face_id;
}



template <int dim, int spacedim>
std::array<unsigned int, 4>
const CatmullClark<dim, spacedim>::rotated_vertices(const unsigned int local_face_id) const
{
    /*    -> u
     *   |     2
     * v V  0-----1
     *      |     |
     *    0 |     | 1
     *      |     |
     *      2-----3
     *         3
     */
  switch (local_face_id)
    {
//      case 0:
//        return {0, 2, 3, 1};
//      case 1:
//        return {3, 1, 0, 2};
//      case 2:
//        return {1, 0, 2, 3};
//      case 3:
//        return {2, 3, 1, 0};
        case 0:
            return {2, 0, 1, 3};
        case 1:
            return {1, 3, 2, 0};
        case 2:
            return {0, 1, 3, 2};
        case 3:
            return {3, 2, 0, 1};
        default:
        throw std::runtime_error("face_id_not_valid.");
        break;
    }
}



template <int dim, int spacedim>
std::vector<unsigned int>
const CatmullClark<dim, spacedim>::get_diagonal_dof_id_to_ex(
  typename hp::DoFHandler<dim, spacedim>::active_cell_iterator cell_0,
  typename hp::DoFHandler<dim, spacedim>::active_cell_iterator cell_neighbour,
  unsigned int                                                 ex_index,
  unsigned int                                                 n_element) const
{
  std::vector<types::global_dof_index> cell_0_indices(
    cell_0->get_fe().dofs_per_cell, 0);
  std::vector<types::global_dof_index> cell_neighbour_indices(
    cell_neighbour->get_fe().dofs_per_cell, 0);
  cell_0->get_dof_indices(cell_0_indices);
  cell_neighbour->get_dof_indices(cell_neighbour_indices);
  types::global_dof_index global_ex_index =
    cell_0_indices[ex_index * n_element];
  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < 4; ++i)
    {
      if (cell_neighbour_indices[i * n_element] == global_ex_index)
        {
          for (unsigned int iel = 0; iel < n_element; ++iel)
            {
              indices.push_back(opposite_vertex_dofs(i) * n_element + iel);
            }
        }
    }
  return indices;
}

template <int dim, int spacedim>
std::vector<types::global_dof_index>
CatmullClark<dim, spacedim>::non_local_dof_indices(
  const DoFCellAccessor<dim, spacedim, false> &accessor) const
{
  std::map<unsigned int,
           typename hp::DoFHandler<dim, spacedim>::active_cell_iterator>
  cells = ordered_cell_patch_vector[accessor.active_cell_index()];
    
  unsigned int                         valence;
  std::vector<types::global_dof_index>
    cell_dof_indices(accessor.get_fe().n_dofs_per_cell(), 0);
  std::vector<types::global_dof_index>
    non_local_dof_indices(accessor.get_fe().n_non_local_dofs_per_cell(), 0);
  accessor.get_dof_indices(cell_dof_indices);
            
  switch (cells.size())
    {
      case 4:
        {
          valence = 1;
          /*
           *-----*-----*
           |     |     |
           |  2  |  3  |
           |     |     |
           *-----*-----*
           |     |     |
           |  0  |  1  |
           |     |     |
           *-----*-----*     */
          /*
           *       2
           *    0-----1
           *    |     |
           *  0 |     | 1
           *    |     |
           *    2-----3
           *       3
           */
          /*
           6-----7-----8
           |     |     |
           |     |     |
           |     |     |
           3-----4-----5
           |     |     |
           |     |     |
           |     |     |
           0-----1-----2     */
          /*
          indices mapping for non-local dofs
          0(4)-----1(5)----2(6)
          |        |        |
          |        |        |
          |        |        |
          ?--------?-------3(7)
          |        |        |
          |        |        |
          |        |        |
          ?--------?-------4(8)     */
          std::vector<unsigned int> edges_on_boundary(0);
          for (unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie)
            {
              if (cells[0]->at_boundary(ie))
                {
                  edges_on_boundary.push_back(ie);
                }
            }
          for (unsigned int iel = 0; iel < n_element; ++iel)
            {
              int face_local_id;
              face_local_id     = common_face_local_id(cells[0], cells[1]);
              auto rotated_iv_1 = rotated_vertices(face_local_id);
              cell_dof_indices.resize(cells[1]->get_fe().dofs_per_cell);
              cells[1]->get_dof_indices(cell_dof_indices);
              non_local_dof_indices[4 * n_element + iel] =
                cell_dof_indices[rotated_iv_1[3] * n_element + iel];
              non_local_dof_indices[3 * n_element + iel] =
                cell_dof_indices[rotated_iv_1[2] * n_element + iel];

              face_local_id     = common_face_local_id(cells[0], cells[2]);
              auto rotated_iv_2 = rotated_vertices(face_local_id);
              cell_dof_indices.resize(cells[2]->get_fe().dofs_per_cell);
              cells[2]->get_dof_indices(cell_dof_indices);
              non_local_dof_indices[1 * n_element + iel] =
                cell_dof_indices[rotated_iv_2[3] * n_element + iel];
              non_local_dof_indices[0 * n_element + iel] =
                cell_dof_indices[rotated_iv_2[2] * n_element + iel];
                
              non_local_dof_indices[2 * n_element + iel] =
                get_neighbour_dofs(cells[0],
                                   cells[3],
                                   n_element)[0 * n_element + iel];
            }
          break;
        }
      case 6:
        {
          valence = 2;
          /*
           *-----*-----*-----*
           |     |     |     |
           |  5  |  2  |  4  |
           |     |     |     |
           *-----*-----*-----*
           |     |     |     |
           |  3  |  0  |  1  |
           |     |     |     |
           *-----*-----*-----*     */
          /*
           *       2
           *    0-----1
           *    |     |
           *  0 |     | 1
           *    |     |
           *    2-----3
           *       3
           */
          /*
           8-----9----10----11
           |     |     |     |
           |     |     |     |
           |     |     |     |
           4-----5-----6-----7
           |     |     |     |
           |     |     |     |
           |     |     |     |
           0-----1-----2-----3     */
          /*
          7(11)----0(4)-----1(5)------2(6)
          |        |         |        |
          |        |         |        |
          |        |         |        |
          6(10)----?---------?--------3(7)
          |        |         |        |
          |        |         |        |
          |        |         |        |
          5(9)-----?---------?--------4(8)     */

          int face_local_id;
          face_local_id     = common_face_local_id(cells[0], cells[1]);
          auto rotated_iv_1 = rotated_vertices(face_local_id);
          cell_dof_indices.resize(cells[1]->get_fe().dofs_per_cell);
          cells[1]->get_dof_indices(cell_dof_indices);
          for (unsigned int iel = 0; iel < n_element; ++iel)
            {
              non_local_dof_indices[6 * n_element + iel] =
                cell_dof_indices[rotated_iv_1[2] * n_element + iel];
              non_local_dof_indices[5 * n_element + iel] =
                cell_dof_indices[rotated_iv_1[3] * n_element + iel];
            }
          face_local_id     = common_face_local_id(cells[0], cells[2]);
          auto rotated_iv_2 = rotated_vertices(face_local_id);
          cell_dof_indices.resize(cells[2]->get_fe().dofs_per_cell);
          cells[2]->get_dof_indices(cell_dof_indices);
          for (unsigned int iel = 0; iel < n_element; ++iel)
            {
              non_local_dof_indices[1 * n_element + iel] =
                cell_dof_indices[rotated_iv_2[2] * n_element + iel];
              non_local_dof_indices[0 * n_element + iel] =
                cell_dof_indices[rotated_iv_2[3] * n_element + iel];
            }
          face_local_id     = common_face_local_id(cells[0], cells[3]);
          auto rotated_iv_3 = rotated_vertices(face_local_id);
          cell_dof_indices.resize(cells[3]->get_fe().dofs_per_cell);
          cells[3]->get_dof_indices(cell_dof_indices);
          for (unsigned int iel = 0; iel < n_element; ++iel)
            {
              non_local_dof_indices[4 * n_element + iel] =
                cell_dof_indices[rotated_iv_3[2] * n_element + iel];
              non_local_dof_indices[3 * n_element + iel] =
                cell_dof_indices[rotated_iv_3[3] * n_element + iel];

              non_local_dof_indices[7 * n_element + iel] =
                get_neighbour_dofs(cells[0],
                                   cells[4],
                                   n_element)[0 * n_element + iel];
              non_local_dof_indices[2 * n_element + iel] =
                get_neighbour_dofs(cells[0],
                                   cells[5],
                                   n_element)[0 * n_element + iel];
            }
          break;
        }
      case 9:
        {
          valence = 4;
          /*
           *-----*-----*-----*
           |     |     |     |
           |  5  |  3  |  6  |
           |     |     |     |
           *-----*-----*-----*
           |     |     |     |
           |  1  |  0  |  2  |
           |     |     |     |
           *-----*-----*-----*
           |     |     |     |
           |  7  |  4  |  8  |
           |     |     |     |
           *-----*-----*-----*
           */
          /*
           *       2
           *    0-----1
           *    |     |
           *  0 |     | 1
           *    |     |
           *    2-----3
           *       3
           */
          /*
           12----13----14----15
           |     |     |     |
           |     |     |     |
           |     |     |     |
           8-----9-----10----11
           |     |     |     |
           |     |     |     |
           |     |     |     |
           4-----5-----6-----7
           |     |     |     |
           |     |     |     |
           |     |     |     |
           0-----1-----2-----3
           */
          /*
          indices mapping for non-local dofs
           11(15)---0(4)-----1(5)-----2(6)
           |        |        |        |
           |        |        |        |
           |        |        |        |
           10(14)--(0)------(1)-------3(7)
           |        |        |        |
           |        |        |        |
           |        |        |        |
           9(13)---(2)------(3)-------4(8)
           |        |        |        |
           |        |        |        |
           |        |        |        |
           8(12)----7(11)----6(10)----5(9)
           */

          auto temp_indices_01 =
            get_neighbour_dofs(cells[0], cells[1], n_element);
          auto temp_indices_02 =
            get_neighbour_dofs(cells[0], cells[2], n_element);
          auto temp_indices_03 =
            get_neighbour_dofs(cells[0], cells[3], n_element);
          auto temp_indices_04 =
            get_neighbour_dofs(cells[0], cells[4], n_element);
          for (unsigned int iel = 0; iel < n_element; ++iel)
            {
              non_local_dof_indices[10 * n_element + iel] =
                temp_indices_01[0 * n_element + iel];
              non_local_dof_indices[9 * n_element + iel] =
                temp_indices_01[1 * n_element + iel];

              non_local_dof_indices[3 * n_element + iel] =
                temp_indices_02[0 * n_element + iel];
              non_local_dof_indices[4 * n_element + iel] =
                temp_indices_02[1 * n_element + iel];

              non_local_dof_indices[0 * n_element + iel] =
                temp_indices_03[0 * n_element + iel];
              non_local_dof_indices[1 * n_element + iel] =
                temp_indices_03[1 * n_element + iel];

              non_local_dof_indices[7 * n_element + iel] =
                temp_indices_04[0 * n_element + iel];
              non_local_dof_indices[6 * n_element + iel] =
                temp_indices_04[1 * n_element + iel];

              non_local_dof_indices[11 * n_element + iel] =
                get_neighbour_dofs(cells[0],
                                   cells[5],
                                   n_element)[0 * n_element + iel];
              non_local_dof_indices[2 * n_element + iel] =
                get_neighbour_dofs(cells[0],
                                   cells[6],
                                   n_element)[0 * n_element + iel];
              non_local_dof_indices[8 * n_element + iel] =
                get_neighbour_dofs(cells[0],
                                   cells[7],
                                   n_element)[0 * n_element + iel];
              non_local_dof_indices[5 * n_element + iel] =
                get_neighbour_dofs(cells[0],
                                   cells[8],
                                   n_element)[0 * n_element + iel];
            }
          break;
        }
      default:
        valence = cells.size() - 5;
        /*                  *
                          / |
                        /   |
                      /     |
         *-----*-----* N-2  *-----*
         |     |     |     /     /
         | N+4 | N-1 |  ..  2  /
         |     |     | /     /
         *-----*-----*-----*
         |     |     |     |
         | N+3 |  0  |  1  |
         |     |     |     |
         *-----*-----*-----*
         |     |     |     |
         |  N  | N+1 | N+2 |
         |     |     |     |
         *-----*-----*-----*         */

        /*                   2v
                           /  |
                          /   |
                         /    |
         2v+7-----2-----1     *------8
         |        |     |    /     /
         |        |     |  ..    /
         |        |     |/     /
         2v+6-----3-----0-----7
         |        |     |     |
         |        |     |     |
         |        |     |     |
         2v+5-----4-----5-----6
         |        |     |     |
         |        |     |     |
         |        |     |     |
         2v+1---2v+2---2v+3---2v+4         */

        /*                   2v-4
                           /  |
                          /   |
                         /    |
         2v+3-----1-----0     *------4
         |        |     |    /     /
         |        |     |  ..    /
         |        |     |/     /
         2v+2-----?-----?-----3
         |        |     |     |
         |        |     |     |
         |        |     |     |
         2v+1-----?-----?-----2
         |        |     |     |
         |        |     |     |
         |        |     |     |
         2v-3---2v-2---2v-1---2v         */



        int ex_vertex_index;
        for (unsigned int i = 0; i < 4; ++i)
          {
            unsigned int n = 0;
            for (unsigned int icell = 1; icell < 3; ++icell)
              {
                std::vector<types::global_dof_index> icell_dof_indices;
                icell_dof_indices.resize(cells[icell]->get_fe().dofs_per_cell);
                cells[icell]->get_dof_indices(icell_dof_indices);
                for (unsigned int j = 0; j < 4; ++j)
                  {
                    if (cell_dof_indices[i * n_element] ==
                        icell_dof_indices[j * n_element])
                      {
                        n += 1;
                      }
                  }
              }
            if (n == 2)
              {
                ex_vertex_index = i;
              }
          }

        for (unsigned int ic = 1; ic < valence - 1; ++ic)
          {
            std::vector<types::global_dof_index> cell_dof_indices;
            cell_dof_indices.resize(cells[ic]->get_fe().dofs_per_cell);
            cells[ic]->get_dof_indices(cell_dof_indices);
            std::vector<unsigned int> dia_dof_id = get_diagonal_dof_id_to_ex(
              cells[0], cells[ic], ex_vertex_index, n_element);
            for (unsigned int iel = 0; iel < n_element; ++iel)
              {
                non_local_dof_indices[2 * ic * n_element + iel] =
                  cell_dof_indices[dia_dof_id[iel]];
              }

            auto v = get_neighbour_dofs(cells[ic - 1], cells[ic], n_element);
            std::vector<unsigned int> on_face_dof(n_element);

            if (v[0] == cell_dof_indices[dia_dof_id[0]])
              {
                for (unsigned int iel = 0; iel < n_element; ++iel)
                  {
                    on_face_dof[iel] = v[1 * n_element + iel];
                  }
              }
            else
              {
                for (unsigned int iel = 0; iel < n_element; ++iel)
                  {
                    on_face_dof[iel] = v[0 * n_element + iel];
                  }
              }
            for (unsigned int iel = 0; iel < n_element; ++iel)
              {
                non_local_dof_indices[(1 + 2 * ic) * n_element + iel] =
                  on_face_dof[iel];
              }
          }
        cell_dof_indices.resize(cells[valence - 1]->get_fe().dofs_per_cell);
        cells[valence - 1]->get_dof_indices(cell_dof_indices);


        std::vector<unsigned int> dia_dof_id = get_diagonal_dof_id_to_ex(
          cells[0], cells[valence - 1], ex_vertex_index, n_element);
        for (unsigned int iel = 0; iel < n_element; ++iel)
          {
            non_local_dof_indices[1 * n_element + iel] =
              cell_dof_indices[dia_dof_id[iel]];
          }
        auto v = get_neighbour_dofs(cells[0], cells[valence - 1], n_element);
        std::vector<unsigned int> on_face_dof(n_element);

        if (v[0] == cell_dof_indices[dia_dof_id[0]])
          {
            for (unsigned int iel = 0; iel < n_element; ++iel)
              {
                on_face_dof[iel] = v[1 * n_element + iel];
              }
          }
        else
          {
            for (unsigned int iel = 0; iel < n_element; ++iel)
              {
                on_face_dof[iel] = v[0 * n_element + iel];
              }
          }
        for (unsigned int iel = 0; iel < n_element; ++iel)
          {
            non_local_dof_indices[0 * n_element + iel] = on_face_dof[iel];
          }

        std::vector<std::vector<unsigned int>> dof_pairs(6);
        dof_pairs[0] =
          get_neighbour_dofs(cells[1], cells[valence + 2], n_element);
        dof_pairs[1] =
          get_neighbour_dofs(cells[0], cells[valence + 1], n_element);
        dof_pairs[2] =
          get_neighbour_dofs(cells[valence + 3], cells[valence], n_element);
        dof_pairs[3] =
          get_neighbour_dofs(cells[valence + 1], cells[valence], n_element);
        dof_pairs[4] =
          get_neighbour_dofs(cells[0], cells[valence + 3], n_element);
        dof_pairs[5] =
          get_neighbour_dofs(cells[valence - 1], cells[valence + 4], n_element);

        std::vector<unsigned int> dof_vec(2 * n_element, 0);

        for (unsigned int i = 0; i < 2; ++i)
          {
            unsigned int idof  = dof_pairs[0][i * n_element];
            bool         is_in = false;
            for (unsigned int j = 0; j < 2; ++j)
              {
                if (idof == dof_pairs[1][j * n_element])
                  {
                    is_in = true;
                  }
              }
            if (is_in == false)
              {
                for (unsigned int iel = 0; iel < n_element; ++iel)
                  {
                    dof_vec[0 * n_element + iel] =
                      dof_pairs[0][i * n_element + iel];
                  }
              }
            else
              {
                for (unsigned int iel = 0; iel < n_element; ++iel)
                  {
                    dof_vec[1 * n_element + iel] =
                      dof_pairs[0][i * n_element + iel];
                  }
              }
          }

        for (unsigned int ip = 1; ip < dof_pairs.size(); ++ip)
          {
            for (unsigned int i = 0; i < 2; ++i)
              {
                unsigned int idof  = dof_pairs[ip][i * n_element];
                bool         is_in = false;
                for (unsigned int j = 0; j < 2; ++j)
                  {
                    if (idof == dof_pairs[ip - 1][j * n_element])
                      {
                        is_in = true;
                      }
                  }
                if (is_in == false)
                  {
                    for (unsigned int iel = 0; iel < n_element; ++iel)
                      {
                        dof_vec.push_back(dof_pairs[ip][i * n_element + iel]);
                      }
                  }
              }
          }

        for (unsigned int iel = 0; iel < n_element; ++iel)
          {
            non_local_dof_indices[(2 * valence) * n_element + iel] =
              dof_vec[0 * n_element + iel];
            non_local_dof_indices[(2 * valence - 1) * n_element + iel] =
              dof_vec[1 * n_element + iel];
            non_local_dof_indices[(2 * valence - 2) * n_element + iel] =
              dof_vec[2 * n_element + iel];
            non_local_dof_indices[(2 * valence - 3) * n_element + iel] =
              dof_vec[3 * n_element + iel];
            non_local_dof_indices[(2 * valence + 1) * n_element + iel] =
              dof_vec[4 * n_element + iel];
            non_local_dof_indices[(2 * valence + 2) * n_element + iel] =
              dof_vec[5 * n_element + iel];
            non_local_dof_indices[(2 * valence + 3) * n_element + iel] =
              dof_vec[6 * n_element + iel];
          }
        break;
    }
    return non_local_dof_indices;
}

template class CatmullClark<2, 3>;
DEAL_II_NAMESPACE_CLOSE
