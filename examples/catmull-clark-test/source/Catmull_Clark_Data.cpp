//
//  Catmull_Clark_DoFs_Implementation.cpp
//  step-4
//
//  Created by zhaowei Liu on 09/12/2019.
//

#include "Catmull_Clark_Data.hpp"

DEAL_II_NAMESPACE_OPEN

hp::FECollection<2, 3>
distribute_catmull_clark_dofs(hp::DoFHandler<2, 3> &dof_handler, const unsigned int fe_dim)
{
    auto catmull_clark = std::make_shared <CatmullClark<2, 3>>(dof_handler,fe_dim);
    return catmull_clark->get_FECollection();
}

template<int dim, int spacedim>
CatmullClark<dim,spacedim>::CatmullClark(hp::DoFHandler<dim, spacedim> &dof_handler, const unsigned int fe_dim)
{
    cell_patch_vector = cell_patches(dof_handler);
    set_FECollection(dof_handler,fe_dim);
    dof_handler.distribute_dofs(fe_collection);
    new_dofs_for_cells(dof_handler);
}

template<int dim, int spacedim>
void CatmullClark<dim,spacedim>::set_FECollection(hp::DoFHandler<dim, spacedim> &dof_handler, const unsigned int fe_dim){
    std::map<int, int> map_valence_to_fe_indices;
    int                i_fe = 0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
    {
        int valence;
        switch (int ncell_in_patch =
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
        std::map<int, int>::iterator iter_fe_valence;
        iter_fe_valence = map_valence_to_fe_indices.find(valence);
        if (iter_fe_valence != map_valence_to_fe_indices.end())
        {
            cell->set_active_fe_index(iter_fe_valence->second);
        }
        else
        {
            map_valence_to_fe_indices.insert(std::pair<int, int>(valence, i_fe));
            FE_Catmull_Clark<dim, spacedim> fe(valence);
            fe_collection.push_back(FESystem<dim,spacedim>(fe,fe_dim));
            cell->set_active_fe_index(i_fe);
            ++i_fe;
        }
    }
}



template<int dim, int spacedim>
std::vector<
std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>> CatmullClark<dim,spacedim>::cell_patches(hp::DoFHandler<dim, spacedim> &dof_handler)
{
    std::multimap<unsigned int,
    typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>
    vertex_to_cell_map;
    for (typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell = dof_handler.begin_active();
         cell != dof_handler.end();
         ++cell)
        for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
            vertex_to_cell_map.insert({cell->vertex_index(v), cell});
    
    std::vector<
    std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>>
    vector_of_sets;
    
    for (typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell = dof_handler.begin_active();
         cell!=dof_handler.end();
         ++cell)
    {
        std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> cells_in_patch;
        for (unsigned int v=0; v< GeometryInfo<dim>::vertices_per_cell;++v){
            typename std::multimap<unsigned int, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>::iterator map_it;
            for (map_it = vertex_to_cell_map.begin();map_it!= vertex_to_cell_map.end();map_it++){
                if(map_it->first == cell->vertex_index(v)){
                    cells_in_patch.insert(map_it->second);
                }
            }
        }
        vector_of_sets.push_back(cells_in_patch);
    }
    return vector_of_sets;
}



template <int dim, int spacedim>
std::map<unsigned int, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> CatmullClark<dim,spacedim>::ordering_cells_in_patch(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell, std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> cells_in_patch){
    std::map<unsigned int, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> cell_patch_map;
    if(cell->at_boundary()==false){
        if(cells_in_patch.size() == 9){
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
            typename std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>::iterator it_cell;
            for(it_cell = cells_in_patch.begin();it_cell!=cells_in_patch.end();++it_cell){
                typename hp::DoFHandler<dim,spacedim>::active_cell_iterator icell = *it_cell;
                if(icell->active_cell_index() == cell->active_cell_index()){
                    cell_patch_map.insert({0,icell});
                }else{
                    bool neighbour_cell = false;
                    for(unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie){
                        if(int(icell->active_cell_index()) == cell->neighbor_index(ie))
                        {
                            cell_patch_map.insert({ie+1,icell});
                            neighbour_cell = true;
                        }
                    }
                    if (!neighbour_cell){
                        for(unsigned int iu = 0; iu < GeometryInfo<2>::vertices_per_cell; ++iu){
                            for(unsigned int iv = 0; iv < GeometryInfo<2>::vertices_per_cell; ++iv){
                                if(icell->vertex_index(iv) == cell->vertex_index(iu)){
                                    cell_patch_map.insert({5+iu, icell});
                                }
                            }
                        }
                    }
                }
            }
        }else{
            std::vector<int> valence(GeometryInfo<2>::vertices_per_cell,0);
            std::multimap<unsigned int, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> vertex_cells_map;
            for(unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell;++v){
                typename std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>::iterator it_cell;
                for(it_cell = cells_in_patch.begin();it_cell!=cells_in_patch.end();++it_cell){
                    typename hp::DoFHandler<dim,spacedim>::active_cell_iterator icell = *it_cell;
                    for (unsigned int iv = 0; iv < GeometryInfo<2>::vertices_per_cell;++iv){
                        if(icell->vertex_index(iv) == cell->vertex_index(v)){
                            valence[v] +=1;
                            vertex_cells_map.insert({v,icell});
                        }
                    }
                }
            }
            int n_extraordinary = 0;
            std::array<unsigned int,4> face_loop;
            int exv; int val;
            for(unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell;++v){
                if (valence[v] != 4){
                    n_extraordinary +=1;
                    exv = v;
                    face_loop = vertex_face_loop(v);
                    val = valence[v];
                    //                    std::cout << "valence = "<< val <<"\n";
                }
            }
            std::array<unsigned int,3> vertices_regular = next_vertices(exv);
            if(n_extraordinary > 1){
                throw std::runtime_error("can not analyse cell with more than one extraordinary vertices, please refine your triangulation.");
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
            std::map<unsigned int,unsigned int> face_to_neighbour,vertex_to_cell;
            face_to_neighbour.insert({0,1});
            face_to_neighbour.insert({1,val+1});
            face_to_neighbour.insert({2,val+3});
            face_to_neighbour.insert({3,val-1});
            vertex_to_cell.insert({0,val+2});
            vertex_to_cell.insert({1,val});
            vertex_to_cell.insert({2,val+4});
            typename std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>::iterator it_cell;
            for(it_cell = cells_in_patch.begin();it_cell!=cells_in_patch.end();++it_cell){
                typename hp::DoFHandler<dim,spacedim>::active_cell_iterator icell = *it_cell;
                if(icell->active_cell_index() == cell->active_cell_index()){
                    cell_patch_map.insert({0,icell});
                }else{
                    bool neighbour_cell = false;
                    for(auto it_face = face_to_neighbour.begin();it_face!=face_to_neighbour.end();++it_face){
                        if(int(icell->active_cell_index()) == cell->neighbor_index(face_loop[it_face->first])){
                            cell_patch_map.insert({it_face->second, icell});
                            neighbour_cell = true;
                        }
                    }
                    
                    if (!neighbour_cell){
                        for(auto it_vert = vertex_to_cell.begin();it_vert != vertex_to_cell.end();++it_vert){
                            for(unsigned int iv = 0; iv < GeometryInfo<2>::vertices_per_cell;++iv){
                                if(icell->vertex_index(iv) == cell->vertex_index(vertices_regular[it_vert->first])){
                                    cell_patch_map.insert({it_vert->second,icell});
                                }
                            }
                        }
                    }
                }
            }
            auto icell = cell_patch_map.find(1)->second;
            std::vector<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> cells_with_exv;
            for (auto cell_with_exv = vertex_cells_map.begin(); cell_with_exv!=vertex_cells_map.end();++cell_with_exv) {
                if (int(cell_with_exv->first) == exv &&
                    cell_with_exv->second->active_cell_index() != cell_patch_map.find(0)->second->active_cell_index() &&
                    cell_with_exv->second->active_cell_index() != cell_patch_map.find(val-1)->second->active_cell_index()) {
                    cells_with_exv.push_back(cell_with_exv->second);
                }
            }
            for (int n=2; n < val-1; ++n){
                auto this_cell = icell;
                for(auto &cell_with_exv : cells_with_exv){
                    if(cell_with_exv->active_cell_index() != this_cell->active_cell_index()){
                        for(unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie){
                            if(this_cell->neighbor_index(ie) == int(cell_with_exv->active_cell_index()) && cell_with_exv->active_cell_index() != cell_patch_map.find(n-2)->second->active_cell_index()){
                                cell_patch_map.insert({n,cell_with_exv});
                                icell=cell_with_exv;
                            }
                        }
                    }
                }
            }
        }
    }
    else{
        if(cells_in_patch.size() == 6){
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
            std::array<unsigned int,3> local_face_indices;
            std::array<unsigned int,2> local_vertex_indices;
            int n_boundary_cell = 0;
            for(unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie){
                if(cell->at_boundary(ie)){
                    local_face_indices = loop_faces(ie);
                    local_vertex_indices = opposite_vertices(ie);
                    n_boundary_cell += 1;
                }
            }
            Assert(n_boundary_cell == 1, ExcMessage("This is not an element with one edge on physical boundary."));
            typename std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>::iterator it_cell;
            for(it_cell = cells_in_patch.begin();it_cell!=cells_in_patch.end();++it_cell){
                typename hp::DoFHandler<dim,spacedim>::active_cell_iterator icell = *it_cell;
                if(icell->active_cell_index() == cell->active_cell_index()){
                    cell_patch_map.insert({0,icell});
                }else{
                    bool neighbour_cell = false;
                    for(unsigned int inei = 0; inei < local_face_indices.size(); ++inei){
                        if(int(icell->active_cell_index()) == cell->neighbor_index(local_face_indices[inei])){
                            cell_patch_map.insert({1+inei,icell});
                            neighbour_cell = true;
                        }
                    }
                    if(!neighbour_cell){
                        for(unsigned int iu = 0; iu < local_vertex_indices.size();++iu){
                            unsigned int id_v = cell->vertex_index(local_vertex_indices[iu]);
                            for(unsigned int iv = 0; iv < GeometryInfo<2>::vertices_per_cell; ++iv){
                                if(icell->vertex_index(iv) == id_v){
                                    cell_patch_map.insert({4+iu,icell});
                                }
                            }
                        }
                    }
                }
            }
        }
        else{
            if(cells_in_patch.size() == 4){
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
                for(unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie){
                    if(cell->at_boundary(ie)){
                        faces_on_boundary.push_back(ie);
                    }
                }
                Assert(faces_on_boundary.size() == 2, ExcMessage("This is not an element with two edges on physical boundary."));
                
                std::array<unsigned int,2> f_n_b = faces_not_on_boundary(faces_on_boundary);
                
                typename std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>::iterator it_cell;
                for(it_cell = cells_in_patch.begin();it_cell!=cells_in_patch.end();++it_cell){
                    typename hp::DoFHandler<dim,spacedim>::active_cell_iterator icell = *it_cell;
                    if(icell->active_cell_index() == cell->active_cell_index()){
                        cell_patch_map.insert({0,icell});
                    }else{
                        bool neighbour_cell = false;
                        for(unsigned int inei = 0; inei < f_n_b.size(); ++inei){
                            if(int(icell->active_cell_index()) == cell->neighbor_index(f_n_b[inei])){
                                cell_patch_map.insert({1+inei,icell});
                                neighbour_cell = true;
                            }
                        }
                        if(!neighbour_cell){
                            cell_patch_map.insert({3,icell});
                        }
                    }
                }
            }
            else{
                throw std::runtime_error("current code can not deal with boundary extraordinary vertex.");
            }
        }
    }
    return cell_patch_map;
}



template<int dim, int spacedim>
//std::vector<std::vector<types::global_dof_index>>
void CatmullClark<dim,spacedim>::new_dofs_for_cells(hp::DoFHandler<dim, spacedim> &dof_handler){
    std::vector<std::vector<unsigned int>> dof_indices_order_vector(dof_handler.get_triangulation().n_active_cells());
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        std::map<unsigned int, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> cells = ordering_cells_in_patch(cell, cell_patch_vector[cell->active_cell_index()]);
        unsigned int valence;
//        std::vector<types::global_dof_index> global_dof_indices(cell->get_fe().dofs_per_cell,0);
        std::vector<types::global_dof_index> cell_dof_indices(cell->get_fe().dofs_per_cell,0);
        std::vector<types::global_dof_index> non_local_dof_indices(cell->get_fe().non_local_dofs_per_cell,0);
        std::vector<unsigned int> reorder_indices(cell->get_fe().dofs_per_cell,0);
        cell-> get_dof_indices(cell_dof_indices);
        for (unsigned int i = 0; i < 4; ++i) {
            indices_mapping.insert({cell_dof_indices[i],cell->vertex_index(i)});
        }
        
        switch (cells.size()){
            case 4:
            {valence = 1;
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
                for(unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie){
                    if(cells[0]->at_boundary(ie)){
                        edges_on_boundary.push_back(ie);
                    }
                }
                auto verts_id = verts_id_on_boundary(edges_on_boundary);
//                global_dof_indices[0] = cell_dof_indices[verts_id[0]];
//                global_dof_indices[1] = cell_dof_indices[verts_id[1]];
//                global_dof_indices[4] = cell_dof_indices[verts_id[2]];
//                global_dof_indices[3] = cell_dof_indices[verts_id[3]];
                
                reorder_indices[0] = verts_id[0];
                reorder_indices[1] = verts_id[1];
                reorder_indices[4] = verts_id[2];
                reorder_indices[3] = verts_id[3];
                reorder_indices[2] = 8;
                reorder_indices[5] = 7;
                reorder_indices[8] = 6;
                reorder_indices[7] = 5;
                reorder_indices[6] = 4;
                
                int face_local_id;
                face_local_id = common_face_local_id(cells[0], cells[1]);
                auto rotated_iv_1 = rotated_vertices(face_local_id);
                cell_dof_indices.resize(cells[1]->get_fe().dofs_per_cell);
                cells[1] -> get_dof_indices(cell_dof_indices);
//                global_dof_indices[2] = cell_dof_indices[rotated_iv_1[2]];
//                global_dof_indices[5] = cell_dof_indices[rotated_iv_1[3]];
                non_local_dof_indices[4] = cell_dof_indices[rotated_iv_1[2]];
                non_local_dof_indices[3] = cell_dof_indices[rotated_iv_1[3]];
                
                face_local_id = common_face_local_id(cells[0], cells[2]);
                auto rotated_iv_2 = rotated_vertices(face_local_id);
                cell_dof_indices.resize(cells[2]->get_fe().dofs_per_cell);
                cells[2] -> get_dof_indices(cell_dof_indices);
//                global_dof_indices[7] = cell_dof_indices[rotated_iv_2[2]];
//                global_dof_indices[6] = cell_dof_indices[rotated_iv_2[3]];
                non_local_dof_indices[1] = cell_dof_indices[rotated_iv_2[2]];
                non_local_dof_indices[0] = cell_dof_indices[rotated_iv_2[3]];
                
//                global_dof_indices[8] = get_neighbour_dofs(cells[0],cells[3])[0];
                non_local_dof_indices[2] = get_neighbour_dofs(cells[0],cells[3])[0];

                break;
            }
            case 6:{
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
                for(unsigned int ie = 0; ie < GeometryInfo<2>::faces_per_cell; ++ie){
                    if(cells[0]->at_boundary(ie)){
                        auto rotated_iv = rotated_vertices(ie);
//                        global_dof_indices[1] = cell_dof_indices[rotated_iv[0]];
//                        global_dof_indices[2] = cell_dof_indices[rotated_iv[1]];
//                        global_dof_indices[6] = cell_dof_indices[rotated_iv[2]];
//                        global_dof_indices[5] = cell_dof_indices[rotated_iv[3]];
                        reorder_indices[1] = rotated_iv[0];
                        reorder_indices[2] = rotated_iv[1];
                        reorder_indices[6] = rotated_iv[2];
                        reorder_indices[5] = rotated_iv[3];
                    }
                }
                reorder_indices[0] = 9;
                reorder_indices[3] = 8;
                reorder_indices[4] = 10;
                reorder_indices[7] = 7;
                reorder_indices[8] = 11;
                reorder_indices[9] = 4;
                reorder_indices[10] = 5;
                reorder_indices[11] = 6;
                
                int face_local_id;
                face_local_id = common_face_local_id(cells[0], cells[1]);
                auto rotated_iv_1 = rotated_vertices(face_local_id);
                cell_dof_indices.resize(cells[1]->get_fe().dofs_per_cell);
                cells[1] -> get_dof_indices(cell_dof_indices);
//                global_dof_indices[3] = cell_dof_indices[rotated_iv_1[2]];
//                global_dof_indices[7] = cell_dof_indices[rotated_iv_1[3]];
                non_local_dof_indices[4] = cell_dof_indices[rotated_iv_1[2]];
                non_local_dof_indices[3] = cell_dof_indices[rotated_iv_1[3]];
                
                face_local_id = common_face_local_id(cells[0], cells[2]);
                auto rotated_iv_2 = rotated_vertices(face_local_id);
                cell_dof_indices.resize(cells[2]->get_fe().dofs_per_cell);
                cells[2] -> get_dof_indices(cell_dof_indices);
//                global_dof_indices[10] = cell_dof_indices[rotated_iv_2[2]];
//                global_dof_indices[9] = cell_dof_indices[rotated_iv_2[3]];
                non_local_dof_indices[1] = cell_dof_indices[rotated_iv_2[2]];
                non_local_dof_indices[0] = cell_dof_indices[rotated_iv_2[3]];
                
                face_local_id = common_face_local_id(cells[0], cells[3]);
                auto rotated_iv_3 = rotated_vertices(face_local_id);
                cell_dof_indices.resize(cells[3]->get_fe().dofs_per_cell);
                cells[3] -> get_dof_indices(cell_dof_indices);
//                global_dof_indices[4] = cell_dof_indices[rotated_iv_3[2]];
//                global_dof_indices[0] = cell_dof_indices[rotated_iv_3[3]];
                non_local_dof_indices[6] = cell_dof_indices[rotated_iv_3[2]];
                non_local_dof_indices[5] = cell_dof_indices[rotated_iv_3[3]];
                
//                global_dof_indices[11] = get_neighbour_dofs(cells[0],cells[4])[0];
//                global_dof_indices[8] = get_neighbour_dofs(cells[0],cells[5])[0];
                non_local_dof_indices[2] = get_neighbour_dofs(cells[0],cells[4])[0];
                non_local_dof_indices[7] = get_neighbour_dofs(cells[0],cells[5])[0];
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
//                global_dof_indices[9] = cell_dof_indices[0];
//                global_dof_indices[10] = cell_dof_indices[1];
//                global_dof_indices[5] = cell_dof_indices[2];
//                global_dof_indices[6] = cell_dof_indices[3];
                std::vector<unsigned int> temp_indices;
                temp_indices = get_neighbour_dofs(cells[0],cells[1]);
//                global_dof_indices[8] = temp_indices[0];
//                global_dof_indices[4] = temp_indices[1];
                non_local_dof_indices[10] = temp_indices[0];
                non_local_dof_indices[9] = temp_indices[1];
                temp_indices = get_neighbour_dofs(cells[0],cells[2]);
//                global_dof_indices[11] = temp_indices[0];
//                global_dof_indices[7] = temp_indices[1];
                non_local_dof_indices[3] = temp_indices[0];
                non_local_dof_indices[4] = temp_indices[1];
                temp_indices = get_neighbour_dofs(cells[0],cells[3]);
//                global_dof_indices[13] = temp_indices[0];
//                global_dof_indices[14] = temp_indices[1];
                non_local_dof_indices[0] = temp_indices[0];
                non_local_dof_indices[1] = temp_indices[1];
                temp_indices = get_neighbour_dofs(cells[0],cells[4]);
//                global_dof_indices[1] = temp_indices[0];
//                global_dof_indices[2] = temp_indices[1];
                non_local_dof_indices[7] = temp_indices[0];
                non_local_dof_indices[6] = temp_indices[1];
                
//                global_dof_indices[12] = get_neighbour_dofs(cells[0],cells[5])[0];
//                global_dof_indices[15] = get_neighbour_dofs(cells[0],cells[6])[0];
//                global_dof_indices[0] = get_neighbour_dofs(cells[0],cells[7])[0];
//                global_dof_indices[3] = get_neighbour_dofs(cells[0],cells[8])[0];
                non_local_dof_indices[11] = get_neighbour_dofs(cells[0],cells[5])[0];
                non_local_dof_indices[2] = get_neighbour_dofs(cells[0],cells[6])[0];
                non_local_dof_indices[8] = get_neighbour_dofs(cells[0],cells[7])[0];
                non_local_dof_indices[5] = get_neighbour_dofs(cells[0],cells[8])[0];
                reorder_indices = {12,11,10,9,13,2,3,8,14,0,1,7,15,4,5,6};
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
                for (unsigned int i = 0; i<4; ++i){
                    unsigned int n = 0;
                    for (unsigned int icell = 0 ; icell < 2; ++icell ){
                        std::vector<types::global_dof_index> icell_dof_indices(4,0);
                        icell_dof_indices.resize(cells[icell+1]->get_fe().dofs_per_cell);
                        cells[icell+1] -> get_dof_indices(icell_dof_indices);
                        for(unsigned int j = 0; j<4; ++j){
                            if(cell_dof_indices[i] == icell_dof_indices[j]){
                                n += 1;
                            }
                        }
                    }
                    if(n == 2){
                        ex_vertex_index = i;
                    }
                }
//                global_dof_indices[0] = cell_dof_indices[ex_vertex_index];
//                global_dof_indices[5] = cell_dof_indices[next_vertices(ex_vertex_index)[0]];
//                global_dof_indices[4] = cell_dof_indices[next_vertices(ex_vertex_index)[1]];
//                global_dof_indices[3] = cell_dof_indices[next_vertices(ex_vertex_index)[2]];
                
                
                for (unsigned int ic = 1; ic < valence - 1; ++ic){
                    std::vector<types::global_dof_index> cell_dof_indices(4,0);
                    cell_dof_indices.resize(cells[ic]->get_fe().dofs_per_cell);
                    cells[ic] -> get_dof_indices(cell_dof_indices);
                    int dia_dof_id = get_diagonal_dof_id_to_ex(cells[0], cells[ic], ex_vertex_index);
//                    global_dof_indices[4+2*ic] = cell_dof_indices[dia_dof_id];
                    non_local_dof_indices[2*ic] = cell_dof_indices[dia_dof_id];

                    auto v = get_neighbour_dofs(cells[ic-1], cells[ic]);
                    int on_face_dof;
                    
                    if (v[0] == cell_dof_indices[dia_dof_id]){
                        on_face_dof =v[1];
                    }else{
                        on_face_dof = v[0];
                    }
//                    global_dof_indices[5+2*ic] = on_face_dof;
                    non_local_dof_indices[1+2*ic] = on_face_dof;

                }
                cell_dof_indices.resize(cells[valence-1]->get_fe().dofs_per_cell);
                cells[valence-1] -> get_dof_indices(cell_dof_indices);
                
                
                int dia_dof_id = get_diagonal_dof_id_to_ex(cells[0], cells[valence-1], ex_vertex_index);
//                global_dof_indices[2] = cell_dof_indices[dia_dof_id];
                non_local_dof_indices[1] = cell_dof_indices[dia_dof_id];

                auto v = get_neighbour_dofs(cells[0], cells[valence-1]);
                int on_face_dof;
                
                if (v[0] == cell_dof_indices[dia_dof_id]){
                    on_face_dof =v[1];
                }else{
                    on_face_dof = v[0];
                }
//                global_dof_indices[1] = on_face_dof;
                non_local_dof_indices[0] = on_face_dof;

                std::vector<std::vector<unsigned int>> dof_pairs(6);
                dof_pairs[0] = get_neighbour_dofs(cells[1], cells[valence+2]);
                dof_pairs[1] = get_neighbour_dofs(cells[0], cells[valence+1]);
                dof_pairs[2] = get_neighbour_dofs(cells[valence+3], cells[valence]);
                dof_pairs[3] = get_neighbour_dofs(cells[valence+1], cells[valence]);
                dof_pairs[4] = get_neighbour_dofs(cells[0], cells[valence+3]);
                dof_pairs[5] = get_neighbour_dofs(cells[valence-1], cells[valence+4]);
                
                std::vector<unsigned int> dof_vec(2,0);
                
                for(unsigned int i = 0; i<2; ++i){
                    unsigned int idof = dof_pairs[0][i];
                    bool is_in = false;
                    for(unsigned int j = 0; j < 2; ++j){
                        if (idof == dof_pairs[1][j]){
                            is_in = true;
                        }
                    }
                    if (is_in == false){
                        dof_vec[0] = idof;
                    }else{
                        dof_vec[1] = idof;
                    }
                }
                
                for(unsigned int ip = 1 ; ip < dof_pairs.size(); ++ip){
                    for(unsigned int i = 0; i<2; ++i){
                        unsigned int idof = dof_pairs[ip][i];
                        bool is_in = false;
                        for (unsigned int j = 0; j < 2 ; ++j){
                            if (idof == dof_pairs[ip-1][j]){
                                is_in = true;
                            }
                        }
                        if (is_in == false){
                            dof_vec.push_back(idof);
                        }
                    }
                }
                
//                global_dof_indices[2*valence+4] = dof_vec[0];
//                global_dof_indices[2*valence+3] = dof_vec[1];
//                global_dof_indices[2*valence+2] = dof_vec[2];
//                global_dof_indices[2*valence+1] = dof_vec[3];
//                global_dof_indices[2*valence+5] = dof_vec[4];
//                global_dof_indices[2*valence+6] = dof_vec[5];
//                global_dof_indices[2*valence+7] = dof_vec[6];
                non_local_dof_indices[2*valence] = dof_vec[0];
                non_local_dof_indices[2*valence-1] = dof_vec[1];
                non_local_dof_indices[2*valence-2] = dof_vec[2];
                non_local_dof_indices[2*valence-3] = dof_vec[3];
                non_local_dof_indices[2*valence+1] = dof_vec[4];
                non_local_dof_indices[2*valence+2] = dof_vec[5];
                non_local_dof_indices[2*valence+3] = dof_vec[6];
                
                std::vector<types::global_dof_index> new_non_local_dof_indices(2*valence+4,0);
//                new_global_dof_indices[0] = global_dof_indices[0];
                if(valence == 3){
//                    new_global_dof_indices[1] = global_dof_indices[1];
                    new_non_local_dof_indices[0] = non_local_dof_indices[0];
                }
                else{
//                    new_global_dof_indices[1] = global_dof_indices[7];
//                    new_global_dof_indices[7] = global_dof_indices[1];
                    new_non_local_dof_indices[0] = non_local_dof_indices[3];
                    new_non_local_dof_indices[3] = non_local_dof_indices[0];
                    for(unsigned int i = 0; i < 2*valence-7; ++i){
//                        new_global_dof_indices[i+8] = global_dof_indices[2*valence-i];
                        new_non_local_dof_indices[i+4] = non_local_dof_indices[2*valence-4-i];
                    }
                }
//                new_global_dof_indices[2] = global_dof_indices[6];
//                new_global_dof_indices[3] = global_dof_indices[5];
//                new_global_dof_indices[4] = global_dof_indices[4];
//                new_global_dof_indices[5] = global_dof_indices[3];
//                new_global_dof_indices[6] = global_dof_indices[2];
//                new_global_dof_indices[2*valence+4] = global_dof_indices[2*valence+7];
//                new_global_dof_indices[2*valence+3] = global_dof_indices[2*valence+6];
//                new_global_dof_indices[2*valence+2] = global_dof_indices[2*valence+5];
//                new_global_dof_indices[2*valence+1] = global_dof_indices[2*valence+1];
//                new_global_dof_indices[2*valence+5] = global_dof_indices[2*valence+2];
//                new_global_dof_indices[2*valence+6] = global_dof_indices[2*valence+3];
//                new_global_dof_indices[2*valence+7] = global_dof_indices[2*valence+4];
//                global_dof_indices = new_global_dof_indices;
                new_non_local_dof_indices[1] = non_local_dof_indices[2];
                new_non_local_dof_indices[2] = non_local_dof_indices[1];
                new_non_local_dof_indices[2*valence] = non_local_dof_indices[2*valence+3];
                new_non_local_dof_indices[2*valence-1] = non_local_dof_indices[2*valence+2];
                new_non_local_dof_indices[2*valence-2] = non_local_dof_indices[2*valence+1];
                new_non_local_dof_indices[2*valence-3] = non_local_dof_indices[2*valence-3];
                new_non_local_dof_indices[2*valence+1] = non_local_dof_indices[2*valence-2];
                new_non_local_dof_indices[2*valence+2] = non_local_dof_indices[2*valence-1];
                new_non_local_dof_indices[2*valence+3] = non_local_dof_indices[2*valence];
                non_local_dof_indices = new_non_local_dof_indices;
                
                reorder_indices[0] = ex_vertex_index;
                reorder_indices[3] = next_vertices(ex_vertex_index)[0];
                reorder_indices[4] = next_vertices(ex_vertex_index)[1];
                reorder_indices[5] = next_vertices(ex_vertex_index)[2];
                reorder_indices[1] = 4;
                reorder_indices[2] = 5;
                for (unsigned int id = 6; id < 2 *valence +8 ; ++id) {
                    reorder_indices[id] = id;
                }
                break;
        }
//        global_dof_vector_vector[cell->active_cell_index()]=global_dof_indices;
        cell->set_non_local_dof_indices(non_local_dof_indices);
        dof_indices_order_vector[cell->active_cell_index()] = reorder_indices;
    }
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
        cell->rearrange_dof_indices(dof_indices_order_vector[cell->active_cell_index()]);
    }
//    return global_dof_vector_vector;
}



template <int dim, int spacedim>
std::vector<unsigned int> CatmullClark<dim,spacedim>::get_neighbour_dofs(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_0, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_neighbour){
    // left to right, up to down
    std::vector<types::global_dof_index> cell_0_indices(cell_0->get_fe().dofs_per_cell,0);
    std::vector<types::global_dof_index> cell_neighbour_indices(cell_neighbour->get_fe().dofs_per_cell,0);
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
    //find how many common DoFs the two cells share
    int n_common = 0; std::vector<int> ith_dof(0);
    for(int i = 0; i<4; ++i){
        for (int j = 0; j < 4; ++j) {
            if (cell_neighbour_indices[i] == cell_0_indices[j]){
                n_common += 1;
                ith_dof.push_back(i);
            }
        }
    }
    if (n_common == 2){
        std::vector<unsigned int> dof_i = opposite_face_dofs(ith_dof[0], ith_dof[1]);
        return {static_cast<unsigned int>(static_cast<int>(cell_neighbour_indices[dof_i[0]])),static_cast<unsigned int>(static_cast<int>(cell_neighbour_indices[dof_i[1]]))};
    }else if(n_common == 1){
        int dof_i = opposite_vertex_dofs(ith_dof[0]);
        return {static_cast<unsigned int>(static_cast<int>(cell_neighbour_indices[dof_i]))};
    }else{
        throw std::runtime_error("The two cells have no common DoFs.");
    }
}



template<int dim, int spacedim>
const std::array<unsigned int,4> CatmullClark<dim,spacedim>::vertex_face_loop(const unsigned int local_vertex_id){
    switch(local_vertex_id)
    {
        case 0:
            return {2,1,3,0};
        case 1:
            return {1,3,0,2};
        case 2:
            return {0,2,1,3};
        case 3:
            return {3,0,2,1};
        default:
            throw std::runtime_error("vertex_id_not_valid.");
            break;
    }
}



template<int dim, int spacedim>
unsigned int CatmullClark<dim,spacedim>::opposite_vertex_dofs(unsigned int i){
    switch(i){
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
std::vector<unsigned int> CatmullClark<dim,spacedim>::opposite_face_dofs(unsigned int i, unsigned int j){
    
    switch(i)
    {
        case 0:
            switch(j){
                case 1:
                    return {2,3};
                case 2:
                    return {1,3};
            }
        case 1:
            return {0,2};
        case 2:
            return {0,1};
            
        default:
            throw std::runtime_error("vertex_id_not_valid.");
            break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int,3> CatmullClark<dim,spacedim>::next_vertices(const unsigned int local_vertex_id) {
    switch(local_vertex_id)
    {
        case 0:
            return {1,3,2};
        case 1:
            return {3,2,0};
        case 2:
            return {0,1,3};
        case 3:
            return {2,0,1};
        default:
            throw std::runtime_error("vertex_id_not_valid.");
            break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int,3> CatmullClark<dim,spacedim>::loop_faces(const unsigned int local_face_id){
    switch(local_face_id)
    {
        case 0:
            return {3,1,2};
        case 1:
            return {2,0,3};
        case 2:
            return {0,3,1};
        case 3:
            return {1,2,0};
        default:
            throw std::runtime_error("face_id_not_valid.");
            break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int,2> CatmullClark<dim,spacedim>::opposite_vertices(const unsigned int local_face_id){
    switch(local_face_id)
    {
        case 0:
            return {3,1};
        case 1:
            return {0,2};
        case 2:
            return {2,3};
        case 3:
            return {1,0};
        default:
            throw std::runtime_error("face_id_not_valid.");
            break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int,2> CatmullClark<dim,spacedim>::faces_not_on_boundary(const std::vector<unsigned int> m){
    /*
     *       2
     *    0-----1
     *    |     |
     *  0 |     | 1
     *    |     |
     *    2-----3
     *       3
     */
    
    if (m.size() != 2){
        throw std::runtime_error("m must has two entries");
    }
    
    switch(m[0])
    {
        case 0:
            switch(m[1]){
                case 3:
                    return {1,2};
                case 2:
                    return {3,1};
            }
        case 1:
            switch(m[1]){
                case 3:
                    return {2,0};
                case 2:
                    return {0,3};
            }
            
        default:
            throw std::runtime_error("faces_id_not_valid.");
            break;
    }
}



template <int dim, int spacedim>
const std::array<unsigned int,4> CatmullClark<dim,spacedim>::verts_id_on_boundary(const std::vector<unsigned int> m){
    /*
     *       2
     *    0-----1
     *    |     |
     *  0 |     | 1
     *    |     |
     *    2-----3
     *       3
     */
    
    if (m.size() != 2){
        throw std::runtime_error("m must has two entries");
    }
    
    switch(m[0])
    {
        case 0:
            switch(m[1]){
                case 3:
                    return {2,3,1,0};
                case 2:
                    return {0,2,3,1};
            }
        case 1:
            switch(m[1]){
                case 3:
                    return {3,1,0,2};
                case 2:
                    return {1,0,2,3};
            }
            
        default:
            throw std::runtime_error("faces_id_not_valid.");
            break;
    }
}



template <int dim, int spacedim>
unsigned int CatmullClark<dim,spacedim>::common_face_local_id(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_0, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell){
    unsigned int face_id;
    for(unsigned i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i){
        for(unsigned j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j){
            if (cell_0->face_index(i) == cell->face_index(j)) {
                face_id = j;
            }
        }
    }
    return face_id;
}



template <int dim, int spacedim>
std::array<unsigned int,4> CatmullClark<dim,spacedim>::rotated_vertices(const unsigned int local_face_id){
    switch(local_face_id)
    {
        case 0:
            return {0,2,3,1};
        case 1:
            return {3,1,0,2};
        case 2:
            return {1,0,2,3};
        case 3:
            return {2,3,1,0};
        default:
            throw std::runtime_error("face_id_not_valid.");
            break;
    }
}



template <int dim, int spacedim>
unsigned int CatmullClark<dim,spacedim>::get_diagonal_dof_id_to_ex(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_0, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_neighbour, unsigned int ex_index){
    std::vector<types::global_dof_index> cell_0_indices(cell_0->get_fe().dofs_per_cell,0);
    std::vector<types::global_dof_index> cell_neighbour_indices(cell_neighbour->get_fe().dofs_per_cell,0);
    cell_0->get_dof_indices(cell_0_indices);
    cell_neighbour->get_dof_indices(cell_neighbour_indices);
    types::global_dof_index global_ex_index = cell_0_indices[ex_index];
    unsigned int index;
    for(unsigned int i = 0; i<4; ++i){
        if (cell_neighbour_indices[i] == global_ex_index){
            index = opposite_vertex_dofs(i);
        }
    }
    return index;
}



template class CatmullClark<2,3>;
DEAL_II_NAMESPACE_CLOSE
