//
//  Catmull_Clark_DoFs_Implementation.hpp
//  step-4
//
//  Created by zhaowei Liu on 09/12/2019.
//

#ifndef Catmull_Clark_DoFs_Implementation_hpp
#define Catmull_Clark_DoFs_Implementation_hpp

#include <stdio.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/dof_handler.h>

#include "FE_Catmull_Clark.hpp"

DEAL_II_NAMESPACE_OPEN

hp::FECollection<2, 3>
distribute_catmull_clark_dofs(hp::DoFHandler<2, 3> &dof_handler, const unsigned int fe_dim);

template<int dim, int spacedim>
class CatmullClark{
public:
    CatmullClark(hp::DoFHandler<dim, spacedim> &dh, const unsigned int fe_dim);
    
//    hp::DoFHandler<dim,spacedim>& ref_DoFHandler(){
//        hp::DoFHandler<dim,spacedim>& df = dof_handler;
//        return df;
//    }
    
   void set_FECollection(hp::DoFHandler<dim, spacedim> &dof_handler, const unsigned int fe_dim);
    
    hp::FECollection<dim,spacedim> get_FECollection(){
        return fe_collection;
    }
    
    std::map<unsigned int, unsigned int> dof_to_vert_indices_mapping(){
        return indices_mapping;
    }
    
//    std::vector<std::vector<types::global_dof_index>> new_dofs_for_cells();
    void new_dofs_for_cells(hp::DoFHandler<dim, spacedim> &dof_handler);

private:
    std::vector<std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>> cell_patch_vector;
    
    std::vector<std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator>> cell_patches(hp::DoFHandler<dim, spacedim> &dof_handler);
    
    std::map<unsigned int, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> ordering_cells_in_patch(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell, std::set<typename hp::DoFHandler<dim,spacedim>::active_cell_iterator> cells_in_patch);
    
//    hp::DoFHandler<dim,spacedim> dof_handler();
    
    hp::FECollection<dim,spacedim> fe_collection;
    
    std::map<unsigned int, unsigned int> indices_mapping;
        
    std::vector<unsigned int> get_neighbour_dofs(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_0, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_neighbour);
    
    std::vector<unsigned int> opposite_face_dofs(unsigned int i, unsigned int j);

    unsigned int opposite_vertex_dofs(unsigned int i);
        
    const std::array<unsigned int,4> vertex_face_loop(const unsigned int local_vertex_id);
    
    const std::array<unsigned int,3> next_vertices(const unsigned int local_vertex_id);
    
    const std::array<unsigned int,3> loop_faces(const unsigned int local_face_id);
       
    const std::array<unsigned int,2> opposite_vertices(const unsigned int local_face_id);
    
    const std::array<unsigned int,2> faces_not_on_boundary(const std::vector<unsigned int> m);
    
    const std::array<unsigned int,4> verts_id_on_boundary(const std::vector<unsigned int> m);
    
    unsigned int common_face_local_id(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_0, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell);
    
    std::array<unsigned int,4> rotated_vertices(const unsigned int local_face_id);

    unsigned int get_diagonal_dof_id_to_ex(typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_0, typename hp::DoFHandler<dim,spacedim>::active_cell_iterator cell_neighbour, unsigned int ex_index);

};

DEAL_II_NAMESPACE_CLOSE

#endif /* Catmull_Clark_DoFs_Implementation_hpp */


