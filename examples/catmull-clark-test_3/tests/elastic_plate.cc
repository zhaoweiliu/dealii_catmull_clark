/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_reordering.h>
 
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/arpack_solver.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/constrained_linear_operator.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <complex>

#include "Catmull_Clark_Data.hpp"
#include "polynomials_Catmull_Clark.hpp"
#include "FE_Catmull_Clark.hpp"
#include "MappingFEField_hp.hpp"

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkQuad.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkXMLUnstructuredGridWriter.h>

using namespace dealii;

template <int dim, int spacedim>
Triangulation<dim,spacedim> set_mesh( std::string type )
{
    Triangulation<dim,spacedim> mesh;
    if (type == "plate")
    {GridGenerator::hyper_cube(mesh);
        mesh.refine_global(4);}
    return mesh;
}



template <int dim, int spacedim>
class Elastic_plate
{
public:
    Elastic_plate(Triangulation<dim,spacedim> &tria);
    ~Elastic_plate();
    void run();
private:
    void setup_system();
    void assemble_system();
    void assemble_boundary_mass_matrix_and_rhs();
    void solve();
    Tensor<4,dim> get_elastic_tensor();
    
    hp::DoFHandler<dim,spacedim> dof_handler;
    hp::FECollection<dim,spacedim> fe_collection;
    hp::MappingCollection<dim,spacedim> mapping_collection;
    hp::QCollection<dim> q_collection;
    hp::QCollection<dim> boundary_q_collection;
    SparsityPattern      sparsity_pattern;
    AffineConstraints<double> constraints;
    SparseMatrix<double> tangent_matrix;
    SparseMatrix<double> boundary_mass_matrix;
    Vector<double> solution;
    Vector<double> vec_values;
    std::vector<types::global_dof_index> constrained_dof_indices;
    const double youngs = 2.1e11,possion = 0.33, thickness = 0.05;
};



template <int dim, int spacedim>
Elastic_plate<dim, spacedim>::Elastic_plate(Triangulation<dim,spacedim> &tria)
:
dof_handler(tria)
{}



template <int dim, int spacedim>
Elastic_plate<dim, spacedim>::~Elastic_plate()
{
    dof_handler.clear();
}



template<int dim, int spacedim>
Tensor<4,dim> Elastic_plate<dim, spacedim>::get_elastic_tensor()
{
    Tensor<2,dim> kronecker;
    kronecker[0][0]  = 1.;
    kronecker[1][1]  = 1.;
    kronecker[0][1]  = 0.;
    kronecker[1][0]  = 0.;

    Tensor<4,dim> elastic;
    for (unsigned int a = 0; a < dim; ++a) {
        for (unsigned int b = 0; b < dim; ++b) {
            for (unsigned int c = 0; c < dim; ++c) {
                for (unsigned int d = 0; d < dim; ++d) {
                    elastic[a][b][c][d] = ( youngs / (1. - possion * possion) ) * (possion * kronecker[a][b] * kronecker[c][d] + 0.5 *(1.-possion) * ( kronecker[a][c] *  kronecker[b][d] +  kronecker[a][d] * kronecker[b][c]));
                }
            }
        }
    }
    return elastic;
}



template <int dim, int spacedim>
void Elastic_plate<dim, spacedim> :: setup_system()
{
    catmull_clark_create_fe_quadrature_and_mapping_collections_and_distribute_dofs(dof_handler,fe_collection,vec_values,mapping_collection,q_collection,boundary_q_collection,3);
    std::cout << "   Number of dofs: " << dof_handler.n_dofs()
    << std::endl;
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
//    constraints.clear();
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern, constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    std::ofstream out("CC_sparsity_pattern.svg");
    sparsity_pattern.print_svg(out);
    tangent_matrix.reinit(sparsity_pattern);
    boundary_mass_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
}



template <int dim, int spacedim>
void Elastic_plate<dim, spacedim> :: assemble_system()
{
    auto elastic_tensor = get_elastic_tensor();
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    FullMatrix<double> cell_tangent_matrix;
    Vector<double>     cell_external_force_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        
        cell_tangent_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_tangent_matrix = 0;
        cell_external_force_rhs.reinit(dofs_per_cell);
        cell_external_force_rhs = 0;
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
        {
            for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
                Tensor<1, spacedim> i_shape_grad = fe_values.shape_grad(i_shape, q_point);
                Tensor<2, spacedim> i_shape_hessian = fe_values.shape_hessian(i_shape, q_point);
                for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                    Tensor<1, spacedim> j_shape_grad = fe_values.shape_grad(j_shape, q_point);
                    Tensor<2, spacedim> j_shape_hessian = fe_values.shape_hessian(j_shape, q_point);
//                    cell_tangent_matrix[local_dof_indices[i_shape]][local_dof_indices[j_shape]] =
                }
            }
        }
    }
}


int main()
{
    const int dim = 2, spacedim = 3;
    Triangulation<dim,spacedim> mesh = set_mesh<dim,spacedim>("elastic_plate");
    Elastic_plate<dim, spacedim> Elastic_plate(mesh);
//    nonlinear_thin_shell.run();
    std::cout <<"finished.\n";
    
    return 0;
}
