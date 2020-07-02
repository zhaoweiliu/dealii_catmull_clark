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
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>

#include "Catmull_Clark_Data.hpp"
#include "polynomials_Catmull_Clark.hpp"
#include "FE_Catmull_Clark.hpp"
#include "MappingFEField_hp.hpp"



// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;



template <int spacedim>
 class RightHandSide : public Function<spacedim>
 {
 public:
   virtual double value(const Point<spacedim> & p,
                        const unsigned int component) const override;
 };
 template <int spacedim>
 double RightHandSide<spacedim>::value(const Point<spacedim> &p,
                                  const unsigned int /*component*/) const
 {
   double product = 1;
//   for (unsigned int d = 0; d < spacedim; ++d)
//     product *= (p[d] + 1);
   return product;
 }




int main()
{
    const int dim = 2, spacedim = 3;
    
    Triangulation<dim,spacedim> mesh;
    static SphericalManifold<dim,spacedim> surface_description;
    {
        Triangulation<spacedim> volume_mesh;
        GridGenerator::half_hyper_ball(volume_mesh);
        std::set<types::boundary_id> boundary_ids;
        boundary_ids.insert (0);
        GridGenerator::extract_boundary_mesh (volume_mesh, mesh,
                                              boundary_ids);
    }
    
    mesh.set_all_manifold_ids(0);
    mesh.set_manifold (0, surface_description);
    mesh.refine_global(2);
    
    std::ofstream gout0("half_sphere.vtu");
    std::ofstream gout1("half_sphere.msh");
    
    GridOut gird_out;
    gird_out.write_vtu(mesh,gout0);
    gird_out.write_msh(mesh,gout1);
    
    hp::DoFHandler<dim,spacedim> dof_handler(mesh);
    hp::FECollection<dim,spacedim> fe_collection;
    hp::MappingCollection<dim,spacedim> mapping_collection;
    hp::QCollection<dim> q_collection;
    Vector<double> vec_values;
    
    catmull_clark_create_fe_quadrature_and_mapping_collections_and_distribute_dofs(dof_handler,fe_collection,vec_values,mapping_collection,q_collection,3);
    
    AffineConstraints<double> constraints;
//    constraints.clear();
//    constraints.close();
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
    
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    std::ofstream out("CC_sparsity_pattern.svg");
    sparsity_pattern.print_svg(out);
    
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_gradients|update_JxW_values);
    
    RightHandSide<spacedim> rhs_function;

    FullMatrix<double> cell_matrix;
    Vector<double>     cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    
    double area = 0;
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;
        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
        {
            area += fe_values.JxW(q_point);
            rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) +=
                    (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                     fe_values.shape_value(j, q_point) * // phi_j(x_q)
                     fe_values.JxW(q_point));           // dx
                }
                cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                                1 *               // f(x_q)
                                fe_values.JxW(q_point));            // dx
            }
        }
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        system_rhs.add(local_dof_indices, cell_rhs);
        system_matrix.add(local_dof_indices, cell_matrix);
    }
    
    std::cout << " area = " << area << std::endl;
    
    
    
    SolverControl solver_control(system_rhs.size(),
                                1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    
    std::cout << solution << std::endl;
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
       for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
            ++q_point){
           double q_sol = 0;
           for (unsigned int i = 0; i < dofs_per_cell; ++i)
           {
               q_sol += solution[local_dof_indices[i]] * fe_values.shape_value(i, q_point);
           }
           std::cout << "quadrature point "<< q_point << " value = "<<q_sol<<std::endl;
       }
    }
    
    DataOut<dim, hp::DoFHandler<dim,spacedim>> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    const std::string filename =
      "solution-CC.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    
    return 0;
}
