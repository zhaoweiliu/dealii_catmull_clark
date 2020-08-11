/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
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
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>

#include "Catmull_Clark_Data.hpp"
#include "polynomials_Catmull_Clark.hpp"
#include "FE_Catmull_Clark.cpp"


using namespace dealii;
template <int dim, int spacedim>
class Step4
{
public:
    Step4();
    void run();
private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void assemble_boundary_system();
    void solve();
    void output_results() const;
    Triangulation<dim, spacedim>             triangulation;
    hp::FECollection<dim, spacedim>          fe;
    hp::DoFHandler<dim, spacedim>            dof_handler;
    SparsityPattern                          sparsity_pattern;
    SparseMatrix<double>                     system_matrix;
    SparseMatrix<double>                     boundary_mass_matrix;
    Vector<double>                           solution;
    Vector<double>                           system_rhs;
    Vector<double>                           boundary_value_rhs;
    double                                   penalty_factor = 1e8;
    std::shared_ptr<CatmullClark<2, 3>>      catmull_clark =
    std::make_shared<CatmullClark<2, 3>>();
};
template <int spacedim>
class RightHandSide : public Function<spacedim>
{
public:
    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;
};
template <int spacedim>
class BoundaryValues : public Function<spacedim>
{
public:
    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;
};
template <int spacedim>
double RightHandSide<spacedim>::value(const Point<spacedim> &p,
                                      const unsigned int /*component*/) const
{
    double return_value = 0.0;
    for (unsigned int i = 0; i < spacedim; ++i)
        return_value += 4.0 * std::pow(p(i), 4.0);
    return return_value;
}
template <int spacedim>
double BoundaryValues<spacedim>::value(const Point<spacedim> &p,
                                       const unsigned int /*component*/) const
{
    return p.square();
}
template <int dim, int spacedim>
Step4<dim, spacedim>::Step4()
: dof_handler(triangulation)
{
    make_grid();
    catmull_clark->set_hp_objects(dof_handler);
    fe = catmull_clark->get_FECollection();
}
template <int dim, int spacedim>
void Step4<dim, spacedim>::make_grid()
{
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(4);
    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
    << std::endl
    << "   Total number of cells: " << triangulation.n_cells()
    << std::endl;
}
template <int dim, int spacedim>
void Step4<dim, spacedim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
    << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    boundary_mass_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    boundary_value_rhs.reinit(dof_handler.n_dofs());
}
template <int dim, int spacedim>
void Step4<dim, spacedim>::assemble_system()
{
    hp::QCollection<dim> quadrature_formula = catmull_clark->get_QCollection();
    RightHandSide<spacedim> right_hand_side;
    hp::FEValues<dim,spacedim> hp_fe_values(fe,
                                            quadrature_formula,
                                            update_values | update_gradients |update_inverse_jacobians |
                                            update_quadrature_points | update_JxW_values);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        
        hp_fe_values.reinit(cell);
        const FEValues<dim, spacedim> &fe_values =
        hp_fe_values.get_present_fe_values();
        cell_matrix = 0;
        cell_rhs    = 0;
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));           // dx
                const auto x_q = fe_values.quadrature_point(q_index);
                cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                right_hand_side.value(x_q) *        // f(x_q)
                                fe_values.JxW(q_index));            // dx
            }
            
            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                    system_matrix.add(local_dof_indices[i],
                                      local_dof_indices[j],
                                      cell_matrix(i, j));
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }
}
template <int dim, int spacedim>
void Step4<dim, spacedim>::assemble_boundary_system()
{
    hp::QCollection<dim> quadrature_boundary_formula = catmull_clark->get_boundary_QCollection();
    BoundaryValues<spacedim> right_hand_side_boundary;
    hp::FEValues<dim,spacedim> hp_fe_values(fe,
                                            quadrature_boundary_formula,
                                            update_values | update_gradients |update_inverse_jacobians|
                                            update_quadrature_points | update_JxW_values | update_jacobians);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        FullMatrix<double> cell_boundary_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_boundary_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        hp_fe_values.reinit(cell);
        const FEValues<dim, spacedim> &fe_values =
        hp_fe_values.get_present_fe_values();
        
                
        cell_boundary_matrix = 0;
        cell_boundary_rhs    = 0;
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            double jxw = 0;
            Tensor<2, spacedim> a_cov; // a_i = x_{,i} , i = 1,2,3
            auto jacobian = fe_values.jacobian(q_index);
            for (unsigned int id = 0; id < spacedim; ++id)
            {
                a_cov[0][id] = jacobian[id][0];
                a_cov[1][id] = jacobian[id][1];
            }
            if(  fe_values.get_quadrature().point(q_index)[1] == 0){
                jxw = a_cov[0].norm() * fe_values.get_quadrature().weight(q_index);
            }else{
                jxw = a_cov[1].norm() * fe_values.get_quadrature().weight(q_index);
            }
            
            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                    cell_boundary_matrix(i, j) +=
                    (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                     fe_values.shape_value(j, q_index) * // phi_j(x_q)
                     jxw);                               // dx
                const auto x_q = fe_values.quadrature_point(q_index);
                cell_boundary_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                         right_hand_side_boundary.value(x_q) *                 // f(x_q)
                                         jxw);                                        // dx
            }
            cell->get_dof_indices(local_dof_indices);
            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                    boundary_mass_matrix.add(local_dof_indices[i],
                                             local_dof_indices[j],
                                             cell_boundary_matrix(i, j));
                boundary_value_rhs(local_dof_indices[i]) += cell_boundary_rhs(i);
            }
        }
    }
}

template <int dim, int spacedim>
void Step4<dim, spacedim>::solve()
{
    std::cout <<"Apply boundary conditions using Penalty method with a factor = "<< penalty_factor <<std::endl;
    system_matrix.add(penalty_factor,boundary_mass_matrix);
    system_rhs.add(penalty_factor,boundary_value_rhs);
    SolverControl            solver_control(3000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    std::cout << "   " << solver_control.last_step()
    << " CG iterations needed to obtain convergence." << std::endl;
}

template <int dim, int spacedim>
void Step4<dim, spacedim>::output_results() const
{
    DataOut<dim, hp::DoFHandler<dim,spacedim>> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream output("solution-2d.vtk");
    data_out.write_vtk(output);
}

template <int dim, int spacedim>
void Step4<dim, spacedim>::run()
{
    std::cout << "Solving problem in " << dim << " space dimensions."
    << std::endl;
    setup_system();
    assemble_system();
    assemble_boundary_system();
    solve();
    output_results();
}

int main()
{
    deallog.depth_console(0);
    {
        Step4<2,3> laplace_problem_2d;
        laplace_problem_2d.run();
    }
    return 0;
}
