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


Tensor<2,3> covariant_to_contravariant (const Tensor<2, 3> a_cov)
{
    Tensor<2,3> a = transpose(a_cov);
    double b11,b12,b13,b21,b22,b23,b31,b32,b33, det;
    int i,j;

    b11 = a[0][0];
    b12 = a[0][1];
    b13 = a[0][2];
    b21 = a[1][0];
    b22 = a[1][1];
    b23 = a[1][2];
    b31 = a[2][0];
    b32 = a[2][1];
    b33 = a[2][2];

    a[0][0] =   b22*b33 - b32*b23;
    a[1][0] = - b21*b33 + b31*b23;
    a[2][0] =   b21*b32 - b31*b22;
    a[0][1] = - b12*b33 + b32*b13;
    a[1][1] =   b11*b33 - b31*b13;
    a[2][1] = - b11*b32 + b31*b12;
    a[0][2] =   b12*b23 - b22*b13;
    a[1][2] = - b11*b23 + b21*b13;
    a[2][2] =   b11*b22 - b21*b12;

    det = b11*a[0][0] + b12*a[1][0] + b13*a[2][0];
    for (i=0; i<3; i++)
      for (j=0; j<3; j++)
        a[i][j] = a[i][j]/det;
                          
    return a;
}



double dot_product_of_bases(const Tensor<1, 3> a, const Tensor<1, 3> b)

{
     double sum = 0;
     for (unsigned int i=0; i<3; i++) sum += a[i]*b[i];
     return sum;
}



Tensor<2, 2> metric_covariant(const Tensor<2, 3> a_cov)
{
    Tensor<2, 2> am_cov;
    for (unsigned int i=0; i<2; i++)
    {
      for (unsigned int j=0; j<2 ; j++)
      {
        am_cov[i][j] =dot_product_of_bases(a_cov[i], a_cov[j]);
      }
    }
    return am_cov;
}



Tensor<2, 2> metric_contravariant(const Tensor<2, 2> am_cov)
{
    Tensor<2, 2> am_contrav;
    double det2 = am_cov[0][0]*am_cov[1][1] - am_cov[1][0]*am_cov[0][1];
    am_contrav[0][0] = am_cov[1][1]/det2;
    am_contrav[1][1] = am_cov[0][0]/det2;
    am_contrav[0][1] =-am_cov[0][1]/det2;
    am_contrav[1][0] = am_contrav[0][1];
    return am_contrav;
}



void constitutive_tensors(Tensor<2, 3> cn, Tensor<2, 3> cm, const Tensor<2, 2> g, const double h, const double emod, const double xnue)

/* constitutive tensor (condensed s33=0)                    */
/*                                                          */
/* g       -->  contravariant metrictensor                  */
/* cn,cm   -->  constitutive tensors                        */
/* propm   -->  material data                               */
/* emod    -->  young's modulus                             */
/* xnue    -->  poisson ratio                               */

{
  double gmod, xa, xb, d, b, bdd;
                        
  gmod = emod/(2.0*(1.0+xnue));
         
  xa = (1.0-xnue)/2.0;
  xb = (1.0+xnue)/2.0;
  d  = emod*h/(1.0-xnue*xnue);
  b  = emod*h*h*h/(12.0*(1.0-xnue*xnue));

/* membrane part */

  cn[0][0] = d*g[0][0]*g[0][0];
  cn[0][1] = d*g[0][0]*g[0][1];
  cn[0][2] = d*(xa*2.0*g[0][1]*g[0][1]+xnue*g[0][0]*g[1][1]);
  cn[1][0] = cn[0][1];
  cn[1][1] = d*(xa*g[0][0]*g[1][1]+xb*g[0][1]*g[0][1]);
  cn[1][2] = d*g[0][1]*g[1][1];
  cn[2][1] = cn[1][2];
  cn[2][2] = d*g[1][1]*g[1][1];
  cn[2][0] = cn[0][2];
 
/* bending part */
                                           
  bdd=b/d;
  cm[0][0] = bdd*cn[0][0];
  cm[0][1] = bdd*cn[0][1];
  cm[0][2] = bdd*cn[0][2];
  cm[1][0] = cm[0][1];
  cm[1][1] = bdd*cn[1][1];
  cm[1][2] = bdd*cn[1][2];
  cm[2][0] = cm[0][2];
  cm[2][1] = cm[1][2];
  cm[2][2] = bdd*cn[2][2];
 
  return;
}



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
//     double product = std::sin(2*numbers::PI * p[0]);
     double product =  p[0];
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
    mesh.refine_global(3);
    
    std::ofstream gout0("half_sphere.vtu");
    std::ofstream gout1("half_sphere.msh");
    
    GridOut gird_out;
    gird_out.write_vtu(mesh,gout0);
    gird_out.write_msh(mesh,gout1);
    
    double youngs = 1e10;
    double possions = 0.2;
    double thinkness = 0.01;
    
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
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> stiffness_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;
    
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    std::ofstream out("CC_sparsity_pattern.svg");
    sparsity_pattern.print_svg(out);
    
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_gradients|update_jacobians|update_jacobian_grads|update_JxW_values|update_normal_vectors);
    
    RightHandSide<spacedim> rhs_function;

    FullMatrix<double> cell_matrix;
    FullMatrix<double> cell_stiff_matrix;
    Vector<double>     cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    
    double area = 0;
        
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_stiff_matrix.reinit(dofs_per_cell,dofs_per_cell);
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
            // covariant base  a1, a2, a3;
            Tensor<2, spacedim> a_cov; // a_i = x_{,i}
            auto jacobian = fe_values.jacobian(q_point);
            for (unsigned int id = 0; id < spacedim; ++id){
                a_cov[0][id] = jacobian[id][0];
                a_cov[1][id] = jacobian[id][1];
            }
            a_cov[2] = cross_product_3d(a_cov[0], a_cov[1]);
            double detJ = a_cov[2].norm();
            a_cov[2] = a_cov[2]/detJ;
            // derivatives of covariant base;
            Tensor<3, spacedim> da_cov; // a_{i,j}
            auto jacobian_grad = fe_values.jacobian_grad(q_point);
            for (unsigned int ialpha = 0; ialpha < dim; ++ialpha)
            {
                for (unsigned int id = 0; id < spacedim; ++id)
                {
                    da_cov[0][ialpha][id] = jacobian_grad[id][0][ialpha];
                    da_cov[1][ialpha][id] = jacobian_grad[id][1][ialpha];
                }
            }
            // contravariant base
            Tensor<2, spacedim> a_contrav = covariant_to_contravariant(a_cov); // a^i
            // covariant metric tensor
            Tensor<2,dim> am_cov = metric_covariant(a_cov);
            // contravariant metric tensor
            Tensor<2,dim> am_contra = metric_contravariant(am_cov);
            
            //constitutive tensors N and M
            Tensor<2, spacedim> constitutive_N, constitutive_M;
            constitutive_tensors(constitutive_N, constitutive_M, am_contra, thinkness, youngs, possions);
            
            Tensor<1,spacedim> a1a11 = cross_product_3d(a_cov[0], da_cov[0][0]);
            Tensor<1,spacedim> a11a2 = cross_product_3d(da_cov[0][0], a_cov[1]);
            Tensor<1,spacedim> a22a2 = cross_product_3d(da_cov[1][1], a_cov[1]);
            Tensor<1,spacedim> a1a22 = cross_product_3d(a_cov[0], da_cov[1][1]);
            Tensor<1,spacedim> a12a2 = cross_product_3d(da_cov[0][1], a_cov[1]);
            Tensor<1,spacedim> a1a12 = cross_product_3d(a_cov[0], da_cov[0][1]);
            Tensor<1,spacedim> a3a1 = cross_product_3d(a_cov[2], a_cov[0]);
            Tensor<1,spacedim> a2a3 = cross_product_3d(a_cov[1], a_cov[2]);
            
            double a3sa11 = dot_product_of_bases(a_cov[2], da_cov[0][0]);
            double a3sa12 = dot_product_of_bases(a_cov[2], da_cov[0][1]);
            double a3sa22 = dot_product_of_bases(a_cov[2], da_cov[1][1]);
            
//            std::vector<Tensor<2, spacedim>>
//                bn_vec(dofs_per_cell/spacedim),
//                bm_vec(dofs_per_cell/spacedim);
//            for (unsigned int i = 0; i < dofs_per_cell/spacedim; ++i) {
////                double shape = fe_values.shape_value(i, q_point);
//                Point<2, double> qpt = fe_values.get_quadrature().get_points()[q_point];
////                Tensor<1, dim> shape_der = fe_collection[cell->active_fe_index()].shape_grad(i, qpt);
////                Tensor<2, dim> shape_der2 = fe_collection[cell->active_fe_index()].shape_grad_grad(i, qpt);
//
//                //  computation of the B operator
//                Tensor<2, spacedim> BN; // membrane part
//                Tensor<2, spacedim> BM; // bending part
//
//                for (unsigned int ii= 0; ii < spacedim; ++ii) {
////                    BN[0][ii] = shape_der[0] * a_cov[0][ii];
////                    BN[1][ii] = shape_der[0] * a_cov[1][ii] + shape_der[1] * a_cov[0][ii];
////                    BN[2][ii] = shape_der[1] * a_cov[1][ii];
////
////                    BM[0][ii] = -shape_der2[0][0]*a_cov[2][ii] + (shape_der[0]*a11a2[ii] + shape_der[1]*a1a11[ii])/detJ + (shape_der[0]*a2a3[ii] + shape_der[1]*a3a1[ii])*a3sa11/detJ;
////                    BM[1][ii] = 2.0*((shape_der[0]*a12a2[ii] + shape_der[1]*a1a12[0])/detJ - shape_der2[0][1]*a_cov[2][ii] + (shape_der[0]*a2a3[ii] + shape_der[1]*a3a1[ii])*a3sa12/detJ);
////                    BM[2][ii] = -shape_der2[1][1]*a_cov[2][ii] + (shape_der[0]*a22a2[ii] + shape_der[1]*a1a22[ii])/detJ + (shape_der[0]*a2a3[ii] + shape_der[1]*a3a1[ii])*a3sa22/detJ;
//                }
//                bn_vec[q_point] = BN;
//                bm_vec[q_point] = BM;
//            }
//
//            for (unsigned int i = 0; i < dofs_per_cell/spacedim; ++i)
//            {
//                auto hn = constitutive_N * bn_vec[i];
//                auto hm = constitutive_M * bm_vec[i];
//
//                for (unsigned int j = 0; j < dofs_per_cell/spacedim; ++j)
//                {
//
//                    auto sn = bn_vec[j/spacedim] * hn;
//                    auto sm = bm_vec[j/spacedim] * hm;
//                    for (unsigned int id = 0; id < spacedim; ++id) {
//                        for (unsigned int jd = 0; jd < spacedim; ++jd) {
//                            cell_stiff_matrix(i*spacedim+id,j*spacedim+jd) += (sn[id][jd] + sm[id][jd])*fe_values.JxW(q_point);
//                        }
//                    }
//                }
//            }
//
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
                                rhs_values[q_point] *               // f(x_q)
                                fe_values.JxW(q_point));            // dx
            }
        }
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        system_rhs.add(local_dof_indices, cell_rhs);
        mass_matrix.add(local_dof_indices, cell_matrix);
        stiffness_matrix.add(local_dof_indices, cell_stiff_matrix);
    }
    
    std::cout << " area = " << area << std::endl;
    
    SolverControl solver_control(system_rhs.size(),
                                1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(mass_matrix, 1.2);
    cg.solve(mass_matrix, solution, system_rhs, preconditioner);
    
    
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
           std::cout << "quadrature point "<< q_point << " x = " << fe_values.get_quadrature_points()[q_point][0]<< " value = "<<q_sol << " error = " <<fe_values.get_quadrature_points()[q_point][0]-q_sol<<std::endl;
       }
    }
    
    DataOut<dim, hp::DoFHandler<dim,spacedim>> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(4);
    const std::string filename =
      "solution-CC.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    
    return 0;
}
