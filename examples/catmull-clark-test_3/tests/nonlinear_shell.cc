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

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;

//template <int spacedim>
//class RightHandSide : public Function<spacedim>
//{
//public:
//    virtual double value(const Point<spacedim> & p,
//                         const unsigned int component) const override;
//};
//template <int spacedim>
//double RightHandSide<spacedim>::value(const Point<spacedim> &p,
//                                      const unsigned int /*component*/) const
//{
//    double product = 1;
//    return product;
//}



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



Tensor<2, 3> metric_covariant(const Tensor<2, 3> a_cov)
{
    Tensor<2, 3> am_cov;
    for (unsigned int ii=0; ii<2; ++ii)
    {
        for (unsigned int jj=0; jj<2 ; ++jj)
        {
            am_cov[ii][jj] =scalar_product(a_cov[ii], a_cov[jj]);
        }
    }
    am_cov[2][2] = 1;
    return am_cov;
}


Tensor<2, 3> metric_contravariant(const Tensor<2, 3> am_cov)
{
    return transpose(invert(am_cov));
}


void vtk_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>()){
    
    //    auto verts = dof_handler.get_triangulation().get_vertices();
    
    const unsigned int ngridpts = 5;
    const unsigned int seg_n = ngridpts-1;
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
    vtkSmartPointer<vtkPoints> points = vtkPoints::New();
    vtkSmartPointer<vtkDoubleArray> function = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> function_2 = vtkDoubleArray::New();
    
    function->SetNumberOfComponents(3);
    function->SetName("disp");
    function->SetComponentName(0, "x");
    function->SetComponentName(1, "y");
    function->SetComponentName(2, "z");
    
    if (potential.size() != 0){
        function_2->SetNumberOfComponents(1);
        function_2->SetName("potential");
        function_2->SetComponentName(0, "value");
    }
    
    int sample_offset = 0;
    int count = 0;
    double seg_length = 1./seg_n;
    int numElem = dof_handler.get_triangulation().n_active_cells();
    
    std::vector<types::global_dof_index> local_dof_indices;
    
    for (auto cell=dof_handler.begin_active();cell!=dof_handler.end(); ++cell){
        
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        for(unsigned int iu = 0; iu < ngridpts; ++iu){
            for(unsigned int iv = 0; iv < ngridpts; ++iv){
                double u = iu*seg_length;
                double v = iv*seg_length;
                //
                Point<3,double> spt = {0,0,0};
                Tensor<1,3,double> disp({0,0,0});
                std::vector<Tensor<1,3>> JJ(3);
                std::vector<Tensor<2,3>> JJ_grad(2);
                double sol = 0;
                for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
                {
                    double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
                    
                    sol += shapes * solution[local_dof_indices[idof]];
                    
                    switch (idof % 3) {
                        case 0:
                            spt[0] += shapes * vertices[local_dof_indices[idof]];
                            disp[0] += shapes * solution[local_dof_indices[idof]];
                            break;
                        case 1:
                            spt[1] += shapes * vertices[local_dof_indices[idof]];
                            disp[1] += shapes * solution[local_dof_indices[idof]];
                            break;
                        case 2:
                            spt[2] += shapes * vertices[local_dof_indices[idof]];
                            disp[2] += shapes * solution[local_dof_indices[idof]];
                            break;
                    }
                }
                double p = 0;
                if (potential.size() != 0){
                    for (unsigned int jdof = 0; jdof < dofs_per_cell/3; ++jdof) {
                        double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(jdof*3, {u,v});
                        p += shapes * potential[local_dof_indices[jdof*3]/3];
                    }
                }
                
                JJ[2] = cross_product_3d(JJ[0],JJ[1]);
                
                double coordsdata [3] = {spt[0],spt[1],spt[2]};
                
                points->InsertPoint(sample_offset+count, coordsdata);
                
                function->InsertComponent(sample_offset+count, 0, disp[0]);
                function->InsertComponent(sample_offset+count, 1, disp[1]);
                function->InsertComponent(sample_offset+count, 2, disp[2]);
                if (potential.size() != 0)
                    function_2->InsertComponent(sample_offset+count, 0, p);
                ++count;
            }
        }
    }
    uint sampleindex = 0;
    //loop over elements
    for(int e = 0; e < numElem; ++e){
        for (unsigned int t = 0 ; t < seg_n; ++t){
            for (unsigned int s = 0; s < seg_n; ++s){
                vtkSmartPointer<vtkCell> cell = vtkQuad::New();
                cell -> GetPointIds() -> SetId(0, sampleindex + t * ngridpts + s);
                cell -> GetPointIds() -> SetId(1, sampleindex + t * ngridpts + s + 1);
                cell -> GetPointIds() -> SetId(2, sampleindex + (t + 1) * ngridpts + s + 1 );
                cell -> GetPointIds() -> SetId(3, sampleindex + (t + 1) * ngridpts + s);
                grid -> InsertNextCell (cell -> GetCellType(), cell -> GetPointIds());
            }
        }
        sampleindex += ngridpts * ngridpts;
    }
    grid -> SetPoints(points);
    grid -> GetPointData() -> AddArray(function);
    if (potential.size() != 0){
        grid -> GetPointData() -> AddArray(function_2);
    }
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
    writer -> SetFileName(filename.c_str());
    writer -> SetInputData(grid);
    if (! writer -> Write()) {
        std::cout<<" Cannot write displacement vtu file! ";
    }
}



template<int dim, int spacedim>
Tensor<2,spacedim> get_tau_mooney_rivlin(const double c_1,
                                         const double c_2,
                                         const Tensor<2,spacedim> gm_contra_ref,
                                         const Tensor<2,spacedim> gm_cov_def,
                                         const Tensor<2,spacedim> gm_contra_def)
{
    double p = 0;
    for (unsigned int k = 0; k < spacedim; ++k)
        for (unsigned int l = 0; l < spacedim; ++l){
            p += 2 * (c_1 + c_2 * gm_contra_ref[k][l] * gm_cov_def[k][l]) - 2*c_2*gm_contra_ref[2][k] * gm_cov_def[k][l] * gm_contra_ref[l][2];
        }
    Tensor<2,spacedim> tau;
    for (unsigned int i = 0; i < spacedim; ++i)
        for (unsigned int j = 0; j < spacedim; ++j)
            for (unsigned int k = 0; k < spacedim; ++k)
                for (unsigned int l = 0; l < spacedim; ++l){
                    tau[i][j] += 2 * (c_1 + c_2 * gm_contra_ref[k][l] * gm_cov_def[k][l]) *gm_contra_ref[i][j] - 2 * c_2 * gm_contra_ref[i][k] * gm_cov_def[k][l] * gm_contra_ref[l][j] - p * gm_contra_def[i][j];
                }
    
    return tau;
}



template<int dim, int spacedim>
class linearisation_derivatives
{
public:
    linearisation_derivatives(const double ishape_fun, const Tensor<1, spacedim> ishape_grad, const Tensor<2,spacedim> ishape_hessian, const double jshape_fun, const Tensor<1, spacedim> jshape_grad, const Tensor<2,spacedim> jshape_hessian, const Tensor<2, spacedim> a_cov, const Tensor<2, dim, Tensor<1,spacedim>> da_cov, const unsigned int dof_i, const unsigned int dof_j)
    :
    i_shape(ishape_fun),
    i_shape_deriv(ishape_grad),
    i_shape_deriv2(ishape_hessian),
    j_shape(jshape_fun),
    j_shape_deriv(jshape_grad),
    j_shape_deriv2(jshape_hessian),
    a_cov(a_cov),
    da_cov(da_cov),
    r(dof_i),
    s(dof_j)
    {
        if (r%3 == 0) {
            u_r = {i_shape, 0, 0};
            r_r = {i_shape, 0, 0};
        }else if(r%3 == 1){
            u_r = {0, i_shape, 0};
            r_r = {0, i_shape, 0};
        }else if(r%3 == 2){
            u_r = {0, 0, i_shape};
            r_r = {0, 0, i_shape};
        }
        
        if (s%3 == 0) {
            u_s = {j_shape, 0, 0};
            r_s = {j_shape, 0, 0};
        }else if(s%3 == 1){
            u_s = {0, j_shape, 0};
            r_s = {0, j_shape, 0};
        }else if(s%3 == 2){
            u_s = {0, 0, j_shape};
            r_s = {0, 0, j_shape};
        }
        
        for (unsigned int i = 0; i < dim; ++i) {
            if (r%3 == 0) {
                a_cov_ar[i] = {i_shape_deriv[i], 0, 0};
            }else if(r%3 == 1){
                a_cov_ar[i] = {0, i_shape_deriv[i], 0};
            }else if(r%3 == 2){
                a_cov_ar[i] = {0, 0, i_shape_deriv[i]};
            }
            if (s%3 == 0) {
                a_cov_as[i] = {j_shape_deriv[i], 0, 0};
            }else if(s%3 == 1){
                a_cov_as[i] = {0, j_shape_deriv[i], 0};
            }else if(s%3 == 2){
                a_cov_as[i] = {0, 0, j_shape_deriv[i]};
            }
            
            for (unsigned int j = 0; j < dim; ++j) {
                if (r%3 == 0) {
                    a_cov_abr[i][j] = {i_shape_deriv2[i][j], 0, 0};
                }else if(r%3 == 1){
                    a_cov_abr[i][j] = {0, i_shape_deriv2[i][j], 0};
                }else if(r%3 == 2){
                    a_cov_abr[i][j] = {0, 0, i_shape_deriv2[i][j]};
                }
                if (s%3 == 0) {
                    a_cov_abs[i][j] = {j_shape_deriv2[i][j], 0, 0};
                }else if(s%3 == 1){
                    a_cov_abs[i][j] = {0, j_shape_deriv2[i][j], 0};
                }else if(s%3 == 2){
                    a_cov_abs[i][j] = {0, 0, j_shape_deriv2[i][j]};
                }
            }
        }
        
        Tensor<1, spacedim> a3_t = cross_product_3d(a_cov[0], a_cov[1]);
        double a3_bar = a3_t.norm();
        Tensor<1, dim, Tensor<1, spacedim>> a3_t_da;
        for (unsigned int i = 0; i < dim; ++i) {
            a3_t_da[i] = cross_product_3d(da_cov[0][i], a_cov[1]) + cross_product_3d(a_cov[0], da_cov[1][i]);
        }
        Tensor<1, dim> a3_bar_da;
        for (unsigned int i = 0; i < dim; ++i) {
            a3_bar_da[i] = scalar_product(a3_t, a3_t_da[i])/a3_bar;
        }
        for (unsigned int i = 0; i < dim; ++i) {
            a3_da[i] = a3_t_da[i] / a3_bar - a3_bar_da * a3_t[i]/ (a3_bar * a3_bar);
        }
        Tensor<2, dim, Tensor<1, spacedim>> a3_t_dab;
        for (unsigned int i = 0; i < dim; ++i) {
            for (unsigned int j = 0; j < dim; ++j) {
                a3_t_dab[i][j] = cross_product_3d(da_cov[0][i], da_cov[1][j]) + cross_product_3d(da_cov[0][j], da_cov[1][i]);
            }
        }
        Tensor<2, dim> a3_bar_dab;
        for (unsigned int i = 0; i < dim; ++i) {
            for (unsigned int j = 0; j < dim; ++j) {
                a3_bar_dab[i][j] = scalar_product(a3_t_da[j], a3_t_da[i])/ a3_bar + scalar_product(a3_t, a3_t_dab[i][j])/ a3_bar - a3_bar_da[j]* scalar_product(a3_t, a3_t_da[i])/ (a3_bar * a3_bar);
            }
        }
        for (unsigned int i = 0; i < dim; ++i) {
            for (unsigned int j = 0; j < dim; ++j) {
                a3_dab = a3_t_dab / a3_bar - a3_bar_da[j] * a3_t_da[i] / (a3_bar * a3_bar) -  a3_bar_dab[i][j] * a3_t / (a3_bar * a3_bar) -  a3_bar_da[i] * a3_t_da[j] / (a3_bar * a3_bar) + 2 * a3_bar_da[j] * a3_bar_da[i] * a3_t / (a3_bar * a3_bar * a3_bar);
            }
        }
        
        Tensor<1, spacedim> a3_t_dr = cross_product_3d(a_cov_ar[0], a_cov[1]) + cross_product_3d(a_cov[0], a_cov_ar[1]);
        Tensor<1, spacedim> a3_t_ds = cross_product_3d(a_cov_as[0], a_cov[1]) + cross_product_3d(a_cov[0], a_cov_as[1]);
        double a3_bar_dr = scalar_product(a3_t, a3_t_dr)/a3_bar;
        double a3_bar_ds = scalar_product(a3_t, a3_t_ds)/a3_bar;

        Tensor<1, spacedim> a3_t_drs = cross_product_3d(a_cov_ar[0], a_cov_as[1]) + cross_product_3d(a_cov_as[0], a_cov_ar[1]);
        double a3_bar_drs = scalar_product(a3_t_ds, a3_t_dr)/ a3_bar + scalar_product(a3_t, a3_t_drs)/ a3_bar - a3_bar_ds* scalar_product(a3_t, a3_t_dr)/ (a3_bar * a3_bar);
        a3_dr = a3_t_dr / a3_bar - a3_t_dr * a3_t/ (a3_bar * a3_bar);
        a3_ds = a3_t_ds / a3_bar - a3_t_ds * a3_t/ (a3_bar * a3_bar);
        a3_drs = a3_t_drs / a3_bar - a3_bar_drs * a3_t /(a3_bar * a3_bar) - a3_bar_dr * a3_t_ds / (a3_bar * a3_bar) - a3_bar_ds * a3_t_dr / (a3_bar * a3_bar) + 2 * a3_bar_dr * a3_bar_ds * a3_t / (a3_bar * a3_bar * a3_bar);
        
        Tensor<1, dim, Tensor<1, spacedim>> a3_t_da_dr;
        Tensor<1, dim, Tensor<1, spacedim>> a3_t_da_ds;
        for (unsigned int i = 0; i < dim; ++i) {
            a3_t_da_dr[i] = cross_product_3d(da_cov[0][i], a_cov_ar[1]) + cross_product_3d(a_cov_ar[0], da_cov[1][i]);
            a3_t_da_ds[i] = cross_product_3d(da_cov[0][i], a_cov_as[1]) + cross_product_3d(a_cov_as[0], da_cov[1][i]);
        }
        Tensor<1, dim, Tensor<1, spacedim>> a3_t_da_drs;
        for (unsigned int i = 0; i < dim; ++i) {
            a3_t_da_drs[i] = cross_product_3d(a_cov_abs[0][i], a_cov_ar[1]) + cross_product_3d(a_cov_ar[0], a_cov_abs[1][i]);
        }
        Tensor<1,dim> a3_bar_da_dr;
        Tensor<1,dim> a3_bar_da_ds;
        Tensor<1,dim> a3_bar_da_drs;
        for (unsigned int i = 0; i < dim; ++i) {
            a3_bar_da_dr[i] = scalar_product(a3_t_dr, a3_t_da[i])/a3_bar + scalar_product(a3_t, a3_t_da_dr[i])/a3_bar - a3_bar_dr * scalar_product(a3_t, a3_t_da[i]) / (a3_bar * a3_bar);
            a3_bar_da_ds[i] = scalar_product(a3_t_ds, a3_t_da[i])/a3_bar + scalar_product(a3_t, a3_t_da_ds[i])/a3_bar - a3_bar_ds * scalar_product(a3_t, a3_t_da[i]) / (a3_bar * a3_bar);
            a3_bar_da_drs[i] = scalar_product(a3_t_drs, a3_t_da[i])/a3_bar + scalar_product(a3_t_dr, a3_t_da_ds[i])/a3_bar - a3_bar_ds * scalar_product(a3_t_dr, a3_t_da[i])/(a3_bar * a3_bar) + scalar_product(a3_t_ds, a3_t_da_dr[i])/a3_bar + scalar_product(a3_t, a3_t_da_drs[i])/a3_bar - a3_bar_ds * scalar_product(a3_t, a3_t_da_dr[i])/(a3_bar * a3_bar) - a3_bar_drs * scalar_product(a3_t, a3_t_da[i]) / (a3_bar * a3_bar) - a3_bar_dr * scalar_product(a3_t_ds, a3_t_da[i]) / (a3_bar * a3_bar) - a3_bar_dr * scalar_product(a3_t, a3_t_da_ds[i]) / (a3_bar * a3_bar) + 2 * a3_bar_ds * a3_bar_dr * scalar_product(a3_t, a3_t_da[i]) / (a3_bar * a3_bar * a3_bar);
        }
        
        for (unsigned int i = 0; i < dim; ++i) {
            a3_da_dr[i] = a3_t_da_dr[i]/a3_bar - a3_bar_dr * a3_t_da[i] /(a3_bar * a3_bar) - a3_bar_da_dr[i] * a3_t /(a3_bar * a3_bar) - a3_bar_da[i] * a3_t_dr /(a3_bar * a3_bar) + 2 * a3_bar_dr * a3_bar_da[i] * a3_t /(a3_bar * a3_bar * a3_bar);
            a3_da_ds[i] = a3_t_da_ds[i]/a3_bar - a3_bar_ds * a3_t_da[i] /(a3_bar * a3_bar) - a3_bar_da_ds[i] * a3_t /(a3_bar * a3_bar) - a3_bar_da[i] * a3_t_ds /(a3_bar * a3_bar) + 2 * a3_bar_ds * a3_bar_da[i] * a3_t /(a3_bar * a3_bar * a3_bar);
            a3_da_drs[i] = a3_t_da_drs[i]/a3_bar - a3_bar_ds * a3_t_da_dr[i]/(a3_bar * a3_bar) - a3_bar_drs * a3_t_da[i] /(a3_bar * a3_bar) - a3_bar_dr * a3_t_da_ds[i] /(a3_bar * a3_bar) +2 * a3_bar_ds * a3_bar_dr * a3_t_da[i] /(a3_bar * a3_bar * a3_bar) - a3_bar_da_drs[i] * a3_t /(a3_bar * a3_bar) - a3_bar_da_dr[i] * a3_t_ds /(a3_bar * a3_bar) + 2 * a3_bar_ds * a3_bar_da_dr[i] * a3_t /(a3_bar * a3_bar * a3_bar) - a3_bar_da_ds[i] * a3_t_dr /(a3_bar * a3_bar) - a3_bar_da[i] * a3_t_drs /(a3_bar * a3_bar) + 2 * a3_bar_ds * a3_bar_da[i] * a3_t_dr /(a3_bar * a3_bar * a3_bar) + 2 * a3_bar_drs * a3_bar_da[i] * a3_t /(a3_bar * a3_bar * a3_bar) + 2 * a3_bar_dr * a3_bar_da_ds[i] * a3_t /(a3_bar * a3_bar * a3_bar) + 2 * a3_bar_dr * a3_bar_da[i] * a3_t_ds /(a3_bar * a3_bar * a3_bar) + 6 * a3_bar_ds  * a3_bar_dr * a3_bar_da[i] * a3_t /(a3_bar * a3_bar * a3_bar * a3_bar);
        }
    }
    
    Tensor<1, dim, Tensor<1, spacedim>> get_a_cov_ar(){return a_cov_ar;};
    Tensor<2, dim, Tensor<1, spacedim>> get_a_cov_abr(){return a_cov_abr;};
    Tensor<1, dim, Tensor<1, spacedim>> get_a_cov_as(){return a_cov_as;};
    Tensor<2, dim, Tensor<1, spacedim>> get_a_cov_abs(){return a_cov_abs;};
    Tensor<1, dim, Tensor<1, spacedim>> get_a3_da(){return a3_da;};
    Tensor<2, dim, Tensor<1, spacedim>> get_a3_dab(){return a3_dab;};
    Tensor<1, spacedim> get_a3_dr(){return a3_dr;};
    Tensor<1, spacedim> get_a3_ds(){return a3_ds;};
    Tensor<1, spacedim> get_a3_drs(){return a3_drs;};
    Tensor<1, dim, Tensor<1, spacedim>> get_a3_da_dr(){return a3_da_dr;};
    Tensor<1, dim, Tensor<1, spacedim>> get_a3_da_ds(){return a3_da_ds;};
    Tensor<1, dim, Tensor<1, spacedim>> get_a3_da_drs(){return a3_da_drs;};
    Tensor<1,spacedim> get_u_r(){return u_r;};
    Tensor<1,spacedim> get_r_r(){return r_r;};
    Tensor<1,spacedim> get_u_s(){return u_s;};
    Tensor<1,spacedim> get_r_s(){return r_s;};
    
private:
    const double i_shape;
    const double i_shape_deriv;
    const double i_shape_deriv2;
    const double j_shape;
    const double j_shape_deriv;
    const double j_shape_deriv2;
    const Tensor<2, spacedim> a_cov;
    const Tensor<2, dim, Tensor<1,spacedim>> da_cov;
    const unsigned int r,s;
    
    Tensor<1, dim, Tensor<1, spacedim>> a_cov_ar;
    Tensor<2, dim, Tensor<1, spacedim>> a_cov_abr;
    Tensor<1, dim, Tensor<1, spacedim>> a_cov_as;
    Tensor<2, dim, Tensor<1, spacedim>> a_cov_abs;
    Tensor<1, dim, Tensor<1, spacedim>> a3_da;
    Tensor<2, dim, Tensor<1, spacedim>> a3_dab;
    Tensor<1, spacedim> a3_dr;
    Tensor<1, spacedim> a3_ds;
    Tensor<1, spacedim> a3_drs;
    Tensor<1, dim, Tensor<1, spacedim>> a3_da_dr;
    Tensor<1, dim, Tensor<1, spacedim>> a3_da_ds;
    Tensor<1, dim, Tensor<1, spacedim>> a3_da_drs;
    Tensor<1,spacedim> u_r, r_r, u_s, r_s;
};



void vtk_eigen_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>()){
    
    //    auto verts = dof_handler.get_triangulation().get_vertices();
    
    const unsigned int ngridpts = 4;
    const unsigned int seg_n = ngridpts-1;
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
    vtkSmartPointer<vtkPoints> points = vtkPoints::New();
    vtkSmartPointer<vtkDoubleArray> function = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> function_2 = vtkDoubleArray::New();
    
    function->SetNumberOfComponents(3);
    function->SetName("disp");
    function->SetComponentName(0, "x");
    function->SetComponentName(1, "y");
    function->SetComponentName(2, "z");
    
    if (potential.size() != 0){
        function_2->SetNumberOfComponents(1);
        function_2->SetName("potential");
        function_2->SetComponentName(0, "value");
    }
    
    int sample_offset = 0;
    int count = 0;
    double seg_length = 1./seg_n;
    int numElem = dof_handler.get_triangulation().n_active_cells();
    
    std::vector<types::global_dof_index> local_dof_indices;
    
    for (auto cell=dof_handler.begin_active();cell!=dof_handler.end(); ++cell){
        
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        for(unsigned int iu = 0; iu < ngridpts; ++iu){
            for(unsigned int iv = 0; iv < ngridpts; ++iv){
                double u = iu*seg_length;
                double v = iv*seg_length;
                //
                Point<3,double> spt = {0,0,0};
                Tensor<1,3,double> disp({0,0,0});
                std::vector<Tensor<1,3>> JJ(3);
                std::vector<Tensor<2,3>> JJ_grad(2);
                double sol = 0;
                for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
                {
                    double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
                    
                    sol += shapes * solution[local_dof_indices[idof]];
                    
                    switch (idof % 3) {
                        case 0:
                            spt[0] += shapes * vertices[local_dof_indices[idof]];
                            disp[0] += shapes * solution[local_dof_indices[idof]];
                            break;
                        case 1:
                            spt[1] += shapes * vertices[local_dof_indices[idof]];
                            disp[1] += shapes * solution[local_dof_indices[idof]];
                            break;
                        case 2:
                            spt[2] += shapes * vertices[local_dof_indices[idof]];
                            disp[2] += shapes * solution[local_dof_indices[idof]];
                            break;
                    }
                }
                double p = 0;
                if (potential.size() != 0){
                    for (unsigned int jdof = 0; jdof < dofs_per_cell/3; ++jdof) {
                        double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(jdof*3, {u,v});
                        p += shapes * potential[local_dof_indices[jdof*3]/3];
                    }
                }
                
                JJ[2] = cross_product_3d(JJ[0],JJ[1]);
                
                double coordsdata [3] = {spt[0],spt[1],spt[2]};
                
                points->InsertPoint(sample_offset+count, coordsdata);
                
                function->InsertComponent(sample_offset+count, 0, disp[0]);
                function->InsertComponent(sample_offset+count, 1, disp[1]);
                function->InsertComponent(sample_offset+count, 2, disp[2]);
                if (potential.size() != 0)
                    function_2->InsertComponent(sample_offset+count, 0, p);
                ++count;
            }
        }
    }
    uint sampleindex = 0;
    //loop over elements
    for(int e = 0; e < numElem; ++e){
        for (unsigned int t = 0 ; t < seg_n; ++t){
            for (unsigned int s = 0; s < seg_n; ++s){
                vtkSmartPointer<vtkCell> cell = vtkQuad::New();
                cell -> GetPointIds() -> SetId(0, sampleindex + t * ngridpts + s);
                cell -> GetPointIds() -> SetId(1, sampleindex + t * ngridpts + s + 1);
                cell -> GetPointIds() -> SetId(2, sampleindex + (t + 1) * ngridpts + s + 1 );
                cell -> GetPointIds() -> SetId(3, sampleindex + (t + 1) * ngridpts + s);
                grid -> InsertNextCell (cell -> GetCellType(), cell -> GetPointIds());
            }
        }
        sampleindex += ngridpts * ngridpts;
    }
    grid -> SetPoints(points);
    grid -> GetPointData() -> AddArray(function);
    if (potential.size() != 0){
        grid -> GetPointData() -> AddArray(function_2);
    }
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
    writer -> SetFileName(filename.c_str());
    writer -> SetInputData(grid);
    if (! writer -> Write()) {
        std::cout<<" Cannot write displacement vtu file! ";
    }
}



template <int dim, int spacedim>
class Nonlinear_shell
{
public:
    Nonlinear_shell();
    ~Nonlinear_shell();
  void run();
private:
  void   setup_fe_and_dof_handler();
  void   setup_system(const bool initial_step);
  void   assemble_system();
  void   solve();
  void   set_mesh( std::string type );
  double compute_residual(const double alpha);
  double determine_step_length() const;
  Triangulation<dim,spacedim> mesh;
  hp::DoFHandler<dim,spacedim> dof_handler;
  hp::FECollection<dim,spacedim> fe_collection;
  hp::MappingCollection<dim,spacedim> mapping_collection;
  hp::QCollection<dim> q_collection;
  hp::QCollection<dim> boundary_q_collection;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> stiffness_matrix;
  Vector<double> newton_update;
  Vector<double> present_solution;
  Vector<double> solution_disp;
  Vector<double> force_rhs;
  const double thickness = 0.01, density = 1000.;
  const double mu = 4.225e5, c_1 = 0.4375*mu, c_2 = 0.0625*mu;
};



template <int dim, int spacedim>
Nonlinear_shell<dim, spacedim>::~Nonlinear_shell()
{
  dof_handler.clear();
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: set_mesh( std::string type )
{
    if (type == "r") {
        static CylindricalManifold<dim,spacedim> surface_description;
        std::string mfile = "/Users/zhaoweiliu/Documents/geometries/roof.msh";
        GridIn<2,3> grid_in;
        grid_in.attach_triangulation(mesh);
        std::ifstream file(mfile.c_str());
        Assert(file, ExcFileNotOpen(mfile.c_str()));
        grid_in.read_msh(file);
        mesh.set_all_manifold_ids(0);
        mesh.set_manifold (0, surface_description);
        mesh.refine_global(1);
    }else{
        if (type == "s") {
            static SphericalManifold<dim,spacedim> surface_description;
            {
                Triangulation<spacedim> volume_mesh;
                GridGenerator::hyper_ball(volume_mesh);
                std::set<types::boundary_id> boundary_ids;
                boundary_ids.insert (0);
                GridGenerator::extract_boundary_mesh (volume_mesh, mesh,
                                                      boundary_ids);
            }
            mesh.set_all_manifold_ids(0);
            mesh.set_manifold (0, surface_description);
            mesh.refine_global(4);
        }else if (type == "c"){
            static CylindricalManifold<dim,spacedim> surface_description;
            {
                Triangulation<spacedim> volume_mesh;
                GridGenerator::cylinder(volume_mesh,300,300);
                std::set<types::boundary_id> boundary_ids;
                boundary_ids.insert (0);
                GridGenerator::extract_boundary_mesh (volume_mesh, mesh, boundary_ids);
            }
            mesh.set_all_manifold_ids(0);
            mesh.set_manifold (0, surface_description);
            mesh.refine_global(4);
        }else {
            if (type == "b")
            {
                static CylindricalManifold<dim,spacedim> surface_description;
                std::string mfile = "/Users/zhaoweiliu/Documents/geometries/beam.msh";
                GridIn<2,3> grid_in;
                grid_in.attach_triangulation(mesh);
                std::ifstream file(mfile.c_str());
                Assert(file, ExcFileNotOpen(mfile.c_str()));
                grid_in.read_msh(file);
                mesh.set_all_manifold_ids(0);
                mesh.set_manifold (0, surface_description);
                mesh.refine_global(1);
                GridTools::rotate(numbers::PI*0.25, 0, mesh);
            }
        }
    }
    std::cout << "   Number of active cells: " << mesh.n_active_cells()
    << std::endl
    << "   Total number of cells: " << mesh.n_cells()
    << std::endl;
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: setup_fe_and_dof_handler()
{
    Vector<double> vec_values;
    catmull_clark_create_fe_quadrature_and_mapping_collections_and_distribute_dofs(dof_handler,fe_collection,vec_values,mapping_collection,q_collection,boundary_q_collection,3);
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    
    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern, constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    solution_disp.reinit(dof_handler.n_dofs());
    force_rhs.reinit(dof_handler.n_dofs());
 }



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: assemble_system()
{
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    FullMatrix<double> cell_stiffness_matrix;
    Vector<double>     cell_force_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
//    std::vector<types::global_dof_index> fix_dof_indices;
    QGauss<1> Qthickness(2);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
                
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        
        cell_stiffness_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_stiffness_matrix = 0;
        
        cell_force_rhs.reinit(dofs_per_cell);
        cell_force_rhs = 0;
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
        {
            // covariant base  a_1, a_2, a_3;
            Tensor<2, spacedim> a_cov_ref; // a_i = x_{,i} , i = 1,2,3
            // derivatives of covariant base;
            Tensor<2, dim, Tensor<1,spacedim>> da_cov_ref; // a_{i,j} = x_{,ij} , i,j = 1,2
            auto jacobian_ref = fe_values.jacobian(q_point);
            
            for (unsigned int id = 0; id < spacedim; ++id){
                a_cov_ref[0][id] = jacobian_ref[id][0];
                a_cov_ref[1][id] = jacobian_ref[id][1];
            }
            a_cov_ref[2] = cross_product_3d(a_cov_ref[0], a_cov_ref[1]);
            double detJ_ref = a_cov_ref[2].norm();
            a_cov_ref[2] = a_cov_ref[2]/detJ_ref;
            
            auto jacobian_grad_ref = fe_values.jacobian_grad(q_point);
            for (unsigned int jj = 0; jj < dim; ++jj)
            {
                for (unsigned int kk = 0; kk < spacedim; ++kk)
                {
                    da_cov_ref[0][jj][kk] = jacobian_grad_ref[kk][0][jj];
                    da_cov_ref[1][jj][kk] = jacobian_grad_ref[kk][1][jj];
                }
            }
            Tensor<2, spacedim> da3_ref;
            da3_ref[0] = cross_product_3d(da_cov_ref[0][0], a_cov_ref[1]) + cross_product_3d(a_cov_ref[0], da_cov_ref[1][0])/detJ_ref;
            da3_ref[1] = cross_product_3d(da_cov_ref[0][1], a_cov_ref[1]) + cross_product_3d(a_cov_ref[0], da_cov_ref[1][1])/detJ_ref;

            // covariant metric tensor
            Tensor<2,spacedim> am_cov_ref = metric_covariant(a_cov_ref);
            // contravariant metric tensor
            Tensor<2,spacedim> am_contra_ref = metric_contravariant(am_cov_ref);
            
            std::vector<double> shape_vec(dofs_per_cell);
            std::vector<Tensor<1, spacedim>> shape_der_vec(dofs_per_cell);
            std::vector<Tensor<2, spacedim>> shape_der2_vec(dofs_per_cell);
            
            Tensor<1, dim, Tensor<1,spacedim>> u_der; // u_{,a}
            Tensor<2, dim, Tensor<1,spacedim>> u_der2; // u_{,ab}
            
            for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
                // compute first and second grad of i_shape function
                double i_shape_vlaue = fe_values.shape_value(i_shape, q_point);
                Tensor<1, spacedim> i_shape_grad = fe_values.shape_grad(i_shape, q_point);
                Tensor<2, spacedim> i_shape_hessian = fe_values.shape_hessian(i_shape, q_point);
                Tensor<1, dim> i_shape_der; // N_{,a}
                Tensor<2, dim> i_shape_der2; // N_{,ab}
                // transform to parametric domain
                for (unsigned int id = 0; id < dim; ++id){
                    for (unsigned int kd = 0; kd < spacedim; ++kd){
                        i_shape_der[id] += i_shape_grad[kd]*jacobian_ref[kd][id];
                        for (unsigned jd = 0; jd < dim; ++jd) {
                            for (unsigned ld = 0; ld < spacedim; ++ld) {
                                i_shape_der2[id][jd] += i_shape_hessian[kd][ld] * jacobian_ref[kd][id] * jacobian_ref[ld][jd];
                            }
                            i_shape_der2[id][jd] += i_shape_grad[kd] * jacobian_grad_ref[kd][id][jd];
                        }
                    }
                }
                shape_vec[i_shape] = i_shape_vlaue;
                shape_der_vec[i_shape] = i_shape_der;
                shape_der2_vec[i_shape] = i_shape_der2;
                for (unsigned int ia = 0; ia < dim; ++ia){
                    switch (i_shape % 3) {
                        case 0:
                            u_der[ia][0] += i_shape_der[ia] * present_solution(local_dof_indices[i_shape]); // u_{,a} = sum N^A_{,a} * U_A
                            break;
                        case 1:
                            u_der[ia][1] += i_shape_der[ia] * present_solution(local_dof_indices[i_shape]);
                            break;
                        case 2:
                            u_der[ia][2] += i_shape_der[ia] * present_solution(local_dof_indices[i_shape]);
                            break;
                        default:
                            break;
                    }
                    for (unsigned int ib = 0; ib < dim; ++ib){
                        switch (i_shape % 3) {
                            case 0:
                                u_der2[ia][ib][0] += i_shape_der2[ia][ib] * present_solution(local_dof_indices[i_shape]); // u_{,ab} = sum N^A_{,ab} * U_A
                                break;
                            case 1:
                                u_der2[ia][ib][1] += i_shape_der2[ia][ib] * present_solution(local_dof_indices[i_shape]);
                                break;
                            case 2:
                                u_der2[ia][ib][2] += i_shape_der2[ia][ib] * present_solution(local_dof_indices[i_shape]);
                                break;
                            default:
                                break;
                        }
                    }
                }
            }
            
            Tensor<2, spacedim> a_cov_def = a_cov_ref;
            Tensor<2, dim, Tensor<1,spacedim>> da_cov_def = da_cov_ref;
            for (unsigned int ia = 0; ia < dim; ++ia){
                a_cov_def[ia] += u_der[ia]; // a_alpha = bar{a_alpha} + u_{,alpha}
                for (unsigned int ib = 0; ib < dim; ++ib){
                    da_cov_def[ia][ib] += u_der2[ia][ib]; // a_{alpha,\beta} = bar{a_{alpha,beta}} + u_{,alpha beta}
                }
            }
            a_cov_def[2] = cross_product_3d(a_cov_def[0], a_cov_def[1]);
            a_cov_def[2] = a_cov_def[2]/a_cov_def.norm();
            
            for (unsigned int s_shape = 0; s_shape < dofs_per_cell; ++s_shape) {
                auto shape_s = shape_vec[s_shape];
                auto shape_s_der = shape_der_vec[s_shape];
                auto shape_s_der2 = shape_der2_vec[s_shape];
                for (unsigned int r_shape = 0; r_shape < dofs_per_cell; ++r_shape) {
                    auto shape_r = shape_vec[r_shape];
                    auto shape_r_der = shape_der_vec[r_shape];
                    auto shape_r_der2 = shape_der2_vec[r_shape];
                    
                    linearisation_derivatives<dim,spacedim> L_derivs(shape_r, shape_r_der, shape_r_der2, shape_s, shape_s_der, shape_s_der2, a_cov_def, da_cov_def, r_shape, s_shape);
                    // terms to compute residual vector
                    Tensor<2, dim, Tensor<1, spacedim>> d2a3_cov_def = L_derivs.get_a3_dab();
                    Tensor<1, dim, Tensor<1, spacedim>> a_cov_def_dr = L_derivs.get_a_cov_ar();
                    Tensor<1, dim, Tensor<1, spacedim>> a3_cov_def_dr = L_derivs.get_a3_dr();
                    Tensor<1, dim, Tensor<1, spacedim>> a3_cov_def_da_dr = L_derivs.get_a3_da_dr();
                    Tensor<1,spacedim> u_r = L_derivs.get_u_r();
                    
                    // terms required in matrix
                    auto a3_cov_def_drs = L_derivs.get_a3_drs();
                    auto a3_cov_def_da_drs = L_derivs.get_a3_da_drs();
                    
                    Tensor<1, dim, Tensor<1, spacedim>> a_cov_def_ds = L_derivs.get_a_cov_as();
                    Tensor<1, dim, Tensor<1, spacedim>> a3_cov_def_ds = L_derivs.get_a3_ds();
                    Tensor<1, dim, Tensor<1, spacedim>> a3_cov_def_da_ds = L_derivs.get_a3_da_ds();

                    
                    double stretch = 0.;
                    Tensor<1, dim> stretch_da;
                    Tensor<2,spacedim> force_resultants;
                    Tensor<2,spacedim> force_resultants_ds;
                    Tensor<1,dim,Tensor<1, spacedim>> moment_resultants;
                    for (unsigned int iq_1d = 0; iq_1d < Qthickness.size(); ++iq_1d) {
                        double u_t = Qthickness.get_points()[iq_1d][0];
                        double w_t = Qthickness.get_weights()[iq_1d];
                        Tensor<2, dim> gm_cov_ref; // g_i = r_{,i} , i = 1,2,3
                        double zeta = thickness * (u_t - 0.5);
                        Tensor<2,spacedim> g_cov_ref;
                        g_cov_ref[0] = a_cov_ref[0] + zeta * da3_ref[0];
                        g_cov_ref[1] = a_cov_ref[1] + zeta * da3_ref[1];
                        g_cov_ref[2] = cross_product_3d(g_cov_ref[0], g_cov_ref[1]);
                        double detJ_g_ref = g_cov_ref[2].norm();
                        double J_ratio = detJ_g_ref/detJ_ref;
        //                g_cov_ref[2] = g_cov_ref[2]/detJ_g;
                        g_cov_ref[2] = a_cov_ref[2]; // Kirchhoff-Love assumption
                        Tensor<2, dim, Tensor<1, spacedim>> dg_cov_def = da_cov_def + zeta * d2a3_cov_def; // g_{a,b} = a_{a,b} + zeta * a3_{,ab}
                        Tensor<1, dim, Tensor<1, spacedim>> g_cov_def_ds = a_cov_def_ds + zeta * a3_cov_def_da_ds; // g_{a,s} = a_{a,s} + zeta * a3_{,a},s
                        for (unsigned id = 0; id < spacedim; ++id) {
                            for (unsigned jd = 0; jd < spacedim; ++jd) {
                                gm_cov_ref[id][jd] += g_cov_ref[id] * g_cov_ref[jd];
                            }
                        }
                        auto gm_contra_ref = transpose(invert(gm_cov_ref));
                        Tensor<2, spacedim> g_cov_def = g_cov_ref;
                        g_cov_def[0] += u_der[0];
                        g_cov_def[1] += u_der[1];
                        Tensor<2, spacedim> gm_cov_def = metric_covariant(g_cov_def);
                        // for imcompressible material
                        double g_33 = determinant(gm_cov_ref)/determinant(gm_cov_def);
                        gm_cov_def[2][2] = g_33;
                        auto gm_contra_def = transpose(invert(gm_cov_def));
                        
                        stretch += sqrt(g_33) * w_t;

                        Tensor<3, dim, Tensor<1, spacedim>> dgm_cov_def; // gm_{ab,c}
                        Tensor<2, dim, Tensor<1, spacedim>> gm_cov_def_ds; // gm_{ab,s}
                        for (unsigned int ia = 0; ia < dim; ++ia){
                            for (unsigned int ib = 0; ib < dim; ++ib){
                                gm_cov_def_ds[ia][ib] += scalar_product(g_cov_def_ds[ia], g_cov_def[ib]) + scalar_product(g_cov_def_ds[ib], g_cov_def[ia]);
                                for (unsigned int ic = 0; ic < dim; ++ic){
                                    dgm_cov_def[ia][ib][ic] += scalar_product(dg_cov_def[ia][ic], g_cov_def[ib])
                                    + scalar_product(dg_cov_def[ib][ic], g_cov_def[ia]); // gm_{ab,c}
                                }
                            }
                        }
                        
                        Tensor<1,dim> sqrt_g33_da;
                        for (unsigned int ia = 0; ia < dim; ++ia) {
                            sqrt_g33_da[ia] = - determinant(gm_cov_ref) *
                            (gm_cov_def[1][1] * dgm_cov_def[0][0][ia] + gm_cov_def[0][0] * dgm_cov_def[1][1][ia]
                             -gm_cov_def[1][0] * dgm_cov_def[0][1][ia] - gm_cov_def[0][1] * dgm_cov_def[1][0][ia])/
                            (2.* pow((gm_cov_def[0][0] * gm_cov_def[1][1] - gm_cov_def[0][1] * gm_cov_def[1][0]), 2) * sqrt(g_33));
                        }
                        double sqrt_g33_ds = - determinant(gm_cov_ref) *
                        (gm_cov_def[1][1] * gm_cov_def_ds[0][0] + gm_cov_def[0][0] * gm_cov_def_ds[1][1]
                         -gm_cov_def[1][0] * gm_cov_def_ds[0][1] - gm_cov_def[0][1] * gm_cov_def_ds[1][0])/
                        (2.* pow((gm_cov_def[0][0] * gm_cov_def[1][1] - gm_cov_def[0][1] * gm_cov_def[1][0]), 2) * sqrt(g_33));
                        
                        Tensor<1, spacedim> g3_cov_def_ds = sqrt(g_33) * a3_cov_def_ds + sqrt_g33_ds * a_cov_def[2];
                        
                        Tensor<2,spacedim> tau = get_tau_mooney_rivlin(c_1, c_2, gm_contra_ref, gm_cov_def, gm_contra_def);
                        
                        for (unsigned int ia = 0; ia < dim; ++ia) {
                            force_resultants[ia] += thickness * tau * g_cov_def[ia] * J_ratio * w_t;
                            moment_resultants[ia] += thickness * tau * g_cov_def[ia] * zeta * J_ratio * w_t;
                        }
                        
                        g_cov_def[2] = sqrt(g_33) * a_cov_def[2]; // now modify the g_3  = lambda_3 * a_3
                        force_resultants[2] += thickness * tau * g_cov_def[2] * J_ratio * w_t;
                    }//loop over thickness quadrature points
                }
            }
        }// loop over surface quadrature points
        
        force_rhs.add(local_dof_indices, cell_force_rhs);
        stiffness_matrix.add(local_dof_indices, local_dof_indices, cell_stiffness_matrix);
    } // loop over cells
}



template <int dim, int spacedim>
double Nonlinear_shell<dim, spacedim>::compute_residual(const double step_length)
{
    Vector<double> residual(dof_handler.n_dofs());
    Vector<double> evaluation_point(dof_handler.n_dofs());
    evaluation_point = present_solution;
    evaluation_point.add(step_length, newton_update);
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    
}



int main()
{
    const int dim = 2, spacedim = 3;
    
    std::string type = "s";
    
    return 0;
}
