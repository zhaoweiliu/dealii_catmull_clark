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
class material_class
{
public:
    virtual ~material_class()
    {}
    
    virtual std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> get_integral_tensors();
    
    virtual void update(const Tensor<1, dim, Tensor<1,spacedim>> delta_u_der, /* du_{,a} */
                        const Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2 /* du_{,ab} */);
    
    virtual Tensor<2, spacedim> get_deformed_covariant_bases();
    
    virtual Tensor<2, dim, Tensor<1,spacedim>> get_deformed_covariant_bases_deriv();
    
};



template<int dim, int spacedim>
class material_neo_hookean : public material_class<dim, spacedim>
{
public:
    material_neo_hookean(const double c1,
                         const double h,
                         const Tensor<2, spacedim> a_cov,
                         const Tensor<2, dim, Tensor<1,spacedim>> da_cov)
    :
    c_1(c1),
    thickness(h),
    a_cov_ref(a_cov),
    da_cov_ref(da_cov),
    am_cov_ref(metric_covariant(a_cov_ref)),
    am_contra_ref ( metric_contravariant(am_cov_ref))
    {
        da3_ref[0] = cross_product_3d(da_cov_ref[0][0], a_cov_ref[1]) + cross_product_3d(a_cov_ref[0], da_cov_ref[1][0])/a_cov_ref[2].norm();
        da3_ref[1] = cross_product_3d(da_cov_ref[0][1], a_cov_ref[1]) + cross_product_3d(a_cov_ref[0], da_cov_ref[1][1])/a_cov_ref[2].norm();
        a_cov_def = a_cov_ref;
        da_cov_def = da_cov_ref;
    }
    
    Tensor<2,dim>  get_tau(const double C_33,
                           const Tensor<2,spacedim> gm_contra_ref,
                           const Tensor<2,spacedim> gm_contra_def);
    
    Tensor<4,dim>  get_elastic_tensor(const double C_33,
                                      const Tensor<2,spacedim> gm_contra_def);
    
    void update(const Tensor<1, dim, Tensor<1,spacedim>> delta_u_der, /* delta u_{,a} */
                const Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2 /* delta u_{,ab} */) override
    {
        u_der += delta_u_der;
        u_der2 += delta_u_der2;
        for (unsigned int ia = 0; ia < dim; ++ia){
            a_cov_def[ia] += delta_u_der[ia]; // a_alpha = bar{a_alpha} + u_{,alpha}
            for (unsigned int ib = 0; ib < dim; ++ib){
                da_cov_def[ia][ib] += delta_u_der2[ia][ib]; // a_{alpha,\beta} = bar{a_{alpha,beta}} + u_{,alpha beta}
            }
        }
    };
    
    std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> get_integral_tensors() override;
    
    Tensor<2, spacedim> get_deformed_covariant_bases() override
    {
        return a_cov_def;
    }
    
    Tensor<2, dim, Tensor<1,spacedim>> get_deformed_covariant_bases_deriv() override
    {
        return da_cov_def;
    }
    
private:
    const double c_1;
    // covariant base  a_1, a_2, a_3;
    const Tensor<2, spacedim> a_cov_ref; // a_i = x_{,i} , i = 1,2,3
    // derivatives of covariant base a_1, a_2;
    const Tensor<2, dim, Tensor<1,spacedim>> da_cov_ref; // a_{i,j} = x_{,ij} , i,j = 1,2
    // deformed covariant base  a_1, a_2, a_3;
    Tensor<2, spacedim> a_cov_def; // a_i = x_{,i} , i = 1,2,3
    // deformed derivatives of covariant base a_1, a_2;
    Tensor<2, dim, Tensor<1,spacedim>> da_cov_def; // a_{i,j} = x_{,ij} , i,j = 1,2
    // derivatives of a_3
    const Tensor<1, dim, Tensor<1,spacedim>> da3_ref; // a_{3,i}, i = 1,2
    // covariant metric tensor
    const Tensor<2,spacedim> am_cov_ref;
    // contravariant metric tensor
    const Tensor<2,spacedim> am_contra_ref;
    // thickness of the shell
    const double thickness;
    
    const QGauss<dim-1> Qh = QGauss<dim-1>(2);
    
    Tensor<1, dim, Tensor<1,spacedim>> u_der = 0;
    
    Tensor<2, dim, Tensor<1,spacedim>> u_der2 = 0;
};



template<int dim, int spacedim>
Tensor<2,dim> material_neo_hookean<dim, spacedim> :: get_tau(const double C_33,
                                                             const Tensor<2,spacedim> gm_contra_ref,
                                                             const Tensor<2,spacedim> gm_contra_def)
{
    Tensor<2,spacedim> tau;
    for (unsigned int ia = 0; ia < dim; ++ia)
        for (unsigned int ib = 0; ib < dim; ++ib)
            tau[ia][ib] += c_1 * (gm_contra_ref[ia][ib] - C_33 * gm_contra_def[ia][ib]);
    
    return tau;
}



template<int dim, int spacedim>
Tensor<4,dim> material_neo_hookean<dim, spacedim> ::get_elastic_tensor(const double C_33,
                                                                       const Tensor<2,spacedim> gm_contra_def)
{
    Tensor<2,spacedim> elastic_tensor;
    for (unsigned int ia = 0; ia < dim; ++ia)
        for (unsigned int ib = 0; ib < dim; ++ib)
            for (unsigned int ic = 0; ic < dim; ++ic)
                for (unsigned int id = 0; id < dim; ++id)
                    elastic_tensor[ia][ib][ic][id] += c_1  * C_33 * (2 * gm_contra_def[ia][ib] * gm_contra_def[ic][id] + gm_contra_def[ia][ic] * gm_contra_def[ib][id] + gm_contra_def[ia][id] * gm_contra_def[ib][ic] );
    
    return elastic_tensor;
}



template<int dim, int spacedim>
std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>>
material_neo_hookean<dim, spacedim> :: get_integral_tensors()
{
    std::vector<Tensor<2,dim>> resultants(2);
    std::vector<Tensor<4,dim>> D_tensors(3);
    for (unsigned int iq_1d = 0; iq_1d < Qh.size(); ++iq_1d) {
        double u_t = Qh.get_points()[iq_1d][0];
        double w_t = Qh.get_weights()[iq_1d];
        double zeta = thickness * (u_t - 0.5);
        Tensor<2,spacedim> g_cov_ref;
        g_cov_ref[0] = a_cov_ref[0] + zeta * da3_ref[0];
        g_cov_ref[1] = a_cov_ref[1] + zeta * da3_ref[1];
        g_cov_ref[2] = cross_product_3d(g_cov_ref[0], g_cov_ref[1]);
        double J_ratio = g_cov_ref[2].norm()/a_cov_ref[2].norm();
        g_cov_ref[2] = a_cov_ref[2]; // Kirchhoff-Love assumption
        
        Tensor<2, dim> gm_cov_ref = metric_covariant(g_cov_ref); // gm_ab
        Tensor<2, dim> gm_contra_ref = metric_contravariant(gm_cov_ref);
        
        Tensor<2, spacedim> g_cov_def = g_cov_ref;
        g_cov_def[0] += u_der[0];
        g_cov_def[1] += u_der[1];
        Tensor<2, spacedim> gm_cov_def = metric_covariant(g_cov_def);
        auto gm_contra_def = metric_contravariant(gm_cov_def);
        
        // for incompressible material
        double g_33 = determinant(gm_cov_ref)/determinant(gm_cov_def); // J_0^{-2}
        
        Tensor<2, dim> stress_tensor = get_tau(g_33, gm_contra_ref, gm_contra_def);
        Tensor<4, dim> elastic_tensor = get_elastic_tensor(g_33, gm_contra_def);
        
        for (unsigned int ia = 0; ia < dim; ++ia) {
            for (unsigned int ib = 0; ib < dim; ++ib) {
                resultants[0][ia][ib] += stress_tensor[ia][ib] * thickness * J_ratio * w_t;
                resultants[1][ia][ib] += stress_tensor[ia][ib] * zeta * thickness * J_ratio * w_t;
                for (unsigned int ic = 0; ic < dim; ++ic) {
                    for (unsigned int id = 0; id < dim; ++id) {
                        D_tensors[0][ia][ib][ic][id] += elastic_tensor[ia][ib][ic][id] * J_ratio * thickness * w_t;
                        D_tensors[1][ia][ib][ic][id] += elastic_tensor[ia][ib][ic][id] * zeta * J_ratio * thickness * w_t;
                        D_tensors[2][ia][ib][ic][id] += elastic_tensor[ia][ib][ic][id] * zeta  * zeta * J_ratio * thickness * w_t;
                    }
                }
            }
        }
    }//loop over thickness quadrature points
    return std::make_pair(resultants, D_tensors);
}



template<int dim, int spacedim>
class material_mooney_rivlin : public material_class<dim, spacedim>
{
public:
    material_mooney_rivlin(const double c1,
                           const double c2,
                           const double h,
                           const Tensor<2, spacedim> a_cov,
                           const Tensor<2, dim, Tensor<1,spacedim>> da_cov)
    :
    c_1(c1),
    c_2(c2),
    thickness(h),
    a_cov_ref(a_cov),
    da_cov_ref(da_cov),
    am_cov_ref(metric_covariant(a_cov_ref)),
    am_contra_ref ( metric_contravariant(am_cov_ref))
    {
        da3_ref[0] = cross_product_3d(da_cov_ref[0][0], a_cov_ref[1]) + cross_product_3d(a_cov_ref[0], da_cov_ref[1][0])/a_cov_ref[2].norm();
        da3_ref[1] = cross_product_3d(da_cov_ref[0][1], a_cov_ref[1]) + cross_product_3d(a_cov_ref[0], da_cov_ref[1][1])/a_cov_ref[2].norm();
        a_cov_def = a_cov_ref;
        da_cov_def = da_cov_ref;
    }
    
    Tensor<2,dim>  get_tau(const double C_33,
                           const Tensor<2,spacedim> gm_contra_ref,
                           const Tensor<2,spacedim> gm_cov_def,
                           const Tensor<2,spacedim> gm_contra_def);
    
    Tensor<4,dim>  get_elastic_tensor(const double C_33,
                                      const Tensor<2,spacedim> gm_contra_ref,
                                      const Tensor<2,spacedim> gm_cov_def,
                                      const Tensor<2,spacedim> gm_contra_def);
    
    void update(const Tensor<1, dim, Tensor<1,spacedim>> delta_u_der, /* du_{,a} */
                const Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2 /* du_{,ab} */) override
    {
        u_der += delta_u_der;
        u_der2 += delta_u_der2;
        for (unsigned int ia = 0; ia < dim; ++ia){
            a_cov_def[ia] += delta_u_der[ia]; // a_alpha = bar{a_alpha} + u_{,alpha}
            for (unsigned int ib = 0; ib < dim; ++ib){
                da_cov_def[ia][ib] += delta_u_der2[ia][ib]; // a_{alpha,\beta} = bar{a_{alpha,beta}} + u_{,alpha beta}
            }
        }
    };
    
    std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> get_integral_tensors() override;
    
    Tensor<2, spacedim> get_deformed_covariant_bases() override
    {
        return a_cov_def;
    }
    
    Tensor<2, dim, Tensor<1,spacedim>> get_deformed_covariant_bases_deriv() override
    {
        return da_cov_def;
    }
    
private:
    const double c_1,c_2;
    // covariant base  a_1, a_2, a_3;
    const Tensor<2, spacedim> a_cov_ref; // a_i = x_{,i} , i = 1,2,3
    // derivatives of covariant base;
    const Tensor<2, dim, Tensor<1,spacedim>> da_cov_ref; // a_{i,j} = x_{,ij} , i,j = 1,2
    // deformed covariant base  a_1, a_2, a_3;
    Tensor<2, spacedim> a_cov_def; // a_i = x_{,i} , i = 1,2,3
    // deformed derivatives of covariant base a_1, a_2;
    Tensor<2, dim, Tensor<1,spacedim>> da_cov_def; // a_{i,j} = x_{,ij} , i,j = 1,2
    // derivatives of a_3
    const Tensor<1, dim, Tensor<1,spacedim>> da3_ref;
    // covariant metric tensor
    const Tensor<2,spacedim> am_cov_ref;
    // contravariant metric tensor
    const Tensor<2,spacedim> am_contra_ref;
    
    const double thickness;
    
    const QGauss<dim-1> Qh = QGauss<dim-1>(3);
    
    Tensor<1, dim, Tensor<1,spacedim>> u_der = 0;
    
    Tensor<2, dim, Tensor<1,spacedim>> u_der2 = 0;
    
};



template<int dim, int spacedim>
Tensor<2,dim> material_mooney_rivlin<dim,spacedim> :: get_tau(const double C_33,
                                                              const Tensor<2,spacedim> gm_contra_ref,
                                                              const Tensor<2,spacedim> gm_cov_def,
                                                              const Tensor<2,spacedim> gm_contra_def)
{
    Tensor<2,spacedim> tau;
    for (unsigned int ia = 0; ia < dim; ++ia)
        for (unsigned int ib = 0; ib < dim; ++ib)
            for (unsigned int ic = 0; ic < dim; ++ic)
                for (unsigned int id = 0; id < dim; ++id){
                    tau[ia][ib] += 2. * (c_1 * gm_contra_ref[ia][ib] + c_2 * ( gm_cov_def[ic][id] * gm_contra_ref[ic][id] * gm_contra_ref[ia][ib] -  gm_cov_def[ic][id] * gm_contra_ref[ia][ic] * gm_contra_ref[id][ib] )) - 2. * ( c_1 + c_2 * ( gm_cov_def[ic][id] * gm_contra_ref[ic][id] - gm_cov_def[ic][id] * gm_contra_ref[2][ic] * gm_contra_ref[id][2] ) ) * C_33 * gm_contra_def[ia][ib];
                }
    
    return tau;
}



template<int dim, int spacedim>
Tensor<4, dim> material_mooney_rivlin<dim,spacedim> ::get_elastic_tensor(const double C_33,
                                                                         const Tensor<2,spacedim> gm_contra_ref,
                                                                         const Tensor<2,spacedim> gm_cov_def,
                                                                         const Tensor<2,spacedim> gm_contra_def)
{
    Tensor<4, dim> elastic_tensor;
    Tensor<4, dim> d2psi_d2;
    Tensor<2, dim> dpsi_d33dab;
    double dpsi_d33;
    for (unsigned int ia = 0; ia < dim; ++ia) {
        for (unsigned int ib = 0; ib < dim; ++ib) {
            dpsi_d33dab[ia][ib] += c_2 * gm_contra_ref[ia][ib] - c_2 * gm_contra_ref[2][ia] * gm_contra_ref[2][ib];
            dpsi_d33 += c_1 + c_2 * ( gm_contra_ref[ia][ib] * gm_contra_ref[ia][ib] - gm_contra_ref[2][ia] * gm_cov_def[ia][ib] * gm_contra_ref[2][ib]  );
            for (unsigned int ic = 0; ic < dim; ++ic) {
                for (unsigned int id = 0; id < dim; ++id) {
                    d2psi_d2 += c_2 * gm_contra_ref[ia][ib] * gm_contra_ref[ic][id] - 0.5 * c_2 * (gm_contra_ref[ia][ic] * gm_contra_ref[ib][id] + gm_contra_ref[ia][id] * gm_contra_ref[ib][ic]);
                }
            }
        }
    }
    
    for (unsigned int ia = 0; ia < dim; ++ia) {
        for (unsigned int ib = 0; ib < dim; ++ib) {
            for (unsigned int ic = 0; ic < dim; ++ic) {
                for (unsigned int id = 0; id < dim; ++id) {
                    elastic_tensor[ia][ib][ic][id] += 4 * d2psi_d2[ia][ib][ic][id] - 4 * dpsi_d33dab[ia][ib] * C_33 * gm_contra_def[ic][id] - 4 * dpsi_d33dab[ic][id] * C_33 * gm_contra_def[ia][ib] + 2 * dpsi_d33 * C_33 * ( 2 * gm_contra_def[ia][ib] * gm_contra_def[ic][id] + gm_contra_def[ia][ic] * gm_contra_def[ib][id] + gm_contra_def[ia][id] * gm_contra_def[ic][ic] );
                }
            }
        }
    }
    
    return elastic_tensor;
}



template<int dim, int spacedim>
std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> material_mooney_rivlin< dim, spacedim >:: get_integral_tensors()
{
    std::vector<Tensor<2,dim>> resultants(2);
    std::vector<Tensor<4,dim>> D_tensors(3);
    for (unsigned int iq_1d = 0; iq_1d < Qh.size(); ++iq_1d) {
        double u_t = Qh.get_points()[iq_1d][0];
        double w_t = Qh.get_weights()[iq_1d];
        double zeta = thickness * (u_t - 0.5);
        Tensor<2,spacedim> g_cov_ref;
        g_cov_ref[0] = a_cov_ref[0] + zeta * da3_ref[0];
        g_cov_ref[1] = a_cov_ref[1] + zeta * da3_ref[1];
        g_cov_ref[2] = cross_product_3d(g_cov_ref[0], g_cov_ref[1]);
        double J_ratio = g_cov_ref[2].norm()/a_cov_ref[2].norm();
        g_cov_ref[2] = a_cov_ref[2]; // Kirchhoff-Love assumption
        
        Tensor<2, dim> gm_cov_ref = metric_covariant(g_cov_ref); // gm_ab
        Tensor<2, dim> gm_contra_ref = metric_contravariant(gm_cov_ref);
        
        Tensor<2, spacedim> g_cov_def = g_cov_ref;
        g_cov_def[0] += u_der[0];
        g_cov_def[1] += u_der[1];
        Tensor<2, spacedim> gm_cov_def = metric_covariant(g_cov_def);
        auto gm_contra_def = metric_contravariant(gm_cov_def);
        
        // for imcompressible material
        double g_33 = determinant(gm_cov_ref)/determinant(gm_cov_def); // J_0^{-2}
        
        Tensor<2, dim> stress_tensor = get_tau(g_33, gm_contra_ref, gm_contra_def);
        Tensor<4, dim> elastic_tensor = get_elastic_tensor(g_33, gm_contra_ref, gm_cov_def, gm_contra_def);
        
        for (unsigned int ia = 0; ia < dim; ++ia) {
            for (unsigned int ib = 0; ib < dim; ++ib) {
                resultants[0][ia][ib] += stress_tensor[ia][ib] * thickness * J_ratio * w_t;
                resultants[1][ia][ib] += stress_tensor[ia][ib] * zeta * thickness * J_ratio * w_t;
                for (unsigned int ic = 0; ic < dim; ++ic) {
                    for (unsigned int id = 0; id < dim; ++id) {
                        D_tensors[0][ia][ib][ic][id] += elastic_tensor[ia][ib][ic][id] * J_ratio * thickness * w_t;
                        D_tensors[1][ia][ib][ic][id] += elastic_tensor[ia][ib][ic][id] * zeta * J_ratio * thickness * w_t;
                        D_tensors[2][ia][ib][ic][id] += elastic_tensor[ia][ib][ic][id] * zeta  * zeta * J_ratio * thickness * w_t;
                    }
                }
            }
        }
    }//loop over thickness quadrature points
    return std::make_pair(resultants, D_tensors);
}



template<int dim, int spacedim>
class PointHistory
{
public:
    PointHistory()
    {}
    
    virtual ~PointHistory()
    {}
    
    void set_material_type(const std::string name){material_type = name;}
    
    void setup_cell_qp (const double h,
                        const Tensor<2, spacedim> a_cov,
                        const Tensor<2, dim, Tensor<1,spacedim>> da_cov,
                        const double c_1,
                        const double c_2 = 0)
    {
        if (material_type == "neo_hookean") {
            material.reset(new material_neo_hookean<dim,spacedim>(c_1, h, a_cov, da_cov));
        }else if(material_type == "mooney_rivlin"){
            material.reset(new material_mooney_rivlin<dim,spacedim>(c_1, c_2, h, a_cov, da_cov));
        }else{
            std::runtime_error("Material type does not supported.");
        }
    }
    
    void update_cell_qp(const Tensor<1, dim, Tensor<1,spacedim>> delta_u_der, /* du_{,a} */
                        const Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2 /* du_{,ab} */)
    {
        material->update(delta_u_der, delta_u_der2);
    }
    
    std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> get_integral_tensors(){
        return material->get_integral_tensors();
    }
    
    Tensor<2, spacedim> get_deformed_covariant_bases(){
        return material->get_deformed_covariant_bases();
    }
    
    Tensor<2, dim, Tensor<1,spacedim>> get_deformed_covariant_bases_deriv(){
        return material->get_deformed_covariant_bases_deriv();
    }
    
private:
    std::shared_ptr< material_class<dim,spacedim> > material;
    std::string material_type = "neo_hookean";
};



template<int dim, int spacedim>
class tangent_derivatives
{
public:
    tangent_derivatives(const double ishape_fun, const Tensor<1, spacedim> ishape_grad, const Tensor<2,spacedim> ishape_hessian, const double jshape_fun, const Tensor<1, spacedim> jshape_grad, const Tensor<2,spacedim> jshape_hessian, const Tensor<2, spacedim> a_cov, const Tensor<2, dim, Tensor<1,spacedim>> da_cov, const unsigned int dof_i, const unsigned int dof_j)
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
        
        Tensor<1, spacedim> a3_t_dr = cross_product_3d(a_cov_ar[0], a_cov[1]) + cross_product_3d(a_cov[0], a_cov_ar[1]);
        Tensor<1, spacedim> a3_t_ds = cross_product_3d(a_cov_as[0], a_cov[1]) + cross_product_3d(a_cov[0], a_cov_as[1]);
        double a3_bar_dr = scalar_product(a3_t, a3_t_dr)/a3_bar;
        double a3_bar_ds = scalar_product(a3_t, a3_t_ds)/a3_bar;
        
        Tensor<1, spacedim> a3_t_drs = cross_product_3d(a_cov_ar[0], a_cov_as[1]) + cross_product_3d(a_cov_as[0], a_cov_ar[1]);
        double a3_bar_drs = scalar_product(a3_t_ds, a3_t_dr)/ a3_bar + scalar_product(a3_t, a3_t_drs)/ a3_bar - a3_bar_ds* scalar_product(a3_t, a3_t_dr)/ (a3_bar * a3_bar);
        a3_dr = a3_t_dr / a3_bar - a3_t_dr * a3_t/ (a3_bar * a3_bar);
        a3_ds = a3_t_ds / a3_bar - a3_t_ds * a3_t/ (a3_bar * a3_bar);
        a3_drs = a3_t_drs / a3_bar - a3_bar_drs * a3_t /(a3_bar * a3_bar) - a3_bar_dr * a3_t_ds / (a3_bar * a3_bar) - a3_bar_ds * a3_t_dr / (a3_bar * a3_bar) + 2 * a3_bar_dr * a3_bar_ds * a3_t / (a3_bar * a3_bar * a3_bar);
        
        for (unsigned int ia = 0; ia < dim; ++ia) {
            for (unsigned int ib = 0; ib < dim; ++ib) {
                membrane_strain_dr[ia][ib] = 0.5 * ( scalar_product( a_cov_ar[ia], a_cov[ib]) +  scalar_product( a_cov_ar[ib], a_cov[ia]) );
                membrane_strain_ds[ia][ib] = 0.5 * ( scalar_product( a_cov_as[ia], a_cov[ib]) +  scalar_product( a_cov_as[ib], a_cov[ia]) );
                membrane_strain_drs[ia][ib] = 0.5 * ( scalar_product( a_cov_ar[ia], a_cov_ar[ib]) + scalar_product( a_cov_ar[ib], a_cov_ar[ia]) );
                
                bending_strain_dr[ia][ib] = - ( scalar_product(a_cov_ar[ia][ib], a_cov[2]) + scalar_product(da_cov[ia][ib], a3_dr) );
                bending_strain_ds[ia][ib] = - ( scalar_product(a_cov_as[ia][ib], a_cov[2]) + scalar_product(da_cov[ia][ib], a3_ds) );
                bending_strain_drs[ia][ib] = - ( scalar_product(a_cov_ar[ia][ib], a3_ds) + scalar_product(a_cov_as[ia][ib], a3_dr) + scalar_product(da_cov[ia][ib], a3_drs) );
            }
        }
    }
    
    Tensor<2, dim> get_membrane_strain_dr(){return membrane_strain_dr;};
    Tensor<2, dim> get_membrane_strain_ds(){return membrane_strain_ds;};
    Tensor<2, dim> get_membrane_strain_drs(){return membrane_strain_drs;};
    
    Tensor<2, dim> get_bending_strain_dr(){return bending_strain_dr;};
    Tensor<2, dim> get_bending_strain_ds(){return bending_strain_ds;};
    Tensor<2, dim> get_bending_strain_drs(){return bending_strain_drs;};
    
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
    
    Tensor<2, dim> membrane_strain_dr;
    Tensor<2, dim> membrane_strain_ds;
    Tensor<2, dim> membrane_strain_drs;
    
    Tensor<2, dim> bending_strain_dr;
    Tensor<2, dim> bending_strain_ds;
    Tensor<2, dim> bending_strain_drs;
    
    Tensor<1, spacedim> a3_dr;
    Tensor<1, spacedim> a3_ds;
    Tensor<1, spacedim> a3_drs;
    
    Tensor<1,spacedim> u_r, r_r, u_s, r_s;
};



template <int dim, int spacedim>
class Nonlinear_shell
{
public:
    Nonlinear_shell();
    ~Nonlinear_shell();
    void run();
private:
    void   set_mesh( std::string type );
    void   setup_system();
    void   assemble_system(const bool initial_step);
    void   solve();
    double compute_residual();
    double determine_step_length() const;
    
    Triangulation<dim,spacedim> mesh;
    hp::DoFHandler<dim,spacedim> dof_handler;
    hp::FECollection<dim,spacedim> fe_collection;
    hp::MappingCollection<dim,spacedim> mapping_collection;
    hp::QCollection<dim> q_collection;
    hp::QCollection<dim> boundary_q_collection;
    SparsityPattern      sparsity_pattern;
    CellDataStorage<typename Triangulation<dim,spacedim>::cell_iterator, PointHistory<dim,spacedim> > quadrature_point_history;
    
    std::string material_type = "neo_hookean";
    SparseMatrix<double> tangent_matrix;
    Vector<double> newton_update;
    Vector<double> present_solution;
    Vector<double> solution_disp;
    Vector<double> force_rhs;
    const double thickness = 0.01, density = 1000.;
    const double mu = 4.225e5, c_1 = 0.4375*mu, c_2 = 0.0625*mu;
    const QGauss<dim-1> Qthickness = QGauss<dim-1>(2);
};



template <int dim, int spacedim>
Nonlinear_shell<dim, spacedim>::~Nonlinear_shell()
{
    dof_handler.clear();
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: set_mesh( std::string type )
{
    if (type == "roof") {
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
        if (type == "sphere") {
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
            GridTools::scale(10., mesh);
        }else if (type == "cylinder"){
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
            if (type == "beam")
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
void Nonlinear_shell<dim, spacedim> :: setup_system()
{
    Vector<double> vec_values;
    catmull_clark_create_fe_quadrature_and_mapping_collections_and_distribute_dofs(dof_handler,fe_collection,vec_values,mapping_collection,q_collection,boundary_q_collection,3);
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    
    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern, constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    tangent_matrix.reinit(sparsity_pattern);
    solution_disp.reinit(dof_handler.n_dofs());
    force_rhs.reinit(dof_handler.n_dofs());
    present_solution.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: assemble_system(const bool initial_step)
{
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    FullMatrix<double> cell_tangent_matrix;
    Vector<double>     cell_force_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    //    std::vector<types::global_dof_index> fix_dof_indices;
    CellDataStorage<typename Triangulation<dim, spacedim>::cell_iterator,
    PointHistory<dim,spacedim> > quadrature_point_history;
    
    if(initial_step == true){
        unsigned int n_q_points = 4;
        std::cout << "    Setting up quadrature point data..." << std::endl;
        quadrature_point_history.initialize(mesh.begin_active(),
                                            mesh.end(),
                                            n_q_points);
    }
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        
        cell_tangent_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_tangent_matrix = 0;
        cell_force_rhs.reinit(dofs_per_cell);
        cell_force_rhs = 0;
        
        std::vector<std::shared_ptr<PointHistory<dim, spacedim>>> lqph = quadrature_point_history.get_data(cell);
        
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
            
            if(initial_step == true){
                lqph[q_point] -> set_material_type(material_type);
                lqph[q_point] -> setup_cell_qp(thickness, a_cov_ref, da_cov_ref, c_1);
            }
            
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
            
            if (initial_step == false) {lqph[q_point] -> update_cell_qp(u_der,u_der2);}
            
            std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> integral_tensors = lqph[q_point] -> get_integral_tensors();
            std::vector<Tensor<2,dim>> resultants = integral_tensors.first;
            Tensor<4,dim> D0 = integral_tensors.second[0];
            Tensor<4,dim> D1 = integral_tensors.second[1];
            Tensor<4,dim> D2 = integral_tensors.second[2];
            
            Tensor<2, spacedim> a_cov_def = lqph[q_point] -> get_deformed_covariant_bases();
            Tensor<2, dim, Tensor<1,spacedim>> da_cov_def = lqph[q_point] -> get_deformed_covariant_bases_deriv();

            for (unsigned int r_shape = 0; r_shape < dofs_per_cell; ++r_shape) {
                auto shape_r = shape_vec[r_shape];
                auto shape_r_der = shape_der_vec[r_shape];
                auto shape_r_der2 = shape_der2_vec[r_shape];
                
                Tensor<1,spacedim> u_r;
                Tensor<2, dim> membrane_strain_dr;
                Tensor<2, dim> bending_strain_dr;
                
                for (unsigned int s_shape = 0; s_shape < dofs_per_cell; ++s_shape) {
                    auto shape_s = shape_vec[s_shape];
                    auto shape_s_der = shape_der_vec[s_shape];
                    auto shape_s_der2 = shape_der2_vec[s_shape];
                    
                    tangent_derivatives<dim,spacedim> T_derivs(shape_r, shape_r_der, shape_r_der2, shape_s, shape_s_der, shape_s_der2, a_cov_def, da_cov_def, r_shape, s_shape);
                    u_r = T_derivs.get_u_r();
                    membrane_strain_dr = T_derivs.get_membrane_strain_dr();
                    bending_strain_dr  = T_derivs.get_bending_strain_dr();
                    Tensor<2, dim> membrane_strain_ds  = T_derivs.get_membrane_strain_ds();
                    Tensor<2, dim> bending_strain_ds   = T_derivs.get_bending_strain_ds();
                    Tensor<2, dim> membrane_strain_drs = T_derivs.get_membrane_strain_drs();
                    Tensor<2, dim> bending_strain_drs  = T_derivs.get_bending_strain_drs();
                    for (unsigned int ia = 0; ia < dim; ++ia) {
                        for (unsigned int ib = 0; ib < dim; ++ib) {
                            cell_tangent_matrix[r_shape][s_shape] += (membrane_strain_drs[ia][ib] * resultants[0][ia][ib] + bending_strain_drs * resultants[1][ia][ib]) * fe_values.JxW(q_point) ;
                            for (unsigned int ic = 0; ic < dim; ++ic) {
                                for (unsigned int id = 0; id < dim; ++id) {
                                    cell_tangent_matrix[r_shape][s_shape] += (membrane_strain_dr[ia][ib] * D0[ia][ib][ic][id] * membrane_strain_ds[ic][id]
                                                                              + bending_strain_dr[ia][ib] * D1[ia][ib][ic][id] * membrane_strain_ds[ic][id]
                                                                              + membrane_strain_dr[ia][ib] * D1[ia][ib][ic][id] * bending_strain_ds[ic][id]
                                                                              + bending_strain_dr[ia][ib] * D2[ia][ib][ic][id] * bending_strain_ds[ic][id])
                                                                            * fe_values.JxW(q_point);
                                }
                            }
                        }
                    }
                }
                
                for (unsigned int ia = 0; ia < dim; ++ia) {
                    for (unsigned int ib = 0; ib < dim; ++ib) {
                        cell_force_rhs[r_shape] += membrane_strain_dr[ia][ib] * resultants[0][ia][ib] * fe_values.JxW(q_point);
                    }
                }
                cell_force_rhs[r_shape] -= 1.0 * scalar_product(fe_values.normal_vector(q_point), u_r) * fe_values.Jxw(q_point); // p = 1.
            }
        }// loop over surface quadrature points
        force_rhs.add(local_dof_indices, cell_force_rhs);
        tangent_matrix.add(local_dof_indices, local_dof_indices, cell_tangent_matrix);
    } // loop over cells
}



template <int dim, int spacedim>
double Nonlinear_shell<dim, spacedim>::determine_step_length() const
{
    return 1.;
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::solve()
{
  SolverControl            solver_control(force_rhs.size(),
                                          force_rhs.l2_norm() * 1e-6);
  SolverCG<Vector<double>> solver(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(tangent_matrix, 1.2);
  solver.solve(tangent_matrix, newton_update, force_rhs, preconditioner);
//  hanging_node_constraints.distribute(newton_update);
  const double alpha = determine_step_length();
  solution_disp.add(alpha, newton_update);
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> ::run()
{
    std::string geometry_type = "sphere";
    set_mesh(geometry_type);
    setup_system();
    
}



int main()
{
    const int dim = 2, spacedim = 3;
    
    Nonlinear_shell<dim, spacedim> nonlinear_thin_shell;
    nonlinear_thin_shell.run();
    
    std::cout <<"finished.\n";
    
    return 0;
}
