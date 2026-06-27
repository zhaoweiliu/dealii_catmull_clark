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

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;

// Define a function to calculate the norm of a vector
double VTV(const Vector<double> vec)
{
    double s = 0;
    for (unsigned i = 0; i < vec.size(); ++i) {
        s += vec[i] * vec[i];
    }
    return s;
}


// Define a function to calculate the dot product of two vectors
double VTW(const Vector<double> v, const Vector<double> w)
{
    double s = 0;
    Assert(v.size() == w.size(),ExcInternalError());
    for (unsigned i = 0; i < v.size(); ++i) {
        s += v[i] * w[i];
    }
    return s;
}


// Define a function to calculate the cross product of two vectors
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



// Define a function to calculate the metric tensor in covariant form
Tensor<2, 2> metric_covariant(const Tensor<2, 3> a_cov)
{
    Tensor<2, 2> am_cov;
    for (unsigned int ii=0; ii<2; ++ii)
    {
        for (unsigned int jj=0; jj<2 ; ++jj)
        {
            am_cov[ii][jj] =scalar_product(a_cov[ii], a_cov[jj]);
        }
    }
    return am_cov;
}


// Define a function to calculate the metric tensor in contravariant form
Tensor<2, 2> metric_contravariant(const Tensor<2, 2> am_cov)
{
    return transpose(invert(am_cov));
}



template <int dim, int spacedim>
std::map<unsigned int,unsigned int> dofs_map(const unsigned int n_dofs, const std::vector<unsigned int> fixed_dofs){
    unsigned int un_fixed_dofs = n_dofs-fixed_dofs.size();
    std::vector<unsigned int> fixed_dofs_copy = fixed_dofs;
    std::map<unsigned int,unsigned int> dof_map;
    unsigned int new_indices = 0;
    for (unsigned int id = 0; id < n_dofs; ++id) {
        bool constrained = false;
        for (unsigned int ifd = 0; ifd < fixed_dofs_copy.size(); ++ifd) {
            if (id == fixed_dofs_copy[ifd]) {
                constrained = true;
                fixed_dofs_copy.erase(fixed_dofs_copy.begin()+ifd);
                break;
            }
        }
        if (constrained == false) {
            dof_map.insert(std::pair<unsigned int, unsigned int>(new_indices,id));
            ++new_indices;
        }
    }
    Assert(fixed_dofs_copy.size() == 0,ExcInternalError());
    Assert(dof_map.size() == un_fixed_dofs,ExcInternalError());
    return dof_map;
}



template <int dim, int spacedim>
LAPACKFullMatrix<double> constrain_dofs_matrix(const LAPACKFullMatrix<double> full_stiffness_matrix, const std::vector<unsigned int> fixed_dofs){
    const unsigned int n_dofs = full_stiffness_matrix.size(0);
    std::map<unsigned int,unsigned int> dof_map = dofs_map<dim, spacedim>(n_dofs, fixed_dofs);
    const unsigned int n_uncons_dofs = dof_map.size();
    LAPACKFullMatrix<double> cons_full_stiffness_matrix(n_uncons_dofs,n_uncons_dofs);
    for (std::map<unsigned int,unsigned int>::iterator iter_i = dof_map.begin(); iter_i != dof_map.end(); ++iter_i) {
        for (std::map<unsigned int,unsigned int>::iterator iter_j = dof_map.begin(); iter_j != dof_map.end(); ++iter_j) {
            cons_full_stiffness_matrix.set(iter_i->first, iter_j->first, full_stiffness_matrix(iter_i->second,iter_j->second));
        }
    }
    return cons_full_stiffness_matrix;
}



template <int dim, int spacedim>
Vector<double> constrain_rhs_vector(const Vector<double> rhs_vector, const std::vector<unsigned int> fixed_dofs){
    const unsigned int n_dofs = rhs_vector.size();
    std::map<unsigned int,unsigned int> dof_map = dofs_map<dim, spacedim>(n_dofs, fixed_dofs);
    const unsigned int n_uncons_dofs = dof_map.size();
    Vector<double> constrained_rhs_vector(n_uncons_dofs);
    for (std::map<unsigned int,unsigned int>::iterator iter_i = dof_map.begin(); iter_i != dof_map.end(); ++iter_i) {
        constrained_rhs_vector[iter_i->first] = rhs_vector[iter_i->second];
    }
    return constrained_rhs_vector;
}



template <int dim, int spacedim>
Vector<double> restore_rhs_vector(const Vector<double> constrained_rhs_vector, const unsigned int n_dofs, const std::vector<unsigned int> fixed_dofs){
    std::map<unsigned int,unsigned int> dof_map = dofs_map<dim, spacedim>(n_dofs, fixed_dofs);
    Vector<double> restored_rhs_vector(n_dofs);
    for (std::map<unsigned int,unsigned int>::iterator iter_i = dof_map.begin(); iter_i != dof_map.end(); ++iter_i) {
        restored_rhs_vector[iter_i->second] = constrained_rhs_vector[iter_i->first];
    }
    return restored_rhs_vector;
}



// vtk plotting function
void vtk_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>(), const double p_t = 0)
{
    
    //    auto verts = dof_handler.get_triangulation().get_vertices();
    
    const unsigned int ngridpts = 10;
    const unsigned int seg_n = ngridpts-1;
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
    vtkSmartPointer<vtkPoints> points = vtkPoints::New();
    vtkSmartPointer<vtkDoubleArray> function = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> function_2 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> normal = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> stretch = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> pressure = vtkDoubleArray::New();

    function->SetNumberOfComponents(3);
    function->SetName("disp");
    function->SetComponentName(0, "x");
    function->SetComponentName(1, "y");
    function->SetComponentName(2, "z");
    
    normal->SetNumberOfComponents(3);
    normal->SetName("normal");
    normal->SetComponentName(0, "x");
    normal->SetComponentName(1, "y");
    normal->SetComponentName(2, "z");
    
    stretch->SetNumberOfComponents(1);
    stretch->SetName("stretch");
    stretch->SetComponentName(0, "value");
    
    pressure->SetNumberOfComponents(1);
    pressure->SetName("pressure");
    pressure->SetComponentName(0, "value");
    
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
                std::vector<Tensor<1,3>> JJ_def(3);
                //                std::vector<Tensor<2,3>> JJ_grad(2);
                double sol = 0;
                for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
                {
                    double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
                    const auto shape_der = dof_handler.get_fe(cell->active_fe_index()).shape_grad(idof, {u,v});
                    
                    sol += shapes * solution[local_dof_indices[idof]];
                    spt[idof % 3] += shapes * vertices[local_dof_indices[idof]];
                    disp[idof % 3] += shapes * solution[local_dof_indices[idof]];
                    
                    JJ[0][idof % 3] += shape_der[0] * vertices[local_dof_indices[idof]];
                    JJ[1][idof % 3] += shape_der[1] * vertices[local_dof_indices[idof]];
                    JJ_def[0][idof % 3] += shape_der[0] * vertices[local_dof_indices[idof]] + shape_der[0] * solution[local_dof_indices[idof]];
                    JJ_def[1][idof % 3] += shape_der[1] * vertices[local_dof_indices[idof]] + shape_der[1] * solution[local_dof_indices[idof]];
                }
                double p = 0;
                if (potential.size() != 0){
                    for (unsigned int jdof = 0; jdof < dofs_per_cell/3; ++jdof) {
                        double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(jdof*3, {u,v});
                        p += shapes * potential[local_dof_indices[jdof*3]/3];
                    }
                }
                
                JJ[2] = cross_product_3d(JJ[0],JJ[1]);
                JJ_def[2] = cross_product_3d(JJ_def[0],JJ_def[1]);
                
                double detJ = JJ[2].norm();
                JJ[2] = JJ[2]/detJ;
                double detJ_def = JJ_def[2].norm();
                JJ_def[2] = JJ_def[2]/detJ_def;
                
                double principle_stretch = (detJ/detJ_def);
                
                double coordsdata [3] = {spt[0],spt[1],spt[2]};
                
                points->InsertPoint(sample_offset+count, coordsdata);
                
                function->InsertComponent(sample_offset+count, 0, disp[0]);
                function->InsertComponent(sample_offset+count, 1, disp[1]);
                function->InsertComponent(sample_offset+count, 2, disp[2]);
                if (potential.size() != 0)
                    function_2->InsertComponent(sample_offset+count, 0, p);
                
                normal->InsertComponent(sample_offset+count, 0, JJ[2][0]);
                normal->InsertComponent(sample_offset+count, 1, JJ[2][1]);
                normal->InsertComponent(sample_offset+count, 2, JJ[2][2]);
                
                stretch->InsertComponent(sample_offset+count, 0, principle_stretch);
                pressure->InsertComponent(sample_offset+count, 0, p_t);
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
    grid -> GetPointData() -> AddArray(normal);
    grid -> GetPointData() -> AddArray(stretch);
    grid -> GetPointData() -> AddArray(pressure);
    
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
    writer -> SetFileName(filename.c_str());
    writer -> SetInputData(grid);
    if (! writer -> Write()) {
        std::cout<<" Cannot write displacement vtu file! ";
    }
}



// vtk plotting function
void vtk_plot_wrinkle(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>(), const double p_t = 0)
{
    const unsigned int ngridpts = 10;
    const unsigned int seg_n = ngridpts-1;
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
    vtkSmartPointer<vtkPoints> points = vtkPoints::New();
    vtkSmartPointer<vtkDoubleArray> function = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> function_2 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> normal = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> stretch = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> pressure = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> I1_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> I2_array = vtkDoubleArray::New();

    // --- 1. 创建新的 VTK 数据数组用于全局 y 方向应变 ---
    // 创建C张量的所有分量数组
    vtkSmartPointer<vtkDoubleArray> C_00_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> C_01_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> C_10_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> C_11_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> C_22_array = vtkDoubleArray::New();

    // 创建e张量的所有分量数组
    vtkSmartPointer<vtkDoubleArray> e_00_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> e_01_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> e_10_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> e_11_array = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> e_22_array = vtkDoubleArray::New();
    
    // 设置C张量分量的数组名称
    C_00_array->SetNumberOfComponents(1);
    C_00_array->SetName("C_00");
    C_00_array->SetComponentName(0, "value");

    C_01_array->SetNumberOfComponents(1);
    C_01_array->SetName("C_01");
    C_01_array->SetComponentName(0, "value");

    C_10_array->SetNumberOfComponents(1);
    C_10_array->SetName("C_10");
    C_10_array->SetComponentName(0, "value");

    C_11_array->SetNumberOfComponents(1);
    C_11_array->SetName("C_11");
    C_11_array->SetComponentName(0, "value");

    C_22_array->SetNumberOfComponents(1);
    C_22_array->SetName("C_22");
    C_22_array->SetComponentName(0, "value");

    // 设置e张量分量的数组名称
    e_00_array->SetNumberOfComponents(1);
    e_00_array->SetName("e_00");
    e_00_array->SetComponentName(0, "value");

    e_01_array->SetNumberOfComponents(1);
    e_01_array->SetName("e_01");
    e_01_array->SetComponentName(0, "value");

    e_10_array->SetNumberOfComponents(1);
    e_10_array->SetName("e_10");
    e_10_array->SetComponentName(0, "value");

    e_11_array->SetNumberOfComponents(1);
    e_11_array->SetName("e_11");
    e_11_array->SetComponentName(0, "value");

    e_22_array->SetNumberOfComponents(1);
    e_22_array->SetName("e_22");
    e_22_array->SetComponentName(0, "value");
    
    // ... 其他数组的设置保持不变 ...
    function->SetNumberOfComponents(3);
    function->SetName("disp");
    function->SetComponentName(0, "x");
    function->SetComponentName(1, "y");
    function->SetComponentName(2, "z");
    
    normal->SetNumberOfComponents(3);
    normal->SetName("normal_def"); // 使用变形后的法向量
    normal->SetComponentName(0, "x");
    normal->SetComponentName(1, "y");
    normal->SetComponentName(2, "z");
    
    stretch->SetNumberOfComponents(1);
    stretch->SetName("stretch");
    stretch->SetComponentName(0, "value");
    
    pressure->SetNumberOfComponents(1);
    pressure->SetName("pressure");
    pressure->SetComponentName(0, "value");
    
    if (potential.size() != 0){
        function_2->SetNumberOfComponents(1);
        function_2->SetName("potential");
        function_2->SetComponentName(0, "value");
    }

    I1_array->SetNumberOfComponents(1);
    I1_array->SetName("I1"); 
    I1_array->SetComponentName(0, "value");
    
    I2_array->SetNumberOfComponents(1);
    I2_array->SetName("I2"); 
    I2_array->SetComponentName(0, "value");
        
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
                std::vector<Tensor<1,3>> JJ(3);      // 参考构型基向量 a_x_bar, a_η_bar, a_ξ_bar
                std::vector<Tensor<1,3>> JJ_def(3);  // 当前构型基向量 A_x, A_η, A_ξ
                
                for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
                {
                    double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
                    const auto shape_der = dof_handler.get_fe(cell->active_fe_index()).shape_grad(idof, {u,v});
                    
                    spt[idof % 3] += shapes * vertices[local_dof_indices[idof]];
                    disp[idof % 3] += shapes * solution[local_dof_indices[idof]];
                    
                    // 参考构型基向量
                    JJ[0][idof % 3] += shape_der[0] * vertices[local_dof_indices[idof]];
                    JJ[1][idof % 3] += shape_der[1] * vertices[local_dof_indices[idof]];
                    // 当前构型基向量
                    JJ_def[0][idof % 3] += shape_der[0] * (vertices[local_dof_indices[idof]] + solution[local_dof_indices[idof]]);
                    JJ_def[1][idof % 3] += shape_der[1] * (vertices[local_dof_indices[idof]] + solution[local_dof_indices[idof]]);
                }

                // 计算法向量
                JJ[2] = cross_product_3d(JJ[0],JJ[1]);
                JJ_def[2] = cross_product_3d(JJ_def[0],JJ_def[1]);
                
                double detJ = JJ[2].norm();
                JJ[2] = JJ[2]/detJ;
                double detJ_def = JJ_def[2].norm();
                JJ_def[2] = JJ_def[2]/detJ_def;
                
                double principle_stretch = ( detJ / detJ_def);

                // --- 2. 计算变形梯度 F 和左Cauchy-Green张量 B ---
                // a. 构造参考构型和当前构型的基向量矩阵 (3x3)
                // 注意：deal.II Tensor 是[row][col]，而向量是[component]
                // A_bar[i][j] 是第j个参考基向量的第i个分量
                Tensor<2, 3> A_bar; 
                Tensor<2, 3> A;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        A_bar[i][j] = JJ[j][i];
                        A[i][j] = JJ_def[j][i];
                    }
                    // 重新定义第三个基向量（未归一化的法向量）
                    A_bar[i][2] = JJ[2][i] * detJ;  // 恢复未归一化的参考构型法向量
                    A[i][2] = JJ_def[2][i] * detJ_def;  // 恢复未归一化的当前构型法向量
                }

                // b. 计算 F = A * A_bar^{-1}
                Tensor<2, 3> A_bar_inv = invert(A_bar);
                Tensor<2, 3> F;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            F[i][j] += A[i][k] * A_bar_inv[k][j];
                        }
                    }
                }

                // c. 计算左 Cauchy-Green 张量 B = F * F^T
                Tensor<2, 3> B;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            B[i][j] += F[i][k] * F[j][k];
                        }
                    }
                }

                // d. 计算 B 的逆
                Tensor<2, 3> B_inv = invert(B);

                // e. 计算 Euler-Almansi 应变张量 e
                Tensor<2, 3> e;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        e[i][j] = 0.5 * ((i == j ? 1.0 : 0.0) - B_inv[i][j]);
                    }
                }

                // --- (可选) 仍然计算不变量 I1, I2 ---
                Tensor<2, 2> am_cov;      // 当前构型的协变度量
                Tensor<2, 2> am_cov_bar;  // 参考构型的协变度量
                for (unsigned int i = 0; i < 2; ++i) {
                    for (unsigned int j = 0; j < 2; ++j) {
                        am_cov[i][j] = JJ_def[i] * JJ_def[j]; 
                        am_cov_bar[i][j] = JJ[i] * JJ[j];     
                    }
                }
                Tensor<2, 2> C = am_cov * invert(am_cov_bar);
                double C_33 = (detJ/detJ_def)*(detJ/detJ_def);
                double I1 = C[0][0] + C[1][1] + C_33;
                double trace_C_squared = C[0][0]*C[0][0] + C[0][1]*C[1][0] + C[1][0]*C[0][1] + C[1][1]*C[1][1] + C_33*C_33;
                double I2 = 0.5 * (I1 * I1 - trace_C_squared);

                double coordsdata [3] = {spt[0],spt[1],spt[2]};
                
                // --- 3. 将数据插入 VTK 数组 ---
                points->InsertPoint(sample_offset+count, coordsdata);
                
                function->InsertComponent(sample_offset+count, 0, disp[0]);
                function->InsertComponent(sample_offset+count, 1, disp[1]);
                function->InsertComponent(sample_offset+count, 2, disp[2]);

                normal->InsertComponent(sample_offset+count, 0, JJ_def[2][0]);
                normal->InsertComponent(sample_offset+count, 1, JJ_def[2][1]);
                normal->InsertComponent(sample_offset+count, 2, JJ_def[2][2]);
                
                stretch->InsertComponent(sample_offset+count, 0, principle_stretch);
                pressure->InsertComponent(sample_offset+count, 0, p_t);

                I1_array->InsertComponent(sample_offset+count, 0, I1);
                I2_array->InsertComponent(sample_offset+count, 0, I2);

                // 插入C张量的所有分量
                C_00_array->InsertComponent(sample_offset+count, 0, C[0][0]);
                C_01_array->InsertComponent(sample_offset+count, 0, C[0][1]);
                C_10_array->InsertComponent(sample_offset+count, 0, C[1][0]);
                C_11_array->InsertComponent(sample_offset+count, 0, C[1][1]);
                C_22_array->InsertComponent(sample_offset+count, 0, C_33);

                // 插入e张量的所有分量
                e_00_array->InsertComponent(sample_offset+count, 0, e[0][0]);
                e_01_array->InsertComponent(sample_offset+count, 0, e[0][1]);
                e_10_array->InsertComponent(sample_offset+count, 0, e[1][0]);
                e_11_array->InsertComponent(sample_offset+count, 0, e[1][1]);
                e_22_array->InsertComponent(sample_offset+count, 0, e[2][2]);

                ++count;
            }
        }
    }
    uint sampleindex = 0;
    //loop over elements
    for(int e = 0; e < numElem; ++e){
        for (unsigned int t = 0 ; t < seg_n; ++t){
            for (unsigned int s = 0; s < seg_n; ++s){
                vtkSmartPointer<vtkCell> cell_vtk = vtkQuad::New();
                cell_vtk -> GetPointIds() -> SetId(0, sampleindex + t * ngridpts + s);
                cell_vtk -> GetPointIds() -> SetId(1, sampleindex + t * ngridpts + s + 1);
                cell_vtk -> GetPointIds() -> SetId(2, sampleindex + (t + 1) * ngridpts + s + 1 );
                cell_vtk -> GetPointIds() -> SetId(3, sampleindex + (t + 1) * ngridpts + s);
                grid -> InsertNextCell (cell_vtk -> GetCellType(), cell_vtk -> GetPointIds());
            }
        }
        sampleindex += ngridpts * ngridpts;
    }
    grid -> SetPoints(points);
    grid -> GetPointData() -> AddArray(function);
    if (potential.size() != 0){
        grid -> GetPointData() -> AddArray(function_2);
    }
    grid -> GetPointData() -> AddArray(normal);
    grid -> GetPointData() -> AddArray(stretch);
    grid -> GetPointData() -> AddArray(pressure);
    grid -> GetPointData() -> AddArray(I1_array);
    grid -> GetPointData() -> AddArray(I2_array);

    // 添加C张量的所有分量
    grid -> GetPointData() -> AddArray(C_00_array);
    grid -> GetPointData() -> AddArray(C_01_array);
    grid -> GetPointData() -> AddArray(C_10_array);
    grid -> GetPointData() -> AddArray(C_11_array);
    grid -> GetPointData() -> AddArray(C_22_array);

    // 添加e张量的所有分量
    grid -> GetPointData() -> AddArray(e_00_array);
    grid -> GetPointData() -> AddArray(e_01_array);
    grid -> GetPointData() -> AddArray(e_10_array);
    grid -> GetPointData() -> AddArray(e_11_array);
    grid -> GetPointData() -> AddArray(e_22_array);

    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
    writer -> SetFileName(filename.c_str());
    writer -> SetInputData(grid);
    if (! writer -> Write()) {
        std::cout<<" Cannot write displacement vtu file! ";
    }
}



template<int dim, int spacedim>
class material_mooney_rivlin_elec
{
public:
    material_mooney_rivlin_elec(const double c1,
                                const double c2,
                                const double h,
                                const double elec_potential,
                                const Tensor<2, spacedim> a_cov,
                                const Tensor<2, dim, Tensor<1,spacedim>> da_cov)
    :
    c_1(c1),
    c_2(c2),
    thickness(h),
    elec_potential(elec_potential),
    a_cov_ref(a_cov),
    da_cov_ref(da_cov),
    am_cov_ref(metric_covariant(a_cov_ref)),
    am_contra_ref ( metric_contravariant(am_cov_ref))
    {
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
            da3_ref[i] = a3_t_da[i] / a3_bar -  ( a3_bar_da[i] * a3_t) / (a3_bar * a3_bar);
        }
        a_cov_def = a_cov_ref;
        da_cov_def = da_cov_ref;
    }
    
    Tensor<2,dim>  get_tau(const double C_33,
                           const Tensor<2,dim> gm_contra_ref,
                           const Tensor<2,dim> gm_cov_def,
                           const Tensor<2,dim> gm_contra_def);
    
    Tensor<4,dim>  get_elastic_tensor(const double C_33,
                                      const Tensor<2,dim> gm_contra_ref,
                                      const Tensor<2,dim> gm_cov_def,
                                      const Tensor<2,dim> gm_contra_def);
    
    void update(const Tensor<1, dim, Tensor<1,spacedim>> delta_u_der, /* du_{,a} */
                const Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2 /* du_{,ab} */)
    {
        u_der += delta_u_der;
        u_der2 += delta_u_der2;
        for (unsigned int ia = 0; ia < dim; ++ia){
            a_cov_def[ia] += delta_u_der[ia]; // a_alpha = bar{a_alpha} + u_{,alpha}
            for (unsigned int ib = 0; ib < dim; ++ib){
                da_cov_def[ia][ib] += delta_u_der2[ia][ib]; // a_{alpha,beta} = bar{a_{alpha,beta}} + u_{,alpha beta}
            }
        }
        a_cov_def[2] = cross_product_3d(a_cov_def[0], a_cov_def[1]);
        a_cov_def[2] = a_cov_def[2]/a_cov_def[2].norm();
    };
    
    std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> get_integral_tensors();
    
    Tensor<2, spacedim> get_deformed_covariant_bases()
    {
        return a_cov_def;
    }
    
    Tensor<2, dim, Tensor<1,spacedim>> get_deformed_covariant_bases_deriv()
    {
        return da_cov_def;
    }

    void reset_elec_load(const double new_elec_load)
    {
        elec_potential = new_elec_load;
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
    Tensor<1, dim, Tensor<1,spacedim>> da3_ref;
    // covariant metric tensor
    const Tensor<2,dim> am_cov_ref;
    // contravariant metric tensor
    const Tensor<2,dim> am_contra_ref;
    
    const double thickness;
    
    const QGauss<dim-1> Qh = QGauss<dim-1>(3);
    
    Tensor<1, dim, Tensor<1,spacedim>> u_der ;
    
    Tensor<2, dim, Tensor<1,spacedim>> u_der2;

    const double epsilon = 4.12e-11;
//    const double elec_potential = 1.5;
    double elec_potential;
};



template<int dim, int spacedim>
Tensor<2,dim> material_mooney_rivlin_elec<dim,spacedim> :: get_tau(const double C_33,
        const Tensor<2,dim> gm_contra_ref,
        const Tensor<2,dim> gm_cov_def,
        const Tensor<2,dim> gm_contra_def)
{
    Tensor<2,dim> tau;
    double trace_C = 0;
    for (unsigned int ic = 0; ic < dim; ++ic){
        for (unsigned int id = 0; id < dim; ++id){
            trace_C += gm_cov_def[ic][id] * gm_contra_ref[ic][id];
        }
    }
    trace_C += C_33;
    Tensor<2,dim> T1;
    for (unsigned int ia = 0; ia < dim; ++ia){
        for (unsigned int ib = 0; ib < dim; ++ib){
            for (unsigned int ic = 0; ic < dim; ++ic){
                for (unsigned int id = 0; id < dim; ++id){
                    T1[ia][ib] += gm_contra_ref[ia][ic] * gm_cov_def[ic][id] * gm_contra_ref[ib][id];
                }
            }
        }
    }
    for (unsigned int ia = 0; ia < dim; ++ia){
        for (unsigned int ib = 0; ib < dim; ++ib){
            // tau[ia][ib] += 2 * c_1 * gm_contra_ref[ia][ib] + 2 * c_2 * (trace_C * gm_contra_ref[ia][ib] - T1[ia][ib]) - 2 * (c_1 + c_2 * (trace_C - C_33)) * C_33 * gm_contra_def[ia][ib] ;
            // tau[ia][ib] += 2 * c_1 * gm_contra_ref[ia][ib] + 2 * c_2 * (trace_C * gm_contra_ref[ia][ib] - T1[ia][ib]) - 2 * (c_1 + c_2 * (trace_C - C_33) + electric_load/(C_33 * C_33)) * C_33 * gm_contra_def[ia][ib] ;
            tau[ia][ib] += 2 * c_1 * gm_contra_ref[ia][ib] + 2 * c_2 * (trace_C * gm_contra_ref[ia][ib] - T1[ia][ib]) - 2 * (c_1 + c_2 * (trace_C - C_33) ) * C_33 * gm_contra_def[ia][ib] - epsilon * (elec_potential * elec_potential)/(thickness*thickness*C_33) * gm_contra_def[ia][ib] ;
        }
    }
    
    return tau;
}



template<int dim, int spacedim>
Tensor<4, dim> material_mooney_rivlin_elec<dim,spacedim> ::get_elastic_tensor(const double C_33,
                                                                         const Tensor<2,dim> gm_contra_ref,
                                                                         const Tensor<2,dim> gm_cov_def,
                                                                         const Tensor<2,dim> gm_contra_def)
{
    double trace_C = 0;
    for (unsigned int ic = 0; ic < dim; ++ic){
        for (unsigned int id = 0; id < dim; ++id){
            trace_C += gm_cov_def[ic][id] * gm_contra_ref[ic][id];
        }
    }
    trace_C += C_33;
    
    Tensor<4, dim> elastic_tensor;
    Tensor<4, dim> d2psi_d2;
    Tensor<2, dim> dpsi_d33dab;
    double dpsi_d33 = c_1 + c_2 * (trace_C - C_33);
    for (unsigned int ia = 0; ia < dim; ++ia) {
        for (unsigned int ib = 0; ib < dim; ++ib) {
            dpsi_d33dab[ia][ib] += c_2 * gm_contra_ref[ia][ib];
            for (unsigned int ic = 0; ic < dim; ++ic) {
                for (unsigned int id = 0; id < dim; ++id) {
                    d2psi_d2[ia][ib][ic][id] += - 0.5 * c_2 * (gm_contra_ref[ia][ic] * gm_contra_ref[ib][id] + gm_contra_ref[ia][id] * gm_contra_ref[ib][ic] - 2 * gm_contra_ref[ia][ib] * gm_contra_ref[ic][id]);
                }
            }
        }
    }
    
    for (unsigned int ia = 0; ia < dim; ++ia) {
        for (unsigned int ib = 0; ib < dim; ++ib) {
            for (unsigned int ic = 0; ic < dim; ++ic) {
                for (unsigned int id = 0; id < dim; ++id) {
                    elastic_tensor[ia][ib][ic][id] += 4 * d2psi_d2[ia][ib][ic][id] - 4 * dpsi_d33dab[ia][ib] * C_33 * gm_contra_def[ic][id] - 4 * dpsi_d33dab[ic][id] * C_33 * gm_contra_def[ia][ib] - (2 * dpsi_d33 * C_33 + epsilon * elec_potential * elec_potential / (thickness*thickness*C_33))* ( gm_contra_def[ia][ib] * gm_contra_def[ic][id] - gm_contra_def[ia][ic] * gm_contra_def[ib][id] - gm_contra_def[ia][id] * gm_contra_def[ib][ic]) + gm_contra_def[ia][ib] * gm_contra_def[ic][id] * (6 * dpsi_d33 * C_33 - epsilon * elec_potential * elec_potential / (thickness*thickness*C_33));
                }
            }
        }
    }
    
    return elastic_tensor;
}



template<int dim, int spacedim>
std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>>
material_mooney_rivlin_elec<dim, spacedim> :: get_integral_tensors()
{
    std::vector<Tensor<2,dim>> resultants(2);
    std::vector<Tensor<4,dim>> D_tensors(3);
    for (unsigned int iq_1d = 0; iq_1d < Qh.size(); ++iq_1d) {
        double u_t = Qh.get_points()[iq_1d][0];
        double w_t = Qh.get_weights()[iq_1d];
        double zeta = thickness * (u_t - 0.5);
        Tensor<dim,spacedim> g_cov_ref;
        g_cov_ref[0] = a_cov_ref[0] + zeta * da3_ref[0];
        g_cov_ref[1] = a_cov_ref[1] + zeta * da3_ref[1];
        double J_ratio = cross_product_3d(g_cov_ref[0], g_cov_ref[1]).norm()/cross_product_3d(a_cov_ref[0],a_cov_ref[1]).norm();
        g_cov_ref[2] = a_cov_ref[2]; // Kirchhoff-Love assumption
        
        Tensor<2, dim> gm_cov_ref = metric_covariant(g_cov_ref); // gm_ab
        Tensor<2, dim> gm_contra_ref = metric_contravariant(gm_cov_ref);
        
        auto a_cov_def = a_cov_ref;
        auto da_cov_def = da_cov_ref;
        a_cov_def[0] += u_der[0];
        a_cov_def[1] += u_der[1];
        da_cov_def[0][0] += u_der2[0][0];
        da_cov_def[0][1] += u_der2[0][1];
        da_cov_def[1][0] += u_der2[1][0];
        da_cov_def[1][1] += u_der2[1][1];
        double a3_norm_ref = cross_product_3d(a_cov_ref[0], a_cov_ref[1]).norm();
        double a3_norm_def = cross_product_3d(a_cov_def[0], a_cov_def[1]).norm();
        auto a3_def =  cross_product_3d(a_cov_def[0], a_cov_def[1])/a3_norm_def;
        double l3 = a3_norm_ref/a3_norm_def;
        Tensor<2, dim> gm_cov_def;
        for (unsigned int ia = 0; ia < dim; ++ia) {
            for (unsigned int ib = 0; ib < dim; ++ib) {
                gm_cov_def[ia][ib] = scalar_product(a_cov_def[ia], a_cov_def[ib]) - 2 * zeta * l3 * scalar_product(da_cov_def[ia][ib], a3_def);
            }
        }
        Tensor<2, dim> gm_contra_def = metric_contravariant(gm_cov_def);
        
        // for incompressible material
        double g_33 = determinant(gm_cov_ref)/determinant(gm_cov_def); // J_0^{-2}
        
//        std::cout << "g_33 = " << g_33 << std::endl;
        
        Tensor<2, dim> stress_tensor = get_tau(g_33, gm_contra_ref, gm_cov_def, gm_contra_def);
        Tensor<4, dim> elastic_tensor = get_elastic_tensor(g_33, gm_contra_ref,gm_cov_def,gm_contra_def);
        
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
class PointHistory_MR
{
public:
    PointHistory_MR()
    {}
    
    virtual ~PointHistory_MR()
    {}
    
    std::string return_material_type(){return material_type;}
    
    void setup_cell_qp (const double h,
                        const double elec_load,
                        const Tensor<2, spacedim> a_cov,
                        const Tensor<2, dim, Tensor<1,spacedim>> da_cov,
                        const double c_1,
                        const double c_2)
    {
        material.reset(new material_mooney_rivlin_elec<dim,spacedim>(c_1, c_2, h, elec_load, a_cov, da_cov));
    }

    void update_cell_elec_load(const double new_elec_load)
    {
        material->reset_elec_load(new_elec_load);
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
    std::shared_ptr< material_mooney_rivlin_elec<dim,spacedim> > material;
    std::string material_type = "mooney_rivlin";
};



template<int dim, int spacedim>
class tangent_derivatives
{
public:
    tangent_derivatives(const double ishape_fun, const Tensor<1, dim> ishape_grad, const Tensor<2,dim> ishape_hessian, const double jshape_fun, const Tensor<1, dim> jshape_grad, const Tensor<2,dim> jshape_hessian, const Tensor<2, spacedim> a_cov, const Tensor<2, dim, Tensor<1,spacedim>> da_cov, const unsigned int dof_i, const unsigned int dof_j)
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
        u_r[r%3] = i_shape;
        r_r[r%3] = i_shape;

        u_s[s%3] = j_shape;
        r_s[s%3] = j_shape;
        
        for (unsigned int i = 0; i < dim; ++i) {
            a_cov_ar[i][r%3] = i_shape_deriv[i];
            a_cov_as[i][s%3] = j_shape_deriv[i];

            for (unsigned int j = 0; j < dim; ++j) {
                a_cov_abr[i][j][r%3] = i_shape_deriv2[i][j];
                a_cov_abs[i][j][s%3] = j_shape_deriv2[i][j];
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
            a3_da[i] = a3_t_da[i] / a3_bar -  ( a3_bar_da[i] * a3_t) / (a3_bar * a3_bar);
        }
        
        a3_t_dr = cross_product_3d(a_cov_ar[0], a_cov[1]) + cross_product_3d(a_cov[0], a_cov_ar[1]);
        a3_t_ds = cross_product_3d(a_cov_as[0], a_cov[1]) + cross_product_3d(a_cov[0], a_cov_as[1]);
        double a3_bar_dr = scalar_product(a3_t, a3_t_dr)/a3_bar;
        double a3_bar_ds = scalar_product(a3_t, a3_t_ds)/a3_bar;
        
        Tensor<1, spacedim> a3_t_drs = cross_product_3d(a_cov_ar[0], a_cov_as[1]) + cross_product_3d(a_cov_as[0], a_cov_ar[1]);
        double a3_bar_drs = scalar_product(a3_t_ds, a3_t_dr)/ a3_bar + scalar_product(a3_t, a3_t_drs)/ a3_bar - (a3_bar_ds * a3_bar_dr)/ a3_bar;
        a3_dr = a3_t_dr / a3_bar - a3_bar_dr * a3_t/ (a3_bar * a3_bar);
        a3_ds = a3_t_ds / a3_bar - a3_bar_ds * a3_t/ (a3_bar * a3_bar);
        a3_drs = a3_t_drs / a3_bar - a3_bar_drs * a3_t /(a3_bar * a3_bar) - a3_bar_dr * a3_t_ds / (a3_bar * a3_bar) - a3_bar_ds * a3_t_dr / (a3_bar * a3_bar) + 2 * a3_bar_dr * a3_bar_ds * a3_t / (a3_bar * a3_bar * a3_bar);
        
        for (unsigned int ia = 0; ia < dim; ++ia) {
            for (unsigned int ib = 0; ib < dim; ++ib) {
                membrane_strain_dr[ia][ib] = 0.5 * ( scalar_product( a_cov_ar[ia], a_cov[ib]) +  scalar_product( a_cov_ar[ib], a_cov[ia]) );
                membrane_strain_ds[ia][ib] = 0.5 * ( scalar_product( a_cov_as[ia], a_cov[ib]) +  scalar_product( a_cov_as[ib], a_cov[ia]) );
                membrane_strain_drs[ia][ib] = 0.5 * ( scalar_product( a_cov_ar[ia], a_cov_as[ib]) + scalar_product( a_cov_ar[ib], a_cov_as[ia]) );
                
                bending_strain_dr[ia][ib] = - ( scalar_product(a_cov_abr[ia][ib], a_cov[2]) + scalar_product(da_cov[ia][ib], a3_dr) );
                bending_strain_ds[ia][ib] = - ( scalar_product(a_cov_abs[ia][ib], a_cov[2]) + scalar_product(da_cov[ia][ib], a3_ds) );
                bending_strain_drs[ia][ib] = - ( scalar_product(a_cov_abr[ia][ib], a3_ds) + scalar_product(a_cov_abs[ia][ib], a3_dr) + scalar_product(da_cov[ia][ib], a3_drs) );
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
    Tensor<1,spacedim> get_a3_t_ds(){return a3_t_ds;};
    Tensor<1,spacedim> get_a3_ds(){return a3_ds;};


private:
    const double i_shape;
    const Tensor<1, dim> i_shape_deriv;
    const Tensor<2, dim> i_shape_deriv2;
    const double j_shape;
    const Tensor<1, dim> j_shape_deriv;
    const Tensor<2, dim> j_shape_deriv2;
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
    Tensor<1, spacedim> a3_t_dr;
    Tensor<1, spacedim> a3_t_ds;
    Tensor<1, spacedim> a3_drs;
    
    Tensor<1,spacedim> u_r, r_r, u_s, r_s;
};



template <int dim, int spacedim>
class Nonlinear_shell
{
public:
    Nonlinear_shell(Triangulation<dim,spacedim> &tria);
    ~Nonlinear_shell();
    void run();
    void fix_pressure();
    void add_perturbation(const double A, const double q);
private:
    void   setup_system();
    void   assemble_system(const bool first_load_step, const bool first_newton_step);
    void   assemble_boundary_mass_matrix_and_rhs(const bool first_newton_step);
    void   solve(const bool first_load_step);
    void   initialise_data(hp::FEValues<dim,spacedim> hp_fe_values);
    double get_error_residual();
    void   nonlinear_solver(const bool initial_step);
    void   make_constrains(const bool first_newton_step);

//    Triangulation<dim,spacedim> mesh;
    hp::DoFHandler<dim,spacedim> dof_handler;
    hp::FECollection<dim,spacedim> fe_collection;
    hp::MappingCollection<dim,spacedim> mapping_collection;
    hp::QCollection<dim> q_collection;
    hp::QCollection<dim> boundary_q_collection;
    SparsityPattern      sparsity_pattern;
    AffineConstraints<double> constraints;
    std::vector<PointHistory_MR<dim,spacedim>>  quadrature_point_history;
    std::string material_type = "MR";
    SparseMatrix<double> tangent_matrix;
    SparseMatrix<double> boundary_mass_matrix;
    Vector<double> solution_newton_update;
    Vector<double> present_solution;
    Vector<double> solution_increment_newton_step;
    Vector<double> solution_increment_load_step;
    Vector<double> internal_force_rhs;
    Vector<double> external_force_ctrl_rhs;
    Vector<double> external_force_const_rhs;
    Vector<double> external_force_rhs;
    Vector<double> residual_vector;
    Vector<double> boundary_value_rhs;
    Vector<double> boundary_edge_load_rhs;
    double weightload;
    bool eigen_analysis = false;

    Vector<double> bifurcation_disp;

    // 位移边界条件分布施加
    double present_displacement_bc;        // 当前加载步的位移边界条件
    double displacement_bc_increment;      // 位移边界条件增量
    double first_displacement_bc;         // 目标位移边界条件
    std::vector<types::boundary_id> displacement_boundary_ids;  // 施加位移的边界ID
    std::vector<unsigned int> displacement_directions;          // 位移方向 (0=x,1=y,2=z)
    std::vector<double> displacement_factors;                   // 各边界位移比例因子
    bool use_displacement_bc;               // 是否使用位移边界条件
    std::vector<types::global_dof_index> displacement_bc_dof_indices;  // 受位移边界条件约束的自由度索引


    Vector<double> vec_values;
    std::vector<types::global_dof_index> constrained_dof_indices;
    double f_load;
    double u_load;
    double elec_load;
    unsigned int total_q_points;
    const double tolerance = 1e-6;
    const double thickness = 0.0005;
    // const double mu = 4.225e3, c_1 = 0.4375*mu, c_2 = 0.0625*mu;
//    const double mu = 4.225e5, c_1 = 0.5*mu, c_2 = 0.;
    // const double mu = 80, c_1 = mu, c_2 = 0.25 * c_1;
    // const double mu = 56e3, c_1 = 8.96e3, c_2 = 19.04e3;
    const double mu = 56e3, c_1 = 28e3, c_2 = 0;
    const QGauss<dim-1> Qthickness = QGauss<dim-1>(2);
    const double penalty_factor = 1e30;
    const unsigned int max_load_step = 50;
    const unsigned int max_newton_step = 20;

    bool instability_check = false;
};


template <int dim, int spacedim>
Nonlinear_shell<dim, spacedim>::Nonlinear_shell(Triangulation<dim,spacedim> &tria)
:
dof_handler(tria)
{}



template <int dim, int spacedim>
Nonlinear_shell<dim, spacedim>::~Nonlinear_shell()
{
    dof_handler.clear();
}


template <int dim, int spacedim>
double Nonlinear_shell<dim, spacedim> :: get_error_residual(){
//    auto residual = force_rhs;
    // 将受约束的自由度的残差设为0
    for (unsigned int ic = 0; ic < constrained_dof_indices.size(); ++ic) {
        residual_vector[constrained_dof_indices[ic]] = 0;
    }
    // 将受位移边界条件约束的自由度的残差设为0
    for (unsigned int ic = 0; ic < displacement_bc_dof_indices.size(); ++ic) {
        residual_vector[displacement_bc_dof_indices[ic]] = 0;
    }
//    std::cout << "residual vector = " << force_rhs << std::endl;
    return residual_vector.l2_norm();
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::add_perturbation(const double A, const double q)
{
    // 遍历所有自由度
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
        // 只处理z方向的自由度（i%3 == 2）
        if (i % 3 == 2) {
            // 获取当前自由度对应的节点索引
            unsigned int node_index = i / 3;

            // 从vec_values中获取节点位置
            // 注意：vec_values中每个节点有3个分量（x,y,z）
            double node_y = vec_values[node_index * 3 + 1];

            // 计算扰动：w = A * sin(q * y)
            double perturbation = A * std::sin(q * node_y);

            // 将扰动添加到当前解中
            present_solution[i] += perturbation;
        }
    }
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: initialise_data(hp::FEValues<dim,spacedim> hp_fe_values)
{
    total_q_points = 0;
    std::cout << "Setting up quadrature point data..." << std::endl;
    for (const auto &cell : dof_handler.active_cell_iterators()){
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        total_q_points += fe_values.n_quadrature_points;
    }
    quadrature_point_history.resize(total_q_points);
    unsigned int history_index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators()){
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        cell->set_user_pointer(&quadrature_point_history[history_index]);
        history_index += fe_values.n_quadrature_points;
    }
    Assert(history_index == quadrature_point_history.size(),ExcInternalError());
    std::cout << "Finish setting up quadrature point data." << std::endl;
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::assemble_boundary_mass_matrix_and_rhs(const bool first_newton_step)
{
    boundary_mass_matrix = 0;
    boundary_value_rhs = 0;
    boundary_edge_load_rhs = 0;

    // 清空位移边界条件约束的自由度索引
    displacement_bc_dof_indices.clear();
    hp::FEValues<dim,spacedim> hp_fe_boundary_values(mapping_collection, fe_collection, boundary_q_collection, update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    FullMatrix<double> cell_b_mass_matrix;
    Vector<double> cell_b_rhs;
    Vector<double> cell_load_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        hp_fe_boundary_values.reinit(cell);
        const FEValues<dim, spacedim> &b_fe_values = hp_fe_boundary_values.get_present_fe_values();
        cell_b_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_b_rhs.reinit(dofs_per_cell);
        cell_load_rhs.reinit(dofs_per_cell);
        cell_b_mass_matrix = 0; cell_b_rhs = 0; cell_load_rhs = 0;
        if (b_fe_values.n_quadrature_points != 1)
        {
            for (unsigned int q_point = 0; q_point < b_fe_values.n_quadrature_points; ++q_point)
            {
                Point<spacedim> qpt = b_fe_values.quadrature_point(q_point);

                // covariant base  a_1, a_2, a_3;
                Tensor<2, spacedim> a_cov; // a_i = x_{,i} , i = 1,2,3
                auto jacobian_ref = b_fe_values.jacobian(q_point);
                
                for (unsigned int id = 0; id < spacedim; ++id){
                    a_cov[0][id] = jacobian_ref[id][0];
                    a_cov[1][id] = jacobian_ref[id][1];
                }
                double tol = 1e-12;

                // x=0边界上的约束：x和z方向自由度为0
                if(std::abs(qpt[0]) < tol){
                    double jxw;
                    if (b_fe_values.get_quadrature().point(q_point)[0] == 0 ) {
                        jxw = a_cov[1].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }else if (b_fe_values.get_quadrature().point(q_point)[1] == 0 ){
                        jxw = a_cov[0].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }

                    for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
                        // 记录受约束的自由度（x和z方向，即i_shape%3==0或i_shape%3==2）
                        if ((i_shape%3 == 0 || i_shape%3 == 2) && b_fe_values.shape_value(i_shape, q_point) > tol) {
                            displacement_bc_dof_indices.push_back(local_dof_indices[i_shape]);
                        }

                        // x方向约束：罚函数项
                        for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                            if (i_shape%3 == 0 && j_shape%3 == 0){
                                cell_b_mass_matrix[i_shape][j_shape] +=
                                    penalty_factor *
                                    b_fe_values.shape_value(i_shape, q_point) *
                                    b_fe_values.shape_value(j_shape, q_point) *
                                    jxw;
                            }
                        }

                        // z方向约束：罚函数项
                        for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                            if (i_shape%3 == 2 && j_shape%3 == 2){
                                cell_b_mass_matrix[i_shape][j_shape] +=
                                    penalty_factor *
                                    b_fe_values.shape_value(i_shape, q_point) *
                                    b_fe_values.shape_value(j_shape, q_point) *
                                    jxw;
                            }
                        }

                        // 右端项：β * 0 * N_i = 0（x和z方向）
                        if (i_shape%3 == 0 || i_shape%3 == 2){
                            cell_b_rhs[i_shape] += 0;  // 位移为0
                        }
                    }
                }

                // y=0边界上的约束：y方向和z方向自由度为0（防止刚体运动）
                if(std::abs(qpt[1]) < tol){
                    double jxw;
                    if (b_fe_values.get_quadrature().point(q_point)[0] == 0 ) {
                        jxw = a_cov[1].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }else if (b_fe_values.get_quadrature().point(q_point)[1] == 0 ){
                        jxw = a_cov[0].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }

                    for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
                        // 记录受约束的自由度（y方向和z方向，即i_shape%3==1或i_shape%3==2）
                        if ((i_shape%3 == 1 || i_shape%3 == 2) && b_fe_values.shape_value(i_shape, q_point) > tol) {
                            displacement_bc_dof_indices.push_back(local_dof_indices[i_shape]);
                        }

                        for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                            // 罚函数项：β * N_i * N_j（y方向和z方向）
                            if ((i_shape%3 == 1 && j_shape%3 == 1) || (i_shape%3 == 2 && j_shape%3 == 2)){
                                cell_b_mass_matrix[i_shape][j_shape] +=
                                    penalty_factor *
                                    b_fe_values.shape_value(i_shape, q_point) *
                                    b_fe_values.shape_value(j_shape, q_point) *
                                    jxw;
                            }
                        }

                        // 右端项：β * 0 * N_i = 0（y方向和z方向）
                        if (i_shape%3 == 1 || i_shape%3 == 2){
                            cell_b_rhs[i_shape] += 0;  // y方向和z方向位移为0
                        }
                    }
                }

                // y=0.1边界上的约束：z方向自由度为0（防止刚体运动）
                if(std::abs(qpt[1] - 0.1) < tol){
                    double jxw;
                    if (b_fe_values.get_quadrature().point(q_point)[0] == 0 ) {
                        jxw = a_cov[1].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }else if (b_fe_values.get_quadrature().point(q_point)[1] == 0 ){
                        jxw = a_cov[0].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }

                    for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
                        // 记录受约束的自由度（z方向，即i_shape%3==2）
                        if ((i_shape%3 == 2) && b_fe_values.shape_value(i_shape, q_point) > tol) {
                            displacement_bc_dof_indices.push_back(local_dof_indices[i_shape]);
                        }

                        for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                            // 罚函数项：β * N_i * N_j（z方向）
                            if (i_shape%3 == 2 && j_shape%3 == 2){
                                cell_b_mass_matrix[i_shape][j_shape] +=
                                    penalty_factor *
                                    b_fe_values.shape_value(i_shape, q_point) *
                                    b_fe_values.shape_value(j_shape, q_point) *
                                    jxw;
                            }
                        }

                        // 右端项：β * 0 * N_i = 0（z方向）
                        if (i_shape%3 == 2){
                            cell_b_rhs[i_shape] += 0;  // y方向和z方向位移为0
                        }
                    }
                }

                // x=0.1边界上的约束：x方向施加位移，z方向为0
                
                if(std::abs(qpt[0] - 0.1) < tol){
                    double jxw;
                    if (b_fe_values.get_quadrature().point(q_point)[0] == 0 ) {
                        jxw = a_cov[1].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }else if (b_fe_values.get_quadrature().point(q_point)[1] == 0 ){
                        jxw = a_cov[0].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }

                    // 弱施加位移边界条件：在x=0.1边界上施加x正方向的位移，z方向为0
                    for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
                        // 记录受位移边界条件约束的自由度（x方向和z方向）
                        if ((i_shape%3 == 0 || i_shape%3 == 2) && b_fe_values.shape_value(i_shape, q_point) > tol) {
                        // if ((i_shape%3 == 2) && b_fe_values.shape_value(i_shape, q_point) > tol) {
                            displacement_bc_dof_indices.push_back(local_dof_indices[i_shape]);
                        }

                        // z方向约束：罚函数项
                        for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                            if (i_shape%3 == 2 && j_shape%3 == 2){
                                cell_b_mass_matrix[i_shape][j_shape] +=
                                    penalty_factor *
                                    b_fe_values.shape_value(i_shape, q_point) *
                                    b_fe_values.shape_value(j_shape, q_point) *
                                    jxw;
                            }
                        }

                        // // x方向位移边界条件：罚函数项
                        for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                            if (i_shape%3 == 0 && j_shape%3 == 0){
                                cell_b_mass_matrix[i_shape][j_shape] +=
                                    penalty_factor *
                                    b_fe_values.shape_value(i_shape, q_point) *
                                    b_fe_values.shape_value(j_shape, q_point) *
                                    jxw;
                            }
                        }

                        // // 右端项：β * ū * N_i
                        // if (i_shape%3 == 0){
                        //     cell_b_rhs[i_shape] += 
                        //         penalty_factor * 
                        //         (present_displacement_bc - 0.0 )* 
                        //         b_fe_values.shape_value(i_shape, q_point) * 
                        //         jxw;
                        // }
                        // 右端项：β * (ū - u_current) * N_i
                        if (i_shape%3 == 0){
                            // 获取当前位移
                            double current_displacement = 0.0;
                            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                                if (i % 3 == 0) {
                                    current_displacement += present_solution[local_dof_indices[i]] * b_fe_values.shape_value(i, q_point) + solution_increment_load_step[local_dof_indices[i]] * b_fe_values.shape_value(i, q_point) ;
                                }
                            }
                            // 计算目标位移和当前位移的差值
                            double displacement_diff = present_displacement_bc - current_displacement;
                            cell_b_rhs[i_shape] +=
                                penalty_factor *
                                displacement_diff *
                                b_fe_values.shape_value(i_shape, q_point) *
                                jxw;
                        }


                        // z方向位移为0
                        if (i_shape%3 == 2){
                            cell_b_rhs[i_shape] += 0;
                        }
                    }
                }
            }
            boundary_mass_matrix.add(local_dof_indices, cell_b_mass_matrix);
            boundary_value_rhs.add(local_dof_indices, cell_b_rhs);
            boundary_edge_load_rhs.add(local_dof_indices, cell_load_rhs);
        }
    }
    // 对位移边界条件约束的自由度索引进行排序和去重
    std::sort(displacement_bc_dof_indices.begin(), displacement_bc_dof_indices.end());
    auto last = std::unique(displacement_bc_dof_indices.begin(), displacement_bc_dof_indices.end());
    displacement_bc_dof_indices.erase(last, displacement_bc_dof_indices.end());
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::make_constrains(const bool first_newton_step){
// 注意：所有位移边界条件（包括防止刚体运动的约束）都已经在
// assemble_boundary_mass_matrix_and_rhs中使用罚函数法施加
// 此函数不再需要做任何处理
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::solve(const bool first_load_step)
{
    SparseDirectUMFPACK K_direct;
    K_direct.initialize(tangent_matrix);
    K_direct.vmult(solution_newton_update, residual_vector);
    std::cout << "solution_newton_update = " << solution_newton_update.l2_norm() << std::endl;
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: assemble_system(const bool first_load_step, const bool first_newton_step)
{
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    FullMatrix<double> cell_tangent_matrix;
    Vector<double>     cell_internal_force_rhs;
    Vector<double>     cell_external_force_ctrl_rhs;
    Vector<double>     cell_external_force_const_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    
    // 清零全局变量
    tangent_matrix = 0;
    internal_force_rhs = 0;
    external_force_ctrl_rhs = 0;
    external_force_const_rhs = 0;

    if(first_load_step == true && first_newton_step == true){
        initialise_data(hp_fe_values);
    }
    double area = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        
        cell_tangent_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_tangent_matrix = 0;
        cell_internal_force_rhs.reinit(dofs_per_cell);
        cell_internal_force_rhs = 0;
        cell_external_force_ctrl_rhs.reinit(dofs_per_cell);
        cell_external_force_ctrl_rhs = 0;
        cell_external_force_const_rhs.reinit(dofs_per_cell);
        cell_external_force_const_rhs = 0;

        PointHistory_MR<dim,spacedim> *lqph = reinterpret_cast<PointHistory_MR<dim,spacedim>*>(cell->user_pointer());
        Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
        Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());
        
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;++q_point)
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
            if(first_load_step == true && first_newton_step == true){
                lqph[q_point].setup_cell_qp(thickness, elec_load, a_cov_ref, da_cov_ref, c_1, c_2);
            }
            if(first_load_step == false && first_newton_step == true){
                lqph[q_point].update_cell_elec_load(elec_load);
            }
            
            std::vector<double> shape_vec(dofs_per_cell);
            std::vector<Tensor<1, dim>> shape_der_vec(dofs_per_cell);
            std::vector<Tensor<2, dim>> shape_der2_vec(dofs_per_cell);
            
            Tensor<1, dim, Tensor<1,spacedim>> delta_u_der; // u_{,a}
            Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2; // u_{,ab}
            
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
                    delta_u_der[ia][i_shape%3] += i_shape_der[ia] * solution_increment_newton_step(local_dof_indices[i_shape]); // u_{,a} = sum N^A_{,a} * U_A
                    if(first_newton_step == true){delta_u_der[ia][i_shape%3] += i_shape_der[ia] * solution_increment_load_step(local_dof_indices[i_shape]);} // u_{,a} = sum N^A_{,a} * U_A

                    for (unsigned int ib = 0; ib < dim; ++ib){
                        delta_u_der2[ia][ib][i_shape%3] += i_shape_der2[ia][ib] * solution_increment_newton_step(local_dof_indices[i_shape]); // u_{,ab} = sum N^A_{,ab} * U_A
                        if(first_newton_step == true){delta_u_der2[ia][ib][i_shape%3] += i_shape_der2[ia][ib] * solution_increment_load_step(local_dof_indices[i_shape]);}
                    }
                }
            }
            if (first_load_step == false || first_newton_step == false) {lqph[q_point].update_cell_qp(delta_u_der,delta_u_der2);}
            
            std::pair<std::vector<Tensor<2,dim>>, std::vector<Tensor<4,dim>>> integral_tensors = lqph[q_point].get_integral_tensors();
            std::vector<Tensor<2,dim>> resultants = integral_tensors.first;
            Tensor<4,dim> D0 = integral_tensors.second[0];
            Tensor<4,dim> D1 = integral_tensors.second[1];
            Tensor<4,dim> D2 = integral_tensors.second[2];
            
            Tensor<2, spacedim> a_cov_def = lqph[q_point].get_deformed_covariant_bases();
            double detJ_def = cross_product_3d(a_cov_def[0], a_cov_def[1]).norm();
            Tensor<2, dim, Tensor<1,spacedim>> da_cov_def = lqph[q_point].get_deformed_covariant_bases_deriv();

            for (unsigned int r_shape = 0; r_shape < dofs_per_cell; ++r_shape) {
                double shape_r = shape_vec[r_shape];
                Tensor<1, dim> shape_r_der = shape_der_vec[r_shape];
                Tensor<2, dim> shape_r_der2 = shape_der2_vec[r_shape];
                
                Tensor<1,spacedim> u_r;
                Tensor<1,spacedim> a3_t_s;
                Tensor<2, dim> membrane_strain_dr;
                Tensor<2, dim> bending_strain_dr;
                
                for (unsigned int s_shape = 0; s_shape < dofs_per_cell; ++s_shape) {
                    double shape_s = shape_vec[s_shape];
                    Tensor<1, dim> shape_s_der = shape_der_vec[s_shape];
                    Tensor<2, dim> shape_s_der2 = shape_der2_vec[s_shape];
                    
                    tangent_derivatives<dim,spacedim> T_derivs(shape_r, shape_r_der, shape_r_der2, shape_s, shape_s_der, shape_s_der2, a_cov_def, da_cov_def, r_shape, s_shape);
                    u_r = T_derivs.get_u_r();
                    a3_t_s = T_derivs.get_a3_t_ds();

                    membrane_strain_dr = T_derivs.get_membrane_strain_dr();
                    bending_strain_dr  = T_derivs.get_bending_strain_dr();
                    Tensor<2, dim> membrane_strain_ds  = T_derivs.get_membrane_strain_ds();
                    Tensor<2, dim> bending_strain_ds   = T_derivs.get_bending_strain_ds();
                    Tensor<2, dim> membrane_strain_drs = T_derivs.get_membrane_strain_drs();
                    Tensor<2, dim> bending_strain_drs  = T_derivs.get_bending_strain_drs();
                    
                    for (unsigned int ia = 0; ia < dim; ++ia) {
                        for (unsigned int ib = 0; ib < dim; ++ib) {
                            cell_tangent_matrix[r_shape][s_shape] += (membrane_strain_drs[ia][ib] * resultants[0][ia][ib] + bending_strain_drs[ia][ib] * resultants[1][ia][ib]) * fe_values.JxW(q_point) ;
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
                    // following pressure load
                    // cell_tangent_matrix[r_shape][s_shape] -= (lambda + pressure_increment_load_step) * reference_load * scalar_product(a3_t_s, u_r) * (1./detJ_ref) * fe_values.JxW(q_point);
                }
                for (unsigned int ia = 0; ia < dim; ++ia) {
                    for (unsigned int ib = 0; ib < dim; ++ib) {
                        cell_internal_force_rhs[r_shape] += (membrane_strain_dr[ia][ib] * resultants[0][ia][ib] + bending_strain_dr[ia][ib] * resultants[1][ia][ib]) * fe_values.JxW(q_point); // f^int
                    }
                }
                // cell_external_force_ctrl_rhs[r_shape] += reference_load * scalar_product(a_cov_def[2], u_r) * (detJ_def/detJ_ref) * fe_values.JxW(q_point); //  f^ext;
                if(r_shape%3 == 2){cell_external_force_const_rhs[r_shape] += weightload * thickness * shape_r * fe_values.JxW(q_point);} //  f^ext;
            }
            area += (detJ_def/detJ_ref) * fe_values.JxW(q_point);
        }// loop over surface quadrature points
        internal_force_rhs.add(local_dof_indices, cell_internal_force_rhs);
        external_force_const_rhs.add(local_dof_indices, cell_external_force_const_rhs);
        tangent_matrix.add(local_dof_indices, local_dof_indices, cell_tangent_matrix);
        
    } // loop over cells

    // 所有牛顿步都组装边界质量矩阵和右端项
    assemble_boundary_mass_matrix_and_rhs(first_newton_step);
    external_force_ctrl_rhs += boundary_value_rhs;
    external_force_ctrl_rhs += boundary_edge_load_rhs;

    // 将边界质量矩阵（罚函数项）添加到切线刚度矩阵中
    tangent_matrix.add(1.0, boundary_mass_matrix);
    
    std::sort(constrained_dof_indices.begin(), constrained_dof_indices.end());
    auto last = std::unique(constrained_dof_indices.begin(), constrained_dof_indices.end());
    constrained_dof_indices.erase(last, constrained_dof_indices.end());
    
    // 对constrained_dof_indices中的自由度施加罚函数约束（限制刚体运动）
    for (unsigned int idof = 0; idof < constrained_dof_indices.size(); ++idof) {
        // 检查当前自由度是否已经在位移边界条件中（避免双重罚函数）
        bool is_displacement_bc_dof = false;
        for (unsigned int ibc = 0; ibc < displacement_bc_dof_indices.size(); ++ibc) {
            if (constrained_dof_indices[idof] == displacement_bc_dof_indices[ibc]) {
                is_displacement_bc_dof = true;
                break;
            }
        }

        // 只对非位移边界条件的自由度使用罚函数法
        if (!is_displacement_bc_dof) {
            for (unsigned int jdof = 0; jdof < dof_handler.n_dofs(); ++jdof) {
                if (constrained_dof_indices[idof] == jdof){
                    tangent_matrix.set(constrained_dof_indices[idof], constrained_dof_indices[idof], penalty_factor);
                }
                else
                {
                    tangent_matrix.set(constrained_dof_indices[idof], jdof, 0);
                    tangent_matrix.set(jdof, constrained_dof_indices[idof], 0);
                }
            }
            residual_vector[constrained_dof_indices[idof]] = 0;
        }
    }

    // K \Delta u = - Residual vector
    // Residual vector = f^int - f^ext
    residual_vector = external_force_ctrl_rhs + external_force_const_rhs - internal_force_rhs;
    
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> :: setup_system()
{
    catmull_clark_create_fe_quadrature_and_mapping_collections_and_distribute_dofs(dof_handler,fe_collection,vec_values,mapping_collection,q_collection,boundary_q_collection,3);
    std::cout << "   Number of dofs: " << dof_handler.n_dofs()
    << std::endl;
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern, constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    std::ofstream out("CC_sparsity_pattern.svg");
    sparsity_pattern.print_svg(out);
    
    tangent_matrix.reinit(sparsity_pattern);
    boundary_mass_matrix.reinit(sparsity_pattern);
    solution_increment_newton_step.reinit(dof_handler.n_dofs());
    internal_force_rhs.reinit(dof_handler.n_dofs());
    external_force_ctrl_rhs.reinit(dof_handler.n_dofs());
    external_force_const_rhs.reinit(dof_handler.n_dofs());
    present_solution.reinit(dof_handler.n_dofs());
    solution_newton_update.reinit(dof_handler.n_dofs());
    boundary_value_rhs.reinit(dof_handler.n_dofs());
    boundary_edge_load_rhs.reinit(dof_handler.n_dofs());
    solution_increment_load_step.reinit(dof_handler.n_dofs());
    bifurcation_disp.reinit(dof_handler.n_dofs());
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> ::nonlinear_solver(const bool first_load_step){
    double initial_residual, residual_error;
    bool first_newton_step;
    for (unsigned int newton_iteration = 0; newton_iteration < max_newton_step; ++ newton_iteration)
    {
        std::cout << " newton_step = " << std::setw(2) << newton_iteration << " " << std::endl;
        double residual_norm = 0;
        if (newton_iteration == 0) {first_newton_step = true;}else{first_newton_step = false;}
        assemble_system(first_load_step, first_newton_step);
        make_constrains(first_newton_step);

        if (first_newton_step == true) {
            residual_error = 1.;
        }
        else
        {
            if (newton_iteration == 1) {
                initial_residual = get_error_residual();
            }
            residual_norm = get_error_residual();
            residual_error = residual_norm / initial_residual;
        }
        std::cout << "residual = " << residual_norm << std::endl;
        
        if (newton_iteration != 0) {
            std::cout << "residual_error = " << residual_error * 100 << "%" <<std::endl;
        }

        if ((residual_error < 1e-4 ) && solution_newton_update.l2_norm() < 1e-6) {
            std::cout << "converged.\n";
            if (eigen_analysis == true)
            {
                // eigenvalues and eigenmodes
                LAPACKFullMatrix<double> full_tangent(dof_handler.n_dofs());
                full_tangent = tangent_matrix;
                // 合并位移边界条件和刚体约束自由度
                std::vector<types::global_dof_index> all_constrained_dofs = displacement_bc_dof_indices;
                all_constrained_dofs.insert(all_constrained_dofs.end(), constrained_dof_indices.begin(), constrained_dof_indices.end());
                std::sort(all_constrained_dofs.begin(), all_constrained_dofs.end());
                auto last_all = std::unique(all_constrained_dofs.begin(), all_constrained_dofs.end());
                all_constrained_dofs.erase(last_all, all_constrained_dofs.end());

                LAPACKFullMatrix<double> reduced_full_tangent = constrain_dofs_matrix<dim, spacedim>(full_tangent, all_constrained_dofs);
                FullMatrix<double>       eigenvectors;
                Vector<double>           eigenvalues;
                Vector<double>           eigenvec(dof_handler.n_dofs()-all_constrained_dofs.size());
                std::vector<Vector<double>>           eigenvecs(0);
                reduced_full_tangent.compute_eigenvalues_symmetric(-10, 10, 1e-5, eigenvalues, eigenvectors);
                
                for(unsigned int ie = 0; ie < eigenvalues.size(); ++ie){
                    std::cout << eigenvalues[ie] << std::endl;
                    for (unsigned int idof = 0; idof < dof_handler.n_dofs()-all_constrained_dofs.size(); ++idof) {
                        eigenvec[idof] = eigenvectors[idof][ie];
                    }
                    Vector<double> restored_eigenvec = restore_rhs_vector<dim, spacedim>(eigenvec, dof_handler.n_dofs(), all_constrained_dofs);
                    eigenvecs.push_back(restored_eigenvec);
                    if (ie == 0 && eigenvalues[ie] < 0) {
                        vtk_plot("plate_eigen_"+std::to_string(ie)+".vtu", dof_handler, mapping_collection, vec_values, restored_eigenvec);
                        bifurcation_disp = eigenvecs[0];
                        // instability_check = true;
                    }
                }
            }

            tangent_matrix.reinit(sparsity_pattern);
            internal_force_rhs.reinit(dof_handler.n_dofs());
            external_force_ctrl_rhs.reinit(dof_handler.n_dofs());
            external_force_const_rhs.reinit(dof_handler.n_dofs());
            solution_newton_update.reinit(dof_handler.n_dofs());
            break;
        }else{
            solve(first_load_step);
        }
        std::cout << "solution_newton_update_norm = " << solution_newton_update.l2_norm() <<std::endl;
        
        solution_increment_newton_step = solution_newton_update;
        solution_increment_load_step += solution_newton_update;
        
        tangent_matrix.reinit(sparsity_pattern);
        internal_force_rhs.reinit(dof_handler.n_dofs());
        external_force_ctrl_rhs.reinit(dof_handler.n_dofs());
        external_force_const_rhs.reinit(dof_handler.n_dofs());
        solution_newton_update.reinit(dof_handler.n_dofs());
    }
}



template <int dim, int spacedim>
Triangulation<dim,spacedim> set_mesh( std::string type )
{
    Triangulation<dim,spacedim> mesh;
    if (type == "plate")
    {
        GridGenerator::hyper_cube(mesh, 0, 0.1);
        mesh.refine_global(4);
    }
    std::cout << "   Number of active cells: " << mesh.n_active_cells()
    << std::endl
    << "   Total number of cells: " << mesh.n_cells()
    << std::endl;
    std::ofstream output_file("test_mesh.vtu");
    GridOut().write_vtu (mesh, output_file);
    return mesh;
}


template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> ::run()
{
    setup_system();
    bool first_load_step;
    elec_load = 0.0;
    // weightload = 1000*9.8;

    // 初始化位移边界条件
    use_displacement_bc = true;  // 启用位移边界条件
    present_displacement_bc = 0.05;  // 初始位移为0
    const double target_elec_load = 4000;  // 第二阶段的电载荷
    

    // 第二阶段：不断增大电压，每一步检查失稳
    const unsigned int elec_steps = 50;  // 第二阶段加载步数
    const double elec_load_increment = target_elec_load / elec_steps;  // 电载荷增量
    
    
    for (unsigned int step = 0; step < elec_steps; ++step) {
        std::cout << "=== Second Phase Loading ===" << std::endl;
        std::cout << "load_step = " << step << std::endl;

        // 继续增加位移边界条件
        std::cout << "present_displacement_bc = " << present_displacement_bc << std::endl;

        // 逐步增加电载荷
        elec_load = (step + 1) * elec_load_increment;
        std::cout << "elec_load = " << elec_load << std::endl;

        // 标记是否为第一个电载荷加载步
        if(step == 0) {
            first_load_step = true;
        } else {
            first_load_step = false;
        }
        // 在基态上叠加扰动
        const double A = 1e-4;  // 扰动幅值
        const double q = 2 * numbers::PI / 0.0025;  // 波数（假设板长为0.1）
        std::cout << "=== Adding perturbation: w = " << A << " * sin(" << q << " * y) ===" << std::endl;
        add_perturbation(A, q);
        
        // 求解非线性方程
        // eigen_analysis = true;
        nonlinear_solver(first_load_step);
        present_solution += solution_increment_load_step;
        // solution_increment_load_step += 2e-2 * bifurcation_disp;

        
        
        // 输出结果
        vtk_plot_wrinkle("plate_MR_" + std::to_string(step) + ".vtu", dof_handler, mapping_collection, vec_values, present_solution, Vector<double>(), elec_load);
    }

    std::cout << "=== Electrical loading completed ===" << std::endl;
    std::cout << "Final elec_load: " << elec_load << std::endl;
    std::cout << "Final present_displacement_bc: " << present_displacement_bc << std::endl;
}


int main()
{
    const int dim = 2, spacedim = 3;
    Triangulation<dim,spacedim> mesh = set_mesh<dim,spacedim>("plate");
    Nonlinear_shell<dim, spacedim> nonlinear_thin_shell(mesh);
    nonlinear_thin_shell.run();
    
    std::cout <<"finished.\n";
    
    return 0;
}