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
#include <deal.II/lac/identity_matrix.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/arpack_solver.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/constrained_linear_operator.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>

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
#include "CatmullClark_subd.hpp"
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

void read_csv(std::string filename, std::vector<Vector<double>> &output, int ncol){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::vector<double>> result;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    int rowIdx = 0;
    // Read data, line by line
    while(std::getline(myFile,line)){
        std::stringstream lineStream(line);
        std::string element;
        std::vector<double> row_vec;
        while(std::getline(lineStream,element,','))
        {
            row_vec.push_back(std::stod(element));
        }
        result.push_back(row_vec);
        ++rowIdx;
    }
    output.resize(ncol);
    for (int icol = 0; icol < ncol; ++icol) {
        output[icol].reinit(rowIdx);
        for (int irow = 0; irow < rowIdx; ++irow) {
            output[icol][irow] = result[irow][icol];
        }
    }
    // Close file
    myFile.close();
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
    double product = 1;
    //     double product =  p[0];
    //   for (unsigned int d = 0; d < spacedim; ++d)
    //     product *= (p[d] + 1);
    return product;
}



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



Tensor<2, 2> metric_contravariant(const Tensor<2, 2> am_cov)
{
//    Tensor<2, 2> am_contrav;
//    double det2 = am_cov[0][0]*am_cov[1][1] - am_cov[1][0]*am_cov[0][1];
//    am_contrav[0][0] = am_cov[1][1]/det2;
//    am_contrav[1][1] = am_cov[0][0]/det2;
//    am_contrav[0][1] =-am_cov[0][1]/det2;
//    am_contrav[1][0] = am_contrav[0][1];
    return transpose(invert(am_cov));
}

void constitutive_fourth_tensors(Tensor<4, 2> &H_tensor, const Tensor<2, 2> g, const double poisson)
/* constitutive tensor (condensed s33=0)                    */
/*                                                          */
/* g       -->  contravariant metrictensor                  */
/* poisson -->  poisson ratio                               */
{
    for(unsigned int ii = 0; ii<2; ++ii){
        for(unsigned int jj = 0; jj<2; ++jj){
            for(unsigned int kk = 0; kk<2; ++kk){
                for(unsigned int ll = 0; ll<2; ++ll){
                    H_tensor[ii][jj][kk][ll] = poisson * g[ii][jj] * g[kk][ll] + 0.5 * (1. - poisson) * (g[ii][kk] * g[jj][ll] + g[ii][ll] * g[jj][kk]);
                }
            }
        }
    }
}



template<int dim, int spacedim>
void make_sparsity_pattern_for_coupling_and_laplace_matrices(hp::DoFHandler<dim, spacedim> &dof_handler, SparsityPattern &sparsity_pattern_couple, SparsityPattern &sparsity_pattern_laplace)
{
    DynamicSparsityPattern dsp1(dof_handler.n_dofs()/spacedim, dof_handler.n_dofs());
    DynamicSparsityPattern dsp2(dof_handler.n_dofs()/spacedim);
    std::vector<types::global_dof_index> local_dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int id = 0; id < dofs_per_cell/spacedim; ++id) {
            for (unsigned int jd = 0; jd < dofs_per_cell/spacedim; ++jd){
                int row = local_dof_indices[spacedim * id]/spacedim;
                int col = local_dof_indices[spacedim * jd]/spacedim;
                dsp2.add(row,col);
                for (unsigned int k = 0; k < spacedim; ++k) {
                    dsp1.add( row, spacedim * col + k);
                }
            }
        }
    }
    sparsity_pattern_couple.copy_from(dsp1);
    sparsity_pattern_laplace.copy_from(dsp2);
}



void constitutive_tensors(Tensor<2, 3> &cn, Tensor<2, 3> &cm, const Tensor<2, 2> g, const double h, const double young, const double poisson)
/* constitutive tensor (condensed s33=0)                    */
/*                                                          */
/* g       -->  contravariant metrictensor                  */
/* cn,cm   -->  constitutive tensors                        */
/* propm   -->  material data                               */
/* young   -->  young's modulus                             */
/* poisson -->  poisson ratio                               */
{
  double xa, xb, d, b, bdd;
                        
//  gmod = young/(2.0*(1.0+poisson));
         
  xa = (1.0-poisson)/2.0;
  xb = (1.0+poisson)/2.0;
  d  = young*h/(1.0-poisson*poisson);
  b  = young*h*h*h/(12.0*(1.0-poisson*poisson));

/* membrane part */

  cn[0][0] = d*g[0][0]*g[0][0];
  cn[0][1] = d*(xa*2.0*g[0][1]*g[0][1]+poisson*g[0][0]*g[1][1]);
  cn[0][2] = d*g[0][0]*g[0][1];
    
  cn[1][0] = d*(xa*2.0*g[0][1]*g[0][1]+poisson*g[0][0]*g[1][1]);
  cn[1][1] = d*g[1][1]*g[1][1];
  cn[1][2] = d*g[0][1]*g[1][1];

  cn[2][0] = d*g[0][0]*g[0][1];
  cn[2][1] = d*g[0][1]*g[1][1];
  cn[2][2] = d*(xa*g[0][0]*g[1][1]+xb*g[0][1]*g[0][1]);
 
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
 }



// void vtk_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>()){
    
// //    auto verts = dof_handler.get_triangulation().get_vertices();
    
//     const unsigned int ngridpts = 10;
//     const unsigned int seg_n = ngridpts-1;
//     vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
//     vtkSmartPointer<vtkPoints> points = vtkPoints::New();
//     vtkSmartPointer<vtkDoubleArray> function = vtkDoubleArray::New();
//     vtkSmartPointer<vtkDoubleArray> function_2 = vtkDoubleArray::New();
//     vtkSmartPointer<vtkDoubleArray> energy_n = vtkDoubleArray::New();
//     vtkSmartPointer<vtkDoubleArray> energy_m = vtkDoubleArray::New();
//     vtkSmartPointer<vtkDoubleArray> energy_total = vtkDoubleArray::New();


//     function->SetNumberOfComponents(3);
//     function->SetName("disp");
//     function->SetComponentName(0, "x");
//     function->SetComponentName(1, "y");
//     function->SetComponentName(2, "z");
    
//     if (potential.size() != 0){
//         function_2->SetNumberOfComponents(1);
//         function_2->SetName("potential");
//         function_2->SetComponentName(0, "value");
//     }
    
//     energy_n->SetNumberOfComponents(1);
//     energy_n->SetName("energy_n");
//     energy_n->SetComponentName(0, "value");
    
//     energy_m->SetNumberOfComponents(1);
//     energy_m->SetName("energy_m");
//     energy_m->SetComponentName(0, "value");
    
//     energy_total->SetNumberOfComponents(1);
//     energy_total->SetName("energy_total");
//     energy_total->SetComponentName(0, "value");
    
//     const double youngs = 1e6;
//     const double possions = 0.5;
//     const double thickness = 0.2;
    
//     int sample_offset = 0;
//     int count = 0;
//     double seg_length = 1./seg_n;
//     int numElem = dof_handler.get_triangulation().n_active_cells();
    
//     std::vector<types::global_dof_index> local_dof_indices;

//     for (auto cell=dof_handler.begin_active();cell!=dof_handler.end(); ++cell){
        
//         const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
//         local_dof_indices.resize(dofs_per_cell);
//         cell->get_dof_indices(local_dof_indices);
//         for(unsigned int iu = 0; iu < ngridpts; ++iu){
//             for(unsigned int iv = 0; iv < ngridpts; ++iv){
//                 double u = iu*seg_length;
//                 double v = iv*seg_length;
// //
//                 Point<3,double> spt = {0,0,0};
//                 Tensor<1,3,double> disp({0,0,0});
//                 Tensor<1,2,Tensor<1, 3>> disp_der;
//                 Tensor<2,2,Tensor<1, 3>> disp_der2;

//                 Tensor<2,3> JJ;
//                 Tensor<2,3,Tensor<1, 3>> J_der;

//                 double sol = 0;
//                 for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
//                 {
//                     double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
//                     const auto shape_der = dof_handler.get_fe(cell->active_fe_index()).shape_grad(idof, {u,v});
//                     const auto shape_der2 = dof_handler.get_fe(cell->active_fe_index()).shape_grad_grad(idof, {u,v});

                    
//                     sol += shapes * solution[local_dof_indices[idof]];
                    
//                     switch (idof % 3) {
//                         case 0:
//                             spt[0] += shapes * vertices[local_dof_indices[idof]];
//                             disp[0] += shapes * solution[local_dof_indices[idof]];
//                             disp_der[0][0] += shape_der[0] * solution[local_dof_indices[idof]];
//                             disp_der[1][0] += shape_der[1] * solution[local_dof_indices[idof]];
//                             disp_der2[0][0][0] += shape_der2[0][0] * solution[local_dof_indices[idof]];
//                             disp_der2[0][1][0] += shape_der2[0][1] * solution[local_dof_indices[idof]];
//                             disp_der2[1][0][0] += shape_der2[1][0] * solution[local_dof_indices[idof]];
//                             disp_der2[1][1][0] += shape_der2[1][1] * solution[local_dof_indices[idof]];
                            
//                             JJ[0][0] += shape_der[0] * vertices[local_dof_indices[idof]];
//                             JJ[1][0] += shape_der[1] * vertices[local_dof_indices[idof]];
//                             J_der[0][0][0] += shape_der2[0][0] * vertices[local_dof_indices[idof]];
//                             J_der[0][1][0] += shape_der2[0][1] * vertices[local_dof_indices[idof]];
//                             J_der[1][0][0] += shape_der2[1][0] * vertices[local_dof_indices[idof]];
//                             J_der[1][1][0] += shape_der2[1][1] * vertices[local_dof_indices[idof]];
                           
//                             break;
//                         case 1:
//                             spt[1] += shapes * vertices[local_dof_indices[idof]];
//                             disp[1] += shapes * solution[local_dof_indices[idof]];
//                             disp_der[0][1] += shape_der[0] * solution[local_dof_indices[idof]];
//                             disp_der[1][1] += shape_der[1] * solution[local_dof_indices[idof]];
//                             disp_der2[0][0][1] += shape_der2[0][0] * solution[local_dof_indices[idof]];
//                             disp_der2[0][1][1] += shape_der2[0][1] * solution[local_dof_indices[idof]];
//                             disp_der2[1][0][1] += shape_der2[1][0] * solution[local_dof_indices[idof]];
//                             disp_der2[1][1][1] += shape_der2[1][1] * solution[local_dof_indices[idof]];
                            
//                             JJ[0][1] += shape_der[0] * vertices[local_dof_indices[idof]];
//                             JJ[1][1] += shape_der[1] * vertices[local_dof_indices[idof]];
//                             J_der[0][0][1] += shape_der2[0][0] * vertices[local_dof_indices[idof]];
//                             J_der[0][1][1] += shape_der2[0][1] * vertices[local_dof_indices[idof]];
//                             J_der[1][0][1] += shape_der2[1][0] * vertices[local_dof_indices[idof]];
//                             J_der[1][1][1] += shape_der2[1][1] * vertices[local_dof_indices[idof]];
                        
//                             break;
//                         case 2:
//                             spt[2] += shapes * vertices[local_dof_indices[idof]];
//                             disp[2] += shapes * solution[local_dof_indices[idof]];
//                             disp_der[0][2] += shape_der[0] * solution[local_dof_indices[idof]];
//                             disp_der[1][2] += shape_der[1] * solution[local_dof_indices[idof]];
//                             disp_der2[0][0][2] += shape_der2[0][0] * solution[local_dof_indices[idof]];
//                             disp_der2[0][1][2] += shape_der2[0][1] * solution[local_dof_indices[idof]];
//                             disp_der2[1][0][2] += shape_der2[1][0] * solution[local_dof_indices[idof]];
//                             disp_der2[1][1][2] += shape_der2[1][1] * solution[local_dof_indices[idof]];
                            
//                             JJ[0][2] += shape_der[0] * vertices[local_dof_indices[idof]];
//                             JJ[1][2] += shape_der[1] * vertices[local_dof_indices[idof]];
//                             J_der[0][0][2] += shape_der2[0][0] * vertices[local_dof_indices[idof]];
//                             J_der[0][1][2] += shape_der2[0][1] * vertices[local_dof_indices[idof]];
//                             J_der[1][0][2] += shape_der2[1][0] * vertices[local_dof_indices[idof]];
//                             J_der[1][1][2] += shape_der2[1][1] * vertices[local_dof_indices[idof]];
                            
//                             break;
//                     }
//                 }
//                 double p = 0;
//                 if (potential.size() != 0){
//                     for (unsigned int jdof = 0; jdof < dofs_per_cell/3; ++jdof) {
//                         double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(jdof*3, {u,v});
//                         p += shapes * potential[local_dof_indices[jdof*3]/3];
//                     }
//                 }
                                
//                 JJ[2] = cross_product_3d(JJ[0],JJ[1]);
//                 double detJ = JJ[2].norm();
//                 JJ[2] = JJ[2]/detJ;
                
//                 // covariant metric tensor
//                 Tensor<2,2> am_cov = metric_covariant(JJ);
//                 // contravariant metric tensor
//                 Tensor<2,2> am_contra = metric_contravariant(am_cov);
//                 // H tensor
//                 Tensor<4,2> H_tensor;
//                 constitutive_fourth_tensors(H_tensor, am_contra, possions);
                
//                 Tensor<2,2> membrane_strain,bending_strain;
                
//                 for (unsigned int ii = 0; ii < 2; ++ii) {
//                     for (unsigned int jj = 0; jj < 2; ++jj) {
//                         membrane_strain[ii][jj] += 0.5 * scalar_product(JJ[ii], disp_der[jj])+scalar_product(JJ[jj], disp_der[ii]);
//                         bending_strain[ii][jj] += -scalar_product(disp_der2[ii][jj], JJ[2]) + (scalar_product(disp_der[0], cross_product_3d(J_der[ii][jj], JJ[1])) +  scalar_product(disp_der[1], cross_product_3d(JJ[0], J_der[ii][jj])))/detJ + scalar_product(JJ[2], J_der[ii][jj]) * (scalar_product(disp_der[0], cross_product_3d(JJ[1], JJ[2])) + scalar_product( disp_der[1], cross_product_3d(JJ[2], JJ[0])))/detJ;
//                     }
//                 }
                
//                 Tensor<2, 2> hn,hm;
//                 double c1 = youngs*thickness/ (1. - possions*possions),
//                 c2 = youngs*thickness*thickness*thickness/(12. * (1. - possions*possions));
//                 for(unsigned int ii = 0; ii < 2 ; ++ii)
//                 for(unsigned int jj = 0; jj < 2 ; ++jj)
//                 for(unsigned int kk = 0; kk < 2 ; ++kk)
//                 for(unsigned int ll = 0; ll < 2 ; ++ll)
//                 for (unsigned int id = 0; id < 3; ++id) {
//                     hn[ii][jj] += c1 * H_tensor[ii][jj][kk][ll] * membrane_strain[ll][kk];
//                     hm[ii][jj] += c2 * H_tensor[ii][jj][kk][ll] * bending_strain[ll][kk];
//                 }
                
//                 double sn = 0,sm = 0;
//                 for(unsigned int ii = 0; ii < 2 ; ++ii){
//                     for(unsigned int jj = 0; jj < 2 ; ++jj)
//                     {
//                         sn += membrane_strain[ii][jj] * hn[jj][ii];
//                         sm += bending_strain[ii][jj] * hm[jj][ii];

//                     }
//                 }
                
//                 double coordsdata [3] = {spt[0],spt[1],spt[2]};

//                 points->InsertPoint(sample_offset+count, coordsdata);
                                
//                 function->InsertComponent(sample_offset+count, 0, disp[0]);
//                 function->InsertComponent(sample_offset+count, 1, disp[1]);
//                 function->InsertComponent(sample_offset+count, 2, disp[2]);
                
//                 energy_n->InsertComponent(sample_offset+count, 0, sn);
//                 energy_m->InsertComponent(sample_offset+count, 0, sm);

//                 if (potential.size() != 0)
//                     function_2->InsertComponent(sample_offset+count, 0, p);
//                 ++count;
//             }
//         }
//     }
//     uint sampleindex = 0;
//     //loop over elements
//     for(int e = 0; e < numElem; ++e){
//         for (unsigned int t = 0 ; t < seg_n; ++t){
//             for (unsigned int s = 0; s < seg_n; ++s){
//                 vtkSmartPointer<vtkCell> cell = vtkQuad::New();
//                 cell -> GetPointIds() -> SetId(0, sampleindex + t * ngridpts + s);
//                 cell -> GetPointIds() -> SetId(1, sampleindex + t * ngridpts + s + 1);
//                 cell -> GetPointIds() -> SetId(2, sampleindex + (t + 1) * ngridpts + s + 1 );
//                 cell -> GetPointIds() -> SetId(3, sampleindex + (t + 1) * ngridpts + s);
//                 grid -> InsertNextCell (cell -> GetCellType(), cell -> GetPointIds());
//             }
//         }
//         sampleindex += ngridpts * ngridpts;
//     }
//     grid -> SetPoints(points);
//     grid -> GetPointData() -> AddArray(function);
//     grid -> GetPointData() -> AddArray(energy_n);
//     grid -> GetPointData() -> AddArray(energy_m);

//     if (potential.size() != 0){
//         grid -> GetPointData() -> AddArray(function_2);
//     }
//     vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
//     writer -> SetFileName(filename.c_str());
//     writer -> SetInputData(grid);
//     if (! writer -> Write()) {
//         std::cout<<" Cannot write displacement vtu file! ";
//     }
// }
void vtk_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>(), const double p_t = 0){
    
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
                
                double principle_stretch = sqrt(detJ_def / detJ);
                
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




template<int dim, int spacedim>
class material_data_linear_elastic
{
public:
    material_data_linear_elastic(const double youngs, const double possions, const double thickness,const Tensor<2, spacedim> a_cov,const Tensor<2, dim, Tensor<1,spacedim>> da_cov)
    :
    E(youngs),
    v(possions),
    h(thickness),
    a_cov_ref(a_cov),
    da_cov_ref(da_cov),
    am_cov_ref(metric_covariant(a_cov_ref)),
    am_contra_ref ( metric_contravariant(am_cov_ref))
    {
        a_cov_def = a_cov_ref;
        da_cov_def = da_cov_ref;
        am_cov_def = am_cov_ref;
        am_contra_def = am_contra_ref;
        
        memStiff = E*h/(1.-v*v);
        benStiff = E*h*h*h/(12. * (1.- v*v));
    }
    
    void update(const Tensor<1, dim, Tensor<1,spacedim>> delta_u_der, /* delta u_{,a} */
                const Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2 /* delta u_{,ab} */)
    {
        u_der += delta_u_der;
        u_der2 += delta_u_der2;
        for (unsigned int ia = 0; ia < dim; ++ia){
            a_cov_def[ia] += delta_u_der[ia]; // a_alpha = bar{a_alpha} + u_{,alpha}
            for (unsigned int ib = 0; ib < dim; ++ib){
                da_cov_def[ia][ib] += delta_u_der2[ia][ib]; // a_{alpha,\beta} = bar{a_{alpha,beta}} + u_{,alpha beta}
            }
        }
        a_cov_def[2] = cross_product_3d(a_cov_def[0], a_cov_def[1]);
        a_cov_def[2] = a_cov_def[2]/a_cov_def[2].norm();
        am_cov_def = metric_covariant(a_cov_def);
        am_contra_def = metric_contravariant(am_cov_def);
    };
    
    void store(){
       a_cov_def_stored = a_cov_def; // a_i = x_{,i} , i = 1,2,3
        // deformed derivatives of covariant base a_1, a_2;
       da_cov_def_stored = da_cov_def; // a_{i,j} = x_{,ij} , i,j = 1,2
        // covariant metric tensor (deformed)
       am_cov_def_stored = am_cov_def;
        // contravariant metric tensor(deformed)
       am_contra_def_stored = am_contra_def;
       u_der_stored = u_der;
       u_der2_stored = u_der2;
    }
    
    void restore(){
        a_cov_def = a_cov_def_stored; // a_i = x_{,i} , i = 1,2,3
         // deformed derivatives of covariant base a_1, a_2;
        da_cov_def = da_cov_def_stored; // a_{i,j} = x_{,ij} , i,j = 1,2
         // covariant metric tensor (deformed)
        am_cov_def = am_cov_def_stored;
         // contravariant metric tensor(deformed)
        am_contra_def = am_contra_def_stored;
        u_der = u_der_stored;
        u_der2 = u_der2_stored;
    }
    
    Tensor<4,dim> get_H_tensor(){
        Tensor<4,dim> H_tensor;
        for(unsigned int ii = 0; ii<dim; ++ii){
            for(unsigned int jj = 0; jj<dim; ++jj){
                for(unsigned int kk = 0; kk<dim; ++kk){
                    for(unsigned int ll = 0; ll<dim; ++ll){
                        H_tensor[ii][jj][kk][ll] = v * am_contra_def[ii][jj] * am_contra_def[kk][ll] + 0.5 * (1. - v) * (am_contra_def[ii][kk] * am_contra_def[jj][ll] + am_contra_def[ii][ll] * am_contra_def[jj][kk]);
                    }
                }
            }
        }
        return H_tensor;
    }
    
    std::pair<Tensor<2,dim>,Tensor<2,dim>> get_strians(){
        Tensor<2,dim> epsilon,kappa;
        for(unsigned int ii = 0; ii<dim; ++ii){
            for(unsigned int jj = 0; jj<dim; ++jj){
                epsilon[ii][jj] += 0.5 * (am_cov_def[ii][jj] - am_cov_ref[ii][jj]);
                kappa[ii][jj] -= scalar_product(da_cov_def[ii][jj],a_cov_def[2]) - scalar_product(da_cov_ref[ii][jj], a_cov_ref[2]);
            }
        }
        return std::make_pair(epsilon,kappa);
    }
    
    std::pair<Tensor<2,dim>,Tensor<2,dim>> get_stresses(){
        Tensor<2,dim> membrane,bending;
        auto strains = get_strians();
        Tensor<2,dim> epsilon = strains.first;
        Tensor<2,dim> kappa = strains.second;
        Tensor<4,dim> Htensor = get_H_tensor();
        for(unsigned int ii = 0; ii<dim; ++ii){
            for(unsigned int jj = 0; jj<dim; ++jj){
                for(unsigned int kk = 0; kk<dim; ++kk){
                    for(unsigned int ll = 0; ll<dim; ++ll){
                        membrane[ii][jj] += memStiff * Htensor[ii][jj][kk][ll] * epsilon[kk][ll];
                        bending[ii][jj] += benStiff * Htensor[ii][jj][kk][ll] * kappa[kk][ll];

                    }
                }
            }
        }
        return std::make_pair(membrane,bending);
    }
    
    Tensor<2, spacedim> get_deformed_covariant_bases()
    {
        return a_cov_def;
    }
    
    Tensor<2, dim, Tensor<1,spacedim>> get_deformed_covariant_bases_deriv()
    {
        return da_cov_def;
    }
    
    double get_membrane_stiffness(){
        return memStiff;
    }
    
    double get_bending_stiffness(){
        return benStiff;
    }
    
private:
    //material and thickness
    const double E,v,h;
    // covariant base  a_1, a_2, a_3;
    const Tensor<2, spacedim> a_cov_ref; // a_i = x_{,i} , i = 1,2,3
    // derivatives of covariant base a_1, a_2;
    const Tensor<2, dim, Tensor<1,spacedim>> da_cov_ref; // a_{i,j} = x_{,ij} , i,j = 1,2
    // covariant metric tensor
    const Tensor<2,dim> am_cov_ref;
    // contravariant metric tensor
    const Tensor<2,dim> am_contra_ref;
    // deformed covariant base  a_1, a_2, a_3;
    Tensor<2, spacedim> a_cov_def, a_cov_def_stored; // a_i = x_{,i} , i = 1,2,3
    // deformed derivatives of covariant base a_1, a_2;
    Tensor<2, dim, Tensor<1,spacedim>> da_cov_def, da_cov_def_stored; // a_{i,j} = x_{,ij} , i,j = 1,2
    // covariant metric tensor (deformed)
    Tensor<2,dim> am_cov_def, am_cov_def_stored;
    // contravariant metric tensor(deformed)
    Tensor<2,dim> am_contra_def, am_contra_def_stored;
        
    Tensor<1, dim, Tensor<1,spacedim>> u_der, u_der_stored;
    
    Tensor<2, dim, Tensor<1,spacedim>> u_der2, u_der2_stored;
    
    double memStiff,benStiff;
};



template<int dim, int spacedim>
class PointHistory
{
public:
    PointHistory()
    {}
    
    virtual ~PointHistory()
    {}
        
    void setup_cell_qp (const double youngs,
                        const double possions,
                        const double thickness,
                        const Tensor<2, spacedim> a_cov,
                        const Tensor<2, dim, Tensor<1,spacedim>> da_cov)
    {
        material.reset(new material_data_linear_elastic<dim,spacedim>(youngs,possions,thickness,a_cov,da_cov));
    }
    
    void update_cell_qp(const Tensor<1, dim, Tensor<1,spacedim>> delta_u_der, /* du_{,a} */
                        const Tensor<2, dim, Tensor<1,spacedim>> delta_u_der2 /* du_{,ab} */)
    {
        material->update(delta_u_der, delta_u_der2);
    }
    
    std::pair<Tensor<2,dim>,Tensor<2,dim>> get_stress_tensors(){
        return material->get_stresses();
    }
    
    Tensor<4,dim> get_H_tensor(){
        return material->get_H_tensor();
    }
    
    void store(){
        material->store();
    }
    
    void restore(){
        material->restore();
    }
    
    Tensor<2, spacedim> get_deformed_covariant_bases(){
        return material->get_deformed_covariant_bases();
    }
    
    Tensor<2, dim, Tensor<1,spacedim>> get_deformed_covariant_bases_deriv(){
        return material->get_deformed_covariant_bases_deriv();
    }
    
    double get_membrane_stiffness(){
        return material->get_membrane_stiffness();
    }
    
    double get_bending_stiffness(){
        return material->get_bending_stiffness();
    }
    
private:
    std::shared_ptr< material_data_linear_elastic<dim,spacedim> > material;
};



template <int dim, int spacedim>
class Nonlinear_shell
{
public:
    Nonlinear_shell(Triangulation<dim,spacedim> &tria);
    ~Nonlinear_shell();
    void run();
private:
    void   setup_system();
    void   assemble_system(const bool first_load_step = false, const bool first_newton_step = false);
    // void   assemble_boundary_force();
    void   assemble_boundary_mass_matrix_and_rhs();
    void   solve();
    void   initialise_data(hp::FEValues<dim,spacedim> hp_fe_values);
    double get_error_residual();
    void   nonlinear_solver(const bool initial_step = false);
    // void   perturbation_test();
    void   make_constrains();

//    Triangulation<dim,spacedim> mesh;
    hp::DoFHandler<dim,spacedim> dof_handler;
    hp::FECollection<dim,spacedim> fe_collection;
    hp::MappingCollection<dim,spacedim> mapping_collection;
    hp::QCollection<dim> q_collection;
    hp::QCollection<dim> boundary_q_collection;
    SparsityPattern      sparsity_pattern;
    AffineConstraints<double> constraints;
    std::vector<PointHistory<dim,spacedim>>  quadrature_point_history;
    
    SparseMatrix<double> tangent_matrix;
    SparseMatrix<double> boundary_mass_matrix;
    Vector<double> solution_newton_update;
    double pressure_newton_update;
    Vector<double> present_solution;
    double present_pressure;
    Vector<double> solution_increment_newton_step;
    Vector<double> solution_increment_load_step;
    double pressure_increment_newton_step;
    double pressure_increment_load_step;
    Vector<double> internal_force_rhs;
    Vector<double> external_force_rhs;
    Vector<double> residual_vector;
    // Vector<double> boundary_value_rhs;
    // Vector<double> boundary_edge_load_rhs;
    // Vector<double> force_rhs;
    // for arclength method
    double lambda;
    Vector<double> a_vector;
    double reference_pressure_VTV;
    double b,A;
    double psi_1 = 5e-7,psi_2 = 1, radius;

    Vector<double> vec_values;
    std::vector<types::global_dof_index> fix_dof_indices;
    double f_load;
    unsigned int total_q_points;
    const double tolerance = 1e-9;
    
    const double youngs = 1e6;
    const double possions = 0.5;
    const double thickness = 0.02;
    const double reference_pressure = 20;

    const unsigned int max_newton_step = 20;
    const unsigned int max_load_step = 100;
    
    bool store_material_data = false;
    bool restore_material_data = false;
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
    for (unsigned int ic = 0; ic < fix_dof_indices.size(); ++ic) {
        residual_vector[fix_dof_indices[ic]] = 0;
    }
    return residual_vector.l2_norm();
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> ::run()
{
    setup_system();
    bool first_load_step;
    for (unsigned int step = 0; step < max_load_step; ++step) {
        if(step < 5 ){
          f_load = 2 + step * 2;
            solution_increment_load_step = 0;
        }else{
            f_load += 2;
        }
        std::cout << "f_load = " << f_load << std::endl;
        std::cout << "step = "<< step << std::endl;
        if(step == 0){
            first_load_step = true;
        }else{
            first_load_step = false;
        }
        nonlinear_solver(first_load_step);
        present_solution += solution_increment_load_step;
//        perturbation_test();

        vtk_plot("cap_06_b_"+std::to_string(step)+".vtu", dof_handler, mapping_collection, vec_values, present_solution,Vector<double>(), f_load);

    }
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
    solution_increment_newton_step.reinit(dof_handler.n_dofs());
    solution_increment_load_step.reinit(dof_handler.n_dofs());
    internal_force_rhs.reinit(dof_handler.n_dofs());
    external_force_rhs.reinit(dof_handler.n_dofs());
    present_solution.reinit(dof_handler.n_dofs());
    solution_newton_update.reinit(dof_handler.n_dofs());
    // boundary_edge_load_rhs.reinit(dof_handler.n_dofs());
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim> ::nonlinear_solver(const bool first_load_step){
    double initial_residual, residual_error;
    bool first_newton_step;
    for (unsigned int newton_iteration = 0; newton_iteration < max_newton_step; ++ newton_iteration)
    {
        std::cout << " newton_step = " << std::setw(2) << newton_iteration << " " << std::endl;
        double residual_norm = 0;
        if (newton_iteration == 0? first_newton_step = true:first_newton_step = false);
        assemble_system(first_load_step, first_newton_step);
//        assemble_boundary_force();
//        residual_vector += boundary_edge_load_rhs;
        make_constrains();
        
        if (first_newton_step == true) {
            std::cout << "first newton iteration " << std::endl;
            residual_error = 1.;
        } else{
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
        if ((residual_error < 1e-2 ) && solution_newton_update.l2_norm() < 1e-9) {
            std::cout << "converged.\n";
            
//            LAPACKFullMatrix<double> full_tangent(dof_handler.n_dofs());
//            full_tangent = tangent_matrix;
//            LAPACKFullMatrix<double> reduced_full_tangent = constrain_dofs_matrix<dim, spacedim>(full_tangent, fix_dof_indices);
//            LAPACKFullMatrix<double> full_tangent_lu = reduced_full_tangent;
//            FullMatrix<double>       eigenvectors;
//            Vector<double>           eigenvalues;
//            Vector<double>           eigenvec(dof_handler.n_dofs()-fix_dof_indices.size());
//            std::vector<Vector<double>>           eigenvecs(0);
//            reduced_full_tangent.compute_eigenvalues_symmetric(-200, 200, 1e-5, eigenvalues, eigenvectors);
//
//            for(unsigned int ie = 0; ie < eigenvalues.size(); ++ie){
//                std::cout << eigenvalues[ie] << std::endl;
//                for (unsigned int idof = 0; idof < dof_handler.n_dofs()-fix_dof_indices.size(); ++idof) {
//                    eigenvec[idof] = eigenvectors[idof][ie];
//                }
//                Vector<double> restored_eigenvec = restore_rhs_vector<dim, spacedim>(eigenvec, dof_handler.n_dofs(), fix_dof_indices);
//                eigenvecs.push_back(restored_eigenvec);
//                if (eigenvalues[ie] <= 0) {
//                    vtk_plot("sphere_eigen_"+std::to_string(eigenvalues[ie])+".vtu", dof_handler, mapping_collection, vec_values, restored_eigenvec);
//                }
//            }
//            if(eigenvecs.size() >= 1){
//                vtk_plot("sphere_eigen_1.vtu", dof_handler, mapping_collection, vec_values, eigenvecs[0]);
//            }

            
            tangent_matrix.reinit(sparsity_pattern);
            internal_force_rhs.reinit(dof_handler.n_dofs());
            external_force_rhs.reinit(dof_handler.n_dofs());
            solution_newton_update.reinit(dof_handler.n_dofs());
            fix_dof_indices.resize(0);
            break;
        }else{
            solve();
        }
//        std::cout << "solution_newton_update_norm = " << newton_update.l2_norm() <<std::endl;
        solution_increment_newton_step = solution_newton_update;
        solution_increment_load_step += solution_newton_update;
        
        tangent_matrix.reinit(sparsity_pattern);
        internal_force_rhs.reinit(dof_handler.n_dofs());
        external_force_rhs.reinit(dof_handler.n_dofs());
        solution_newton_update.reinit(dof_handler.n_dofs());
        fix_dof_indices.resize(0);
    }
}



// template <int dim, int spacedim>
// void Nonlinear_shell<dim, spacedim>::assemble_boundary_force()
// {
//     boundary_edge_load_rhs = 0;
//     hp::FEValues<dim,spacedim> hp_fe_boundary_values(mapping_collection, fe_collection, boundary_q_collection, update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
//     Vector<double> cell_load_rhs;
//     std::vector<types::global_dof_index> local_dof_indices;
//     for (const auto &cell : dof_handler.active_cell_iterators())
//     {
//         const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
//         local_dof_indices.resize(dofs_per_cell);
//         cell->get_dof_indices(local_dof_indices);
//         hp_fe_boundary_values.reinit(cell);
//         const FEValues<dim, spacedim> &b_fe_values = hp_fe_boundary_values.get_present_fe_values();
//         cell_load_rhs.reinit(dofs_per_cell);
//         cell_load_rhs = 0;
//         if (b_fe_values.n_quadrature_points != 1)
//         {
//             for (unsigned int q_point = 0; q_point < b_fe_values.n_quadrature_points;
//                  ++q_point)
//             {
//                 Point<spacedim> qpt = b_fe_values.quadrature_point(q_point);
//                 // covariant base  a_1, a_2, a_3;
//                 Tensor<2, spacedim> a_cov; // a_i = x_{,i} , i = 1,2,3
//                 auto jacobian_ref = b_fe_values.jacobian(q_point);
                
//                 for (unsigned int id = 0; id < spacedim; ++id){
//                     a_cov[0][id] = jacobian_ref[id][0];
//                     a_cov[1][id] = jacobian_ref[id][1];
//                 }
//                 double tol = 1e-9;
//                 if(std::abs(qpt[0]) < tol){
//                     for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
//                         double jxw;
//                         if (b_fe_values.get_quadrature().point(q_point)[0] == 0 ) {
//                             jxw = a_cov[1].norm() * b_fe_values.get_quadrature().weight(q_point);
//                         }else if (b_fe_values.get_quadrature().point(q_point)[1] == 0 ){
//                             jxw = a_cov[0].norm() * b_fe_values.get_quadrature().weight(q_point);
//                         }
//                         if (i_shape%3 == 0){
//                             cell_load_rhs[i_shape] += 100 * b_fe_values.shape_value(i_shape, q_point) * jxw;
//                         }
//                     }
//                 }
//             }
//             boundary_edge_load_rhs.add(local_dof_indices, cell_load_rhs);
//         }
//     }
// }



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::solve()
{
  SolverControl            solver_control(50000, 1e-8);
  SolverCG<Vector<double>> solver(solver_control);
//  PreconditionSSOR<SparseMatrix<double>> preconditioner;
    PreconditionJacobi<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(tangent_matrix);
  solver.solve(tangent_matrix, solution_newton_update, residual_vector, preconditioner);
}



template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::make_constrains()
{
    std::sort(fix_dof_indices.begin(), fix_dof_indices.end());
    auto last = std::unique(fix_dof_indices.begin(), fix_dof_indices.end());
    fix_dof_indices.erase(last, fix_dof_indices.end());
    for (unsigned int idof = 0; idof <fix_dof_indices.size(); ++idof) {
        for (unsigned int jdof = 0; jdof <dof_handler.n_dofs(); ++jdof) {
            if (fix_dof_indices[idof] == jdof){
                tangent_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1);
            }
            else
            {
                tangent_matrix.set(fix_dof_indices[idof], jdof, 0);
                tangent_matrix.set(jdof, fix_dof_indices[idof], 0);
            }
        }
        residual_vector[fix_dof_indices[idof]] = 0;
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
void Nonlinear_shell<dim, spacedim> :: assemble_system(const bool first_load_step, const bool first_newton_step)
{
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    FullMatrix<double> cell_tangent_matrix;
    Vector<double>     cell_internal_force_rhs;
    Vector<double>     cell_external_force_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    if(first_load_step == true && first_newton_step == true){
        initialise_data(hp_fe_values);
    }
    bool load_added = false;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        // resize vectors and matrices
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        
        cell_tangent_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_tangent_matrix = 0;
        cell_internal_force_rhs.reinit(dofs_per_cell);
        cell_internal_force_rhs = 0;
        cell_external_force_rhs.reinit(dofs_per_cell);
        cell_external_force_rhs = 0;
        
        //Point history
        PointHistory<dim, spacedim> *lqph = reinterpret_cast<PointHistory<dim, spacedim>*>(cell->user_pointer());
        Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
        Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());
        
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
            if(first_load_step == true && first_newton_step == true){
                lqph[q_point].setup_cell_qp(youngs,possions,thickness,a_cov_ref, da_cov_ref);
            }
            if (store_material_data == true) {
                lqph[q_point].store();
            }else if (restore_material_data == true){
                lqph[q_point].restore();
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
                // add increments from previous newton step and load step
                for (unsigned int ia = 0; ia < dim; ++ia){
                    delta_u_der[ia][i_shape%3] += i_shape_der[ia] * solution_increment_newton_step(local_dof_indices[i_shape]); // u_{,a} = sum N^A_{,a} * U_A
                    if(first_newton_step == true){delta_u_der[ia][i_shape%3] += i_shape_der[ia] * solution_increment_load_step(local_dof_indices[i_shape]);} // u_{,a} = sum N^A_{,a} * U_A
                    for (unsigned int ib = 0; ib < dim; ++ib){
                        delta_u_der2[ia][ib][i_shape%3] += i_shape_der2[ia][ib] * solution_increment_newton_step(local_dof_indices[i_shape]); // u_{,ab} = sum N^A_{,ab} * U_A
                        if(first_newton_step == true){delta_u_der2[ia][ib][i_shape%3] += i_shape_der2[ia][ib] * solution_increment_load_step(local_dof_indices[i_shape]);}
                    }
                }
            }// loop over shapes for i
            
            if (first_load_step == false || first_newton_step == false) {lqph[q_point].update_cell_qp(delta_u_der,delta_u_der2);}
            
            auto stresses = lqph[q_point].get_stress_tensors();
            Tensor<2,dim> membrane_stress = stresses.first;
            Tensor<2,dim> bending_stress = stresses.second;
            Tensor<4,dim> H_tensor =lqph[q_point].get_H_tensor();
            double memstiff = lqph[q_point].get_membrane_stiffness();
            double benstiff = lqph[q_point].get_bending_stiffness();
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
                            cell_tangent_matrix[r_shape][s_shape] += ( membrane_strain_drs[ia][ib] * membrane_stress[ia][ib] + bending_strain_drs[ia][ib] * bending_stress[ia][ib]) * fe_values.JxW(q_point) ;
                            for (unsigned int ic = 0; ic < dim; ++ic) {
                                for (unsigned int id = 0; id < dim; ++id) {
                                    cell_tangent_matrix[r_shape][s_shape] += (membrane_strain_dr[ia][ib] * memstiff * H_tensor[ia][ib][ic][id] * membrane_strain_ds[ic][id] + bending_strain_dr[ia][ib] * benstiff * H_tensor[ia][ib][ic][id] * bending_strain_ds[ic][id]) * fe_values.JxW(q_point);
                                }
                            }
                        }
                    }
                    // following pressure load
                    // cell_tangent_matrix[r_shape][s_shape] -= -f_load * scalar_product(a3_t_s, u_r) * (1./detJ_ref) * fe_values.JxW(q_point);


                }// loop s_shape
                for (unsigned int ia = 0; ia < dim; ++ia) {
                    for (unsigned int ib = 0; ib < dim; ++ib) {
                        cell_internal_force_rhs[r_shape] += (membrane_strain_dr[ia][ib] * membrane_stress[ia][ib] + bending_strain_dr[ia][ib] * bending_stress[ia][ib]) * fe_values.JxW(q_point); // f^int
                    }
                }
                // cell_external_force_rhs[r_shape] += -f_load * scalar_product(a_cov_def[2], u_r) * (detJ_def/detJ_ref) * fe_values.JxW(q_point); //  f^ext;
                if(r_shape%3 == 0){
                    cell_external_force_rhs[r_shape] += 1 *  fe_values.JxW(q_point); //  f^ext;
                }
            
            }//loop r_shape
        }// loop over quadratures

        // internal_force_rhs.add(local_dof_indices, cell_internal_force_rhs);
        // external_force_rhs.add(local_dof_indices, cell_external_force_rhs);
        // tangent_matrix.add(local_dof_indices, local_dof_indices, cell_tangent_matrix); 
            // double tol = 1e-9;
        for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_cell; ++ivert){
            if (cell->vertex(ivert)[0] == 0 && cell->vertex(ivert)[1] == 0 && load_added == false){
                for(unsigned int idof = 0; idof < dofs_per_cell; ++idof ){
                    double shape_val;
                    if (idof%3 == 2){
                    switch (ivert) {
                        case 0:
                            shape_val = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {0,0});
                            break;
                        case 1:
                            shape_val = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {1,0});
                            break;
                        case 2:
                            shape_val = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {0,1});
                            break;
                        case 3:
                            shape_val = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {1,1});
                            break;
                        default:
                            shape_val = 0;
                            break;
                        }
                    cell_external_force_rhs[idof] = - f_load * shape_val;
                    }
                }
                load_added = true;
            }
        }
                    
            //             cell_external_force_rhs(r_shape) += - f_load * shape_val;
            //             load_added = true;
                        //  std::cout << "shape_val = " << shape_val <<endl;
            //     }
            // }
        
        for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_cell; ++ivert)
        {
            if (cell->vertex(ivert)[0] == 0 && cell->vertex(ivert)[1] == 0)
            {
                unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                // std::cout << "dof_id = " << dof_id << std::endl;
                fix_dof_indices.push_back(dof_id);
                fix_dof_indices.push_back(dof_id+1);
                // fix_dof_indices.push_back(dof_id+2);
            }
             if (cell->vertex(ivert)[2] == 0 )
             {
                unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                // std::cout << "dof_id = " << dof_id << std::endl;
                // fix_dof_indices.push_back(dof_id);
                // fix_dof_indices.push_back(dof_id+1);
                fix_dof_indices.push_back(dof_id+2);
             }
        }
        internal_force_rhs.add(local_dof_indices, cell_internal_force_rhs);
        external_force_rhs.add(local_dof_indices, cell_external_force_rhs);

        tangent_matrix.add(local_dof_indices, local_dof_indices, cell_tangent_matrix); 
    }//loop over cells
    residual_vector = external_force_rhs - internal_force_rhs;
    store_material_data = false;
    restore_material_data = false;

    double sum_load = 0.0;
    for(unsigned int iv = 0; iv < external_force_rhs.size(); ++iv){
        sum_load += external_force_rhs(iv);
    }
    std::cout << "sum_load = " << sum_load << std::endl;
}

  
    
template <int dim, int spacedim>
void Nonlinear_shell<dim, spacedim>::assemble_boundary_mass_matrix_and_rhs()
{
    boundary_mass_matrix = 0;
    // boundary_value_rhs = 0;
    // boundary_edge_load_rhs = 0;
    hp::FEValues<dim,spacedim> hp_fe_boundary_values(mapping_collection, fe_collection, boundary_q_collection, update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    FullMatrix<double> cell_b_mass_matrix;
    Vector<double> cell_b_rhs;
    Vector<double> cell_load_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    double boundary_length = 0.0;
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
//        std::cout << "cell " << cell->active_cell_index() << " has "<< b_fe_values.n_quadrature_points <<" boundary qpts."<< std::endl;
        if (b_fe_values.n_quadrature_points != 1)
        {
            for (unsigned int q_point = 0; q_point < b_fe_values.n_quadrature_points;
                 ++q_point)
            {
                Point<spacedim> qpt = b_fe_values.quadrature_point(q_point);
//                std::cout << "gauss point " << q_point << " = "<< b_fe_values.get_quadrature().point(q_point) << " weight = " << b_fe_values.get_quadrature().weight(q_point) << " qpt = " << qpt << std::endl;
                // covariant base  a_1, a_2, a_3;
                Tensor<2, spacedim> a_cov; // a_i = x_{,i} , i = 1,2,3
                auto jacobian_ref = b_fe_values.jacobian(q_point);
                
                for (unsigned int id = 0; id < spacedim; ++id){
                    a_cov[0][id] = jacobian_ref[id][0];
                    a_cov[1][id] = jacobian_ref[id][1];
                }
                double tol = 1e-9;
                if (qpt[0] < tol|| qpt[0] - 100 <tol || qpt[1] < tol || qpt[1] - 100 < 1e-9) {
//                if (std::abs(qpt[0]) < tol || std::abs(qpt[1]) < tol || std::abs(qpt[2]) < tol) {
                    double jxw;
                    
                    if (b_fe_values.get_quadrature().point(q_point)[0] == 0 ) {
                        jxw = a_cov[1].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }else if (b_fe_values.get_quadrature().point(q_point)[1] == 0 ){
                        jxw = a_cov[0].norm() * b_fe_values.get_quadrature().weight(q_point);
                    }
//                    boundary_length += jxw;
                    
                    for (unsigned int i_shape = 0; i_shape < dofs_per_cell; ++i_shape) {
                        if ( b_fe_values.shape_value(i_shape, q_point) > tol) {
                            for (unsigned int j_shape = 0; j_shape < dofs_per_cell; ++j_shape) {
                                if (std::abs(qpt[0]) < tol && b_fe_values.shape_value(j_shape, q_point) > tol) {
                                    if (i_shape%3 == j_shape%3 && i_shape%3 == 0){
                                        cell_b_mass_matrix[i_shape][j_shape] += b_fe_values.shape_value(i_shape, q_point) * b_fe_values.shape_value(j_shape, q_point) * jxw;
                                    }
                                }
                                if (std::abs(qpt[1]) < tol && b_fe_values.shape_value(j_shape, q_point) > tol) {
                                    if (i_shape%3 == j_shape%3 && i_shape%3 == 1){
                                        cell_b_mass_matrix[i_shape][j_shape] += b_fe_values.shape_value(i_shape, q_point) * b_fe_values.shape_value(j_shape, q_point) * jxw;
                                    }
                                }
                                if (std::abs(qpt[2]) < tol && b_fe_values.shape_value(j_shape, q_point) > tol) {
                                    if (i_shape%3 == j_shape%3 && i_shape%3 == 2){
                                        cell_b_mass_matrix[i_shape][j_shape] += b_fe_values.shape_value(i_shape, q_point) * b_fe_values.shape_value(j_shape, q_point) * jxw;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            boundary_mass_matrix.add(local_dof_indices, cell_b_mass_matrix);
        }
    }
}



template <int dim, int spacedim>
Triangulation<dim,spacedim> set_mesh( std::string type )
{
    Triangulation<dim,spacedim> mesh;
    if (type == "plate"){
        GridGenerator::hyper_cube<dim,spacedim>(mesh,0,1);
        mesh.refine_global(3);
    }
    else if (type == "circle_cap")
    {
         std::string mfile = "/Users/zhaoweiliu/Documents/geometries/cap_06.msh";
        GridIn<2,3> grid_in;
        grid_in.attach_triangulation(mesh);
        std::ifstream file(mfile.c_str());
        Assert(file, ExcFileNotOpen(mfile.c_str()));
        grid_in.read_msh(file);
        // mesh.refine_global(1);
        Catmull_Clark_subdivision(mesh);
        // GridTools::scale(1.0260, mesh);
    }
    std::cout << "   Number of active cells: " << mesh.n_active_cells()
    << std::endl
    << "   Total number of cells: " << mesh.n_cells()
    << std::endl;
    std::ofstream output_file("cap_mesh.vtu");
    GridOut().write_vtu (mesh, output_file);
    return mesh;
}



int main()
{
    const int dim = 2, spacedim = 3;
    // mesh
    Triangulation<dim,spacedim> mesh = set_mesh<dim,spacedim>("circle_cap");
    Nonlinear_shell<dim, spacedim> nonlinear_thin_shell(mesh);
    nonlinear_thin_shell.run();
    
    std::cout <<"finished.\n";
    
    return 0;
    
}
