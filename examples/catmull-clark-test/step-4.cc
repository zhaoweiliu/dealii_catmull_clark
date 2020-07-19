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
#include <deal.II/lac/sparse_direct.h>

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

void vtk_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution){
    
//    auto verts = dof_handler.get_triangulation().get_vertices();
    
    const unsigned int ngridpts = 4;
    const unsigned int seg_n = ngridpts-1;
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
    vtkSmartPointer<vtkPoints> points = vtkPoints::New();
    vtkSmartPointer<vtkDoubleArray> function = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a1 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a2 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a3 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a11 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a12 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a22 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> dN1a1 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> dN1a2 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> dN2a1 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> dN2a2 = vtkDoubleArray::New();


    function->SetNumberOfComponents(3);
    function->SetName("disp");
    function->SetComponentName(0, "x");
    function->SetComponentName(1, "y");
    function->SetComponentName(2, "z");
//    function->SetComponentName(3, "magnitude");

    a1->SetNumberOfComponents(3);
    a1->SetName("a1");
    a1->SetComponentName(0, "dxdxi");
    a1->SetComponentName(1, "dydxi");
    a1->SetComponentName(2, "dzdxi");
    
    a2->SetNumberOfComponents(3);
    a2->SetName("a2");
    a2->SetComponentName(0, "dxdeta");
    a2->SetComponentName(1, "dydeta");
    a2->SetComponentName(2, "dzdeta");
    
    a3->SetNumberOfComponents(3);
    a3->SetName("a3");
    a3->SetComponentName(0, "n1");
    a3->SetComponentName(1, "n2");
    a3->SetComponentName(2, "n3");
    
    a11->SetNumberOfComponents(3);
    a11->SetName("a11");
    a11->SetComponentName(0, "d2xdxi2");
    a11->SetComponentName(1, "d2ydxi2");
    a11->SetComponentName(2, "d2zdxi2");
    
    a12->SetNumberOfComponents(3);
    a12->SetName("a12");
    a12->SetComponentName(0, "d2xdxideta");
    a12->SetComponentName(1, "d2ydxideta");
    a12->SetComponentName(2, "d2zdxideta");
    
    a22->SetNumberOfComponents(3);
    a22->SetName("a22");
    a22->SetComponentName(0, "d2xdeta2");
    a22->SetComponentName(1, "d2ydeta2");
    a22->SetComponentName(2, "d2zdeta2");
    
    dN1a1->SetNumberOfComponents(3);
    dN1a1->SetName("dN1a1");
    dN1a1->SetComponentName(0, "x");
    dN1a1->SetComponentName(1, "y");
    dN1a1->SetComponentName(2, "z");

    dN1a2->SetNumberOfComponents(3);
    dN1a2->SetName("dN1a2");
    dN1a2->SetComponentName(0, "x");
    dN1a2->SetComponentName(1, "y");
    dN1a2->SetComponentName(2, "z");
    
    dN2a1->SetNumberOfComponents(3);
    dN2a1->SetName("dN2a1");
    dN2a1->SetComponentName(0, "x");
    dN2a1->SetComponentName(1, "y");
    dN2a1->SetComponentName(2, "z");
    
    dN2a2->SetNumberOfComponents(3);
    dN2a2->SetName("dN2a2");
    dN2a2->SetComponentName(0, "x");
    dN2a2->SetComponentName(1, "y");
    dN2a2->SetComponentName(2, "z");
    
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
//                Point<3,double> spt = mapping[cell->active_fe_index()].transform_unit_to_real_cell(cell, {u,v});
                Point<3,double> spt = {0,0,0};
                Tensor<1,3,double> disp({0,0,0});
                std::vector<Tensor<1,3>> JJ(3);
                std::vector<Tensor<2,3>> JJ_grad(2);
                double sol = 0;
                for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
                {
                    double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
                    Tensor<1, 2> shape_grad = dof_handler.get_fe(cell->active_fe_index()).shape_grad(idof, {u,v});
                    Tensor<2, 2> shape_grad_grad = dof_handler.get_fe(cell->active_fe_index()).shape_grad_grad(idof, {u,v});
                    
//                    std::cout << shape_grad_grad<<"\n";

                    for (unsigned int jj = 0; jj < 2; ++jj) {
                        switch (idof % 3) {
                            case 0:
                                JJ[jj][0] += shape_grad[jj] * vertices[local_dof_indices[idof]];
                                break;
                            case 1:
                                JJ[jj][1] += shape_grad[jj] * vertices[local_dof_indices[idof]];
                                break;
                            case 2:
                                JJ[jj][2] += shape_grad[jj] * vertices[local_dof_indices[idof]];
                                break;
                        }
                        for (unsigned int ii = 0; ii < 2; ++ii) {
                            switch (idof % 3) {
                                case 0:
                                    JJ_grad[jj][ii][0] += shape_grad_grad[jj][ii] * vertices[local_dof_indices[idof]];
                                    break;
                                case 1:
                                    JJ_grad[jj][ii][1] += shape_grad_grad[jj][ii] * vertices[local_dof_indices[idof]];
                                    break;
                                case 2:
                                    JJ_grad[jj][ii][2] += shape_grad_grad[jj][ii] * vertices[local_dof_indices[idof]];
                                    break;
                            }
                        }
                    }
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
                
                Tensor<1, 2> shape_grad = dof_handler.get_fe(cell->active_fe_index()).shape_grad(0, {u,v});
                
                JJ[2] = cross_product_3d(JJ[0],JJ[1]);
                
                double coordsdata [3] = {spt[0],spt[1],spt[2]};

                points->InsertPoint(sample_offset+count, coordsdata);
                
                dN1a1->InsertComponent(sample_offset+count, 0, shape_grad[0]*JJ[0][0]);
                dN1a1->InsertComponent(sample_offset+count, 1, shape_grad[0]*JJ[0][1]);
                dN1a1->InsertComponent(sample_offset+count, 2, shape_grad[0]*JJ[0][2]);
                
                dN1a2->InsertComponent(sample_offset+count, 0, shape_grad[0]*JJ[1][0]);
                dN1a2->InsertComponent(sample_offset+count, 1, shape_grad[0]*JJ[1][1]);
                dN1a2->InsertComponent(sample_offset+count, 2, shape_grad[0]*JJ[1][2]);

                dN2a1->InsertComponent(sample_offset+count, 0, shape_grad[1]*JJ[0][0]);
                dN2a1->InsertComponent(sample_offset+count, 1, shape_grad[1]*JJ[0][1]);
                dN2a1->InsertComponent(sample_offset+count, 2, shape_grad[1]*JJ[0][2]);
                
                dN2a2->InsertComponent(sample_offset+count, 0, shape_grad[1]*JJ[1][0]);
                dN2a2->InsertComponent(sample_offset+count, 1, shape_grad[1]*JJ[1][1]);
                dN2a2->InsertComponent(sample_offset+count, 2, shape_grad[1]*JJ[1][2]);
                
                function->InsertComponent(sample_offset+count, 0, disp[0]);
                function->InsertComponent(sample_offset+count, 1, disp[1]);
                function->InsertComponent(sample_offset+count, 2, disp[2]);
                
                a1->InsertComponent(sample_offset+count, 0, JJ[0][0]);
                a1->InsertComponent(sample_offset+count, 1, JJ[0][1]);
                a1->InsertComponent(sample_offset+count, 2, JJ[0][2]);
                a2->InsertComponent(sample_offset+count, 0, JJ[1][0]);
                a2->InsertComponent(sample_offset+count, 1, JJ[1][1]);
                a2->InsertComponent(sample_offset+count, 2, JJ[1][2]);
                a3->InsertComponent(sample_offset+count, 0, JJ[2][0]);
                a3->InsertComponent(sample_offset+count, 1, JJ[2][1]);
                a3->InsertComponent(sample_offset+count, 2, JJ[2][2]);
                
                a11->InsertComponent(sample_offset+count, 0, JJ_grad[0][0][0]);
                a11->InsertComponent(sample_offset+count, 1, JJ_grad[0][0][1]);
                a11->InsertComponent(sample_offset+count, 2, JJ_grad[0][0][2]);
                
                a12->InsertComponent(sample_offset+count, 0, JJ_grad[0][1][0]);
                a12->InsertComponent(sample_offset+count, 1, JJ_grad[0][1][1]);
                a12->InsertComponent(sample_offset+count, 2, JJ_grad[0][1][2]);
                
                a22->InsertComponent(sample_offset+count, 0, JJ_grad[1][1][0]);
                a22->InsertComponent(sample_offset+count, 1, JJ_grad[1][1][1]);
                a22->InsertComponent(sample_offset+count, 2, JJ_grad[1][1][2]);

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
    grid -> GetPointData() -> AddArray(a1);
    grid -> GetPointData() -> AddArray(a2);
    grid -> GetPointData() -> AddArray(a3);
    grid -> GetPointData() -> AddArray(a11);
    grid -> GetPointData() -> AddArray(a12);
    grid -> GetPointData() -> AddArray(dN1a1);
    grid -> GetPointData() -> AddArray(dN1a2);
    grid -> GetPointData() -> AddArray(dN2a1);
    grid -> GetPointData() -> AddArray(dN2a2);


    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
    writer -> SetFileName(filename.c_str());
    writer -> SetInputData(grid);
    if (! writer -> Write()) {
        std::cout<<" Cannot write displacement vtu file! ";
    }
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
     double product = std::sin(2*numbers::PI * p[0]);
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
    for (unsigned int ii=0; ii<2; ii++)
    {
      for (unsigned int jj=0; jj<2 ; jj++)
      {
        am_cov[ii][jj] =scalar_product(a_cov[ii], a_cov[jj]);
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



int main()
{
    const int dim = 2, spacedim = 3;
    
    Triangulation<dim,spacedim> mesh;
//    static SphericalManifold<dim,spacedim> surface_description;
//    {
//        Triangulation<spacedim> volume_mesh;
//        GridGenerator::half_hyper_ball(volume_mesh);
//        std::set<types::boundary_id> boundary_ids;
//        boundary_ids.insert (0);
//        GridGenerator::extract_boundary_mesh (volume_mesh, mesh,
//                                              boundary_ids);
//    }
    
    GridGenerator::hyper_cube(mesh, 0, 1);
    
//    mesh.set_all_manifold_ids(0);
//    mesh.set_manifold (0, surface_description);
    mesh.refine_global(4);
    std::cout << "   Number of active cells: " << mesh.n_active_cells()
               << std::endl
               << "   Total number of cells: " << mesh.n_cells()
               << std::endl;
    
    double youngs = 1e10;
    double possions = 0.2;
    double thickness = 0.01;
    
    hp::DoFHandler<dim,spacedim> dof_handler(mesh);
    hp::FECollection<dim,spacedim> fe_collection;
    hp::MappingCollection<dim,spacedim> mapping_collection;
    hp::QCollection<dim> q_collection;
    hp::QCollection<dim> boundary_q_collection;

    Vector<double> vec_values;
    
    catmull_clark_create_fe_quadrature_and_mapping_collections_and_distribute_dofs(dof_handler,fe_collection,vec_values,mapping_collection,q_collection,boundary_q_collection,3);

    AffineConstraints<double> constraints;
//    constraints.clear();
//    constraints.close();
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> stiffness_matrix;

    Vector<double> solution;
    Vector<double> solution_disp;

    Vector<double> system_rhs;
    Vector<double> force_rhs;

    
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    std::ofstream out("CC_sparsity_pattern.svg");
    sparsity_pattern.print_svg(out);
    
    system_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    solution_disp.reinit(dof_handler.n_dofs());

    system_rhs.reinit(dof_handler.n_dofs());
    force_rhs.reinit(dof_handler.n_dofs());

     hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    RightHandSide<spacedim> rhs_function;

    FullMatrix<double> cell_matrix;
    Vector<double>     cell_rhs;
    
    FullMatrix<double> cell_stiffness_matrix;
    Vector<double>     cell_force_rhs;
    
    std::vector<types::global_dof_index> local_dof_indices;
    
    std::vector<types::global_dof_index> fix_dof_indices;
    
    double area = 0;
        
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
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
                                rhs_values[q_point] *               // f(x_q)
                                fe_values.JxW(q_point));            // dx
            }
        }

        system_rhs.add(local_dof_indices, cell_rhs);
        system_matrix.add(local_dof_indices, cell_matrix);
        
        cell_stiffness_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_stiffness_matrix = 0;
        cell_force_rhs.reinit(dofs_per_cell);
        cell_force_rhs = 0;
        
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
        {
            auto qp = q_collection[cell->active_fe_index()].point(q_point);
                        
            // covariant base  a_1, a_2, a_3;
            Tensor<2, spacedim> a_cov; // a_i = x_{,i} , i = 1,2,3
            // derivatives of covariant base;
            std::vector<Tensor<2, spacedim>> da_cov(2); // a_{i,j} = x_{,ij} , i,j = 1,2,3
            auto jacobian = fe_values.jacobian(q_point);
            
            for (unsigned int id = 0; id < spacedim; ++id){
                a_cov[0][id] = jacobian[id][0];
                a_cov[1][id] = jacobian[id][1];
            }
            a_cov[2] = cross_product_3d(a_cov[0], a_cov[1]);
            double detJ = a_cov[2].norm();
            a_cov[2] = a_cov[2]/detJ;
            
            auto jacobian_grad = fe_values.jacobian_grad(q_point);
            for (unsigned int K = 0; K < dim; ++K)
            {
                for (unsigned int ii = 0; ii < spacedim; ++ii)
                {
                    da_cov[0][K][ii] = jacobian_grad[ii][0][K];
                    da_cov[1][K][ii] = jacobian_grad[ii][1][K];
                }
            }
            
            // covariant metric tensor
            Tensor<2,dim> am_cov = metric_covariant(a_cov);
            
            // contravariant metric tensor
            Tensor<2,dim> am_contra = metric_contravariant(am_cov);
            
            //constitutive tensors N and M (H tensor)
            Tensor<2, spacedim> constitutive_N, constitutive_M;
            constitutive_tensors(constitutive_N, constitutive_M, am_contra, thickness, youngs, possions);
            
//            std::cout << constitutive_N <<"\n" << constitutive_M<<"\n";
            
            Tensor<1,spacedim> a1ca11 = cross_product_3d(a_cov[0], da_cov[0][0]);
            Tensor<1,spacedim> a11ca2 = cross_product_3d(da_cov[0][0], a_cov[1]);
            Tensor<1,spacedim> a22ca2 = cross_product_3d(da_cov[1][1], a_cov[1]);
            Tensor<1,spacedim> a1ca22 = cross_product_3d(a_cov[0], da_cov[1][1]);
            Tensor<1,spacedim> a12ca2 = cross_product_3d(da_cov[0][1], a_cov[1]);
            Tensor<1,spacedim> a1ca12 = cross_product_3d(a_cov[0], da_cov[0][1]);
            Tensor<1,spacedim> a3ca1 = cross_product_3d(a_cov[2], a_cov[0]);
            Tensor<1,spacedim> a2ca3 = cross_product_3d(a_cov[1], a_cov[2]);
            
            double a3sa11 = scalar_product(a_cov[2], da_cov[0][0]);
            double a3sa12 = scalar_product(a_cov[2], da_cov[0][1]);
            double a3sa22 = scalar_product(a_cov[2], da_cov[1][1]);
            
            
            std::vector<Tensor<2, 3>>
            bn_vec(dofs_per_cell/spacedim),
            bm_vec(dofs_per_cell/spacedim);
            
            for (unsigned int i_shape = 0; i_shape < dofs_per_cell/spacedim; ++i_shape) {
                // compute first and second grad of i_shape function
                Tensor<1, spacedim> shape_grad = fe_values.shape_grad(i_shape*spacedim, q_point);
                Tensor<2, spacedim> shape_hessian = fe_values.shape_hessian(i_shape*spacedim, q_point);
                Tensor<1, dim> shape_der;
                Tensor<2, dim> shape_der2;
                // transform to parametric domain
                for (unsigned int id = 0; id < dim; ++id){
                    for (unsigned int kd = 0; kd < spacedim; ++kd){
                        shape_der[id] += shape_grad[kd]*jacobian[kd][id];
                        for (unsigned jd = 0; jd < dim; ++jd) {
                            for (unsigned ld = 0; ld < spacedim; ++ld) {
                                shape_der2[id][jd] += shape_hessian[kd][ld] * jacobian[kd][id] * jacobian[ld][jd];
                            }
                            shape_der2[id][jd] += shape_grad[kd] * jacobian_grad[kd][id][jd];
                        }
                    }
                }
                
//                auto qp = q_collection[cell->active_fe_index()].point(q_point);
//                Tensor<1, dim> shape_der = fe_collection[cell->active_fe_index()].shape_grad(i_shape*spacedim, qp);
//                Tensor<2, dim> shape_der2 = fe_collection[cell->active_fe_index()].shape_grad_grad(i_shape*spacedim, qp);
                
                //  computation of the B operator (strains) for i_shape function
                Tensor<2, spacedim> BN; // membrane part
                Tensor<2, spacedim> BM; // bending part
                
                for (unsigned int ii= 0; ii < spacedim; ++ii) {
                    BN[0][ii] = shape_der[0] * a_cov[0][ii]; // alpha = beta = 0
                    BN[1][ii] = shape_der[1] * a_cov[1][ii]; // alpha = beta = 1
                    BN[2][ii] = shape_der[0] * a_cov[1][ii] + shape_der[1] * a_cov[0][ii]; // alpha = 0, beta = 1 and alpha = 1, beta = 0
                    
                    BM[0][ii] = -shape_der2[0][0]*a_cov[2][ii] + (shape_der[0]*a11ca2[ii] + shape_der[1]*a1ca11[ii])/detJ + (shape_der[0]*a2ca3[ii] + shape_der[1]*a3ca1[ii])*a3sa11/detJ;
                    BM[1][ii] = -shape_der2[1][1]*a_cov[2][ii] + (shape_der[0]*a22ca2[ii] + shape_der[1]*a1ca22[ii])/detJ + (shape_der[0]*a2ca3[ii] + shape_der[1]*a3ca1[ii])*a3sa22/detJ;
                    BM[2][ii] = 2.0*((shape_der[0]*a12ca2[ii] + shape_der[1]*a1ca12[ii])/detJ - shape_der2[0][1] * a_cov[2][ii] + (shape_der[0]*a2ca3[ii] + shape_der[1]*a3ca1[ii])*a3sa12/detJ);
                }
                
                bn_vec[i_shape] = BN;
                bm_vec[i_shape] = BM;
                
//                std::cout <<  BM << "\n";

                
            } // loop over shape functions
            
            for (unsigned int j_node = 0; j_node < dofs_per_cell/spacedim; ++j_node)
            {
                Tensor<2, spacedim> hn,hm;
                for(unsigned int ii = 0; ii < 3 ; ++ii)
                    for(unsigned int jj = 0; jj < 3 ; ++jj)
                        for(unsigned int kk = 0; kk < spacedim ; ++kk){
                            hn[ii][kk] += constitutive_N[ii][jj] * bn_vec[j_node][jj][kk];
                            hm[ii][kk] += constitutive_M[ii][jj] * bm_vec[j_node][jj][kk];
                        }
                
                for (unsigned int i_node = 0; i_node < dofs_per_cell/spacedim; ++i_node)
                {
                    Tensor<2, spacedim> sn,sm;
                    
                    for(unsigned int ii = 0; ii < spacedim ; ++ii)
                        for(unsigned int jj = 0; jj < 3 ; ++jj)
                            for(unsigned int kk = 0; kk < spacedim ; ++kk){
                                sn[ii][kk] += bn_vec[i_node][jj][ii] * hn[jj][kk];
                                sm[ii][kk] += bm_vec[i_node][jj][ii] * hm[jj][kk];
                            }
                    
//                    std::cout << sn << "\n" << sm << "\n";
                    
                    for (unsigned int id = 0; id < spacedim; ++id) {
                        for (unsigned int jd = 0; jd < spacedim; ++jd) {
                            cell_stiffness_matrix(i_node*spacedim+id, j_node*spacedim+jd) += (sn[id][jd] + sm[id][jd]) * fe_values.JxW(q_point);
                        }
                    }
                }// loop over nodes (i)
                cell_force_rhs(j_node*spacedim + 2) += 1 * fe_values.shape_value(j_node*spacedim + 2 , q_point)* fe_values.JxW(q_point); // f_z = 1
//                cell_force_rhs(j_node*spacedim+1) += 1 * fe_values.shape_value(j_node*spacedim + 2 , q_point)* fe_values.JxW(q_point); // f_y = 1
            }// loop over nodes (j)
        }// loop over quadrature points
        
        force_rhs.add(local_dof_indices, cell_force_rhs);
        stiffness_matrix.add(local_dof_indices, local_dof_indices, cell_stiffness_matrix);
        
        for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_cell; ++ivert)
        {
            if (cell->vertex(ivert)[0] == 0 || cell->vertex(ivert)[0] == 1 || cell->vertex(ivert)[1] == 0 || cell->vertex(ivert)[1] == 1 )
//            if ( cell->vertex(ivert)[1] == 1)
            {
                unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                //                std:: cout << cell->vertex(ivert)[0] <<" ";
                //                std:: cout << cell->vertex(ivert)[1] <<std::endl;
                fix_dof_indices.push_back(dof_id);
                fix_dof_indices.push_back(dof_id+1);
                fix_dof_indices.push_back(dof_id+2);
            }
        }
    } // loop over cells
    
    std::cout << " area = " << area << std::endl;
        
    std::sort(fix_dof_indices.begin(), fix_dof_indices.end());
    auto last = std::unique(fix_dof_indices.begin(), fix_dof_indices.end());
    fix_dof_indices.erase(last, fix_dof_indices.end());
    for (unsigned int idof = 0; idof <fix_dof_indices.size(); ++idof) {
        for (unsigned int jdof = 0; jdof <dof_handler.n_dofs(); ++jdof) {
            if (fix_dof_indices[idof] == jdof){
                stiffness_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1);
            }
            else
            {
                stiffness_matrix.set(fix_dof_indices[idof], jdof, 0);
                stiffness_matrix.set(jdof, fix_dof_indices[idof], 0);
            }
        }
        force_rhs[fix_dof_indices[idof]] = 0;
    }
    
    
    SolverControl solver_control(system_rhs.size(),
                                1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    
    
    SolverControl solver_control_2(5000,
                                1e-12 * force_rhs.l2_norm());
    SolverCG<Vector<double>> cg_2(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner_2;
    preconditioner_2.initialize(stiffness_matrix, 1.2);
    cg_2.solve(stiffness_matrix, solution_disp, force_rhs, preconditioner_2);
    
//        SparseDirectUMFPACK direct_solver;
//        direct_solver.initialize(sparsity_pattern);
//        solution_disp = force_rhs;
//        direct_solver.solve(stiffness_matrix,solution_disp);
    
//    std::cout << solution_disp << std::endl;
    
//    for (const auto &cell : dof_handler.active_cell_iterators())
//    {
//        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
//        local_dof_indices.resize(dofs_per_cell);
//        cell->get_dof_indices(local_dof_indices);
//        hp_fe_values.reinit(cell);
//        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
//       for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
//            ++q_point){
//           double q_sol = 0;
//           for (unsigned int i = 0; i < dofs_per_cell; ++i)
//           {
//               q_sol += solution[local_dof_indices[i]] * fe_values.shape_value(i, q_point);
//           }
//           std::cout << "quadrature point "<< q_point << " x = " << fe_values.get_quadrature_points()[q_point][0]<< " value = "<<q_sol<<std::endl;
//       }
//    }
    
//    DataOut<dim, hp::DoFHandler<dim,spacedim>> data_out;
//    data_out.attach_dof_handler(dof_handler);
//    data_out.add_data_vector(solution, "solution");
//    data_out.build_patches(4);
//    const std::string filename =
//      "solution-CC.vtk";
//    std::ofstream output(filename);
//    data_out.write_vtk(output);
    vtk_plot("cc-solution.vtu", dof_handler, mapping_collection,vec_values, solution_disp);

    return 0;
}
