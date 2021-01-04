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

void Catmull_Clark_subdivision(Triangulation<2,3> &mesh){
    int ncells = mesh.n_active_cells();
    int nfaces = mesh.n_faces();
    int nvertices = mesh.n_vertices();
    // initialise cell data
    std::vector<CellData<2> >                  new_cells;
    SubCellData                                subcelldata;
    // initialise topology data vectors
    std::vector<Point<3>> vertices(nvertices+ncells+nfaces);
    std::vector<Point<3>> cvs(ncells);
    std::vector<Point<3>> fvs(nfaces);
    std::vector<Point<3>> fvs_mid(nfaces);
    std::vector<Point<3>> vvs(nvertices);
    std::vector<std::vector<int>> v_in_cells(nvertices);
    std::vector<std::vector<int>> v_in_faces(nvertices);
    std::vector<bool> v_at_boundary(nvertices,false);
    std::vector<bool> f_at_boundary(nfaces,false);
    
    //Loop over cells to compute new cell and face vertices
    for (Triangulation<2,3>::active_cell_iterator cell=mesh.begin_active();cell!= mesh.end(); ++cell){
        //initialise a point
        Point<3> v{0,0,0};
        //loop over vertices in cell iterator
        for(unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i){
            v_in_cells[cell->vertex_index(i)].push_back(cell->active_cell_index());
            v += cell->vertex(i);
        }
        // compute face point
        v= v/4.0;
        // same it into vector
        cvs[cell->active_cell_index()] = v;
        
        //loop over faces in cell iterator
        for(unsigned int i = 0; i<GeometryInfo<2>::faces_per_cell;++i){
            
            // avoid duplicated faces
            if(cell->neighbor_index(i)<int(cell->active_cell_index())){
                //loop over vertices in each face
                for(unsigned int j=0; j<GeometryInfo<2>::vertices_per_face; ++j){
                    //vertices id for each face
                    v_in_faces[cell->face(i)->vertex_index(j)].push_back(cell->face_index(i));
                }
                fvs_mid[cell->face_index(i)] = cell->face(i)->center();
                // determine whether face on boundary
                if(cell->face(i)->at_boundary()==false){
                    // compute new face points and face midpoints
                    fvs[cell->face_index(i)] = (cvs[cell->neighbor_index(i)] + cvs[cell->active_cell_index()])/4.0 + cell->face(i)->vertex(0)/4.0 + cell->face(i)->vertex(1)/4.0;
                }
                else{
                    f_at_boundary[cell->face_index(i)] = true;
                    // if on the boundary, the new face points are face midpoints
                    for(unsigned int j=0; j<GeometryInfo<2>::vertices_per_face; ++j){
                        v_at_boundary[cell->face(i)->vertex_index(j)] = true;
                    }
                    fvs[cell->face_index(i)] = cell->face(i)->center();
                }
            }
        }
        
        // compute cell data (connectivity)
        std::vector<std::vector<unsigned int>> connectivity(4);
        connectivity[0] = {cell->vertex_index(0),nvertices+cell->face_index(2),nfaces+nvertices+cell->active_cell_index(),nvertices+cell->face_index(0)};
        connectivity[1] = { nvertices + cell->face_index(2),cell->vertex_index(1), nvertices+cell->face_index(1), nfaces+nvertices+cell->active_cell_index()};
        connectivity[2] = {nvertices + cell->face_index(0),nfaces+nvertices+cell->active_cell_index(),nvertices+cell->face_index(3),cell->vertex_index(2)};
        connectivity[3] = {nfaces+nvertices+cell->active_cell_index(),nvertices+ cell->face_index(1),cell->vertex_index(3),nvertices+cell->face_index(3)};
        
        for(unsigned int j = 0; j<GeometryInfo<2>::vertices_per_cell;++j){
            new_cells.emplace_back();
            for (unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i){
                new_cells.back().vertices[i] = connectivity[j][i];
            }
            new_cells.back().material_id = cell->material_id();
        }
    }
    
    //Loop over old vertices and modify them
    for(Triangulation<2,3>::active_vertex_iterator vertex = mesh.begin_active_vertex();vertex!= mesh.end_vertex();++vertex){
        Point<3> cv{0,0,0};
        Point<3> fv(0,0,0);
        double nc = 0.0, nf = 0.0;
        for (unsigned int i = 0; i < v_in_cells[vertex->index()].size(); ++i) {
            cv +=cvs[v_in_cells[vertex->index()][i]];
            nc+=1;
        }
        cv=cv/nc;
        
        for (unsigned int i = 0; i < v_in_faces[vertex->index()].size(); ++i){
            fv += fvs_mid[v_in_faces[vertex->index()][i]];
            nf += 1;
        }
        fv = fv/nf;
        // averaging
        if(v_at_boundary[vertex->index()]==false){
            if (nf == nc){
                vvs[vertex->index()] = vertex->center() + 1./nc * (cv - vertex->center()) + 2./nf * (fv-vertex->center());
                //            vvs[vertex->index()] = (nf-3.)/nf * vertex->center() + 1./nc * cv + 2./nf * fv;
            }else{
                throw std::runtime_error("number of faces and cells are not equal.");
            }
        }
        else{
            Point<3>f_b{0,0,0};
            for (unsigned int i = 0; i < v_in_faces[vertex->index()].size(); ++i){
                if (f_at_boundary[v_in_faces[vertex->index()][i]]==true){
                    f_b += fvs_mid[v_in_faces[vertex->index()][i]];
                }
            }
            //            vvs[vertex->index()] = 1./2.*vertex->center() + 1./4.*f_b;
            vvs[vertex->index()] = vertex->center();
        }
    }
    
    for(int i = 0; i<nvertices;++i)
    vertices[i] = vvs[i];
    for(int i = 0; i<nfaces;++i)
    vertices[i+nvertices] = fvs[i];
    for(int i = 0; i<ncells;++i)
    vertices[i+nvertices+nfaces] = cvs[i];
    
    GridTools::delete_unused_vertices(vertices,new_cells,subcelldata);
    Triangulation<2,3> new_mesh;
    GridReordering<2,3>::reorder_cells (new_cells);
    new_mesh.create_triangulation_compatibility(vertices, new_cells, subcelldata);
    mesh.clear();
    mesh.copy_triangulation(new_mesh);
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
    //    H_tensor[0][0][0][0] = 1.;
    //    H_tensor[1][1][1][1] = 1.;
    //    H_tensor[0][0][1][1] = poisson;
    //    H_tensor[1][1][0][0] = poisson;
    //    H_tensor[0][1][0][1] = 0.5*(1. - poisson);
    //    H_tensor[1][0][1][0] = 0.5*(1. - poisson);
    //    H_tensor[0][1][1][0] = 0.5*(1. - poisson);
    //    H_tensor[1][0][0][1] = 0.5*(1. - poisson);
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
//
//
//template<int dim, int spacedim>
//void make_constraints(const hp::DoFHandler<dim,spacedim> &dof_handler, AffineConstraints<double> &constraints)
//{
//
//}



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



void vtk_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>()){
    
    //    auto verts = dof_handler.get_triangulation().get_vertices();
    
    const unsigned int ngridpts = 4;
    const unsigned int seg_n = ngridpts-1;
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
    vtkSmartPointer<vtkPoints> points = vtkPoints::New();
    vtkSmartPointer<vtkDoubleArray> function = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> function_2 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> normal = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> membrane_part = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> bending_part = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> local_coords = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> shape_func = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> shape_func_dx = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> shape_func_dy = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a1 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> a2 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> da11 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> da12 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> da22 = vtkDoubleArray::New();
    vtkSmartPointer<vtkDoubleArray> curvature = vtkDoubleArray::New();
    
    
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
    
    normal->SetNumberOfComponents(3);
    normal->SetName("normal");
    normal->SetComponentName(0, "x");
    normal->SetComponentName(1, "y");
    normal->SetComponentName(2, "z");
    
    shape_func->SetNumberOfComponents(50);
    shape_func->SetName("shape_func");
    for (unsigned int i = 0; i < 50; ++i) {
        std::string s = std::to_string(i);
        char const *pchar = s.c_str();
        shape_func->SetComponentName(i, pchar);
    }
    
    shape_func_dx->SetNumberOfComponents(50);
    shape_func_dx->SetName("shape_dx");
    for (unsigned int i = 0; i < 50; ++i) {
        std::string s = std::to_string(i);
        char const *pchar = s.c_str();
        shape_func_dx->SetComponentName(i, pchar);
    }
    
    shape_func_dy->SetNumberOfComponents(50);
    shape_func_dy->SetName("shape_dy");
    for (unsigned int i = 0; i < 50; ++i) {
        std::string s = std::to_string(i);
        char const *pchar = s.c_str();
        shape_func_dy->SetComponentName(i, pchar);
    }
    
    local_coords->SetNumberOfComponents(2);
    local_coords->SetName("local_coords");
    local_coords->SetComponentName(0, "xi");
    local_coords->SetComponentName(1, "eta");
    
    membrane_part->SetNumberOfComponents(9);
    membrane_part->SetName("membrane_part");
    membrane_part->SetComponentName(0, "xx");
    membrane_part->SetComponentName(1, "yy");
    membrane_part->SetComponentName(2, "zz");
    membrane_part->SetComponentName(3, "xy");
    membrane_part->SetComponentName(4, "xz");
    membrane_part->SetComponentName(5, "yz");
    membrane_part->SetComponentName(6, "yx");
    membrane_part->SetComponentName(7, "zx");
    membrane_part->SetComponentName(8, "zy");
    
    bending_part->SetNumberOfComponents(9);
    bending_part->SetName("bending_part");
    bending_part->SetComponentName(0, "xx");
    bending_part->SetComponentName(1, "yy");
    bending_part->SetComponentName(2, "zz");
    bending_part->SetComponentName(3, "xy");
    bending_part->SetComponentName(4, "xz");
    bending_part->SetComponentName(5, "yz");
    bending_part->SetComponentName(6, "yx");
    bending_part->SetComponentName(7, "zx");
    bending_part->SetComponentName(8, "zy");
    
    a1->SetNumberOfComponents(3);
    a1->SetName("a1");
    a1->SetComponentName(0, "x");
    a1->SetComponentName(1, "y");
    a1->SetComponentName(2, "z");
    
    a2->SetNumberOfComponents(3);
    a2->SetName("a2");
    a2->SetComponentName(0, "x");
    a2->SetComponentName(1, "y");
    a2->SetComponentName(2, "z");
    
    da11->SetNumberOfComponents(3);
    da11->SetName("da11");
    da11->SetComponentName(0, "x");
    da11->SetComponentName(1, "y");
    da11->SetComponentName(2, "z");
    
    da12->SetNumberOfComponents(3);
    da12->SetName("da12");
    da12->SetComponentName(0, "x");
    da12->SetComponentName(1, "y");
    da12->SetComponentName(2, "z");
    
    da22->SetNumberOfComponents(3);
    da22->SetName("da22");
    da22->SetComponentName(0, "x");
    da22->SetComponentName(1, "y");
    da22->SetComponentName(2, "z");
    
    curvature->SetNumberOfComponents(2);
    curvature->SetName("curvature");
    curvature->SetComponentName(0, "1");
    curvature->SetComponentName(1, "2");
    
    double youngs = 1e7;
    double possions = 0.3;
    double thickness = 3;
    
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
                Tensor<1,2,Tensor<1, 3>> disp_der;
                Tensor<2,2,Tensor<1, 3>> disp_der2;
                Tensor<2,3> JJ;
                Tensor<2,3,Tensor<1, 3>> J_der;
                //                std::vector<Tensor<2,3>> JJ_grad(2);
                double p = 0;
                for (unsigned int ishape = 0; ishape < 50; ++ishape) {
                    shape_func->InsertComponent(sample_offset+count, ishape, 0);
                    shape_func_dx->InsertComponent(sample_offset+count, ishape, 0);
                    shape_func_dy->InsertComponent(sample_offset+count, ishape, 0);
                }
                for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
                {
                    const double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
                    const auto shape_der = dof_handler.get_fe(cell->active_fe_index()).shape_grad(idof, {u,v});
                    const auto shape_der2 = dof_handler.get_fe(cell->active_fe_index()).shape_grad_grad(idof, {u,v});
                    
                    switch (idof % 3) {
                        case 0:
                            spt[0] += shapes * vertices[local_dof_indices[idof]];
                            disp[0] += shapes * solution[local_dof_indices[idof]];
                            disp_der[0][0] += shape_der[0] * solution[local_dof_indices[idof]];
                            disp_der[1][0] += shape_der[1] * solution[local_dof_indices[idof]];
                            disp_der2[0][0][0] += shape_der2[0][0] * solution[local_dof_indices[idof]];
                            disp_der2[0][1][0] += shape_der2[0][1] * solution[local_dof_indices[idof]];
                            disp_der2[1][0][0] += shape_der2[1][0] * solution[local_dof_indices[idof]];
                            disp_der2[1][1][0] += shape_der2[1][1] * solution[local_dof_indices[idof]];
                            
                            JJ[0][0] += shape_der[0] * vertices[local_dof_indices[idof]];
                            JJ[1][0] += shape_der[1] * vertices[local_dof_indices[idof]];
                            J_der[0][0][0] += shape_der2[0][0] * vertices[local_dof_indices[idof]];
                            J_der[0][1][0] += shape_der2[0][1] * vertices[local_dof_indices[idof]];
                            J_der[1][0][0] += shape_der2[1][0] * vertices[local_dof_indices[idof]];
                            J_der[1][1][0] += shape_der2[1][1] * vertices[local_dof_indices[idof]];
                            
                            for (unsigned int ishape = 0; ishape < 12; ++ishape) {
                                if (local_dof_indices[idof]/3 == ishape) {
                                    shape_func->InsertComponent(sample_offset+count, ishape, shapes);
                                }
                            }
                            
                            break;
                        case 1:
                            spt[1] += shapes * vertices[local_dof_indices[idof]];
                            disp[1] += shapes * solution[local_dof_indices[idof]];
                            disp_der[0][1] += shape_der[0] * solution[local_dof_indices[idof]];
                            disp_der[1][1] += shape_der[1] * solution[local_dof_indices[idof]];
                            disp_der2[0][0][1] += shape_der2[0][0] * solution[local_dof_indices[idof]];
                            disp_der2[0][1][1] += shape_der2[0][1] * solution[local_dof_indices[idof]];
                            disp_der2[1][0][1] += shape_der2[1][0] * solution[local_dof_indices[idof]];
                            disp_der2[1][1][1] += shape_der2[1][1] * solution[local_dof_indices[idof]];
                            
                            JJ[0][1] += shape_der[0] * vertices[local_dof_indices[idof]];
                            JJ[1][1] += shape_der[1] * vertices[local_dof_indices[idof]];
                            J_der[0][0][1] += shape_der2[0][0] * vertices[local_dof_indices[idof]];
                            J_der[0][1][1] += shape_der2[0][1] * vertices[local_dof_indices[idof]];
                            J_der[1][0][1] += shape_der2[1][0] * vertices[local_dof_indices[idof]];
                            J_der[1][1][1] += shape_der2[1][1] * vertices[local_dof_indices[idof]];
                            break;
                        case 2:
                            spt[2] += shapes * vertices[local_dof_indices[idof]];
                            disp[2] += shapes * solution[local_dof_indices[idof]];
                            disp_der[0][2] += shape_der[0] * solution[local_dof_indices[idof]];
                            disp_der[1][2] += shape_der[1] * solution[local_dof_indices[idof]];
                            disp_der2[0][0][2] += shape_der2[0][0] * solution[local_dof_indices[idof]];
                            disp_der2[0][1][2] += shape_der2[0][1] * solution[local_dof_indices[idof]];
                            disp_der2[1][0][2] += shape_der2[1][0] * solution[local_dof_indices[idof]];
                            disp_der2[1][1][2] += shape_der2[1][1] * solution[local_dof_indices[idof]];
                            
                            JJ[0][2] += shape_der[0] * vertices[local_dof_indices[idof]];
                            JJ[1][2] += shape_der[1] * vertices[local_dof_indices[idof]];
                            J_der[0][0][2] += shape_der2[0][0] * vertices[local_dof_indices[idof]];
                            J_der[0][1][2] += shape_der2[0][1] * vertices[local_dof_indices[idof]];
                            J_der[1][0][2] += shape_der2[1][0] * vertices[local_dof_indices[idof]];
                            J_der[1][1][2] += shape_der2[1][1] * vertices[local_dof_indices[idof]];
                            break;
                    }
                }
                //                std::cout << J_der[0][0] << std::endl;
                //                std::cout << J_der[0][1] << std::endl;
                //                std::cout << J_der[1][1] << std::endl;
                
                JJ[2] = cross_product_3d(JJ[0],JJ[1]);
                double detJ = JJ[2].norm();
                JJ[2] = JJ[2]/detJ;
                auto J_inv = invert(JJ);
                Tensor<1,3> f_grad;
                for (unsigned int idof = 0; idof < dofs_per_cell; ++idof)
                {
                    //                    const double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(idof, {u,v});
                    const Tensor<1,2> shape_der = dof_handler.get_fe(cell->active_fe_index()).shape_grad(idof, {u,v});
                    double shape_dx = shape_der[0]*J_inv[0][0] + shape_der[1]*J_inv[0][1];
                    double shape_dy = shape_der[0]*J_inv[1][0] + shape_der[1]*J_inv[1][1];
                    double shape_dz = shape_der[0]*J_inv[2][0] + shape_der[1]*J_inv[2][1];
                    for (unsigned int ishape = 0; ishape < 12; ++ishape) {
                        if (local_dof_indices[idof]/3 == ishape && local_dof_indices[idof]%3 == 0) {
                            shape_func_dx->InsertComponent(sample_offset+count, ishape, shape_dx);
                            shape_func_dy->InsertComponent(sample_offset+count, ishape, shape_dy);
                        }
                    }
                    switch (idof%3) {
                        case 0:
                            f_grad[0] += shape_dx * 1;
                            break;
                        case 1:
                            f_grad[1] += shape_dy * 1;
                            break;
                        case 2:
                            f_grad[2] += shape_dz * 1;
                            break;
                            
                        default:
                            break;
                    }
                    
                }
                
                // covariant metric tensor
                Tensor<2,2> am_cov = metric_covariant(JJ);
                // contravariant metric tensor
                Tensor<2,2> am_contra = metric_contravariant(am_cov);
                // H tensor
                Tensor<4,2> H_tensor;
                constitutive_fourth_tensors(H_tensor, am_contra, possions);
                
                Tensor<2,2, Tensor<1, 3>> membrane_strain,bending_strain;
                //                Tensor<2,2, Tensor<1, 3>> bending_strain;
                for (unsigned int ii = 0; ii < 2; ++ii) {
                    for (unsigned int jj = 0; jj < 2; ++jj) {
                        for (unsigned int id = 0; id < 3; ++id) {
                            membrane_strain[ii][jj][id] += 0.5 * (JJ[ii][id] * disp_der[jj][id] + JJ[jj][id] * disp_der[ii][id]);
                            bending_strain[ii][jj][id] += - disp_der2[ii][jj][id] * JJ[2][id] + (disp_der[0][id] * cross_product_3d(J_der[ii][jj], JJ[1])[id] + disp_der[1][id] * cross_product_3d(JJ[0], J_der[ii][jj])[id])/detJ + scalar_product(JJ[2], J_der[ii][jj]) * (disp_der[0][id] * cross_product_3d(JJ[1], JJ[2])[id] + disp_der[1][id] * cross_product_3d(JJ[2], JJ[0])[id])/detJ;
                        }
                    }
                }
                
                Tensor<2, 2, Tensor<1, 3>> hn,hm;
                double c1 = youngs*thickness/ (1. - possions*possions),
                c2 = youngs*thickness*thickness*thickness/(12. * (1. - possions*possions));
                for(unsigned int ii = 0; ii < 2 ; ++ii)
                for(unsigned int jj = 0; jj < 2 ; ++jj)
                for(unsigned int kk = 0; kk < 2 ; ++kk)
                for(unsigned int ll = 0; ll < 2 ; ++ll)
                for (unsigned int id = 0; id < 3; ++id) {
                    hn[ii][jj][id] += c1 * H_tensor[ii][jj][kk][ll] * membrane_strain[ll][kk][id];
                    hm[ii][jj][id] += c2 * H_tensor[ii][jj][kk][ll] * bending_strain[ll][kk][id];
                }
                Tensor<2, 3> sn,sm;
                for(unsigned int ii = 0; ii < 2 ; ++ii)
                for(unsigned int jj = 0; jj < 2 ; ++jj)
                for (unsigned int id = 0; id < 3; ++id)
                for (unsigned int jd = 0; jd < 3; ++jd) {
                    sn[id][jd] += membrane_strain[ii][jj][id] * hn[jj][ii][jd];
                    sm[id][jd] += bending_strain[ii][jj][id] * hm[jj][ii][jd];
                }
                
                if (potential.size() != 0){
                    for (unsigned int jdof = 0; jdof < dofs_per_cell/3; ++jdof) {
                        double shapes = dof_handler.get_fe(cell->active_fe_index()).shape_value(jdof*3, {u,v});
                        p += shapes * potential[local_dof_indices[jdof*3]/3];
                    }
                }
                
                double coordsdata [3] = {spt[0],spt[1],spt[2]};
                
                double curvature_1 = 1./detJ * scalar_product(cross_product_3d(J_der[0][0], JJ[1]) + cross_product_3d(JJ[0], J_der[1][0]), covariant_to_contravariant(JJ)[0]);
                double curvature_2 = 1./detJ * scalar_product(cross_product_3d(J_der[0][1], JJ[1]) + cross_product_3d(JJ[0], J_der[1][1]), covariant_to_contravariant(JJ)[1]);
                
                points->InsertPoint(sample_offset+count, coordsdata);
                
                function->InsertComponent(sample_offset+count, 0, disp[0]);
                function->InsertComponent(sample_offset+count, 1, disp[1]);
                function->InsertComponent(sample_offset+count, 2, disp[2]);
                if (potential.size() != 0){
                    function_2->InsertComponent(sample_offset+count, 0, p);
                }
                normal->InsertComponent(sample_offset+count, 0, JJ[2][0]);
                normal->InsertComponent(sample_offset+count, 1, JJ[2][1]);
                normal->InsertComponent(sample_offset+count, 2, JJ[2][2]);
                
                local_coords->InsertComponent(sample_offset+count, 0, u);
                local_coords->InsertComponent(sample_offset+count, 1, v);
                
                a1->InsertComponent(sample_offset+count, 0, JJ[0][0]);
                a1->InsertComponent(sample_offset+count, 1, JJ[0][1]);
                a1->InsertComponent(sample_offset+count, 2, JJ[0][2]);
                
                a2->InsertComponent(sample_offset+count, 0, JJ[1][0]);
                a2->InsertComponent(sample_offset+count, 1, JJ[1][1]);
                a2->InsertComponent(sample_offset+count, 2, JJ[1][2]);
                
                da11->InsertComponent(sample_offset+count, 0, J_der[0][0][0]);
                da11->InsertComponent(sample_offset+count, 1, J_der[0][0][1]);
                da11->InsertComponent(sample_offset+count, 2, J_der[0][0][2]);
                
                da12->InsertComponent(sample_offset+count, 0, J_der[0][1][0]);
                da12->InsertComponent(sample_offset+count, 1, J_der[0][1][1]);
                da12->InsertComponent(sample_offset+count, 2, J_der[0][1][2]);
                
                da22->InsertComponent(sample_offset+count, 0, J_der[1][1][0]);
                da22->InsertComponent(sample_offset+count, 1, J_der[1][1][1]);
                da22->InsertComponent(sample_offset+count, 2, J_der[1][1][2]);
                
                curvature->InsertComponent(sample_offset+count, 0, curvature_1);
                curvature->InsertComponent(sample_offset+count, 1, curvature_2);
                
                membrane_part->InsertComponent(sample_offset+count, 0, sn[0][0]);
                membrane_part->InsertComponent(sample_offset+count, 1, sn[1][1]);
                membrane_part->InsertComponent(sample_offset+count, 2, sn[2][2]);
                membrane_part->InsertComponent(sample_offset+count, 3, sn[0][1]);
                membrane_part->InsertComponent(sample_offset+count, 4, sn[0][2]);
                membrane_part->InsertComponent(sample_offset+count, 5, sn[1][2]);
                membrane_part->InsertComponent(sample_offset+count, 6, sn[1][0]);
                membrane_part->InsertComponent(sample_offset+count, 7, sn[2][0]);
                membrane_part->InsertComponent(sample_offset+count, 8, sn[2][1]);
                
                bending_part->InsertComponent(sample_offset+count, 0, sm[0][0]);
                bending_part->InsertComponent(sample_offset+count, 1, sm[1][1]);
                bending_part->InsertComponent(sample_offset+count, 2, sm[2][2]);
                bending_part->InsertComponent(sample_offset+count, 3, sm[0][1]);
                bending_part->InsertComponent(sample_offset+count, 4, sm[0][2]);
                bending_part->InsertComponent(sample_offset+count, 5, sm[1][2]);
                bending_part->InsertComponent(sample_offset+count, 6, sm[1][0]);
                bending_part->InsertComponent(sample_offset+count, 7, sm[2][0]);
                bending_part->InsertComponent(sample_offset+count, 8, sm[2][1]);
                
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
    grid -> GetPointData() -> AddArray(membrane_part);
    grid -> GetPointData() -> AddArray(bending_part);
    //    grid -> GetPointData() -> AddArray(u_i);
    grid -> GetPointData() -> AddArray(local_coords);
    grid -> GetPointData() -> AddArray(shape_func);
    grid -> GetPointData() -> AddArray(shape_func_dx);
    grid -> GetPointData() -> AddArray(shape_func_dy);
    grid -> GetPointData() -> AddArray(a1);
    grid -> GetPointData() -> AddArray(a2);
    grid -> GetPointData() -> AddArray(da11);
    grid -> GetPointData() -> AddArray(da12);
    grid -> GetPointData() -> AddArray(da22);
    grid -> GetPointData() -> AddArray(curvature);
    
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
    writer -> SetFileName(filename.c_str());
    writer -> SetInputData(grid);
    if (! writer -> Write()) {
        std::cout<<" Cannot write displacement vtu file! ";
    }
}


int main()
{
    const int dim = 2, spacedim = 3;
    
    std::string type = "s";
    
    Triangulation<dim,spacedim> mesh;
    if (type == "r") {
        //        GridGenerator::hyper_cube(mesh, 0, 100);
        //        std::string mfile = "/Users/zhaoweiliu/Documents/geometries/plate_5_2.msh";
        static CylindricalManifold<dim,spacedim> surface_description;
        std::string mfile = "/Users/zhaoweiliu/Documents/geometries/roof.msh";
        GridIn<2,3> grid_in;
        grid_in.attach_triangulation(mesh);
        std::ifstream file(mfile.c_str());
        Assert(file, ExcFileNotOpen(mfile.c_str()));
        grid_in.read_msh(file);
        //        GridTools::scale(50., mesh);
        //
        mesh.set_all_manifold_ids(0);
        mesh.set_manifold (0, surface_description);
        //
        mesh.refine_global(2);
        //        GridTools::rotate(numbers::PI*0.25, 0, mesh);
    }else{
        if (type == "s") {
            static SphericalManifold<dim,spacedim> surface_description;
            {
                Triangulation<spacedim> volume_mesh;
                //                GridGenerator::half_hyper_ball(volume_mesh);
                GridGenerator::hyper_ball(volume_mesh);
                std::set<types::boundary_id> boundary_ids;
                boundary_ids.insert (0);
                GridGenerator::extract_boundary_mesh (volume_mesh, mesh,
                                                      boundary_ids);
            }
            mesh.set_all_manifold_ids(0);
            mesh.set_manifold (0, surface_description);
            mesh.refine_global(4);
            GridTools::scale(4.4688, mesh);
        }else if (type == "c"){
            static CylindricalManifold<dim,spacedim> surface_description;
            {
                Triangulation<spacedim> volume_mesh;
                GridGenerator::cylinder(volume_mesh,300,300);
                std::set<types::boundary_id> boundary_ids;
                boundary_ids.insert (0);
                GridGenerator::extract_boundary_mesh (volume_mesh, mesh, boundary_ids);
            }
            //            std::string mfile = "cylinder_coarse_distorted.msh";
            //            GridIn<2,3> grid_in;
            //            grid_in.attach_triangulation(mesh);
            //            std::ifstream file(mfile.c_str());
            //            Assert(file, ExcFileNotOpen(mfile.c_str()));
            //            grid_in.read_msh(file);
            mesh.set_all_manifold_ids(0);
            mesh.set_manifold (0, surface_description);
            mesh.refine_global(6);
        }else {
            if (type == "b")
            {
                //            Point<dim> p1({-10,-1}), p2({10,1});
                //            GridGenerator::subdivided_hyper_rectangle(mesh,{10,1}, p1, p2);
                //            mesh.refine_global(2);
                
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
    
    
    
    //    mesh.refine_global(3);
    
    std::ofstream output_file("sphere_mesh.vtu");
    GridOut().write_vtu(mesh, output_file);
    
    std::cout << "   Number of active cells: " << mesh.n_active_cells()
    << std::endl
    << "   Total number of cells: " << mesh.n_cells()
    << std::endl;
    double youngs, possions, thickness, density;
    if (type == "r" || type == "b") {
        youngs = 4.32e8;
        possions = 0.;
        thickness = 0.25;
        density = 1000.;
        
    }else if(type == "s"){
        //         youngs = 6.825e7;
        //         possions = 0.3;
        //         thickness = 0.04;
        //         density = 10000.;
        youngs = 28e6;
        possions = 0.28;
        thickness = 0.0625;
        density = 0.000751;
        
    }else{
        youngs = 3e6;
        possions = 0.3;
        thickness = 3;
        density = 10000.;
    }
    
    //    Tensor<4,spacedim> elastic_tensor;
    //    elastic_tensor[0][0][0][0] = 139e9;
    //    elastic_tensor[1][1][1][1] = 139e9;
    //
    //    elastic_tensor[0][0][1][1] = 77.8e9;
    //    elastic_tensor[1][1][0][0] = 77.8e9;
    //
    //    elastic_tensor[0][1][0][1] = 30.6e9;
    //    elastic_tensor[1][0][1][0] = 30.6e9;
    
    Tensor<2,spacedim> dielectric_tensor;
    dielectric_tensor[0][0] = 1.306e-6;
    dielectric_tensor[1][1] = 1.306e-6;
    dielectric_tensor[2][2] = 1.151e-6;
    
    Tensor<3,spacedim> piezoelectric_tensor;
    piezoelectric_tensor[2][0][0] = -1;
    piezoelectric_tensor[2][1][1] = -1;
    piezoelectric_tensor[2][2][2] = 3;
    piezoelectric_tensor[0][0][2] = 2;
    piezoelectric_tensor[1][1][2] = 2;
    
    
    hp::DoFHandler<dim,spacedim> dof_handler(mesh);
    hp::FECollection<dim,spacedim> fe_collection;
    hp::MappingCollection<dim,spacedim> mapping_collection;
    hp::QCollection<dim> q_collection;
    hp::QCollection<dim> boundary_q_collection;
    
    Vector<double> vec_values;
    
    catmull_clark_create_fe_quadrature_and_mapping_collections_and_distribute_dofs(dof_handler,fe_collection,vec_values,mapping_collection,q_collection,boundary_q_collection,3);
    
    
    AffineConstraints<double> constraints;
    constraints.clear();
    //    constraints.close();
    SparsityPattern      sparsity_pattern;
    SparsityPattern      sparsity_pattern_couple;
    SparsityPattern      sparsity_pattern_laplace;
    
    SparseMatrix<double> stiffness_matrix;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> mass_l_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> coupling_matrix;
    SparseMatrix<double> dielectric_matrix;
    //    SparseMatrix<double> boundary_mass_matrix;
    
    Vector<double> solution_disp;
    Vector<double> solution_disp_coupled;
    Vector<double> force_rhs;
    Vector<double> force_coefficient;
    //    Vector<double> boundary_rhs;
    
    
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    
    make_sparsity_pattern_for_coupling_and_laplace_matrices(dof_handler, sparsity_pattern_couple, sparsity_pattern_laplace);
    
    std::ofstream out1("sparsity_pattern_couple.svg");
    sparsity_pattern_couple.print_svg(out1);
    
    std::ofstream out2("sparsity_pattern_laplace.svg");
    sparsity_pattern_laplace.print_svg(out2);
    
    stiffness_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    coupling_matrix.reinit(sparsity_pattern_couple);
    dielectric_matrix.reinit(sparsity_pattern_laplace);
    laplace_matrix.reinit(sparsity_pattern_laplace);
    mass_l_matrix.reinit(sparsity_pattern_laplace);
    //    boundary_mass_matrix.reinit(sparsity_pattern);
    
    solution_disp.reinit(dof_handler.n_dofs());
    force_rhs.reinit(dof_handler.n_dofs());
    force_coefficient.reinit(dof_handler.n_dofs());
    //    boundary_rhs.reinit(dof_handler.n_dofs());
    
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    //    hp::FEValues<dim,spacedim> hp_fe_boundary_values(mapping_collection, fe_collection,
    //                                                     boundary_q_collection,
    //                                            update_values | update_gradients |update_inverse_jacobians|
    //                                            update_quadrature_points | update_JxW_values | update_jacobians);
    
    RightHandSide<spacedim> rhs_function;
    
    FullMatrix<double> cell_mass_matrix;
    FullMatrix<double> cell_stiffness_matrix;
    FullMatrix<double> cell_dielectric_matrix;
    FullMatrix<double> cell_laplace_matrix;
    FullMatrix<double> cell_mass_l_matrix;
    FullMatrix<double> cell_coupling_matrix;
    //    FullMatrix<double> cell_boundary_matrix;
    
    Vector<double>     cell_force_rhs;
    //    Vector<double>     cell_boundary_rhs;
    
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> local_electric_dof_indices;
    
    std::vector<types::global_dof_index> fix_dof_indices;
    std::vector<types::global_dof_index> laplace_fix_dof_indices;
    
    
    double area = 0;
    double curvature = 0;
    
    QGauss<1> Qthickness(2);
    
    bool load_1 = false, load_2 = false, load_3 = false, load_4 = false;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        
        local_electric_dof_indices.resize(dofs_per_cell/spacedim);
        
        hp_fe_values.reinit(cell);
        const FEValues<dim,spacedim> &fe_values = hp_fe_values.get_present_fe_values();
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        
        //        hp_fe_boundary_values.reinit(cell);
        //        const FEValues<dim, spacedim> &fe_boundary_values =
        //        hp_fe_boundary_values.get_present_fe_values();
        
        cell_stiffness_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_dielectric_matrix.reinit(dofs_per_cell/spacedim, dofs_per_cell/spacedim);
        cell_laplace_matrix.reinit(dofs_per_cell/spacedim, dofs_per_cell/spacedim);
        cell_mass_l_matrix.reinit(dofs_per_cell/spacedim, dofs_per_cell/spacedim);
        cell_coupling_matrix.reinit(dofs_per_cell/spacedim, dofs_per_cell);
        //        cell_boundary_matrix.reinit(dofs_per_cell, dofs_per_cell);
        
        
        cell_stiffness_matrix = 0;
        cell_mass_matrix = 0;
        cell_mass_l_matrix = 0;
        cell_laplace_matrix = 0;
        cell_dielectric_matrix = 0;
        cell_coupling_matrix = 0;
        //        cell_boundary_matrix = 0;
        
        cell_force_rhs.reinit(dofs_per_cell);
        cell_force_rhs = 0;
        //        cell_boundary_rhs.reinit(dofs_per_cell);
        //        cell_boundary_rhs = 0;
        
        //        for (const unsigned int q_index : fe_boundary_values.quadrature_point_indices())
        //        {
        //            double jxw = 0;
        //            Tensor<2, spacedim> a_cov; // a_i = x_{,i} , i = 1,2,3
        //            auto jacobian = fe_boundary_values.jacobian(q_index);
        //            for (unsigned int id = 0; id < spacedim; ++id)
        //            {
        //                a_cov[0][id] = jacobian[id][0];
        //                a_cov[1][id] = jacobian[id][1];
        //            }
        //            if(  fe_boundary_values.get_quadrature().point(q_index)[1] == 0){
        //                jxw = a_cov[0].norm() * fe_boundary_values.get_quadrature().weight(q_index);
        //            }else{
        //                jxw = a_cov[1].norm() * fe_boundary_values.get_quadrature().weight(q_index);
        //            }
        //
        //            for (unsigned int i = 0; i < dofs_per_cell/spacedim; ++i)
        //            {
        //                for (unsigned int id = 0; id <spacedim; ++id)
        //                {
        //                    for (unsigned int j = 0; j < dofs_per_cell/spacedim; ++j)
        //                    {
        //                        for (unsigned int jd = 0; jd <spacedim; ++jd) {
        //                            if (id != 2 && jd != 2) {
        //                                cell_boundary_matrix(i*spacedim + id, j*spacedim + jd) +=
        //                                (fe_boundary_values.shape_value(i*spacedim + id, q_index) * // phi_i(x_q)
        //                                 fe_boundary_values.shape_value(j*spacedim + jd, q_index) * // phi_j(x_q)
        //                                 jxw);// dx
        //                            }
        //                        }
        //                    }
        //                }
        //            }
        //            boundary_mass_matrix.add(local_dof_indices, local_dof_indices, cell_boundary_matrix);
        //        }
        //        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        //        {
        //            for (unsigned int j = 0; j < dofs_per_cell; ++j)
        //            {
        //                std::cout << cell_boundary_matrix[i][j] <<" ";
        //            }
        //            std::cout << std::endl;
        //        }
        //        std::cout << std::endl;
        
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
        {
            // covariant base  a_1, a_2, a_3;
            Tensor<2, spacedim> a_cov; // a_i = x_{,i} , i = 1,2,3
            // derivatives of covariant base;
            Tensor<2, dim, Tensor<1,spacedim>> da_cov; // a_{i,j} = x_{,ij} , i,j = 1,2
            auto jacobian = fe_values.jacobian(q_point);
            
            for (unsigned int id = 0; id < spacedim; ++id){
                a_cov[0][id] = jacobian[id][0];
                a_cov[1][id] = jacobian[id][1];
            }
            a_cov[2] = cross_product_3d(a_cov[0], a_cov[1]);
            double detJ = a_cov[2].norm();
            a_cov[2] = a_cov[2]/detJ;
            
            auto jacobian_grad = fe_values.jacobian_grad(q_point);
            for (unsigned int jj = 0; jj < dim; ++jj)
            {
                for (unsigned int kk = 0; kk < spacedim; ++kk)
                {
                    da_cov[0][jj][kk] = jacobian_grad[kk][0][jj];
                    da_cov[1][jj][kk] = jacobian_grad[kk][1][jj];
                }
            }
            
            // covariant metric tensor
            Tensor<2,dim> am_cov = metric_covariant(a_cov);
            Tensor<2,spacedim> am_cov_3d;
            for (unsigned id = 0; id < dim; ++id) {
                for (unsigned jd = 0; jd < dim; ++jd) {
                    am_cov_3d[id][jd] += am_cov[id][jd];
                }
            }
            am_cov_3d[2][2] = 1.;
            // contravariant metric tensor
            Tensor<2,dim> am_contra = metric_contravariant(am_cov);
            Tensor<2,spacedim> am_contra_3d;
            am_contra_3d = transpose(invert(am_cov_3d));
            
            // H tensor
            //            Tensor<4,2> H_tensor;
            //            constitutive_fourth_tensors(H_tensor, am_contra, possions);
            // rotated_permittivity
            Tensor<2,spacedim> rotated_dielectric_tensor;
            // rotated_piezoelectric_tensor
            Tensor<3,spacedim> rotated_piezoelectric_tensor;
            
            Tensor<2,spacedim> rotate_tensor;
            rotate_tensor[0] = a_cov[0]/a_cov[0].norm();
            rotate_tensor[1] = a_cov[1]/a_cov[1].norm();
            rotate_tensor[2] = a_cov[2];
            
            for (unsigned int id = 0; id < spacedim; ++id) {
                for (unsigned int jd = 0; jd < spacedim; ++jd) {
                    for (unsigned int kd = 0; kd < spacedim; ++kd) {
                        for (unsigned int ld = 0; ld < spacedim; ++ld) {
                            rotated_dielectric_tensor[id][jd] += dielectric_tensor[kd][ld] * rotate_tensor[kd][id] * rotate_tensor[ld][jd];
                            rotated_piezoelectric_tensor[id][jd][kd] += piezoelectric_tensor[ld][jd][kd] * rotate_tensor[ld][id] * am_contra_3d[jd][kd];
                        }
                    }
                }
            }
            //
            //            if(dofs_per_cell/spacedim == 12){
            //                double curvature_1 = 1./detJ * scalar_product(cross_product_3d(da_cov[0][0], a_cov[1]) + cross_product_3d(a_cov[0], da_cov[1][0]), covariant_to_contravariant(a_cov)[0]);
            //                double curvature_2 = 1./detJ * scalar_product(cross_product_3d(da_cov[0][1], a_cov[1]) + cross_product_3d(a_cov[0], da_cov[1][1]), covariant_to_contravariant(a_cov)[1]);
            //
            //                std::cout << "curvature = " << curvature_1 << " "<< curvature_2 << std::endl;
            //                //            std::cout << "mean curvature = " << 0.5 * trace(curvature_tensor) << std::endl;
            //                //            std::cout << "gaussian curvature = " << determinant(curvature_tensor) << std::endl;
            //                std::cout << "curvature error = "<< numbers::NumberTraits<double>::abs( curvature_1 - 1./300.)/(1./300.)<<std::endl;
            //            }
            
            area += fe_values.JxW(q_point);
            
            std::vector<Tensor<2, dim, Tensor<1, spacedim>>>
            bn_vec(dofs_per_cell/spacedim),
            bm_vec(dofs_per_cell/spacedim);
            
            for (unsigned int i_shape = 0; i_shape < dofs_per_cell/spacedim; ++i_shape) {
                local_electric_dof_indices[i_shape] = local_dof_indices[spacedim*i_shape]/spacedim;
                // compute first and second grad of i_shape function
                Tensor<1, spacedim> shape_grad = fe_values.shape_grad(i_shape * spacedim, q_point);
                Tensor<2, spacedim> shape_hessian = fe_values.shape_hessian(i_shape * spacedim, q_point);
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
                //                std::cout << "shape_der = " << shape_der <<std::endl;
                //                std::cout << "shape_der2 = " << shape_der2 << std::endl;
                //  computation of the B operator (strains) for i_shape function
                Tensor<2,dim, Tensor<1, spacedim>> membrane_strain;
                Tensor<2,dim, Tensor<1, spacedim>> bending_strain;
                for (unsigned int ii = 0; ii < dim; ++ii) {
                    for (unsigned int jj = 0; jj < dim; ++jj) {
                        for (unsigned int id = 0; id < spacedim; ++id) {
                            membrane_strain[ii][jj][id] = 0.5 * (a_cov[ii][id] * shape_der[jj] + a_cov[jj][id] * shape_der[ii]);
                            bending_strain[ii][jj][id] = - shape_der2[ii][jj] * a_cov[2][id] + (shape_der[0] * cross_product_3d(da_cov[ii][jj], a_cov[1])[id] + shape_der[1] * cross_product_3d(a_cov[0], da_cov[ii][jj])[id])/detJ + scalar_product(a_cov[2], da_cov[ii][jj]) * (shape_der[0] * cross_product_3d(a_cov[1], a_cov[2])[id] + shape_der[1] * cross_product_3d(a_cov[2], a_cov[0])[id]) / detJ;
                            //                            bending_strain[ii][jj][id] = - shape_der2[ii][jj] * a_cov[2][id] + (shape_der[0] * cross_product_3d(da_cov[ii][jj], a_cov[1])[id] + shape_der[1] * cross_product_3d(a_cov[0], da_cov[ii][jj])[id])/detJ;
                            
                        }
                        //                        std::cout << "m = "<< membrane_strain[ii][jj] << "\n";
                        //                        std::cout <<" b1 = "<<- shape_der2[ii][jj] * a_cov[2] <<"\n";
                        //                        std::cout <<" b2 = "<< (shape_der[0] * cross_product_3d(da_cov[ii][jj], a_cov[1]) + shape_der[1] * cross_product_3d(a_cov[0], da_cov[ii][jj]))/detJ <<"\n";
                        //                        std::cout <<" b3 = "<<  scalar_product(a_cov[2], da_cov[ii][jj]) * (shape_der[0] * cross_product_3d(a_cov[1], a_cov[2]) + shape_der[1] * cross_product_3d(a_cov[2], a_cov[0])) / detJ <<"\n";
                    }
                }
                
                bn_vec[i_shape] = membrane_strain;
                bm_vec[i_shape] = bending_strain;
            } // loop over shape functions
            
            for (unsigned int iq_1d = 0; iq_1d < Qthickness.size(); ++iq_1d) {
                double u_t = Qthickness.get_points()[iq_1d][0];
                double w_t = Qthickness.get_weights()[iq_1d];
                Tensor<2, dim> gm_cov; // g_i = r_{,i} , i = 1,2,3
                double zeta = thickness * (u_t - 0.5);
                for (unsigned id = 0; id < dim; ++id) {
                    for (unsigned jd = 0; jd < dim; ++jd) {
                        gm_cov[id][jd] += am_cov[id][jd] - 2*zeta * scalar_product(da_cov[id][jd],a_cov[2]);
                    }
                }
                auto gm_contra = transpose(invert(gm_cov));
                
                // H tensor
                Tensor<4,2> H_tensor;
                constitutive_fourth_tensors(H_tensor, gm_contra, possions);
                
                for (unsigned int i_node = 0; i_node < dofs_per_cell/spacedim; ++i_node)
                {
                    Tensor<2, dim, Tensor<1, spacedim>> hn,hm;
                    double coeff = youngs/ (1. - possions*possions);
                    
                    for(unsigned int ii = 0; ii < dim ; ++ii)
                    for(unsigned int jj = 0; jj < dim ; ++jj)
                    for(unsigned int kk = 0; kk < dim ; ++kk)
                    for(unsigned int ll = 0; ll < dim ; ++ll)
                    for (unsigned int id = 0; id < spacedim; ++id) {
                        hn[ii][jj][id] += coeff * H_tensor[ii][jj][kk][ll] * bn_vec[i_node][kk][ll][id];
                        hm[ii][jj][id] += coeff * H_tensor[ii][jj][kk][ll] * bm_vec[i_node][kk][ll][id];
                    }
                    
                    for (unsigned int j_node = 0; j_node < dofs_per_cell/spacedim; ++j_node)
                    {
                        Tensor<2, spacedim> sn,sm;
                        for(unsigned int ii = 0; ii < dim ; ++ii)
                        for(unsigned int jj = 0; jj < dim ; ++jj)
                        for (unsigned int id = 0; id < spacedim; ++id)
                        for (unsigned int jd = 0; jd < spacedim; ++jd) {
                            sn[id][jd] += hn[ii][jj][id] * bn_vec[j_node][ii][jj][jd];
                            sm[id][jd] += hm[ii][jj][id] * bm_vec[j_node][ii][jj][jd];
                        }
                        
                        for (unsigned int id = 0; id < spacedim; ++id) {
                            for (unsigned int jd = 0; jd < spacedim; ++jd) {
                                cell_stiffness_matrix(i_node * spacedim + id, j_node * spacedim + jd) += (thickness * sn[id][jd] + thickness * zeta * zeta * sm[id][jd]) * w_t * fe_values.JxW(q_point);
                                if(id == jd)
                                    cell_mass_matrix(i_node*spacedim+id, j_node*spacedim+jd) += density * thickness * fe_values.shape_value(i_node*spacedim+id, q_point) * fe_values.shape_value(j_node*spacedim+jd, q_point) * w_t * fe_values.JxW(q_point);
                            }
                        }
                    }
                }
            }
            
            for (unsigned int i_node = 0; i_node < dofs_per_cell/spacedim; ++i_node)
            {
                //                Tensor<2, dim, Tensor<1, spacedim>> hn,hm;
                //                double c1 = youngs*thickness/ (1. - possions*possions),
                //                c2 = youngs*thickness*thickness*thickness/(12. * (1. - possions*possions));
                //                for(unsigned int ii = 0; ii < dim ; ++ii)
                //                    for(unsigned int jj = 0; jj < dim ; ++jj)
                //                        for(unsigned int kk = 0; kk < dim ; ++kk)
                //                            for(unsigned int ll = 0; ll < dim ; ++ll)
                //                                for (unsigned int id = 0; id < spacedim; ++id) {
                //                                    hn[ii][jj][id] += c1 * H_tensor[ii][jj][kk][ll] * bn_vec[i_node][kk][ll][id];
                //                                    hm[ii][jj][id] += c2 * H_tensor[ii][jj][kk][ll] * bm_vec[i_node][kk][ll][id];
                //                                }
                
                //                for (unsigned int j_node = 0; j_node < dofs_per_cell/spacedim; ++j_node)
                //                {
                ////                    Tensor<2, spacedim> sn,sm;
                ////                    for(unsigned int ii = 0; ii < dim ; ++ii)
                ////                        for(unsigned int jj = 0; jj < dim ; ++jj)
                ////                            for (unsigned int id = 0; id < spacedim; ++id)
                ////                                for (unsigned int jd = 0; jd < spacedim; ++jd) {
                ////                                    sn[id][jd] += hn[ii][jj][id] * bn_vec[j_node][ii][jj][jd];
                ////                                    sm[id][jd] += hm[ii][jj][id] * bm_vec[j_node][ii][jj][jd];
                ////                                }
                //
                ////                    for (unsigned int id = 0; id < spacedim; ++id) {
                ////                        for (unsigned int jd = 0; jd < spacedim; ++jd) {
                ////                            cell_stiffness_matrix(i_node * spacedim + id, j_node * spacedim + jd) += (sn[id][jd] + sm[id][jd]) * fe_values.JxW(q_point);
                ////                            cell_mass_matrix(i_node*spacedim+id, j_node*spacedim+jd) += density * thickness * fe_values.shape_value(i_node*spacedim+id, q_point) * fe_values.shape_value(j_node*spacedim+jd, q_point) * fe_values.JxW(q_point);
                ////                        }
                ////                    }
                //
                //                    for (unsigned int ii = 0; ii < spacedim; ++ii) {
                //                        for (unsigned int jj = 0; jj < spacedim; ++jj) {
                //                            cell_dielectric_matrix(i_node, j_node) += (pow(thickness,5.) / 80. * rotated_dielectric_tensor[ii][jj] * fe_values.shape_grad(i_node * spacedim, q_point)[ii] * fe_values.shape_grad(j_node * spacedim, q_point)[jj] + pow(thickness, 3.)/3. * rotated_dielectric_tensor[ii][jj] * fe_values.shape_value(i_node*spacedim, q_point) * fe_values.shape_value(j_node*spacedim, q_point) * fe_values.normal_vector(q_point)[ii] * fe_values.normal_vector(q_point)[jj]) *
                //                            fe_values.JxW(q_point);
                //                        }
                //                    }
                //
                //                    cell_laplace_matrix(i_node, j_node) += fe_values.shape_grad(i_node * spacedim, q_point) * fe_values.shape_grad(j_node * spacedim, q_point) * fe_values.JxW(q_point);
                //
                //                    cell_mass_l_matrix(i_node, j_node) += fe_values.shape_value(i_node * spacedim, q_point) * fe_values.shape_value(j_node * spacedim, q_point) * fe_values.JxW(q_point);
                //
                //                    for (unsigned int ii = 0; ii < spacedim; ++ii) {
                //                        for (unsigned int jj = 0; jj < dim; ++jj) {
                //                            for (unsigned int kk = 0; kk < dim; ++kk){
                //                                for (unsigned int jd = 0; jd < spacedim; ++jd) {
                //                                    cell_coupling_matrix(i_node, j_node * spacedim + jd) -= (pow(thickness, 3.) / 12. * rotated_piezoelectric_tensor[ii][jj][kk] * fe_values.shape_grad(i_node * spacedim, q_point)[ii] * bn_vec[j_node][jj][kk][jd] + pow(thickness, 3.)/6. * rotated_piezoelectric_tensor[ii][jj][kk] * fe_values.shape_value(i_node * spacedim, q_point) * fe_values.normal_vector(q_point)[ii] * bm_vec[j_node][jj][kk][jd] )*
                //                                    fe_values.JxW(q_point);
                //                                }
                //                            }
                //                        }
                //                    }
                //                }// loop over nodes (i)
                // uniform load
                if (type == "r") {
                    //                    cell_force_rhs(i_node*spacedim + 1) += 45 * sqrt(2) * fe_values.shape_value(i_node*spacedim + 1 , q_point)* fe_values.JxW(q_point); // f_y = -45*sqrt(2)
                    //                    cell_force_rhs(i_node*spacedim + 2) += - 45 * sqrt(2) * fe_values.shape_value(i_node*spacedim + 2 , q_point)* fe_values.JxW(q_point); // f_z = -45*sqrt(2)
                    //                    cell_force_rhs(i_node*spacedim + 1) += - 90 * fe_values.shape_value(i_node*spacedim + 1 , q_point)* fe_values.JxW(q_point); // f_y = -90
                    cell_force_rhs(i_node*spacedim + 2) += - 90 * fe_values.shape_value(i_node*spacedim + 2 , q_point)* fe_values.JxW(q_point); // f_z = -90
                    
                }else if (type == "s" ){
                    //                    cell_force_rhs(i_node*spacedim) += 1 * fe_values.shape_value(i_node*spacedim , q_point)* fe_values.JxW(q_point); // f_x = 1
                }else if (type == "b" ){
                    cell_force_rhs(i_node*spacedim + 2) += 1 * fe_values.shape_value(i_node*spacedim + 2, q_point)* fe_values.JxW(q_point); // f_z = 1
                }else if(type == "c" ){
                    // uniform load
                    //                    cell_force_rhs(i_node*spacedim + 2) += 1 * fe_values.shape_value(i_node*spacedim +2 , q_point)* fe_values.JxW(q_point); // f_y = 1
                }
            }// loop over nodes (j)
        }// loop over quadrature points
        if(type == "c" ){
            if(load_1 == false){
                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
                    if( numbers::NumberTraits<double>::abs( cell->vertex(ivert)[0]) < 1e-2 && numbers::NumberTraits<double>::abs( cell->vertex(ivert)[1]) < 1e-4 && cell->vertex(ivert)[2] >299.5){
                        load_1 = true;
                        for (unsigned int idof = 0; idof < dofs_per_cell/spacedim; ++idof) {
                            double shape = 0;
                            unsigned int vert_dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                            
                            if (vert_dof_id == local_dof_indices[0]) {
                                shape = cell->get_fe().shape_value(idof*spacedim, {0,0});
                            }else if (vert_dof_id == local_dof_indices[3]){
                                shape = cell->get_fe().shape_value(idof*spacedim, {1,0});
                            }else if (vert_dof_id == local_dof_indices[6]){
                                shape = cell->get_fe().shape_value(idof*spacedim, {0,1});
                            }else if (vert_dof_id == local_dof_indices[9]){
                                shape = cell->get_fe().shape_value(idof*spacedim, {1,1});
                            }
                            cell_force_rhs(idof*spacedim+2) = -1.*shape;
                        }
                    }
                }
            }
            if(load_2 == false){
                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
                    if( numbers::NumberTraits<double>::abs( cell->vertex(ivert)[0]) < 1e-4 && numbers::NumberTraits<double>::abs( cell->vertex(ivert)[1]) < 1e-4 && cell->vertex(ivert)[2] < -299.5){
                        load_2 = true;
                        for (unsigned int idof = 0; idof < dofs_per_cell/spacedim; ++idof) {
                            double shape = 0;
                            unsigned int vert_dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                            
                            if (vert_dof_id == local_dof_indices[0]) {
                                shape = cell->get_fe().shape_value(idof*spacedim, {0,0});
                            }else if (vert_dof_id == local_dof_indices[3]){
                                shape = cell->get_fe().shape_value(idof*spacedim, {1,0});
                            }else if (vert_dof_id == local_dof_indices[6]){
                                shape = cell->get_fe().shape_value(idof*spacedim, {0,1});
                            }else if (vert_dof_id == local_dof_indices[9]){
                                shape = cell->get_fe().shape_value(idof*spacedim, {1,1});
                            }
                            cell_force_rhs(idof*spacedim+2) = 1.*shape;
                        }
                    }
                }
            }
        }
        
        //        if(type == "s" ){
        //            if(load_1 == false){
        //                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
        //                    if( cell->vertex(ivert)[1] >= 10.- 1e-4){
        //                        load_1 = true;
        //                        for (unsigned int idof = 0; idof < dofs_per_cell/spacedim; ++idof) {
        //                            double shape = 0;
        //                            unsigned int vert_dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
        //
        //                            if (vert_dof_id == local_dof_indices[0]) {
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,0});
        //                            }else if (vert_dof_id == local_dof_indices[3]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,0});
        //                            }else if (vert_dof_id == local_dof_indices[6]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,1});
        //                            }else if (vert_dof_id == local_dof_indices[9]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,1});
        //                            }
        //                            cell_force_rhs(idof*spacedim+1) = 2.*shape;
        //                        }
        //                    }
        //                }
        //            }
        //            if(load_2 == false){
        //                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
        //                    if( cell->vertex(ivert)[1] <= -10.+ 1e-4){
        //                        load_2 = true;
        //                        for (unsigned int idof = 0; idof < dofs_per_cell/spacedim; ++idof) {
        //                            double shape = 0;
        //                            unsigned int vert_dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
        //
        //                            if (vert_dof_id == local_dof_indices[0]) {
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,0});
        //                            }else if (vert_dof_id == local_dof_indices[3]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,0});
        //                            }else if (vert_dof_id == local_dof_indices[6]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,1});
        //                            }else if (vert_dof_id == local_dof_indices[9]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,1});
        //                            }
        //                            cell_force_rhs(idof*spacedim+1) = -2.*shape;
        //                        }
        //                    }
        //                }
        //            }
        //            if(load_3 == false){
        //                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
        //                    if( cell->vertex(ivert)[2] >= 10.- 1e-4){
        //                        load_3 = true;
        //                        for (unsigned int idof = 0; idof < dofs_per_cell/spacedim; ++idof) {
        //                            double shape = 0;
        //                            unsigned int vert_dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
        //
        //                            if (vert_dof_id == local_dof_indices[0]) {
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,0});
        //                            }else if (vert_dof_id == local_dof_indices[3]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,0});
        //                            }else if (vert_dof_id == local_dof_indices[6]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,1});
        //                            }else if (vert_dof_id == local_dof_indices[9]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,1});
        //                            }
        //                            cell_force_rhs(idof*spacedim+2) = -2.*shape;
        //                        }
        //                    }
        //                }
        //            }
        //            if(load_4 == false){
        //                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
        //                    if( cell->vertex(ivert)[2] <= -10.+ 1e-4){
        //                        load_4 = true;
        //                        for (unsigned int idof = 0; idof < dofs_per_cell/spacedim; ++idof) {
        //                            double shape = 0;
        //                            unsigned int vert_dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
        //
        //                            if (vert_dof_id == local_dof_indices[0]) {
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,0});
        //                            }else if (vert_dof_id == local_dof_indices[3]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,0});
        //                            }else if (vert_dof_id == local_dof_indices[6]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {0,1});
        //                            }else if (vert_dof_id == local_dof_indices[9]){
        //                                shape = cell->get_fe().shape_value(idof*spacedim, {1,1});
        //                            }
        //                            cell_force_rhs(idof*spacedim+2) = 2.*shape;
        //                        }
        //                    }
        //                }
        //            }
        //        }
        
        force_rhs.add(local_dof_indices, cell_force_rhs);
        stiffness_matrix.add(local_dof_indices, local_dof_indices, cell_stiffness_matrix);
        mass_matrix.add(local_dof_indices, local_dof_indices, cell_mass_matrix);
        dielectric_matrix.add(local_electric_dof_indices, local_electric_dof_indices, cell_dielectric_matrix);
        coupling_matrix.add(local_electric_dof_indices, local_dof_indices, cell_coupling_matrix);
        
        laplace_matrix.add(local_electric_dof_indices, local_electric_dof_indices, cell_laplace_matrix);
        mass_l_matrix.add(local_electric_dof_indices, local_electric_dof_indices, cell_mass_l_matrix);
        
        // boundary conditions
        for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_cell; ++ivert)
        {
            if (type == "r") {
                //                if (cell->vertex(ivert)[0] == 0 || cell->vertex(ivert)[0] == 100 || cell->vertex(ivert)[1] == 0 || cell->vertex(ivert)[1] == 100 )
                if (cell->vertex(ivert)[0] >= 50 || cell->vertex(ivert)[0] <= 1e-6)
                    //                if ( cell->vertex(ivert)[0] <= 0 || cell->vertex(ivert)[0] >= 50 || cell->vertex(ivert)[1] > 16 || cell->vertex(ivert)[1] < -16)
                {
                    unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
//                    fix_dof_indices.push_back(dof_id);
                    fix_dof_indices.push_back(dof_id+1);
                    fix_dof_indices.push_back(dof_id+2);
                    laplace_fix_dof_indices.push_back(dof_id/3);
                }
            }else if (type == "s"){
                if ( cell->vertex(ivert)[0] == 0){
                    unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                    //                    fix_dof_indices.push_back(dof_id);
                    //                    fix_dof_indices.push_back(dof_id+1);
                    //                    fix_dof_indices.push_back(dof_id+2);
                    laplace_fix_dof_indices.push_back(dof_id/3);
                }
            }else if (type == "c"){
                if ( cell->vertex(ivert)[0] <= -300. || cell->vertex(ivert)[0] >= 300.){
                    //                if ( cell->vertex(ivert)[0] == -300.){
                    unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                    fix_dof_indices.push_back(dof_id);
                    fix_dof_indices.push_back(dof_id+1);
                    //                    fix_dof_indices.push_back(dof_id+2);
                    laplace_fix_dof_indices.push_back(dof_id/3);
                }
            }else if (type == "b"){
                //                if ( cell->vertex(ivert)[0] >= 10 || cell->vertex(ivert)[0] <= -10)
                //                if ( cell->vertex(ivert)[1] > 16 || cell->vertex(ivert)[1] < -16)
                if ( cell->vertex(ivert)[2] > 24.8 || cell->vertex(ivert)[1] < - 24.8)
                {
                    unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                    fix_dof_indices.push_back(dof_id);
                    fix_dof_indices.push_back(dof_id+1);
                    fix_dof_indices.push_back(dof_id+2);
                    laplace_fix_dof_indices.push_back(dof_id/3);
                }
            }
        }
        
    } // loop over cells
    
    std::cout << " area = " << area << std::endl;
    //    std::cout << " error of area = " << (area - 2*numbers::PI*300*600) / (2*numbers::PI*300*600) << std::endl;
    
    
    std::sort(fix_dof_indices.begin(), fix_dof_indices.end());
    auto last = std::unique(fix_dof_indices.begin(), fix_dof_indices.end());
    fix_dof_indices.erase(last, fix_dof_indices.end());
    for (unsigned int idof = 0; idof <fix_dof_indices.size(); ++idof) {
        constraints.add_line(fix_dof_indices[idof]);
    }
    constraints.close();
    
    std::sort(laplace_fix_dof_indices.begin(), laplace_fix_dof_indices.end());
    auto last2 = std::unique(laplace_fix_dof_indices.begin(), laplace_fix_dof_indices.end());
    laplace_fix_dof_indices.erase(last2, laplace_fix_dof_indices.end());
    
    //    for (unsigned int idof = 0; idof <laplace_fix_dof_indices.size(); ++idof) {
    //        for (unsigned int jdof = 0; jdof <dof_handler.n_dofs()/3; ++jdof) {
    //            if (laplace_fix_dof_indices[idof] == jdof){
    //                laplace_matrix.set(laplace_fix_dof_indices[idof], laplace_fix_dof_indices[idof], 1e20);
    //                mass_l_matrix.set(laplace_fix_dof_indices[idof], laplace_fix_dof_indices[idof], 1e20);
    ////                coupled_system_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e20);
    //            }
    //            else
    //            {
    //                laplace_matrix.set(laplace_fix_dof_indices[idof], jdof, 0);
    //                laplace_matrix.set(jdof, laplace_fix_dof_indices[idof], 0);
    //                mass_l_matrix.set(laplace_fix_dof_indices[idof], jdof, 0);
    //                mass_l_matrix.set(jdof, laplace_fix_dof_indices[idof], 0);
    //            }
    //        }
    //    }
    //
    //    std::vector<Vector<double> >        l_eigenvectors(dof_handler.n_dofs()/3);
    //    std::vector<std::complex<double>>   l_eigenvalues(10);
    //    for (unsigned int i = 0; i < l_eigenvectors.size(); ++i){
    //        l_eigenvectors[i].reinit(dof_handler.n_dofs()/3);
    //    }
    
    //    SolverControl solver_control(dof_handler.n_dofs()/3, 1e-12);
    //    SparseDirectUMFPACK inverse;
    //    inverse.initialize (laplace_matrix);
    //    const unsigned int num_arnoldi_vectors = 50;
    //    ArpackSolver::AdditionalData additional_data(num_arnoldi_vectors, ArpackSolver::algebraically_largest, true);
    //    ArpackSolver eigensolver(solver_control,additional_data);
    //    eigensolver.solve(laplace_matrix, mass_l_matrix, inverse, l_eigenvalues, l_eigenvectors);
    //    for (unsigned int i = 0; i < l_eigenvalues.size(); ++i) {
    //        std::cout <<"Eigenvalue "<<i<<" = "<<l_eigenvalues[i]<<std::endl;
    //        vtk_plot("sphere_laplace_eigen_solution_"+std::to_string(i)+".vtu", dof_handler, mapping_collection, vec_values, solution_disp ,l_eigenvectors[i]);
    //    }
    
    
    std::cout << "solve elastic system of equations.\n";
    //    stiffness_matrix.add(1e20,boundary_mass_matrix);
    const auto op_k = linear_operator(stiffness_matrix);
    const auto op_kmod = constrained_linear_operator(constraints, op_k);
    Vector<double> rhs_mod = constrained_right_hand_side(constraints, op_k, force_rhs);
    //
    PreconditionJacobi<SparseMatrix<double>> preconditioner_k;
    preconditioner_k.initialize(stiffness_matrix);
    ReductionControl reduction_control_K(dof_handler.n_dofs()*3, 1.0e-10, 1.0e-10);
    SolverCG<Vector<double>>    solver_K(reduction_control_K);
    const auto op_k_inv = inverse_operator(op_kmod, solver_K, preconditioner_k);
    solution_disp = op_k_inv * rhs_mod;
    ////    constraints.distribute(solution_disp);
    ////    SolverControl            ela_solver_control(dof_handler.n_dofs()*3, 1e-10);
    ////    SolverCG<Vector<double>> ela_solver(ela_solver_control);
    ////    ela_solver.solve(op_kmod, solution_disp, rhs_mod, PreconditionIdentity());
    ////    constraints.distribute(solution_disp);
    std::cout << "   " << reduction_control_K.last_step()
    << " CG iterations needed to obtain convergence." << std::endl;
    ////
    if(type == "r")
        vtk_plot("curved_roof_solution.vtu", dof_handler, mapping_collection, vec_values, solution_disp);
    if(type == "s")
        vtk_plot("hemisphere_uniform_solution.vtu", dof_handler, mapping_collection, vec_values, solution_disp);
    if(type == "c")
        vtk_plot("cylinder_pinched_solution.vtu", dof_handler, mapping_collection, vec_values, solution_disp);
    if(type == "b")
        vtk_plot("curved_beam_solution.vtu", dof_handler, mapping_collection, vec_values, solution_disp);
    
    std::cout << "eigen analysis:\n"<<std::endl;
    //    std::cout << "solve coupled system of equations.\n";
    //
    const auto op_m = linear_operator(dielectric_matrix);
    PreconditionJacobi<SparseMatrix<double>> preconditioner_m;
    preconditioner_m.initialize(dielectric_matrix);
    ReductionControl reduction_control_M(4000, 1.0e-12, 1.0e-12);
    SolverCG<Vector<double>>    solver_M(reduction_control_M);
    const auto op_m_inv = inverse_operator(op_m, solver_M, preconditioner_m);
    const auto op_c = linear_operator(coupling_matrix);
    const auto op_s = op_k - transpose_operator(op_c) * op_m_inv * op_c;
    
    //    const auto op_smod = constrained_linear_operator(constraints, op_s);
    //    rhs_mod = constrained_right_hand_side(constraints, op_s, force_rhs);
    //
    //    SolverControl            solver_control_s(4000, 1.e-10);
    //    SolverCG<Vector<double>> solver_s(solver_control_s);
    //    const auto op_s_inv = inverse_operator(op_smod, solver_s, PreconditionIdentity());
    //
    //    solution_disp_coupled = op_s_inv * rhs_mod;
    //
    //    Vector<double> potential = op_m_inv * op_c * solution_disp_coupled;
    //    std::cout <<"potential = " << potential << std::endl;
    //
    //    vtk_plot("coupled_shell_solution.vtu", dof_handler, mapping_collection, vec_values, solution_disp_coupled, potential);
    
    // eigenvalue problem
    
    LAPACKFullMatrix<double> coupled_system_matrix(dof_handler.n_dofs(),dof_handler.n_dofs());
    LAPACKFullMatrix<double> mass_matrix_full(dof_handler.n_dofs(),dof_handler.n_dofs());
    
    for (unsigned int icol = 0; icol < dof_handler.n_dofs(); ++icol) {
        std::cout <<"col = " << icol+1 << "/" << dof_handler.n_dofs() <<"\n";
        Vector<double> ith_vec(dof_handler.n_dofs());
        ith_vec[icol] = 1.;
        Vector<double> column_vec = op_s*ith_vec;
        Vector<double> column_vec_mass(dof_handler.n_dofs());
        mass_matrix.vmult(column_vec_mass, ith_vec);
        for (unsigned int irow = 0; irow < dof_handler.n_dofs(); ++irow) {
            coupled_system_matrix.set(irow, icol, column_vec[irow]);
            mass_matrix_full.set(irow, icol, column_vec_mass[irow]);
        }
    }
    
    std::vector<Vector<double> >        eigenvectors(dof_handler.n_dofs());
    std::vector<std::complex<double>>   eigenvalues(30);
    for (unsigned int i = 0; i < eigenvectors.size(); ++i){
        eigenvectors[i].reinit(dof_handler.n_dofs());
    }
    
    for (unsigned int idof = 0; idof <fix_dof_indices.size(); ++idof) {
        for (unsigned int jdof = 0; jdof <dof_handler.n_dofs(); ++jdof) {
            if (fix_dof_indices[idof] == jdof){
                stiffness_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);
                mass_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);
                coupled_system_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);
                mass_matrix_full.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);
            }
            else
            {
                stiffness_matrix.set(fix_dof_indices[idof], jdof, 0);
                stiffness_matrix.set(jdof, fix_dof_indices[idof], 0);
                mass_matrix.set(fix_dof_indices[idof], jdof, 0);
                mass_matrix.set(jdof, fix_dof_indices[idof], 0);
                coupled_system_matrix.set(fix_dof_indices[idof], jdof, 0);
                coupled_system_matrix.set(jdof, fix_dof_indices[idof], 0);
                mass_matrix_full.set(fix_dof_indices[idof], jdof, 0);
                mass_matrix_full.set(jdof, fix_dof_indices[idof], 0);
            }
        }
    }
    
    stiffness_matrix.compress(VectorOperation::add);
    mass_matrix.compress(VectorOperation::add);
    double min_spurious_eigenvalue = std::numeric_limits<double>::max(),
    max_spurious_eigenvalue = -std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    if (constraints.is_constrained(i))
    {
        const double ev         = stiffness_matrix(i, i) / mass_matrix(i, i);
        min_spurious_eigenvalue = std::min(min_spurious_eigenvalue, ev);
        max_spurious_eigenvalue = std::max(max_spurious_eigenvalue, ev);
    }
    std::cout << "   Spurious eigenvalues are all in the interval "
    << "[" << min_spurious_eigenvalue << ","
    << max_spurious_eigenvalue << "]" << std::endl;
    
    SolverControl solver_control(dof_handler.n_dofs(), 1e-16);
    SparseDirectUMFPACK inverse;
    inverse.initialize (stiffness_matrix);
    const unsigned int num_arnoldi_vectors = 200;
    ArpackSolver::AdditionalData additional_data(num_arnoldi_vectors, ArpackSolver::algebraically_largest, true);
    ArpackSolver eigensolver(solver_control,additional_data);
    eigensolver.solve(stiffness_matrix, mass_matrix, inverse, eigenvalues, eigenvectors);
    
    for (unsigned int i = 0; i < eigenvalues.size(); ++i) {
        std::cout <<"Eigenvalue "<<i<<" = "<<eigenvalues[i] << " frequence = " << std::sqrt(eigenvalues[i].real()) << " (rad/s) " << " = " << std::sqrt(eigenvalues[i]).real() /(2.* numbers::PI) << "(Hz)" <<std::endl;
        if (type == "c") {
            vtk_eigen_plot("cylinder_eigen_solution_"+std::to_string(i)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i]);
        }else if (type == "s"){
            vtk_eigen_plot("sphere_eigen_solution_"+std::to_string(i)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i]);
        }else if (type == "r"){
            vtk_eigen_plot("curved_roof_eigen_solution_"+std::to_string(i)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i]);
        }else if (type == "b"){
            vtk_eigen_plot("curved_beam_eigen_solution_"+std::to_string(i)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i]);
        }
    }
    
    //    const auto system_lop = linear_operator(coupled_system_matrix);
    //    SolverCG<Vector<double>> solver_inverse(solver_control);
    //    const auto s_inv = inverse_operator(system_lop, solver_inverse, preconditioner_k);
    //
    //    eigensolver.solve(coupled_system_matrix,mass_matrix,s_inv,eigenvalues,eigenvectors);
    //    for (unsigned int i = 0; i < eigenvalues.size(); ++i) {
    //        std::cout <<"Eigenvalue "<<i<<" = "<<eigenvalues[i]<<std::endl;
    //        vtk_plot("coupled_plate_eigen_solution_"+std::to_string(i)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i]);
    //    }
    //        inverse.initialize (coupled_system_matrix);
    //
    //        eigensolver.solve(coupled_system_matrix,mass_matrix,inverse,eigenvalues,eigenvectors);
    //
//    coupled_system_matrix.compute_eigenvalues();
    coupled_system_matrix.compute_generalized_eigenvalues_symmetric(mass_matrix_full, eigenvectors);
    unsigned int i_e = 0;
    for (unsigned int i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues[i] = coupled_system_matrix.eigenvalue(i);
        if (eigenvalues[i].real() > 0 && eigenvalues[i].real() != 1) {
            ++i_e;
            std::cout <<"Coupled Eigenvalue "<<i<<" = "<<eigenvalues[i] << " frequence = " << std::sqrt(eigenvalues[i].real()) << " (rad/s) " << " = " << std::sqrt(eigenvalues[i]).real() /(2.* numbers::PI) << "(Hz)" <<std::endl;
            vtk_eigen_plot("coupled_eigen_solution_"+std::to_string(i_e)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i]);
        }
    }
    return 0;
}
