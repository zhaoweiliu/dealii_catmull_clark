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

void vtk_plot(const std::string &filename, const hp::DoFHandler<2, 3> &dof_handler, const hp::MappingCollection<2, 3> mapping, const Vector<double> vertices, const Vector<double> solution, const Vector<double> potential = Vector<double>()){
    
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



int main()
{
    const int dim = 2, spacedim = 3;
    
    std::string type = "r";
    
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
        mesh.refine_global(3);
//        GridTools::rotate(numbers::PI*0.25, 0, mesh);
    }else{
        if (type == "s") {
            static SphericalManifold<dim,spacedim> surface_description;
            {
                Triangulation<spacedim> volume_mesh;
                GridGenerator::half_hyper_ball(volume_mesh);
//                GridGenerator::hyper_ball(volume_mesh);
                std::set<types::boundary_id> boundary_ids;
                boundary_ids.insert (0);
                GridGenerator::extract_boundary_mesh (volume_mesh, mesh,
                                                      boundary_ids);
            }
            mesh.set_all_manifold_ids(0);
            mesh.set_manifold (0, surface_description);
            mesh.refine_global(5);
            GridTools::scale(10., mesh);
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
            mesh.refine_global(4);
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
            else{
                if (type == "pb") {
                    std::string mfile = "/Users/zhaoweiliu/Documents/geometries/piezo_buzzer.msh";
                    GridIn<2,3> grid_in;
                    grid_in.attach_triangulation(mesh);
                    std::ifstream file(mfile.c_str());
                    Assert(file, ExcFileNotOpen(mfile.c_str()));
                    grid_in.read_msh(file);
                }
            }
        }
    }
        
    Tensor<2,spacedim> dielectric_tensor;
    dielectric_tensor[0][0] = 1.306e-9;
    dielectric_tensor[1][1] = 1.306e-9;
    dielectric_tensor[2][2] = 1.151e-9;
    
    Tensor<3,spacedim> piezoelectric_tensor;
    piezoelectric_tensor[2][0][0] = -5.2;
    piezoelectric_tensor[2][1][1] = -5.2;
    piezoelectric_tensor[2][2][2] = 15.1;
    piezoelectric_tensor[0][0][2] = 12.7;
    piezoelectric_tensor[1][1][2] = 12.7;
    
//    mesh.refine_global(3);
    
    std::ofstream output_file("test_mesh.vtk");
    GridOut().write_vtk (mesh, output_file);

    std::cout << "   Number of active cells: " << mesh.n_active_cells()
    << std::endl
    << "   Total number of cells: " << mesh.n_cells()
    << std::endl;
    double youngs, possions, thickness, density;
    if (type == "r" || type == "b") {
//        youngs = 4.32e8;
        youngs = 200e9;
        possions = 0.;
        thickness = 0.25;
        density = 36.;
        
    }else if(type == "s"){
         youngs = 6.825e7;
         possions = 0.3;
         thickness = 0.04;
         density = 10000.;
    }else{
        youngs = 3e6;
        possions = 0.3;
        thickness = 3;
        density = 10000.;
    }

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
    SparsityPattern      sparsity_pattern, sparsity_pattern_couple, sparsity_pattern_laplace;
    
    SparseMatrix<double> stiffness_matrix, mass_matrix, coupling_matrix, dielectric_matrix;

    Vector<double> solution_disp;
    Vector<double> solution_disp_coupled;
    Vector<double> force_rhs;
    Vector<double> force_coefficient;
    
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,constraints);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    
    make_sparsity_pattern_for_coupling_and_laplace_matrices(dof_handler, sparsity_pattern_couple, sparsity_pattern_laplace);
    
    stiffness_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    coupling_matrix.reinit(sparsity_pattern_couple);
    dielectric_matrix.reinit(sparsity_pattern_laplace);

    solution_disp.reinit(dof_handler.n_dofs());
    force_rhs.reinit(dof_handler.n_dofs());
    force_coefficient.reinit(dof_handler.n_dofs());
    
    hp::FEValues<dim,spacedim> hp_fe_values(mapping_collection, fe_collection, q_collection,update_values|update_quadrature_points|update_jacobians|update_jacobian_grads|update_inverse_jacobians| update_gradients|update_hessians|update_jacobian_pushed_forward_grads|update_JxW_values|update_normal_vectors);
    
    RightHandSide<spacedim> rhs_function;
    
    FullMatrix<double> cell_mass_matrix;
    FullMatrix<double> cell_stiffness_matrix;
    FullMatrix<double> cell_coupling_matrix;
    FullMatrix<double> cell_dielectric_matrix;
    
    Vector<double>     cell_force_rhs;
    
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> local_electric_dof_indices;

    std::vector<types::global_dof_index> fix_dof_indices;

    
    double area = 0;
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
        
        cell_stiffness_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_dielectric_matrix.reinit(dofs_per_cell/spacedim, dofs_per_cell/spacedim);
        cell_coupling_matrix.reinit(dofs_per_cell/spacedim, dofs_per_cell);

        cell_stiffness_matrix = 0;
        cell_mass_matrix = 0;
        cell_dielectric_matrix = 0;
        cell_coupling_matrix = 0;
        
        cell_force_rhs.reinit(dofs_per_cell);
        cell_force_rhs = 0;

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
            
            // contravariant metric tensor
            Tensor<2,dim> am_contra = metric_contravariant(am_cov);
            
            Tensor<2,spacedim> am_cov_3d;
            for (unsigned id = 0; id < dim; ++id) {
                for (unsigned jd = 0; jd < dim; ++jd) {
                    am_cov_3d[id][jd] += am_cov[id][jd];
                }
            }
            am_cov_3d[2][2] = 1.;
            // contravariant metric tensor
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
                //  computation of the B operator (strains) for i_shape function
                Tensor<2,dim, Tensor<1, spacedim>> membrane_strain;
                Tensor<2,dim, Tensor<1, spacedim>> bending_strain;
                for (unsigned int ii = 0; ii < dim; ++ii) {
                    for (unsigned int jj = 0; jj < dim; ++jj) {
                        for (unsigned int id = 0; id < spacedim; ++id) {
                            membrane_strain[ii][jj][id] = 0.5 * (a_cov[ii][id] * shape_der[jj] + a_cov[jj][id] * shape_der[ii]);
                            bending_strain[ii][jj][id] = - shape_der2[ii][jj] * a_cov[2][id] + (shape_der[0] * cross_product_3d(da_cov[ii][jj], a_cov[1])[id] + shape_der[1] * cross_product_3d(a_cov[0], da_cov[ii][jj])[id])/detJ + scalar_product(a_cov[2], da_cov[ii][jj]) * (shape_der[0] * cross_product_3d(a_cov[1], a_cov[2])[id] + shape_der[1] * cross_product_3d(a_cov[2], a_cov[0])[id]) / detJ;
                        }
                    }
                }
                
                bn_vec[i_shape] = membrane_strain;
                bm_vec[i_shape] = bending_strain;
            } // loop over shape functions
            
            // H tensor
            Tensor<4,2> H_tensor;
            constitutive_fourth_tensors(H_tensor, am_contra, possions);
            
            for (unsigned int i_node = 0; i_node < dofs_per_cell/spacedim; ++i_node)
            {
                Tensor<2, dim, Tensor<1, spacedim>> hn,hm;
                double coeff = youngs/ (1. - possions*possions);

                for(unsigned int ii = 0; ii < dim ; ++ii)
                    for(unsigned int jj = 0; jj < dim ; ++jj)
                        for(unsigned int kk = 0; kk < dim ; ++kk)
                            for(unsigned int ll = 0; ll < dim ; ++ll)
                                for (unsigned int id = 0; id < spacedim; ++id) {
                                    hn[ii][jj][id] += thickness * coeff * H_tensor[ii][jj][kk][ll] * bn_vec[i_node][kk][ll][id];
                                    hm[ii][jj][id] += 1./12. * thickness * thickness * thickness * coeff * H_tensor[ii][jj][kk][ll] * bm_vec[i_node][kk][ll][id];
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
                            cell_stiffness_matrix(i_node * spacedim + id, j_node * spacedim + jd) += (sn[id][jd] + sm[id][jd]) * fe_values.JxW(q_point);
                            if (id == jd) {
                                cell_mass_matrix(i_node*spacedim+id, j_node*spacedim+jd) += density * thickness * fe_values.shape_value(i_node*spacedim+id, q_point) * fe_values.shape_value(j_node*spacedim+jd, q_point) * fe_values.JxW(q_point);
                            }
                        }
                    }
                }
            }
            
            for (unsigned int i_node = 0; i_node < dofs_per_cell/spacedim; ++i_node)
            {
                for (unsigned int j_node = 0; j_node < dofs_per_cell/spacedim; ++j_node)
                {
                    for (unsigned int ii = 0; ii < spacedim; ++ii) {
                        for (unsigned int jj = 0; jj < spacedim; ++jj) {
                            cell_dielectric_matrix(i_node, j_node) -= (pow(thickness,5.) / 80. * rotated_dielectric_tensor[ii][jj] * fe_values.shape_grad(i_node * spacedim, q_point)[ii] * fe_values.shape_grad(j_node * spacedim, q_point)[jj] + pow(thickness, 3.)/3. * rotated_dielectric_tensor[ii][jj] * fe_values.shape_value(i_node*spacedim, q_point) * fe_values.shape_value(j_node*spacedim, q_point) * fe_values.normal_vector(q_point)[ii] * fe_values.normal_vector(q_point)[jj]) *
                            fe_values.JxW(q_point);
                        }
                    }
                    for (unsigned int ii = 0; ii < spacedim; ++ii) {
                        for (unsigned int jj = 0; jj < dim; ++jj) {
                            for (unsigned int kk = 0; kk < dim; ++kk){
                                for (unsigned int jd = 0; jd < spacedim; ++jd) {
                                    cell_coupling_matrix(i_node, j_node * spacedim + jd) += (pow(thickness, 3.) / 12. * rotated_piezoelectric_tensor[ii][jj][kk] * fe_values.shape_grad(i_node * spacedim, q_point)[ii] * bn_vec[j_node][jj][kk][jd] + pow(thickness, 3.)/6. * rotated_piezoelectric_tensor[ii][jj][kk] * fe_values.shape_value(i_node * spacedim, q_point) * fe_values.normal_vector(q_point)[ii] * bm_vec[j_node][jj][kk][jd] )*
                                    fe_values.JxW(q_point);
                                }
                            }
                        }
                    }
                }
                // uniform load
                if (type == "r") {
//                    cell_force_rhs(i_node*spacedim + 1) += 45 * sqrt(2) * fe_values.shape_value(i_node*spacedim + 1 , q_point)* fe_values.JxW(q_point); // f_z = -90
                    cell_force_rhs(i_node*spacedim + 2) += - 90 * fe_values.shape_value(i_node*spacedim + 2 , q_point)* fe_values.JxW(q_point); // f_z = -90

                }else if (type == "s" ){
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
        
        if(type == "s" ){
            if(load_1 == false){
                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
                    if( cell->vertex(ivert)[1] >= 10.- 1e-4){
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
                            cell_force_rhs(idof*spacedim+1) = 2.*shape;
                        }
                    }
                }
            }
            if(load_2 == false){
                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
                    if( cell->vertex(ivert)[1] <= -10.+ 1e-4){
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
                            cell_force_rhs(idof*spacedim+1) = -2.*shape;
                        }
                    }
                }
            }
            if(load_3 == false){
                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
                    if( cell->vertex(ivert)[2] >= 10.- 1e-4){
                        load_3 = true;
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
                            cell_force_rhs(idof*spacedim+2) = -2.*shape;
                        }
                    }
                }
            }
            if(load_4 == false){
                for (unsigned int ivert = 0; ivert < 4; ++ivert) {
                    if( cell->vertex(ivert)[2] <= -10.+ 1e-4){
                        load_4 = true;
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
                            cell_force_rhs(idof*spacedim+2) = 2.*shape;
                        }
                    }
                }
            }
        }
        
        force_rhs.add(local_dof_indices, cell_force_rhs);
        stiffness_matrix.add(local_dof_indices, local_dof_indices, cell_stiffness_matrix);
        mass_matrix.add(local_dof_indices, local_dof_indices, cell_mass_matrix);
        dielectric_matrix.add(local_electric_dof_indices, local_electric_dof_indices, cell_dielectric_matrix);
        coupling_matrix.add(local_electric_dof_indices, local_dof_indices, cell_coupling_matrix);
                
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
                }
            }else if (type == "s"){
                if ( cell->vertex(ivert)[0] == 0){
                    unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                    fix_dof_indices.push_back(dof_id);
                    fix_dof_indices.push_back(dof_id+1);
                    fix_dof_indices.push_back(dof_id+2);
                }
            }else if (type == "c"){
                if ( cell->vertex(ivert)[0] <= -300. || cell->vertex(ivert)[0] >= 300.){
//                if ( cell->vertex(ivert)[0] == -300.){
                    unsigned int dof_id = cell->vertex_dof_index(ivert,0, cell->active_fe_index());
                    fix_dof_indices.push_back(dof_id);
                    fix_dof_indices.push_back(dof_id+1);
//                    fix_dof_indices.push_back(dof_id+2);
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
    
//    std::cout << "solve elastic system of equations.\n";
    
    const auto op_k = linear_operator(stiffness_matrix);
    const auto op_m = linear_operator(dielectric_matrix);
    PreconditionJacobi<SparseMatrix<double>> preconditioner_k;
    PreconditionJacobi<SparseMatrix<double>> preconditioner_m;
    preconditioner_k.initialize(stiffness_matrix);
    preconditioner_m.initialize(dielectric_matrix);
    ReductionControl reduction_control_M(4000, 1.0e-12, 1.0e-12);
    SolverCG<Vector<double>>    solver_M(reduction_control_M);
    const auto op_m_inv = inverse_operator(op_m, solver_M, preconditioner_m);
    const auto op_c = linear_operator(coupling_matrix);
    const auto op_s = op_k - transpose_operator(op_c) * op_m_inv * op_c;
    
    const auto op_smod = constrained_linear_operator(constraints, op_s);
    const auto op_kmod = constrained_linear_operator(constraints, op_k);

    auto rhs_mod = constrained_right_hand_side(constraints, op_s, force_rhs);
    
    SolverControl            solver_control_k(40000, 1.e-9);
    SolverCG<Vector<double>> solver_k(solver_control_k);
    const auto op_k_inv = inverse_operator(op_kmod, solver_k, preconditioner_k);

    SolverControl            solver_control_s(40000, 1.e-9);
    SolverCG<Vector<double>> solver_s(solver_control_s);
    const auto op_s_inv = inverse_operator(op_smod, solver_s, preconditioner_k);

    solution_disp_coupled = op_s_inv * rhs_mod;
    solution_disp = op_k_inv * rhs_mod;

    Vector<double> potential = op_m_inv * op_c * solution_disp_coupled;
//    std::cout <<"potential = " << potential << std::endl;

    vtk_plot("coupled_shell_solution.vtu", dof_handler, mapping_collection, vec_values, solution_disp_coupled, potential);
    vtk_plot("elastic_shell_solution.vtu", dof_handler, mapping_collection, vec_values, solution_disp);

    
    for (unsigned int idof = 0; idof <fix_dof_indices.size(); ++idof) {
        for (unsigned int jdof = 0; jdof <dof_handler.n_dofs(); ++jdof) {
            if (fix_dof_indices[idof] == jdof){
                stiffness_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);
                mass_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);

            }
            else
            {
                stiffness_matrix.set(fix_dof_indices[idof], jdof, 0);
                stiffness_matrix.set(jdof, fix_dof_indices[idof], 0);
                mass_matrix.set(fix_dof_indices[idof], jdof, 0);
                mass_matrix.set(jdof, fix_dof_indices[idof], 0);
            }
        }
    }
    
    // eigenvalue problem
    std::vector<Vector<double> >        eigenvectors(20);
    std::vector<std::complex<double>>   eigenvalues(20);
    
    for (unsigned int i = 0; i < eigenvectors.size(); ++i){
        eigenvectors[i].reinit(dof_handler.n_dofs());
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
    const unsigned int num_arnoldi_vectors = 50;
    ArpackSolver::AdditionalData additional_data(num_arnoldi_vectors, ArpackSolver::algebraically_largest, true);
    ArpackSolver eigensolver(solver_control,additional_data);
    eigensolver.solve(stiffness_matrix, mass_matrix, inverse, eigenvalues, eigenvectors);
    
    for (unsigned int i = 0; i < eigenvectors.size(); ++i){
        if(eigenvalues[i].real() > 1.+1e-4){
            std::cout <<"Elastic Eigenvalue "<<i<<" = "<<eigenvalues[i] << " frequence = " << std::sqrt(eigenvalues[i].real()) << " (rad/s) " << " = " << std::sqrt(eigenvalues[i]).real() /(2.* numbers::PI) << "(Hz)" <<std::endl;
            vtk_plot("test_eigen_solution_2_"+std::to_string(i)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i]);
        }
    }
    
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
//            std:: cout << column_vec[irow] << " ";
        }
//        std::cout << std::endl;
    }

    for (unsigned int idof = 0; idof <fix_dof_indices.size(); ++idof) {
        for (unsigned int jdof = 0; jdof <dof_handler.n_dofs(); ++jdof) {
            if (fix_dof_indices[idof] == jdof){
                coupled_system_matrix.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);
                mass_matrix_full.set(fix_dof_indices[idof], fix_dof_indices[idof], 1e50);

            }
            else
            {
                coupled_system_matrix.set(fix_dof_indices[idof], jdof, 0);
                coupled_system_matrix.set(jdof, fix_dof_indices[idof], 0);
                mass_matrix_full.set(fix_dof_indices[idof], jdof, 0);
                mass_matrix_full.set(jdof, fix_dof_indices[idof], 0);
            }
        }
    }
    eigenvalues.resize(dof_handler.n_dofs());
    eigenvectors.resize(dof_handler.n_dofs());
    coupled_system_matrix.compute_generalized_eigenvalues_symmetric(mass_matrix_full, eigenvectors);
    unsigned n_eign = 1;
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
        eigenvalues[i] = coupled_system_matrix.eigenvalue(i);
        if(eigenvalues[i].real() > 1.+1e-4 && n_eign < 20){
            std::cout <<"Coupled Eigenvalue "<<n_eign<<" = "<<eigenvalues[i] << " frequence = " << std::sqrt(eigenvalues[i].real()) << " (rad/s) " << " = " << std::sqrt(eigenvalues[i]).real() /(2.* numbers::PI) << "(Hz)" <<std::endl;
            Vector<double> eigen_potential = op_m_inv * op_c * eigenvectors[i];
            vtk_plot("coupled_test_eigen_solution_2_"+std::to_string(n_eign)+".vtu", dof_handler, mapping_collection, vec_values, eigenvectors[i],eigen_potential);
            ++n_eign;
        }
    }
    return 0;
}
