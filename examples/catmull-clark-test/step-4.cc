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

#include <deal.II/hp/dof_handler.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include "polynomials_Catmull_Clark.hpp"
#include "FE_Catmull_Clark.hpp"
#include "Catmull_Clark_Data.hpp"



// This is new, however: in the previous example we got some unwanted output
// from the linear solvers. If we want to suppress it, we have to include this
// file and add a single line somewhere to the program (see the main()
// function below for that):
#include <deal.II/base/logstream.h>

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;



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
        
    auto fe_collection = distribute_catmull_clark_dofs(dof_handler,1);
    
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    AffineConstraints<double> constraints;
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,constraints);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    std::ofstream out("CC_sparsity_pattern.svg");
    sparsity_pattern.print_svg(out);
//
    std::vector<types::global_dof_index> dof_indices;
    
    for(auto cell = dof_handler.begin_active(); cell!=dof_handler.end(); ++ cell){
        dof_indices.resize(cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        std::cout<< " n non local dofs per cells = "<<cell->get_fe().non_local_dofs_per_cell<<"\n";
        for (unsigned int i = 0; i < dof_indices.size(); ++i) {
            std::cout << dof_indices[i]<<" ";
        }
        std::cout <<std::endl;
    }

    std::cout << "number of dofs = " << dof_handler.n_dofs()<<"\n";
    
  return 0;
}
