/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Abner Salgado, Texas A&M University 2009
 */


// @sect3{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones. Each one of them has been discussed in previous tutorial
// programs, so we will not get into details here.
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

// Finally this is as in all previous programs:
namespace Step35
{
  using namespace dealii;



  // @sect3{Run time parameters}
  //
  // Since our method has several parameters that can be fine-tuned we put
  // them into an external file, so that they can be determined at run-time.
  //
  // This includes, in particular, the formulation of the equation for the
  // auxiliary variable $\phi$, for which we declare an <code>enum</code>.
  // Next, we declare a class that is going to read and store all the
  // parameters that our program needs to run.
  namespace RunTimeParameters
  {
    enum MethodFormulation
    {
      METHOD_STANDARD,
      METHOD_ROTATIONAL
    };

    class Data_Storage
    {
    public:
      Data_Storage();

      void read_data (const char *filename);
      MethodFormulation form;
      double initial_time,
             final_time,
             viscosity;
      double length, width, hole_radius, max_velocity;
      double head_length, head_width, centre_x, centre_y, tail_length, tail_thickness;
      double dt;
      unsigned int n_global_refines,
               pressure_degree;
      unsigned int vel_max_iterations,
               vel_Krylov_size,
               vel_off_diagonals,
               vel_update_prec;
      double vel_eps,
             vel_diag_strength;
      bool verbose;
      unsigned int output_interval;
    protected:
      ParameterHandler prm;
    };

    // In the constructor of this class we declare all the parameters. The
    // details of how this works have been discussed elsewhere, for example in
    // step-19 and step-29.
    Data_Storage::Data_Storage() :
      form (METHOD_ROTATIONAL),
      initial_time (0.),
      final_time (1.),
      viscosity (1.),
      length (25.),
      width (4.1),
      hole_radius (0.2),
      max_velocity (1.5),
      dt (5e-4),
      n_global_refines (0),
      pressure_degree (1),
      vel_max_iterations (1000),
      vel_Krylov_size (30),
      vel_off_diagonals (60),
      vel_update_prec (15),
      vel_eps (1e-12),
      vel_diag_strength (0.01),
      verbose (true),
      output_interval (15)
    {
      prm.declare_entry ("Method_Form", "rotational",
                         Patterns::Selection ("rotational|standard"),
                         " Used to select the type of method that we are going "
                         "to use. ");
      prm.enter_subsection ("Physical data");
      {
        prm.declare_entry ("initial_time", "0.",
                           Patterns::Double (0.),
                           " The initial time of the simulation. ");
        prm.declare_entry ("final_time", "1.",
                           Patterns::Double (0.),
                           " The final time of the simulation. ");
        prm.declare_entry ("viscosity", "1.",
                           Patterns::Double (0.),
                           " The viscosity. ");
      }
        prm.leave_subsection();

        prm.enter_subsection ("Boundary data");
        {
            prm.declare_entry ("length", "25.",
                               Patterns::Double (0.),
                               " The length of the tank. ");
            prm.declare_entry ("width", "4.1",
                               Patterns::Double (0.),
                               " The width of the tank. ");
            prm.declare_entry ("hole_radius", "0.2",
                               Patterns::Double (0.),
                               " The radius of the inlet and outlet hole. ");
            prm.declare_entry ("max_velocity", "1.5",
                               Patterns::Double (0.),
                               " The maxmium velocity of the flow. ");
        }
      prm.leave_subsection();
        
        prm.enter_subsection ("Diamond data");
        {
            prm.declare_entry ("head_length", "2.",
                               Patterns::Double (0.),
                               " The length of the diamond. ");
            prm.declare_entry ("head_width", "2.",
                               Patterns::Double (0.),
                               " The width of the diamond. ");
            prm.declare_entry ("centre_x", "3.",
                               Patterns::Double (0.),
                               " The x coordinate of diamond centre. ");
            prm.declare_entry ("centre_y", "3.",
                               Patterns::Double (0.),
                               " The y coordinate of diamond centre. ");
            prm.declare_entry ("tail_length", "5.",
                               Patterns::Double (0.),
                               " The length of plate. ");
            prm.declare_entry ("tail_thickness", "0.1",
                               Patterns::Double (0.),
                               " The thinkness of plate. ");
        }
        prm.leave_subsection();

      prm.enter_subsection ("Time step data");
      {
        prm.declare_entry ("dt", "5e-4",
                           Patterns::Double (0.),
                           " The time step size. ");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Space discretization");
      {
        prm.declare_entry ("n_of_refines", "0",
                           Patterns::Integer (0, 15),
                           " The number of global refines we do on the mesh. ");
        prm.declare_entry ("pressure_fe_degree", "1",
                           Patterns::Integer (1, 5),
                           " The polynomial degree for the pressure space. ");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Data solve velocity");
      {
        prm.declare_entry ("max_iterations", "1000",
                           Patterns::Integer (1, 1000),
                           " The maximal number of iterations GMRES must make. ");
        prm.declare_entry ("eps", "1e-12",
                           Patterns::Double (0.),
                           " The stopping criterion. ");
        prm.declare_entry ("Krylov_size", "30",
                           Patterns::Integer(1),
                           " The size of the Krylov subspace to be used. ");
        prm.declare_entry ("off_diagonals", "60",
                           Patterns::Integer(0),
                           " The number of off-diagonal elements ILU must "
                           "compute. ");
        prm.declare_entry ("diag_strength", "0.01",
                           Patterns::Double (0.),
                           " Diagonal strengthening coefficient. ");
        prm.declare_entry ("update_prec", "15",
                           Patterns::Integer(1),
                           " This number indicates how often we need to "
                           "update the preconditioner");
      }
      prm.leave_subsection();

      prm.declare_entry ("verbose", "true",
                         Patterns::Bool(),
                         " This indicates whether the output of the solution "
                         "process should be verbose. ");

      prm.declare_entry ("output_interval", "1",
                         Patterns::Integer(1),
                         " This indicates between how many time steps we print "
                         "the solution. ");
    }



    void Data_Storage::read_data (const char *filename)
    {
      std::ifstream file (filename);
      AssertThrow (file, ExcFileNotOpen (filename));

      prm.parse_input (file);

      if (prm.get ("Method_Form") == std::string ("rotational"))
        form = METHOD_ROTATIONAL;
      else
        form = METHOD_STANDARD;

      prm.enter_subsection ("Physical data");
      {
        initial_time = prm.get_double ("initial_time");
        final_time   = prm.get_double ("final_time");
        viscosity     = prm.get_double ("viscosity");
      }
      prm.leave_subsection();
      
        prm.enter_subsection ("Boundary data");
        {
            length = prm.get_double ("length");
            width   = prm.get_double ("width");
            hole_radius   = prm.get_double ("hole_radius");

            max_velocity = prm.get_double ("max_velocity");
        }
        prm.leave_subsection();
        
        prm.enter_subsection ("Diamond data");
        {
            head_length = prm.get_double ("head_length");
            head_width   = prm.get_double ("head_width");
            centre_x = prm.get_double ("centre_x");
            centre_y = prm.get_double ("centre_y");
            tail_length = prm.get_double ("tail_length");
            tail_thickness   = prm.get_double ("tail_thickness");
        }
        prm.leave_subsection();
    

      prm.enter_subsection ("Time step data");
      {
        dt = prm.get_double ("dt");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Space discretization");
      {
        n_global_refines = prm.get_integer ("n_of_refines");
        pressure_degree     = prm.get_integer ("pressure_fe_degree");
      }
      prm.leave_subsection();

      prm.enter_subsection ("Data solve velocity");
      {
        vel_max_iterations = prm.get_integer ("max_iterations");
        vel_eps            = prm.get_double ("eps");
        vel_Krylov_size    = prm.get_integer ("Krylov_size");
        vel_off_diagonals  = prm.get_integer ("off_diagonals");
        vel_diag_strength  = prm.get_double ("diag_strength");
        vel_update_prec    = prm.get_integer ("update_prec");
      }
      prm.leave_subsection();

      verbose = prm.get_bool ("verbose");

      output_interval = prm.get_integer ("output_interval");
    }
  }
    
  namespace GmeshGenerator
    {
        class MeshGenerator
        {
        public:
            MeshGenerator(const RunTimeParameters::Data_Storage &data);
            void output();
        private:
            double t_length, t_width;
            double h_length, h_width;
            std::pair<double, double> centre;
            double r,t,s;
            
        };
    
    
        MeshGenerator::MeshGenerator (const RunTimeParameters::Data_Storage &data)
        {
            t_length = data.length;
            t_width = data.width;
            r = data.hole_radius;
            h_length = data.head_length;
            h_width = data.head_width;
            centre.first = data.centre_x;
            centre.second = data.centre_y;
            t = data.tail_thickness;
            s = data.tail_length;
        }
        
        void MeshGenerator::output(){
            FILE * meshfile;
            meshfile = fopen("diamond_example.geo","w");
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 1. ,0. ,0.);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 2. ,0. ,t_width);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 3. ,t_length ,t_width);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 4. ,t_length ,0.);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 5. ,centre.first ,centre.second);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 6. , 0. ,t_width/2.0 - r);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 7. , 0. ,t_width/2.0 + r);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 8. ,t_length ,t_width/2.0 + r);
            fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", 9. ,t_length ,t_width/2.0 - r);
            std::vector<std::vector<double>> points(7);
            points[0] =  std::vector<double> {centre.first , centre.second - t/2.};
            points[1] =  std::vector<double> {centre.first , centre.second + t/2.};
            points[2] =  std::vector<double> {centre.first + s, centre.second - t/2.};
            points[3] =  std::vector<double> {centre.first + s, centre.second + t/2.};
            points[4] =  std::vector<double> {centre.first, centre.second - h_width/2.};
            points[5] =  std::vector<double> {centre.first - h_length, centre.second};
            points[6] =  std::vector<double> {centre.first, centre.second + h_width/2.};
            for (int i = 0; i<points.size();++i){
                fprintf(meshfile,"Point(%2.f) = {%4.6f , %4.6f, 0.0, 0.2};\n", double(i) + 10. , points[i][0] ,points[i][1]);
            }
            
            std::vector<std::vector<double>> lines(11);
            lines[0] = std::vector<double> {1,6};
            lines[1] = std::vector<double> {6,7};
            lines[2] = std::vector<double> {7,2};
            lines[3] = std::vector<double> {2,3};
            lines[4] = std::vector<double> {3,8};
            lines[5] = std::vector<double> {8,9};
            lines[6] = std::vector<double> {9,4};
            lines[7] = std::vector<double> {4,1};
            lines[8] = std::vector<double> {14,16};
            lines[9] = std::vector<double> {16,15};
            lines[10] = std::vector<double> {15,14};
            
            for (int i = 0; i < lines.size(); ++i) {
                fprintf(meshfile,"Line(%2.f) = {%2.f , %2.f};\n", double(i)+1.,lines[i][0],lines[i][1]);
            }
            
            fprintf(meshfile,"Line Loop(%2.f) = {%2.f, %2.f, %2.f, %2.f, %2.f, %2.f, %2.f, %2.f};\n", 1.,1.,2.,3.,4.,5.,6.,7.,8.);

            fprintf(meshfile,"Line Loop(%2.f) = {%2.f, %2.f, %2.f};\n", 2., 9., 10., 11.);
            fprintf(meshfile,"Plane Surface(1) = {1,2};\n");
            fprintf(meshfile,"Physical Surface(1) = {1};\n");
            fprintf(meshfile,"Physical Line(1) = { 4 };\n");
            fprintf(meshfile,"Physical Line(2) = { 2 };\n");
            fprintf(meshfile,"Physical Line(3) = { 6 };\n");
            fprintf(meshfile,"Physical Line(4) = {1,3,5,7,9,10,11};\n");
            fprintf(meshfile,"Physical Line(5) = {8};\n");

            fprintf(meshfile,"Transfinite Line{1,3,5,7} = 5;\n ");
            fprintf(meshfile,"Transfinite Line{2,6} = 10;\n");
            fprintf(meshfile,"Transfinite Line{4,8} = 25;\n");

            fprintf(meshfile,"Transfinite Line{9,10,11} = 10;\n");

            fprintf(meshfile,"Mesh.Algorithm = 8;\n");
            fprintf(meshfile,"Mesh.RecombineAll = 1;\n");
            fprintf(meshfile,"Mesh.CharacteristicLengthFactor = 5;\n");
            fprintf(meshfile,"Mesh.SubdivisionAlgorithm = 1;\n");
            fprintf(meshfile,"Mesh.Smoothing = 1;\n");
            
            fclose (meshfile);
            
            const int ierr = std::system("gmsh -2 diamond_example.geo ");
        }
    }


  // @sect3{Equation data}

  // In the next namespace, we declare the initial and boundary conditions:
  namespace EquationData
  {
    // As we have chosen a completely decoupled formulation, we will not take
    // advantage of deal.II's capabilities to handle vector valued
    // problems. We do, however, want to use an interface for the equation
    // data that is somehow dimension independent. To be able to do that, our
    // functions should be able to know on which spatial component we are
    // currently working, and we should be able to have a common interface to
    // do that. The following class is an attempt in that direction.
    template <int dim>
    class MultiComponentFunction: public Function<dim>
    {
    public:
      MultiComponentFunction (const double initial_time);
      void set_component (const unsigned int d);
    protected:
      unsigned int comp;
    };

    template <int dim>
    MultiComponentFunction<dim>::
    MultiComponentFunction (const double initial_time)
      :
      Function<dim> (1, initial_time), comp(0)
    {}


    template <int dim>
    void MultiComponentFunction<dim>::set_component(const unsigned int d)
    {
      Assert (d<dim, ExcIndexRange (d, 0, dim));
      comp = d;
    }


    // With this class defined, we declare classes that describe the boundary
    // conditions for velocity and pressure:
    template <int dim>
    class Velocity : public MultiComponentFunction<dim>
    {
    public:
        
      Velocity (const double initial_time, const double width, const double max_velocity);

      virtual double value (const Point<dim> &p,
                            const unsigned int component = 0) const;

      virtual void value_list (const std::vector< Point<dim> > &points,
                               std::vector<double> &values,
                               const unsigned int component = 0) const;
    private:
        
        const double H;
        const double Um;
        
    };


    template <int dim>
    Velocity<dim>::Velocity (const double initial_time, const double width, const double max_velocity)
      :
      MultiComponentFunction<dim> (initial_time),
      H(width),
      Um(max_velocity)
    {}

    template <int dim>
    void Velocity<dim>::value_list (const std::vector<Point<dim> > &points,
                                    std::vector<double> &values,
                                    const unsigned int) const
    {
      const unsigned int n_points = points.size();
      Assert (values.size() == n_points,
              ExcDimensionMismatch (values.size(), n_points));
      for (unsigned int i=0; i<n_points; ++i)
        values[i] = Velocity<dim>::value (points[i]);
    }

    template <int dim>
    double Velocity<dim>::value(const Point<dim> &p,
                                 const unsigned int) const
    {
      if (this->comp == 0)
        {
          return 4. * Um * p(1) * (H - p(1)) / ( H * H );
        }
      else
        return 0.;
    }


    template <int dim>
    class Pressure: public Function<dim>
    {
    public:
        
      Pressure (const double initial_time = 0.0, const double length = 25.0);

      virtual double value (const Point<dim> &p,
                            const unsigned int component = 0) const;

      virtual void value_list (const std::vector< Point<dim> > &points,
                               std::vector<double> &values,
                               const unsigned int component = 0) const;
        
    private:
        
        const double W;

    };

    template <int dim>
    Pressure<dim>::Pressure (const double initial_time, const double length)
      :
      Function<dim> (1, initial_time),
      W(length)
    {}


    template <int dim>
    double Pressure<dim>::value (const Point<dim> &p,
                                 const unsigned int) const
    {
      return W-p(0);
    }

    template <int dim>
    void Pressure<dim>::value_list (const std::vector<Point<dim> > &points,
                                    std::vector<double> &values,
                                    const unsigned int) const
    {
      const unsigned int n_points = points.size();
      Assert (values.size() == n_points, ExcDimensionMismatch (values.size(), n_points));
      for (unsigned int i=0; i<n_points; ++i)
        values[i] = Pressure<dim>::value (points[i]);
    }
  }



  // @sect3{The <code>NavierStokesProjection</code> class}

  // Now for the main class of the program. It implements the various versions
  // of the projection method for Navier-Stokes equations.  The names for all
  // the methods and member variables should be self-explanatory, taking into
  // account the implementation details given in the introduction.
  template <int dim>
  class NavierStokesProjection
  {
  public:
    NavierStokesProjection (const RunTimeParameters::Data_Storage &data);

    void run (const bool         verbose    = false,
              const unsigned int n_plots = 10);
  protected:
    RunTimeParameters::MethodFormulation type;

    const unsigned int deg;
    const double       dt;
    const double       t_0, T, Rey;
    const double       t_length, t_width;

    EquationData::Velocity<dim>       vel_exact;
    std::map<types::global_dof_index, double>    boundary_values;
    std::vector<types::boundary_id> boundary_ids;

    Triangulation<dim> triangulation;

    FE_Q<dim>          fe_velocity;
    FE_Q<dim>          fe_pressure;

    DoFHandler<dim>    dof_handler_velocity;
    DoFHandler<dim>    dof_handler_pressure;

    QGauss<dim>        quadrature_pressure;
    QGauss<dim>        quadrature_velocity;
    QGauss<dim-1>      qv_face;
    QGauss<dim-1>      qp_face;

    SparsityPattern    sparsity_pattern_velocity;
    SparsityPattern    sparsity_pattern_pressure;
    SparsityPattern    sparsity_pattern_pres_vel;

    SparseMatrix<double> vel_Laplace_plus_Mass;
    SparseMatrix<double> vel_it_matrix[dim];
    SparseMatrix<double> vel_Mass;
    SparseMatrix<double> vel_Laplace;
    SparseMatrix<double> vel_Advection;
    SparseMatrix<double> pres_Laplace;
    SparseMatrix<double> pres_Mass;
    SparseMatrix<double> pres_Diff[dim];
    SparseMatrix<double> pres_iterative;

    Vector<double> pres_n;
    Vector<double> pres_n_minus_1;
    Vector<double> phi_n;
    Vector<double> phi_n_minus_1;
    Vector<double> u_n[dim];
    Vector<double> u_n_minus_1[dim];
    Vector<double> u_star[dim];
    Vector<double> force[dim];
    Vector<double> v_tmp;
    Vector<double> pres_tmp;
    Vector<double> rot_u;
    Vector<double>  shear_stress;

    SparseILU<double> prec_velocity[dim];
    SparseILU<double> prec_pres_Laplace;
    SparseDirectUMFPACK prec_mass;
    SparseDirectUMFPACK prec_vel_mass;

    DeclException2 (ExcInvalidTimeStep,
                    double, double,
                    << " The time step " << arg1 << " is out of range."
                    << std::endl
                    << " The permitted range is (0," << arg2 << "]");

    void create_triangulation_and_dofs (const unsigned int n_refines);

    void initialize();

    void interpolate_velocity ();

    void diffusion_step (const bool reinit_prec);

    void projection_step (const bool reinit_prec);

    void update_pressure (const bool reinit_prec);

  private:
    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    void initialize_velocity_matrices();

    void initialize_pressure_matrices();

    // The next few structures and functions are for doing various things in
    // parallel. They follow the scheme laid out in @ref threads, using the
    // WorkStream class. As explained there, this requires us to declare two
    // structures for each of the assemblers, a per-task data and a scratch
    // data structure. These are then handed over to functions that assemble
    // local contributions and that copy these local contributions to the
    // global objects.
    //
    // One of the things that are specific to this program is that we don't
    // just have a single DoFHandler object that represents both the
    // velocities and the pressure, but we use individual DoFHandler objects
    // for these two kinds of variables. We pay for this optimization when we
    // want to assemble terms that involve both variables, such as the
    // divergence of the velocity and the gradient of the pressure, times the
    // respective test functions. When doing so, we can't just anymore use a
    // single FEValues object, but rather we need two, and they need to be
    // initialized with cell iterators that point to the same cell in the
    // triangulation but different DoFHandlers.
    //
    // To do this in practice, we declare a "synchronous" iterator -- an
    // object that internally consists of several (in our case two) iterators,
    // and each time the synchronous iteration is moved up one step, each of
    // the iterators stored internally is moved up one step as well, thereby
    // always staying in sync. As it so happens, there is a deal.II class that
    // facilitates this sort of thing.
    typedef std::tuple< typename DoFHandler<dim>::active_cell_iterator,
            typename DoFHandler<dim>::active_cell_iterator
            > IteratorTuple;

    typedef SynchronousIterators<IteratorTuple> IteratorPair;

    void initialize_gradient_operator();

    struct InitGradPerTaskData
    {
      unsigned int              d;
      unsigned int              vel_dpc;
      unsigned int              pres_dpc;
      FullMatrix<double>        local_grad;
      std::vector<types::global_dof_index> vel_local_dof_indices;
      std::vector<types::global_dof_index> pres_local_dof_indices;

      InitGradPerTaskData (const unsigned int dd,
                           const unsigned int vdpc,
                           const unsigned int pdpc)
        :
        d(dd),
        vel_dpc (vdpc),
        pres_dpc (pdpc),
        local_grad (vdpc, pdpc),
        vel_local_dof_indices (vdpc),
        pres_local_dof_indices (pdpc)
      {}
    };

    struct InitGradScratchData
    {
      unsigned int  nqp;
      FEValues<dim> fe_val_vel;
      FEValues<dim> fe_val_pres;
      InitGradScratchData (const FE_Q<dim> &fe_v,
                           const FE_Q<dim> &fe_p,
                           const QGauss<dim> &quad,
                           const UpdateFlags flags_v,
                           const UpdateFlags flags_p)
        :
        nqp (quad.size()),
        fe_val_vel (fe_v, quad, flags_v),
        fe_val_pres (fe_p, quad, flags_p)
      {}
      InitGradScratchData (const InitGradScratchData &data)
        :
        nqp (data.nqp),
        fe_val_vel (data.fe_val_vel.get_fe(),
                    data.fe_val_vel.get_quadrature(),
                    data.fe_val_vel.get_update_flags()),
        fe_val_pres (data.fe_val_pres.get_fe(),
                     data.fe_val_pres.get_quadrature(),
                     data.fe_val_pres.get_update_flags())
      {}
    };

    void assemble_one_cell_of_gradient (const IteratorPair  &SI,
                                        InitGradScratchData &scratch,
                                        InitGradPerTaskData &data);

    void copy_gradient_local_to_global (const InitGradPerTaskData &data);

    // The same general layout also applies to the following classes and
    // functions implementing the assembly of the advection term:
    void assemble_advection_term();

    struct AdvectionPerTaskData
    {
      FullMatrix<double>        local_advection;
      std::vector<types::global_dof_index> local_dof_indices;
      AdvectionPerTaskData (const unsigned int dpc)
        :
        local_advection (dpc, dpc),
        local_dof_indices (dpc)
      {}
    };

    struct AdvectionScratchData
    {
      unsigned int                 nqp;
      unsigned int                 dpc;
      std::vector< Point<dim> >    u_star_local;
      std::vector< Tensor<1,dim> > grad_u_star;
      std::vector<double>          u_star_tmp;
      FEValues<dim>                fe_val;
      AdvectionScratchData (const FE_Q<dim> &fe,
                            const QGauss<dim> &quad,
                            const UpdateFlags flags)
        :
        nqp (quad.size()),
        dpc (fe.dofs_per_cell),
        u_star_local (nqp),
        grad_u_star (nqp),
        u_star_tmp (nqp),
        fe_val (fe, quad, flags)
      {}

      AdvectionScratchData (const AdvectionScratchData &data)
        :
        nqp (data.nqp),
        dpc (data.dpc),
        u_star_local (nqp),
        grad_u_star (nqp),
        u_star_tmp (nqp),
        fe_val (data.fe_val.get_fe(),
                data.fe_val.get_quadrature(),
                data.fe_val.get_update_flags())
      {}
    };

    void assemble_one_cell_of_advection (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         AdvectionScratchData &scratch,
                                         AdvectionPerTaskData &data);

    void copy_advection_local_to_global (const AdvectionPerTaskData &data);

    // The final few functions implement the diffusion solve as well as
    // postprocessing the output, including computing the curl of the
    // velocity:
    void diffusion_component_solve (const unsigned int d);

    void output_results (const unsigned int step);
      
    std::vector<std::vector<double>> output_results_boundary ();

    void assemble_vorticity (const bool reinit_prec);
  };



  // @sect4{ <code>NavierStokesProjection::NavierStokesProjection</code> }

  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify
  // that the data we read is reasonable and, finally, create the
  // triangulation and load the initial data.
  template <int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(const RunTimeParameters::Data_Storage &data)
    :
    type (data.form),
    deg (data.pressure_degree),
    dt (data.dt),
    t_0 (data.initial_time),
    T (data.final_time),
    Rey (1.0/data.viscosity),
    t_length (data.length),
    t_width (data.width),
    vel_exact (data.initial_time, data.width, data.max_velocity),
    fe_velocity (deg+1),
    fe_pressure (deg),
    dof_handler_velocity (triangulation),
    dof_handler_pressure (triangulation),
    quadrature_pressure (deg+1),
    quadrature_velocity (deg+2),
    qv_face (deg+3),
    qp_face (deg+3),
    vel_max_its (data.vel_max_iterations),
    vel_Krylov_size (data.vel_Krylov_size),
    vel_off_diagonals (data.vel_off_diagonals),
    vel_update_prec (data.vel_update_prec),
    vel_eps (data.vel_eps),
    vel_diag_strength (data.vel_diag_strength)
  {
    if (deg < 1)
      std::cout << " WARNING: The chosen pair of finite element spaces is not stable."
                << std::endl
                << " The obtained results will be nonsense"
                << std::endl;

    AssertThrow (!  ( (dt <= 0.) || (dt > .5*T)), ExcInvalidTimeStep (dt, .5*T));

    create_triangulation_and_dofs (data.n_global_refines);
    initialize();
  }


  // @sect4{ <code>NavierStokesProjection::create_triangulation_and_dofs</code> }

  // The method that creates the triangulation and refines it the needed
  // number of times.  After creating the triangulation, it creates the mesh
  // dependent data, i.e. it distributes degrees of freedom and renumbers
  // them, and initializes the matrices and vectors that we will use.
  template <int dim>
  void
  NavierStokesProjection<dim>::
  create_triangulation_and_dofs (const unsigned int n_refines)
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (triangulation);
    {
//      std::string filename = "/Users/Benjamin/Documents/researchCode/deal/dealii/examples/step-35/nsbench2.inp";
      std::string filename = "diamond_example.msh";

      std::ifstream file (filename.c_str());
      Assert (file, ExcFileNotOpen (filename.c_str()));
//      grid_in.read_ucd (file);
      grid_in.read_msh(file);
    }
//      Triangulation<2> tria1;
//      GridGenerator::hyper_cube_with_cylindrical_hole (tria1, 0.5, 2.05);
//      Point<2> shift_vector{2.05,2.05};
//      GridTools::shift(shift_vector, tria1);
//
//      Triangulation<2> tria2;
//      std::vector< unsigned int > repetitions(2);
//      repetitions[0]=5;
//      repetitions[1]=2;
//      GridGenerator::subdivided_hyper_rectangle (tria2, repetitions,
//                                                 Point<2>(4.0,0.0),
//                                                 Point<2>(25.0,4.0));
//
//      Triangulation<2> triangulation;
//      GridGenerator::merge_triangulations (tria1, tria2, triangulation);
//
      std::cout << "Number of refines = " << n_refines
      << std::endl;
      triangulation.refine_global (n_refines);
      std::cout << "Number of active cells: " << triangulation.n_active_cells()
      << std::endl;
//
//
    std::ofstream out ("circle.eps");
    GridOut grid_out;
    grid_out.write_eps (triangulation, out);
      
    std::cout << "Grid written to circle.eps" << std::endl;


    boundary_ids = triangulation.get_boundary_ids();

    dof_handler_velocity.distribute_dofs (fe_velocity);
    DoFRenumbering::boost::Cuthill_McKee (dof_handler_velocity);
    dof_handler_pressure.distribute_dofs (fe_pressure);
    DoFRenumbering::boost::Cuthill_McKee (dof_handler_pressure);

    initialize_velocity_matrices();
    initialize_pressure_matrices();
    initialize_gradient_operator();

    pres_n.reinit (dof_handler_pressure.n_dofs());
    pres_n_minus_1.reinit (dof_handler_pressure.n_dofs());
    phi_n.reinit (dof_handler_pressure.n_dofs());
    phi_n_minus_1.reinit (dof_handler_pressure.n_dofs());
    pres_tmp.reinit (dof_handler_pressure.n_dofs());
    for (unsigned int d=0; d<dim; ++d)
      {
        u_n[d].reinit (dof_handler_velocity.n_dofs());
        u_n_minus_1[d].reinit (dof_handler_velocity.n_dofs());
        u_star[d].reinit (dof_handler_velocity.n_dofs());
        force[d].reinit (dof_handler_velocity.n_dofs());
      }
    v_tmp.reinit (dof_handler_velocity.n_dofs());
    rot_u.reinit (dof_handler_velocity.n_dofs());
    shear_stress.reinit(dof_handler_velocity.n_dofs());

    std::cout << "dim (X_h) = " << (dof_handler_velocity.n_dofs()*dim)
              << std::endl
              << "dim (M_h) = " << dof_handler_pressure.n_dofs()
              << std::endl
              << "Re        = " << Rey
              << std::endl
              << std::endl;
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method creates the constant matrices and loads the initial data
  template <int dim>
  void
  NavierStokesProjection<dim>::initialize()
  {
    vel_Laplace_plus_Mass = 0.;
    vel_Laplace_plus_Mass.add (1.0/Rey, vel_Laplace);
    vel_Laplace_plus_Mass.add (1.5/dt, vel_Mass);

    EquationData::Pressure<dim> pres (t_0, t_length);
    VectorTools::interpolate (dof_handler_pressure, pres, pres_n_minus_1);
    pres.advance_time (dt);
    VectorTools::interpolate (dof_handler_pressure, pres, pres_n);
    phi_n = 0.;
    phi_n_minus_1 = 0.;
    for (unsigned int d=0; d<dim; ++d)
      {
        vel_exact.set_time (t_0);
        vel_exact.set_component(d);
        VectorTools::interpolate (dof_handler_velocity, Functions::ZeroFunction<dim>(), u_n_minus_1[d]);
        vel_exact.advance_time (dt);
        VectorTools::interpolate (dof_handler_velocity, Functions::ZeroFunction<dim>(), u_n[d]);
      }
  }


  // @sect4{ The <code>NavierStokesProjection::initialize_*_matrices</code> methods }

  // In this set of methods we initialize the sparsity patterns, the
  // constraints (if any) and assemble the matrices that do not depend on the
  // timestep <code>dt</code>. Note that for the Laplace and mass matrices, we
  // can use functions in the library that do this. Because the expensive
  // operations of this function -- creating the two matrices -- are entirely
  // independent, we could in principle mark them as tasks that can be worked
  // on in %parallel using the Threads::new_task functions. We won't do that
  // here since these functions internally already are parallelized, and in
  // particular because the current function is only called once per program
  // run and so does not incur a cost in each time step. The necessary
  // modifications would be quite straightforward, however.
  template <int dim>
  void
  NavierStokesProjection<dim>::initialize_velocity_matrices()
  {
    {
      DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(), dof_handler_velocity.n_dofs());
      DoFTools::make_sparsity_pattern (dof_handler_velocity, dsp);
      sparsity_pattern_velocity.copy_from (dsp);
    }
    vel_Laplace_plus_Mass.reinit (sparsity_pattern_velocity);
    for (unsigned int d=0; d<dim; ++d)
      vel_it_matrix[d].reinit (sparsity_pattern_velocity);
    vel_Mass.reinit (sparsity_pattern_velocity);
    vel_Laplace.reinit (sparsity_pattern_velocity);
    vel_Advection.reinit (sparsity_pattern_velocity);

    MatrixCreator::create_mass_matrix (dof_handler_velocity,
                                       quadrature_velocity,
                                       vel_Mass);
    MatrixCreator::create_laplace_matrix (dof_handler_velocity,
                                          quadrature_velocity,
                                          vel_Laplace);
  }

  // The initialization of the matrices that act on the pressure space is
  // similar to the ones that act on the velocity space.
  template <int dim>
  void
  NavierStokesProjection<dim>::initialize_pressure_matrices()
  {
    {
      DynamicSparsityPattern dsp(dof_handler_pressure.n_dofs(), dof_handler_pressure.n_dofs());
      DoFTools::make_sparsity_pattern (dof_handler_pressure, dsp);
      sparsity_pattern_pressure.copy_from (dsp);
    }

    pres_Laplace.reinit (sparsity_pattern_pressure);
    pres_iterative.reinit (sparsity_pattern_pressure);
    pres_Mass.reinit (sparsity_pattern_pressure);

    MatrixCreator::create_laplace_matrix (dof_handler_pressure,
                                          quadrature_pressure,
                                          pres_Laplace);
    MatrixCreator::create_mass_matrix (dof_handler_pressure,
                                       quadrature_pressure,
                                       pres_Mass);
  }


  // For the gradient operator, we start by initializing the sparsity pattern
  // and compressing it.  It is important to notice here that the gradient
  // operator acts from the pressure space into the velocity space, so we have
  // to deal with two different finite element spaces. To keep the loops
  // synchronized, we use the <code>typedef</code>'s that we have defined
  // before, namely <code>PairedIterators</code> and
  // <code>IteratorPair</code>.
  template <int dim>
  void
  NavierStokesProjection<dim>::initialize_gradient_operator()
  {
    {
      DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(), dof_handler_pressure.n_dofs());
      DoFTools::make_sparsity_pattern (dof_handler_velocity, dof_handler_pressure, dsp);
      sparsity_pattern_pres_vel.copy_from (dsp);
    }

    InitGradPerTaskData per_task_data (0, fe_velocity.dofs_per_cell,
                                       fe_pressure.dofs_per_cell);
    InitGradScratchData scratch_data (fe_velocity,
                                      fe_pressure,
                                      quadrature_velocity,
                                      update_gradients | update_JxW_values,
                                      update_values);

    for (unsigned int d=0; d<dim; ++d)
      {
        pres_Diff[d].reinit (sparsity_pattern_pres_vel);
        per_task_data.d = d;
        WorkStream::run (IteratorPair (IteratorTuple (dof_handler_velocity.begin_active(),
                                                      dof_handler_pressure.begin_active()
                                                     )
                                      ),
                         IteratorPair (IteratorTuple (dof_handler_velocity.end(),
                                                      dof_handler_pressure.end()
                                                     )
                                      ),
                         *this,
                         &NavierStokesProjection<dim>::assemble_one_cell_of_gradient,
                         &NavierStokesProjection<dim>::copy_gradient_local_to_global,
                         scratch_data,
                         per_task_data
                        );
      }
  }

  template <int dim>
  void
  NavierStokesProjection<dim>::
  assemble_one_cell_of_gradient (const IteratorPair  &SI,
                                 InitGradScratchData &scratch,
                                 InitGradPerTaskData &data)
  {
    scratch.fe_val_vel.reinit (std::get<0> (*SI));
    scratch.fe_val_pres.reinit (std::get<1> (*SI));

    std::get<0> (*SI)->get_dof_indices (data.vel_local_dof_indices);
    std::get<1> (*SI)->get_dof_indices (data.pres_local_dof_indices);

    data.local_grad = 0.;
    for (unsigned int q=0; q<scratch.nqp; ++q)
      {
        for (unsigned int i=0; i<data.vel_dpc; ++i)
          for (unsigned int j=0; j<data.pres_dpc; ++j)
            data.local_grad (i, j) += -scratch.fe_val_vel.JxW(q) *
                                      scratch.fe_val_vel.shape_grad (i, q)[data.d] *
                                      scratch.fe_val_pres.shape_value (j, q);
      }
  }


  template <int dim>
  void
  NavierStokesProjection<dim>::
  copy_gradient_local_to_global(const InitGradPerTaskData &data)
  {
    for (unsigned int i=0; i<data.vel_dpc; ++i)
      for (unsigned int j=0; j<data.pres_dpc; ++j)
        pres_Diff[data.d].add (data.vel_local_dof_indices[i], data.pres_local_dof_indices[j],
                               data.local_grad (i, j) );
  }


  // @sect4{ <code>NavierStokesProjection::run</code> }

  // This is the time marching function, which starting at <code>t_0</code>
  // advances in time using the projection method with time step
  // <code>dt</code> until <code>T</code>.
  //
  // Its second parameter, <code>verbose</code> indicates whether the function
  // should output information what it is doing at any given moment: for
  // example, it will say whether we are working on the diffusion, projection
  // substep; updating preconditioners etc. Rather than implementing this
  // output using code like
  // @code
  //   if (verbose) std::cout << "something";
  // @endcode
  // we use the ConditionalOStream class to do that for us. That
  // class takes an output stream and a condition that indicates whether the
  // things you pass to it should be passed through to the given output
  // stream, or should just be ignored. This way, above code simply becomes
  // @code
  //   verbose_cout << "something";
  // @endcode
  // and does the right thing in either case.
  template <int dim>
  void
  NavierStokesProjection<dim>::run (const bool verbose,
                                    const unsigned int output_interval)
  {
    ConditionalOStream verbose_cout (std::cout, verbose);

    const unsigned int n_steps =  static_cast<unsigned int>((T - t_0)/dt);
    vel_exact.set_time (2.*dt);
    output_results(1);
    
    for (unsigned int n = 2; n<=n_steps; ++n)
      {
        if (n % output_interval == 0)
          {
            verbose_cout << "Plotting Solution" << std::endl;
            output_results(n);
          }
//          if (n*dt == 2.5 || n*dt == 5. || n*dt == 7.5 || n*dt == 10. || n*dt == 12.5 ){
//              std::cout << "Step = " << n << " Time = " << (n*dt) << std::endl;
//          }
        
        verbose_cout << "  Interpolating the velocity " << std::endl;

        interpolate_velocity();
        verbose_cout << "  Diffusion Step" << std::endl;
        if (n % vel_update_prec == 0)
          verbose_cout << "    With reinitialization of the preconditioner"
                       << std::endl;
        diffusion_step ((n%vel_update_prec == 0) || (n == 2));
        verbose_cout << "  Projection Step" << std::endl;
        projection_step ( (n == 2));
        verbose_cout << "  Updating the Pressure" << std::endl;
        update_pressure ( (n == 2));
        vel_exact.advance_time(dt);
      }
    output_results (n_steps);
      }



  template <int dim>
  void
  NavierStokesProjection<dim>::interpolate_velocity()
  {
    for (unsigned int d=0; d<dim; ++d)
      {
        u_star[d].equ (2., u_n[d]);
        u_star[d] -=  u_n_minus_1[d];
      }
  }


  // @sect4{<code>NavierStokesProjection::diffusion_step</code>}

  // The implementation of a diffusion step. Note that the expensive operation
  // is the diffusion solve at the end of the function, which we have to do
  // once for each velocity component. To accelerate things a bit, we allow
  // to do this in %parallel, using the Threads::new_task function which makes
  // sure that the <code>dim</code> solves are all taken care of and are
  // scheduled to available processors: if your machine has more than one
  // processor core and no other parts of this program are using resources
  // currently, then the diffusion solves will run in %parallel. On the other
  // hand, if your system has only one processor core then running things in
  // %parallel would be inefficient (since it leads, for example, to cache
  // congestion) and things will be executed sequentially.
  template <int dim>
  void
  NavierStokesProjection<dim>::diffusion_step (const bool reinit_prec)
  {
    pres_tmp.equ (-1., pres_n);
    pres_tmp.add (-4./3., phi_n, 1./3., phi_n_minus_1);

    assemble_advection_term();

    for (unsigned int d=0; d<dim; ++d)
      {
        force[d] = 0.;
        v_tmp.equ (2./dt,u_n[d]);
        v_tmp.add (-.5/dt,u_n_minus_1[d]);
        vel_Mass.vmult_add (force[d], v_tmp);

        pres_Diff[d].vmult_add (force[d], pres_tmp);
        u_n_minus_1[d] = u_n[d];

        vel_it_matrix[d].copy_from (vel_Laplace_plus_Mass);
        vel_it_matrix[d].add (1., vel_Advection);

        vel_exact.set_component(d);
        boundary_values.clear();
        for (std::vector<types::boundary_id>::const_iterator
             boundaries = boundary_ids.begin();
             boundaries != boundary_ids.end();
             ++boundaries)
          {
            switch (*boundaries)
              {
              case 1:
                VectorTools::
                interpolate_boundary_values (dof_handler_velocity,
                                             *boundaries,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
                break;
              case 2:
                VectorTools::
                interpolate_boundary_values (dof_handler_velocity,
                                             *boundaries,
                                             vel_exact,
                                             boundary_values);
                break;
              case 3:
                if (d != 0)
                  VectorTools::
                  interpolate_boundary_values (dof_handler_velocity,
                                               *boundaries,
                                               Functions::ZeroFunction<dim>(),
                                               boundary_values);
                break;
              case 4:
                VectorTools::
                interpolate_boundary_values (dof_handler_velocity,
                                             *boundaries,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
                break;
              case 5:
                VectorTools::
                interpolate_boundary_values (dof_handler_velocity,
                                            *boundaries,
                                            Functions::ZeroFunction<dim>(),
                                            boundary_values);
                break;
                      
              default:
                Assert (false, ExcNotImplemented());
              }
          }
        MatrixTools::apply_boundary_values (boundary_values,
                                            vel_it_matrix[d],
                                            u_n[d],
                                            force[d]);
      }


    Threads::TaskGroup<void> tasks;
    for (unsigned int d=0; d<dim; ++d)
      {
        if (reinit_prec)
          prec_velocity[d].initialize (vel_it_matrix[d],
                                       SparseILU<double>::
                                       AdditionalData (vel_diag_strength,
                                                       vel_off_diagonals));
        tasks += Threads::new_task (&NavierStokesProjection<dim>::
                                    diffusion_component_solve,
                                    *this, d);
      }
    tasks.join_all();
  }



  template <int dim>
  void
  NavierStokesProjection<dim>::diffusion_component_solve (const unsigned int d)
  {
    SolverControl solver_control (vel_max_its, vel_eps*force[d].l2_norm());
    SolverGMRES<> gmres (solver_control,
                         SolverGMRES<>::AdditionalData (vel_Krylov_size));
    gmres.solve (vel_it_matrix[d], u_n[d], force[d], prec_velocity[d]);
  }


  // @sect4{ The <code>NavierStokesProjection::assemble_advection_term</code> method and related}

  // The following few functions deal with assembling the advection terms,
  // which is the part of the system matrix for the diffusion step that
  // changes at every time step. As mentioned above, we will run the assembly
  // loop over all cells in %parallel, using the WorkStream class and other
  // facilities as described in the documentation module on @ref threads.
  template <int dim>
  void
  NavierStokesProjection<dim>::assemble_advection_term()
  {
    vel_Advection = 0.;
    AdvectionPerTaskData data (fe_velocity.dofs_per_cell);
    AdvectionScratchData scratch (fe_velocity, quadrature_velocity,
                                  update_values |
                                  update_JxW_values |
                                  update_gradients);
    WorkStream::run (dof_handler_velocity.begin_active(),
                     dof_handler_velocity.end(), *this,
                     &NavierStokesProjection<dim>::assemble_one_cell_of_advection,
                     &NavierStokesProjection<dim>::copy_advection_local_to_global,
                     scratch,
                     data);
  }



  template <int dim>
  void
  NavierStokesProjection<dim>::
  assemble_one_cell_of_advection(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 AdvectionScratchData &scratch,
                                 AdvectionPerTaskData &data)
  {
    scratch.fe_val.reinit(cell);
    cell->get_dof_indices (data.local_dof_indices);
    for (unsigned int d=0; d<dim; ++d)
      {
        scratch.fe_val.get_function_values (u_star[d], scratch.u_star_tmp);
        for (unsigned int q=0; q<scratch.nqp; ++q)
          scratch.u_star_local[q](d) = scratch.u_star_tmp[q];
      }

    for (unsigned int d=0; d<dim; ++d)
      {
        scratch.fe_val.get_function_gradients (u_star[d], scratch.grad_u_star);
        for (unsigned int q=0; q<scratch.nqp; ++q)
          {
            if (d==0)
              scratch.u_star_tmp[q] = 0.;
            scratch.u_star_tmp[q] += scratch.grad_u_star[q][d];
          }
      }

    data.local_advection = 0.;
    for (unsigned int q=0; q<scratch.nqp; ++q)
      for (unsigned int i=0; i<scratch.dpc; ++i)
        for (unsigned int j=0; j<scratch.dpc; ++j)
          data.local_advection(i,j) += (scratch.u_star_local[q] *
                                        scratch.fe_val.shape_grad (j, q) *
                                        scratch.fe_val.shape_value (i, q)
                                        +
                                        0.5 *
                                        scratch.u_star_tmp[q] *
                                        scratch.fe_val.shape_value (i, q) *
                                        scratch.fe_val.shape_value (j, q))
                                       *
                                       scratch.fe_val.JxW(q) ;
  }



  template <int dim>
  void
  NavierStokesProjection<dim>::
  copy_advection_local_to_global(const AdvectionPerTaskData &data)
  {
    for (unsigned int i=0; i<fe_velocity.dofs_per_cell; ++i)
      for (unsigned int j=0; j<fe_velocity.dofs_per_cell; ++j)
        vel_Advection.add (data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.local_advection(i,j));
  }



  // @sect4{<code>NavierStokesProjection::projection_step</code>}

  // This implements the projection step:
  template <int dim>
  void
  NavierStokesProjection<dim>::projection_step (const bool reinit_prec)
  {
    pres_iterative.copy_from (pres_Laplace);

    pres_tmp = 0.;
    for (unsigned d=0; d<dim; ++d)
      pres_Diff[d].Tvmult_add (pres_tmp, u_n[d]);

    phi_n_minus_1 = phi_n;

    static std::map<types::global_dof_index, double> bval;
    if (reinit_prec)
      VectorTools::interpolate_boundary_values (dof_handler_pressure, 3,
                                                Functions::ZeroFunction<dim>(), bval);

    MatrixTools::apply_boundary_values (bval, pres_iterative, phi_n, pres_tmp);

    if (reinit_prec)
      prec_pres_Laplace.initialize(pres_iterative,
                                   SparseILU<double>::AdditionalData (vel_diag_strength,
                                       vel_off_diagonals) );

    SolverControl solvercontrol (vel_max_its, vel_eps*pres_tmp.l2_norm());
    SolverCG<> cg (solvercontrol);
    cg.solve (pres_iterative, phi_n, pres_tmp, prec_pres_Laplace);

    phi_n *= 1.5/dt;
  }


  // @sect4{ <code>NavierStokesProjection::update_pressure</code> }

  // This is the pressure update step of the projection method. It implements
  // the standard formulation of the method, that is @f[ p^{n+1} = p^n +
  // \phi^{n+1}, @f] or the rotational form, which is @f[ p^{n+1} = p^n +
  // \phi^{n+1} - \frac{1}{Re} \nabla\cdot u^{n+1}.  @f]
  template <int dim>
  void
  NavierStokesProjection<dim>::update_pressure (const bool reinit_prec)
  {
    pres_n_minus_1 = pres_n;
    switch (type)
      {
      case RunTimeParameters::METHOD_STANDARD:
        pres_n += phi_n;
        break;
      case RunTimeParameters::METHOD_ROTATIONAL:
        if (reinit_prec)
          prec_mass.initialize (pres_Mass);
        pres_n = pres_tmp;
        prec_mass.solve (pres_n);
        pres_n.sadd(1./Rey, 1., pres_n_minus_1);
        pres_n += phi_n;
        break;
      default:
        Assert (false, ExcNotImplemented());
      };
  }


  // @sect4{ <code>NavierStokesProjection::output_results</code> }

  // This method plots the current solution. The main difficulty is that we
  // want to create a single output file that contains the data for all
  // velocity components, the pressure, and also the vorticity of the flow. On
  // the other hand, velocities and the pressure live on separate DoFHandler
  // objects, and so can't be written to the same file using a single DataOut
  // object. As a consequence, we have to work a bit harder to get the various
  // pieces of data into a single DoFHandler object, and then use that to
  // drive graphical output.
  //
  // We will not elaborate on this process here, but rather refer to step-32,
  // where a similar procedure is used (and is documented) to
  // create a joint DoFHandler object for all variables.
  //
  // Let us also note that we here compute the vorticity as a scalar quantity
  // in a separate function, using the $L^2$ projection of the quantity
  // $\text{curl} u$ onto the finite element space used for the components of
  // the velocity. In principle, however, we could also have computed as a
  // pointwise quantity from the velocity, and do so through the
  // DataPostprocessor mechanism discussed in step-29 and step-33.
  template <int dim>
  void NavierStokesProjection<dim>::output_results (const unsigned int step)
  {
    assemble_vorticity ( (step == 1));
    const FESystem<dim> joint_fe (fe_velocity, dim,
                                  fe_pressure, 1,
                                  fe_velocity, 1);
    DoFHandler<dim> joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Assert (joint_dof_handler.n_dofs() ==
            ((dim + 1)*dof_handler_velocity.n_dofs() +
             dof_handler_pressure.n_dofs()),
            ExcInternalError());
    static Vector<double> joint_solution (joint_dof_handler.n_dofs());
    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
        loc_vel_dof_indices (fe_velocity.dofs_per_cell),
        loc_pres_dof_indices (fe_pressure.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
    joint_cell = joint_dof_handler.begin_active(),
    joint_endc = joint_dof_handler.end(),
    vel_cell   = dof_handler_velocity.begin_active(),
    pres_cell  = dof_handler_pressure.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell)
      {
        joint_cell->get_dof_indices (loc_joint_dof_indices);
        vel_cell->get_dof_indices (loc_vel_dof_indices),
                 pres_cell->get_dof_indices (loc_pres_dof_indices);
        for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
          switch (joint_fe.system_to_base_index(i).first.first)
            {
            case 0:
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                u_n[ joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 1:
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                pres_n (loc_pres_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 2:
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                rot_u (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 3:
                Assert (joint_fe.system_to_base_index(i).first.second == 0,
                            ExcInternalError());
                joint_solution (loc_joint_dof_indices[i]) =
                    std::abs(shear_stress (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]));
                break;
            default:
              Assert (false, ExcInternalError());
            }
      }
    std::vector<std::string> joint_solution_names (dim, "v");
    joint_solution_names.push_back ("p");
    joint_solution_names.push_back ("rot_u");
//    joint_solution_names.push_back ("shear_stress");
    DataOut<dim> data_out;
    data_out.attach_dof_handler (joint_dof_handler);
    std::vector< DataComponentInterpretation::DataComponentInterpretation >
    component_interpretation (dim+2,
                              DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim]
      = DataComponentInterpretation::component_is_scalar;
    component_interpretation[dim+1]
      = DataComponentInterpretation::component_is_scalar;
    data_out.add_data_vector (joint_solution,
                              joint_solution_names,
                              DataOut<dim>::type_dof_data,
                              component_interpretation);
    data_out.build_patches (deg + 1);
    std::ofstream output (("diamond_1_test_" +
                           Utilities::int_to_string (step, 5) +
                           ".vtk").c_str());
    data_out.write_vtk (output);
      
      data_out.build_patches (deg + 1);
//    std::ofstream ofs (("boundary_solution_xshape_0.00175" +
//                             Utilities::int_to_string (step, 5) +
//                             ".txt").c_str());
    std::ofstream ofs2 (("boundary_solution_diamond_1_test_xy " +
                          Utilities::int_to_string (step, 5) +
                          ".txt").c_str());
      
    std::cout << "Step = " << step << " Time = " << (step*dt) << std::endl;
//
    std::vector<std::vector<double>> data = output_results_boundary();
//    ofs.precision( 18 );
    ofs2.precision( 18 );
      double x_coord = 0;
      double x_coord_2 = 0.03;
      double x_coord_3 = 0.06;
      double x_coord_4 = 0.09;
      double x_coord_5 = 0.12;
      bool wss_out = false;
      bool wss_out_2 = false;
      bool wss_out_3 = false;
      bool wss_out_4 = false;
      bool wss_out_5 = false;
      std::cout.precision( 18 );
      for(auto const& idata : data){
          ofs2 << idata[0]<< "\t" << idata[1] << std::endl;
          if (idata[0] > x_coord && idata[0] < x_coord+0.001 && wss_out == false){
              wss_out = true;
//              std::cout << step*dt <<" "<< idata[1]<<std::endl;
          }
          if (idata[0] > x_coord_2 && idata[0] < x_coord_2+0.001 && wss_out_2 == false){
              wss_out_2 = true;
//              std::cout <<" "<< idata[1]<<std::endl;
          }
          if (idata[0] > x_coord_3 && idata[0] < x_coord_3+0.001 && wss_out_3 == false){
              wss_out_3 = true;
//              std::cout <<" "<< idata[1]<<std::endl;
          }
          if (idata[0] > x_coord_4 && idata[0] < x_coord_4+0.001 && wss_out_4 == false){
              wss_out_4 = true;
//              std::cout <<" "<< idata[1]<<std::endl;
          }
          if (idata[0] > x_coord_5 && idata[0] < x_coord_5+0.001 && wss_out_5 == false){
              wss_out_5 = true;
//              std::cout <<" "<< idata[1]<<std::endl;
          }
      }
  }
    
    template <int dim>
    std::vector<std::vector<double>> NavierStokesProjection<dim>::output_results_boundary()
    {
        std::vector<std::vector<double>> data(0);
        std::vector<Point<dim> >  points;
        std::vector<Tensor<1, dim> > WSS_qp;
        std::vector<double>  OSI_qp;
        double length = 0;
        double wss_x = 0, wss_y = 0;
        static int out_index_b = 0;
        const UpdateFlags uv_face(update_values | update_quadrature_points | update_normal_vectors | update_gradients| update_JxW_values);
        FEFaceValues<dim> fe_facev_values(fe_velocity, qv_face, uv_face);
        const unsigned int dpf = fe_velocity.dofs_per_cell, nqp = qv_face.size();

        std::vector< Tensor<1,dim> > grad_u1 (nqp), grad_u2 (nqp);
        int nf = 0;
    
        typename DoFHandler<dim>::active_cell_iterator cell =
        dof_handler_velocity.begin_active(), endc = dof_handler_velocity.end();
        for (unsigned int cell_count = 0; cell != endc; ++cell, ++cell_count){
            if(cell->is_locally_owned()){
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f){
                    if (cell->face(f)->boundary_id()==5){
                        fe_facev_values.reinit(cell, f);
                        fe_facev_values.get_function_gradients (u_n[0], grad_u1);
                        fe_facev_values.get_function_gradients (u_n[1], grad_u2);
                        nf += 1;
                        for(unsigned int f_q_point = 0; f_q_point < nqp; ++f_q_point)
                        {
                            Point<dim> qpt = fe_facev_values.quadrature_point(f_q_point);
                            points.push_back(qpt);
                            double dudx= 0.0,dudy= 0.0,dvdx= 0.0,dvdy = 0.0;
//                            double qp_pres = 0.0, qp_u1 = 0.0, qp_u2 = 0.0;
                            for(unsigned int id = 0; id < dpf; ++id){
                                auto shape = fe_facev_values.shape_value(id,f_q_point);
                                dudx += grad_u1[f_q_point][0]*shape;
                                dudy += grad_u1[f_q_point][1]*shape;
                                dvdx += grad_u2[f_q_point][0]*shape;
                                dvdy += grad_u2[f_q_point][1]*shape;
//                                qp_pres += pres[f_q_point] * shape;
//                                qp_u1 += u1[f_q_point] * shape;
//                                qp_u2 += u2[f_q_point] * shape;
                            }
                            Tensor<2,dim> strain;
                            strain[0][0] = dudx;
                            strain[0][1] = (dudy+dvdx)/2.0;
                            strain[1][0] = (dudy+dvdx)/2.0;
                            strain[1][1] = dvdy;
                            Tensor<2,dim> stress = (1.0/Rey) *1000* strain;
                            Tensor<1,dim> norm = fe_facev_values.normal_vector(f_q_point);
                            Tensor<1,dim> traction,traction_n,WSS_qpt;
                            traction[0] = stress[0][0]*norm[0]+stress[0][1]*norm[1];
                            traction[1] = stress[1][0]*norm[0]+stress[1][1]*norm[1];
                            traction_n = (traction[0]*norm[0] + traction[1]*norm[1]) * norm;
//                            double v_n = qp_u1 * norm[0] + qp_u2*norm[1];
//                            double v_t = (qp_u1 - v_n * norm[0])*norm[0] + (qp_u2 - v_n * norm[1]) * norm[1];
                            
                            WSS_qpt = traction - traction_n;
                            
                            wss_x += WSS_qpt[0] * fe_facev_values.JxW(f_q_point);
                            wss_y += WSS_qpt[1] * fe_facev_values.JxW(f_q_point);
                            
                            if (WSS_qpt.norm() < 0.008){
                                length += fe_facev_values.JxW(f_q_point);
                            }
                            data.push_back({qpt[0],WSS_qpt.norm()});
                            std::cout << qpt[0] <<" "<< WSS_qpt.norm() << std::endl;
                        }
                    }
                }
            }
        }
        return data;
    }
  

  // Following is the helper function that computes the vorticity by
  // projecting the term $\text{curl} u$ onto the finite element space used
  // for the components of the velocity. The function is only called whenever
  // we generate graphical output, so not very often, and as a consequence we
  // didn't bother parallelizing it using the WorkStream concept as we do for
  // the other assembly functions. That should not be overly complicated,
  // however, if needed. Moreover, the implementation that we have here only
  // works for 2d, so we bail if that is not the case.
  template <int dim>
  void NavierStokesProjection<dim>::assemble_vorticity (const bool reinit_prec)
  {
    Assert (dim == 2, ExcNotImplemented());
    if (reinit_prec)
      prec_vel_mass.initialize (vel_Mass);

    FEValues<dim> fe_val_vel (fe_velocity, quadrature_velocity,
                              update_gradients |
                              update_JxW_values |
                              update_values);
    const unsigned int dpc = fe_velocity.dofs_per_cell,
                       nqp = quadrature_velocity.size();
    std::vector<types::global_dof_index> ldi (dpc);
    Vector<double> loc_rot (dpc);
    Vector<double> loc_shear_stress (dpc);
    std::vector< Tensor<1,dim> > grad_u1 (nqp), grad_u2 (nqp);
    rot_u = 0.;
    shear_stress = 0.;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_velocity.begin_active(),
    end  = dof_handler_velocity.end();
      double area = 0.0;
    for (; cell != end; ++cell)
      {
        fe_val_vel.reinit (cell);
        cell->get_dof_indices (ldi);
        fe_val_vel.get_function_gradients (u_n[0], grad_u1);
        fe_val_vel.get_function_gradients (u_n[1], grad_u2);
        loc_rot = 0.;
        loc_shear_stress = 0.;
          for (unsigned int q=0; q<nqp; ++q){
            area += fe_val_vel.JxW(q);
            for (unsigned int i=0; i<dpc; ++i){
                loc_rot(i) += (grad_u2[q][0] - grad_u1[q][1]) *
                            fe_val_vel.shape_value (i, q) *
                            fe_val_vel.JxW(q);
                loc_shear_stress(i) += (1./Rey) * (1./2.) * (grad_u1[q][1] + grad_u2[q][0])*
                            fe_val_vel.shape_value (i, q) *
                            fe_val_vel.JxW(q);
            }
          }

          for (unsigned int i=0; i<dpc; ++i){
              rot_u (ldi[i]) += loc_rot(i);
              shear_stress (ldi[i]) += loc_shear_stress(i);
          }
      }
//      std::cout << "area = "<< (25*8.1) - area << "\n";
    prec_vel_mass.solve (rot_u);
    prec_vel_mass.solve(shear_stress);
  }
}


// @sect3{ The main function }

// The main function looks very much like in all the other tutorial programs,
// so there is little to comment on here:
int main()
{
  try
    {
      using namespace dealii;
      using namespace Step35;

      RunTimeParameters::Data_Storage data;
      data.read_data ("parameter-file.prm");

      deallog.depth_console (data.verbose ? 2 : 0);
        
      GmeshGenerator::MeshGenerator gmsh(data);
      gmsh.output();
        
      NavierStokesProjection<2> test (data);
      test.run (data.verbose, data.output_interval);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!"
            << std::endl
            << "Don't forget to brush your teeth :-)"
            << std::endl << std::endl;
  return 0;
}
