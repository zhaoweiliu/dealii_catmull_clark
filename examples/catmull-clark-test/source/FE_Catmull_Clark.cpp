//
//  FE_Catmull_Clark.cpp
//  step-4
//
//  Created by zhaowei Liu on 05/11/2019.
//

#include "FE_Catmull_Clark.hpp"

#include <deal.II/base/std_cxx14/memory.h>

#include <deal.II/fe/fe_nothing.h>

#include "polynomials_Catmull_Clark.hpp"


DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim>
FE_Catmull_Clark<dim,spacedim>::FE_Catmull_Clark(const unsigned int val, const unsigned int n_components, const bool dominate)
: FiniteElement<dim,spacedim> (
    FiniteElementData<dim>({0,0,0,0,2*val+8},
                            n_components,
                            0,
                            FiniteElementData<dim>::H2),
                               
    std::vector<bool>(),
    std::vector<ComponentMask>()),
    valence(val),
    dominate(dominate)
{}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_Catmull_Clark<dim, spacedim>::clone() const
{
  return std_cxx14::make_unique<FE_Catmull_Clark<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_Catmull_Clark<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_Catmull_Clark<" << dim << ">(";
  if (this->n_components() > 1)
    {
      namebuf << this->n_components();
      if (dominate)
        namebuf << ", dominating";
    }
  else if (dominate)
    namebuf << "dominating";
  namebuf << ")";
  return namebuf.str();
}



template <int dim, int spacedim>
UpdateFlags
FE_Catmull_Clark<dim, spacedim>::requires_update_flags(const UpdateFlags flags) const
{
  return flags;
}



template<int dim, int spacedim>
double FE_Catmull_Clark<dim, spacedim>::shape_value (const unsigned int i, const Point< dim > &p) const
{
    if (valence == 4){
        // i in [0,15];
        return poly_reg.value(i,p);
    }else if(valence == 2){
        // i in [0,11];
        return poly_one_end.value(i,p);
    }else if (valence == 1){
        // i in [0,8];
        return poly_two_ends.value(i,p);
    }else{
        // i in [0, 2*valence + 7]
    }
}


template<int dim, int spacedim>
FullMatrix<double> FE_Catmull_Clark<dim, spacedim>::compute_subd_matrix(const Point<dim> p, Point<dim> &p_mapped, double &Jacobian){
    double u = p[0], v = p[1];
    double eps = 10e-10;
    if (u < eps && v < eps){
        u += eps;
        v += eps;
    }
    int n = int(std::floor(std::min(-std::log2(u), -std::log2(v))+1));
    double pow2 = pow(2.,n-1.);
    int k = -1;
    u *= pow2;
    v *= pow2;
    if (v < 0.5) {
        k = 0; u = 2. * u - 1.; v = 2. * v;
    }else if(u < 0.5){
        k = 2; u = 2. * u; v = 2. * v - 1.;
    }else{
        k = 1; u = 2. * u - 1; v = 2. * v - 1.;
    }
    // mapping p into the sub paramentric domian
    p_mapped = {u,v};
    
    FullMatrix<double> P;
    switch (k) {
        case 0:
             P = pickmtrx1();
        case 1:
             P = pickmtrx2();
        case 2:
             P = pickmtrx3();
        default:
            throw std::runtime_error("no picking matrix returned.");
            break;
    }
    FullMatrix<double> D(16,2*valence+8);
    FullMatrix<double> A_bar = A_bar_matrix();
    FullMatrix<double> A_n = A_bar;
    for(int i = 1;i<n;++i){
        A_bar.mmult(A_n, A_matrix());
        A_bar = A_n;
    }
    P.mmult(D,A_n);
    Jacobian = pow(2,n);
    return D;
};



template <int dim, int spacedim>
std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
FE_Catmull_Clark<dim, spacedim>::
get_data(
         const UpdateFlags update_flags,
         const Mapping<dim, spacedim> & mapping,
         const Quadrature<dim> & quadrature,
         dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,spacedim>& output_data
         ) const
{
    // Create a default data object.  Normally we would then
    // need to resize things to hold the appropriate numbers
    // of dofs, but in this case all data fields are empty.
    return std_cxx14::make_unique<
    typename FiniteElement<dim, spacedim>::InternalDataBase>();
}

DEAL_II_NAMESPACE_CLOSE
