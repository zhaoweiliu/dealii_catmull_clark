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
double FE_Catmull_Clark<dim, spacedim>::shape_value (const unsigned int i, const Point< dim > &p) const{
    
    
}

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
