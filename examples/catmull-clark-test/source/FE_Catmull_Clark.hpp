//
//  FE_Catmull_Clark.hpp
//  step-4
//
//  Created by zhaowei Liu on 05/11/2019.
//

#ifndef FE_Catmull_Clark_hpp
#define FE_Catmull_Clark_hpp

#include <stdio.h>

#include <deal.II/fe/fe.h>

DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim = dim>
class FE_Catmull_Clark : public FiniteElement<dim, spacedim>
{
public:
    
    FE_Catmull_Clark(const unsigned int val, const unsigned int n_components = 1, const bool dominate = false);
    
    virtual std::unique_ptr<FiniteElement<dim, spacedim>>
    clone() const override;
    
    /**
     * Return a string that uniquely identifies a finite element. In this case
     * it is <code>FE_Catmull_Clark@<dim@></code>.
     */
    virtual std::string
    get_name() const override;
    
    // for documentation, see the FiniteElement base class
    virtual UpdateFlags
    requires_update_flags(const UpdateFlags update_flags) const override;

    
private:

    const bool dominate;
};

DEAL_II_NAMESPACE_CLOSE

#endif
