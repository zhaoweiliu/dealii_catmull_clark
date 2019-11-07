//
//  polynomials_Catmull_Clark.cpp
//  step-4
//
//  Created by zhaowei Liu on 07/11/2019.
//

#include "polynomials_Catmull_Clark.hpp"

DEAL_II_NAMESPACE_OPEN


template<int dim>
polynomials_Catmull_Clark<dim>::polynomials_Catmull_Clark()
{
    for (int i = 0 ; i < 3 ; ++i){
        polys_1d.push_back(PolynomialsCubicBSpline(i)) ;
    }
}

template<int dim>
void polynomials_Catmull_Clark<dim>::
compute(const Point<dim> &unit_point,
        std::vector<double>&values,
        std::vector<Tensor<1,dim>> &grads,
        std::vector<Tensor<2,dim>> &grad_grads,
        std::vector<Tensor<3,dim>> &third_derivatives,
        std::vector<Tensor<4,dim>> &fourth_derivatives )const
{
    
}


DEAL_II_NAMESPACE_CLOSE
