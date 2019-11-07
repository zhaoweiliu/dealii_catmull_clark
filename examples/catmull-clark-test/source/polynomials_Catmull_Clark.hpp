//
//  polynomials_Catmull_Clark.hpp
//  step-4
//
//  Created by zhaowei Liu on 07/11/2019.
//

#ifndef polynomials_Catmull_Clark_hpp
#define polynomials_Catmull_Clark_hpp

#include <stdio.h>
#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include "polynomials_CubicBSpline.hpp"
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_product_polynomials.h>

DEAL_II_NAMESPACE_OPEN


template<int dim>
class polynomials_Catmull_Clark
{
public:
    polynomials_Catmull_Clark();
    
    void compute(const Point<dim> &unit_point,
                 std::vector<double>&values,
                 std::vector<Tensor<1,dim>> &grads,
                 std::vector<Tensor<2,dim>> &grad_grads,
                 std::vector<Tensor<3,dim>> &third_derivatives,
                 std::vector<Tensor<4,dim>> &fourth_derivatives )const;
        
    unsigned int
    degree() const;
    
    std::string
    name() const;
    
private:
    const unsigned int my_degree = 3;
    
    std::vector<PolynomialsCubicBSpline> polys_1d;
};



template <int dim>
inline unsigned int
polynomials_Catmull_Clark<dim>::degree() const
{
    return my_degree;
}



template <int dim>
inline std::string
polynomials_Catmull_Clark<dim>::name() const
{
    return "Catmull_Clark";
}


DEAL_II_NAMESPACE_CLOSE

#endif /* polynomials_Catmull_Clark_hpp */
