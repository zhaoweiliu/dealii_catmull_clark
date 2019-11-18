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
    if (dim == 1) {
        double u = unit_point[0];
        for (unsigned int i = 0; i < 4; ++i) {
            PolynomialsCubicBSpline poly(i);
            values[i] = poly.value(u);
            auto poly_der = poly.derivative();
            grads[i][0] = poly_der.value(u);
            auto poly_der_der = poly_der.derivative();
            grad_grads[i][0][0] = poly_der_der.value(u);
            auto poly_third_der = poly_der_der.derivative();
            third_derivatives[i][0][0][0] = poly_third_der.value(u);
            fourth_derivatives[i] = 0;
        }
    }
    else if(dim == 2){
        std::vector<PolynomialsCubicBSpline> pols;
        for (unsigned int i = 0; i < 4 ; ++i) {
            PolynomialsCubicBSpline p_i(i);
            pols.push_back(p_i);
        }
        TensorProductPolynomials<dim> tp_poly(pols);
        for (unsigned int i = 0; i < 4*4; ++i) {
            values[i] = tp_poly.compute_value(i, unit_point);
            grads[i] = tp_poly.compute_grad(i, unit_point);
            grad_grads[i] = tp_poly.compute_grad_grad(i, unit_point);
//            third_derivatives[i] = tp_poly.compute_derivative<3>(i, unit_point);
        }
    }
};

template class polynomials_Catmull_Clark<1>;
template class polynomials_Catmull_Clark<2>;

DEAL_II_NAMESPACE_CLOSE
