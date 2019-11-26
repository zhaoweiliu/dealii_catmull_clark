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
    for (int i = 0 ; i < 2 ; ++i){
        polys_1d_end.push_back(PolynomialsCubicBSplineEnd(i)) ;
    }
}

template<int dim>
void polynomials_Catmull_Clark<dim>::regular::
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



template<int dim>
void polynomials_Catmull_Clark<dim>::two_ends_truncated::
compute(const Point<dim> &unit_point,
        std::vector<double>&values,
        std::vector<Tensor<1,dim>> &grads,
        std::vector<Tensor<2,dim>> &grad_grads,
        std::vector<Tensor<3,dim>> &third_derivatives,
        std::vector<Tensor<4,dim>> &fourth_derivatives )const
{
    AssertDimension(dim, 2);
    std::vector<PolynomialsCubicBSplineEnd> pols;
    for (unsigned int i = 0; i < 3 ; ++i) {
        PolynomialsCubicBSplineEnd p_i(i);
        pols.push_back(p_i);
    }
    TensorProductPolynomials<dim> tp_poly(pols);
    for (unsigned int i = 0; i < 3*3; ++i) {
        values[i] = tp_poly.compute_value(i, unit_point);
        grads[i] = tp_poly.compute_grad(i, unit_point);
        grad_grads[i] = tp_poly.compute_grad_grad(i, unit_point);
//        third_derivatives[i] = tp_poly.compute_derivative<3>(i, unit_point);
    }
};

template<int dim>
void polynomials_Catmull_Clark<dim>::one_end_truncated::
compute(const Point<dim> &unit_point,
        std::vector<double>&values,
        std::vector<Tensor<1,dim>> &grads,
        std::vector<Tensor<2,dim>> &grad_grads,
        std::vector<Tensor<3,dim>> &third_derivatives,
        std::vector<Tensor<4,dim>> &fourth_derivatives )const
{
    AssertDimension(dim, 2);
    std::vector<PolynomialsCubicBSpline> pols_1;
    std::vector<PolynomialsCubicBSplineEnd> pols_2;
    for (unsigned int i = 0; i < 4 ; ++i) {
        PolynomialsCubicBSpline p_i(i);
        pols_1.push_back(p_i);
    }
    for (unsigned int i = 0; i < 3 ; ++i) {
        PolynomialsCubicBSplineEnd p_i(i);
        pols_2.push_back(p_i);
    }
    for (unsigned int i = 0; i < 4; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            values[j*4+i] = pols_1[i].value(unit_point[0])*pols_2[j].value(unit_point[1]);
            
            auto poly_1_der = pols_1[i].derivative();
            auto poly_2_der = pols_2[j].derivative();
            Tensor<1,dim> grad;
            grad[0] = poly_1_der.value(unit_point[0]) * pols_2[j].value(unit_point[1]);
            grad[1] = pols_1[i].value(unit_point[0]) * poly_2_der.value(unit_point[1]);
            grads[j*4 + i] = grad;
            auto poly_1_der2 = poly_1_der.derivative();
            auto poly_2_der2 = poly_2_der.derivative();
            Tensor<2,dim> grad_grad;
            grad_grad[0][0] = poly_1_der2.value(unit_point[0]) * pols_2[j].value(unit_point[1]);
            grad_grad[0][1] = poly_1_der.value(unit_point[0]) * poly_2_der.value(unit_point[1]);
            grad_grad[1][0] = grad_grad[0][1];
            grad_grad[1][1] = pols_1[i].value(unit_point[0]) * poly_2_der2.value(unit_point[1]);
            grad_grads[j*4+i] = grad_grad;
        }
    }
};



template class polynomials_Catmull_Clark<1>;
template class polynomials_Catmull_Clark<2>;

DEAL_II_NAMESPACE_CLOSE
