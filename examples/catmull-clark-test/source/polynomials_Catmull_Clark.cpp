//
//  polynomials_Catmull_Clark.cpp
//  step-4
//
//  Created by zhaowei Liu on 07/11/2019.
//

#include "polynomials_Catmull_Clark.hpp"

DEAL_II_NAMESPACE_OPEN

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
            values[i] = polys_1d[i].value(u);
            auto poly_der = polys_1d[i].derivative();
            grads[i][0] = poly_der.value(u);
            auto poly_der_der = poly_der.derivative();
            grad_grads[i][0][0] = poly_der_der.value(u);
            auto poly_third_der = poly_der_der.derivative();
            third_derivatives[i][0][0][0] = poly_third_der.value(u);
            fourth_derivatives[i] = 0;
        }
    }
    else if(dim == 2){
        TensorProductPolynomials<dim> tp_poly(polys_1d);
        for (unsigned int i = 0; i < 4*4; ++i) {
            values[i] = tp_poly.compute_value(i, unit_point);
            grads[i] = tp_poly.compute_grad(i, unit_point);
            grad_grads[i] = tp_poly.compute_grad_grad(i, unit_point);
        }
    }
};



template<int dim>
double polynomials_Catmull_Clark<dim>::regular::value(const unsigned int i, const Point<dim> &unit_point) const
{
    TensorProductPolynomials<dim> tp_poly(polys_1d);
    return tp_poly.compute_value(i, unit_point);
}

template<int dim>
Tensor<1,dim> polynomials_Catmull_Clark<dim>::regular::grads( const unsigned int i, const Point<dim> &unit_point) const
{
    TensorProductPolynomials<dim> tp_poly(polys_1d);
    return tp_poly.compute_grad(i, unit_point);
}

template<int dim>
Tensor<2,dim> polynomials_Catmull_Clark<dim>::regular::grad_grads( const unsigned int i, const Point<dim> &unit_point) const
{
    TensorProductPolynomials<dim> tp_poly(polys_1d);
    return tp_poly.compute_grad_grad(i, unit_point);
}

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
    TensorProductPolynomials<dim> tp_poly(polys_1d_end);
    for (unsigned int i = 0; i < 3*3; ++i) {
        values[i] = tp_poly.compute_value(i, unit_point);
        grads[i] = tp_poly.compute_grad(i, unit_point);
        grad_grads[i] = tp_poly.compute_grad_grad(i, unit_point);
    }
};



template<int dim>
double polynomials_Catmull_Clark<dim>::two_ends_truncated::value(const unsigned int i, const Point<dim> &unit_point) const
{
    TensorProductPolynomials<dim> tp_poly(polys_1d_end);
    return tp_poly.compute_value(i, unit_point);
}



template<int dim>
Tensor<1,dim> polynomials_Catmull_Clark<dim>::two_ends_truncated::grads(const unsigned int i, const Point<dim> &unit_point) const
{
    TensorProductPolynomials<dim> tp_poly(polys_1d_end);
    return tp_poly.compute_grad(i, unit_point);
}



template<int dim>
Tensor<2,dim> polynomials_Catmull_Clark<dim>::two_ends_truncated::grad_grads(const unsigned int i, const Point<dim> &unit_point) const
{
    TensorProductPolynomials<dim> tp_poly(polys_1d_end);
    return tp_poly.compute_grad_grad(i, unit_point);
}



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



template<int dim>
double polynomials_Catmull_Clark<dim>::one_end_truncated::value(const unsigned int i, const Point<dim> &unit_point) const{
    unsigned int a = i/4;
    unsigned int b = i - a*4;
    return pols_1[b].value(unit_point[0])*pols_2[a].value(unit_point[1]);
}

template<int dim>
Tensor<1,dim> polynomials_Catmull_Clark<dim>::one_end_truncated::grads(const unsigned int i, const Point<dim> &unit_point) const
{
    unsigned int a = i/4;
    unsigned int b = i - a*4;
    Tensor<1,dim> grads;
    grads[0] = pols_1[b].derivative().value(unit_point[0])*pols_2[a].value(unit_point[1]);
    grads[1] = pols_1[b].value(unit_point[0])*pols_2[a].derivative().value(unit_point[1]);
    return grads;
}

template<int dim>
Tensor<2,dim> polynomials_Catmull_Clark<dim>::one_end_truncated::grad_grads(const unsigned int i, const Point<dim> &unit_point) const
{
    unsigned int a = i/4;
    unsigned int b = i - a*4;
    Tensor<2,dim> grad_grads;
    grad_grads[0][0] = pols_1[b].derivative().derivative().value(unit_point[0])*pols_2[a].value(unit_point[1]);
    
    grad_grads[0][1] = pols_1[b].derivative().value(unit_point[0])*pols_2[a].derivative().value(unit_point[1]);
    
    grad_grads[1][0] = pols_1[b].derivative().value(unit_point[0])*pols_2[a].derivative().value(unit_point[1]);

    grad_grads[1][1] = pols_1[b].value(unit_point[0])*pols_2[a].derivative().derivative().value(unit_point[1]);
    return grad_grads;
}

template class polynomials_Catmull_Clark<1>;
template class polynomials_Catmull_Clark<2>;

DEAL_II_NAMESPACE_CLOSE
