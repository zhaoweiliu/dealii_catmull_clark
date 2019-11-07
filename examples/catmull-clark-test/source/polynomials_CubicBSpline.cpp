//
//  polynomials_CubicBSpline.cpp
//  step-4
//
//  Created by zhaowei Liu on 07/11/2019.
//

#include "polynomials_CubicBSpline.hpp"

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace
    {
    std::vector<double> get_cubicbspline_coefficients(const unsigned int index){
        
        switch (index) {
            case 0:
                return {1./6.,-1./2.,1./2.,-1./6.};
                break;
            case 1:
                return {4./6.,0.,1.,-1./2.};
                break;
            case 2:
                return {1./6.,1./2.,1./2.,-1./2.};
                break;
            case 3:
                return {0.,0.,0.,-1.};
                break;
            default:
                Assert(index > 3, ExcMessage("cubic polynomial coefficient index needs to be < 4."));
                return {0,0,0,0};
                break;
        }
    }
    }

PolynomialsCubicBSpline :: PolynomialsCubicBSpline(const unsigned int index)
: Polynomials::Polynomial<double>(get_cubicbspline_coefficients(index))
{}

DEAL_II_NAMESPACE_CLOSE
