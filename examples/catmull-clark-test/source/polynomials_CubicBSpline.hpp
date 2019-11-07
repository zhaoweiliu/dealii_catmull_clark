//
//  polynomials_CubicBSpline.hpp
//  step-4
//
//  Created by zhaowei Liu on 07/11/2019.
//

#ifndef polynomials_CubicBSpline_hpp
#define polynomials_CubicBSpline_hpp

#include <stdio.h>

#include <deal.II/base/polynomial.h>

#include <fstream>
#include <iostream>


DEAL_II_NAMESPACE_OPEN
class PolynomialsCubicBSpline : public Polynomials::Polynomial<double>
{
public:
    PolynomialsCubicBSpline(const unsigned int index);
    
};

DEAL_II_NAMESPACE_CLOSE

#endif /* polynomials_CubicBSpline_hpp */
