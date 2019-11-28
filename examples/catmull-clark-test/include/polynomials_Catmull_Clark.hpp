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
    polynomials_Catmull_Clark(){};
    
    unsigned int
    degree() const;
    
    std::string
    name() const;
    
    class regular{
    public:
        regular(){
            polynomials_Catmull_Clark();
            for (int i = 0 ; i < 4 ; ++i){
                polys_1d.push_back(PolynomialsCubicBSpline(i)) ;
            }
        };
        
        void compute(const Point<dim> &unit_point,
        std::vector<double>&values,
        std::vector<Tensor<1,dim>> &grads,
        std::vector<Tensor<2,dim>> &grad_grads,
        std::vector<Tensor<3,dim>> &third_derivatives,
        std::vector<Tensor<4,dim>> &fourth_derivatives )const;
        
        double value( const unsigned int i, const Point<dim> &unit_point) const;
        
    private:
        std::vector<PolynomialsCubicBSpline> polys_1d;
        
    };
    
    class one_end_truncated{
       public:
           one_end_truncated(){
               polynomials_Catmull_Clark();
               for (int i = 0 ; i < 4 ; ++i){
                   pols_1.push_back(PolynomialsCubicBSpline(i)) ;
                   if (i < 3){
                   pols_2.push_back(PolynomialsCubicBSplineEnd(i)) ;
                   }
               }
           };
        
           void compute(const Point<dim> &unit_point,
           std::vector<double>&values,
           std::vector<Tensor<1,dim>> &grads,
           std::vector<Tensor<2,dim>> &grad_grads,
           std::vector<Tensor<3,dim>> &third_derivatives,
           std::vector<Tensor<4,dim>> &fourth_derivatives )const;
        
        double value( const unsigned int i, const Point<dim> &unit_point) const;
        
        private:
        std::vector<PolynomialsCubicBSpline> pols_1;

        std::vector<PolynomialsCubicBSplineEnd> pols_2;

       };
    
    class two_ends_truncated{
    public:
        two_ends_truncated(){
            polynomials_Catmull_Clark();
            for (int i = 0 ; i < 3 ; ++i){
                polys_1d_end.push_back(PolynomialsCubicBSplineEnd(i)) ;
            }
        };
        
        void compute(const Point<dim> &unit_point,
        std::vector<double>&values,
        std::vector<Tensor<1,dim>> &grads,
        std::vector<Tensor<2,dim>> &grad_grads,
        std::vector<Tensor<3,dim>> &third_derivatives,
        std::vector<Tensor<4,dim>> &fourth_derivatives )const;
        
        double value( const unsigned int i, const Point<dim> &unit_point) const;
        
    private:
        std::vector<PolynomialsCubicBSplineEnd> polys_1d_end;

    };
    
private:
    const unsigned int my_degree = 3;
    

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
