//
//  FE_Catmull_Clark.hpp
//  step-4
//
//  Created by zhaowei Liu on 05/11/2019.
//

#ifndef FE_Catmull_Clark_hpp
#define FE_Catmull_Clark_hpp

#include <stdio.h>

 #include <deal.II/base/config.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_poly_tensor.h>

#include "polynomials_CubicBSpline.hpp"
#include "polynomials_Catmull_Clark.hpp"


DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim>
class FE_Catmull_Clark : public FiniteElement<dim, spacedim>
{
public:
    
    FE_Catmull_Clark(const unsigned int valence, const unsigned int n_components = 1, const bool dominate = false);
    
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
    
    virtual double shape_value (const unsigned int i, const Point< dim > &p) const;
    
    
    virtual std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
    get_data(
             const UpdateFlags update_flags,
             const Mapping<dim, spacedim> &mapping,
             const Quadrature<dim> &       quadrature,
         dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,spacedim>
             &output_data) const override;
    
private:
    
    const unsigned int valence;

    const bool dominate;
    
    const typename polynomials_Catmull_Clark<dim>::regular poly_reg;
    
    const typename polynomials_Catmull_Clark<dim>::one_end_truncated poly_one_end;

    const typename polynomials_Catmull_Clark<dim>::two_ends_truncated poly_two_ends;
    
    FullMatrix<double> compute_subd_matrix(const Point<dim> p, Point<dim> &p_mapped, double &Jacobian);
    
    constexpr static double mS_12[7][7] = {
        {1./64., 3./32., 1./64., 0., 3./32., 1./64., 0. },
        {0., 1./16., 1./16., 0., 0., 0., 0.},
        {0., 1./64., 3./32., 1./64., 0., 0., 0.},
        {0., 0., 1./16., 1./16., 0., 0., 0.},
        {0., 0., 0., 0., 1./16., 1./16., 0.},
        {0., 0., 0., 0., 1./64., 3./32., 1./64.},
        {0., 0., 0., 0., 0., 1./16., 1./16.}
    };
    // size of matrix S_11 is not fixed but no-zero entries are fixed
    
    constexpr static double mS_11[7][8] = {
        {1./64., 0., 0., 3./32., 9./16., 3./32., 0., 0.},
        {1./16., 0., 0., 1./16., 3./8., 3./8., 0., 0.},
        {3./32., 0., 0., 1./64., 3./32., 9./16., 3./32., 1./64.},
        {1./16., 0., 0., 0., 0., 3./8., 3./8., 1./16.},
        {1./16., 0., 0., 3./8., 3./8., 1./16., 0., 0.},
        {3./32., 1./64., 3./32., 9./16., 3./32., 1./64., 0., 0.},
        {1./16., 1./16., 3./8., 3./8., 0., 0., 0., 0.}
    };
    
    constexpr static double mS_21[9][7] = {
        { 0., 0., 0., 0., 1./4., 0., 0.},
        { 0., 0., 0., 0., 3./8., 1./16., 0.},
        { 0., 0., 0., 0., 1./4., 1./4., 0.},
        { 0., 0., 0., 0., 1./16., 3./8., 1./16.},
        { 0., 0., 0., 0., 0., 1./4., 1./4.},
        { 0., 0., 0., 1./16., 3./8., 0., 0.},
        { 0., 0., 0., 1./4., 1./4., 0., 0.},
        { 0., 0., 1./16., 3./8., 1./16., 0., 0.},
        { 0., 0., 1./4., 1./4., 0., 0., 0.}
    };
    
    constexpr static double mS_22[9][7] = {
        { 1./4., 1./4., 0., 0., 1./4., 0., 0.},
        { 1./16., 3./8., 1./16., 0., 1./16., 0., 0.},
        { 0., 1./4., 1./4., 0., 0., 0., 0.},
        { 0., 1./16., 3./8., 1./16., 0., 0., 0.},
        { 0., 0., 1./4., 1./4., 0., 0., 0.},
        { 1./16., 1./16., 0., 0., 3./8., 1./16., 0.},
        { 0., 0., 0., 0., 1./4., 1./4., 0.},
        { 0., 0., 0., 0., 1./16., 3./8., 1./16.},
        { 0., 0., 0., 0., 0., 1./4., 1./4.}
    };
    
    const FullMatrix<double> S_matrix(){
        FullMatrix<double> S(2 * valence + 1);
        double a_N = 1. - (7.)/(4. * valence);
        double b_N = 3./(2. * valence * valence);
        double c_N = 1./(4. * valence * valence);
        double d = 3./8.;
        double e = 1./16.;
        double f = 1./4.;
        S.set(0, 0, a_N);
        S.set(0,1,b_N); S.set(0, 2, c_N);
        S.set(1, 0, d); S.set(2, 0, f);
        S.set(1, 1, d); S.set(1, 2, e); S.set(1, 3, e);
        S.set(1, 2*valence-1, e); S.set(1, 2*valence, e);
        S.set(2, 1, f); S.set(2, 2, f); S.set(2, 3, f);
        for(int iv = 1; iv < valence-1; ++iv){
            S.set(0,2*iv+1,b_N);
            S.set(0,2*iv+2,c_N);
            S.set(2*iv+1,0,d);
            S.set(2*iv+2,0,f);
            S.set(2*iv+1,2*iv+1,d);
            S.set(2*iv+1,2*iv,e);S.set(2*iv+1,2*iv-1,e);
            S.set(2*iv+1,2*iv+2,e);S.set(2*iv+1,2*iv+3,e);
            S.set(2*iv+2, 2*iv+2, f);
            S.set(2*iv+2, 2*iv+1, f);S.set(2*iv+2, 2*iv+3, f);
        }
        S.set(0, 2*valence-1, b_N);
        S.set(0, 2*valence, c_N);
        S.set(2*valence-1, 0, d);
        S.set(2*valence, 0, f);
        S.set(2*valence-1, 1, e);
        S.set(2*valence, 1, f);
        S.set(2*valence-1, 2*valence-1, d);
        S.set(2*valence-1, 2*valence-3, e);
        S.set(2*valence-1, 2*valence-2, e);
        S.set(2*valence-1, 2*valence, e);
        S.set(2*valence, 2*valence-1, f);
        S.set(2*valence, 2*valence, f);
        return S;
    };
    
    const FullMatrix<double> A_matrix(){
        FullMatrix<double> A(2*valence+8);
        FullMatrix<double> S = S_matrix();
        //S has size 2*val+1 x 2*val+1
        A.fill(S);
        if(valence != 3){
            for (int i = 0; i<7; ++i) {
                for(int j =0; j<8; ++j){
                    A.set(2*valence+1+i, j, mS_11[i][j]);
                }
            }
        }else{
            for (int i = 0; i<7; ++i) {
                for(int j =0; j<7; ++j){
                    A.set(2*valence+1+i, j, mS_11[i][j]);
                }
                A.set(2*valence+3,1,1./64.);
                A.set(2*valence+4,1,1./16.);
            }
        }
        
        for (int i = 0; i<7; ++i) {
            for(int j =0; j<7; ++j){
                A.set(2*valence+1+i, 2*valence+1+j, mS_12[i][j]);
            }
        }
        return A;
    };
    
    const FullMatrix<double> A_bar_matrix(){
        FullMatrix<double> A_bar(2*valence+17,2*valence+8);
        FullMatrix<double> A = A_matrix();
        A_bar.fill(A);
        for (int i = 0; i<9; ++i) {
            for(int j =0; j<7; ++j){
                A_bar.set(2*valence+8+i, j, mS_21[i][j]);
            }
        }
        for (int i = 0; i<9; ++i) {
            for(int j =0; j<7; ++j){
                A_bar.set(2*valence+8+i, 2*valence+1+j, mS_22[i][j]);
            }
        }
        return A_bar;
    };
    
    const FullMatrix<double> pickmtrx1(){
        FullMatrix<double> P(16,2*valence+17);
        if (valence == 3) {
            P.set(0, 1, 1.0);
        }else{
            P.set(0, 7, 1.0);
        }
        P.set(1, 6, 1.0);
        P.set(2, 2*valence+4, 1.0); P.set(3, 2*valence+12, 1.0);
        P.set(4, 0, 1.0); P.set(5, 5, 1.0);
        P.set(6, 2*valence+3, 1.0); P.set(7, 2*valence+11, 1.0);
        P.set(8, 3, 1.0); P.set(9, 4, 1.0);
        P.set(10, 2*valence+2, 1.0); P.set(11, 2*valence+10, 1.0);
        P.set(12, 2*valence+6, 1.0); P.set(13, 2*valence+5, 1.0);
        P.set(14, 2*valence+1, 1.0); P.set(15, 2*valence+9, 1.0);
        return P;
    };
    
    const FullMatrix<double> pickmtrx2(){
        FullMatrix<double> P(16,2*valence+17);
        P.set(0, 0, 1.0); P.set(1, 5, 1.0);
        P.set(2, 2*valence+3, 1.0); P.set(3, 2*valence+11, 1.0);
        P.set(4, 3, 1.0); P.set(5, 4, 1.0);
        P.set(6, 2*valence+2, 1.0); P.set(7, 2*valence+10, 1.0);
        P.set(8, 2*valence+6, 1.0); P.set(9, 2*valence+5, 1.0);
        P.set(10, 2*valence+1, 1.0); P.set(11, 2*valence+9, 1.0);
        P.set(12, 2*valence+15, 1.0); P.set(13, 2*valence+14, 1.0);
        P.set(14, 2*valence+13, 1.0); P.set(15, 2*valence+8, 1.0);
        return P;
    };
    
    const FullMatrix<double> pickmtrx3(){
        FullMatrix<double> P(16,2*valence+17);
        P.set(0, 1, 1.0); P.set(1, 0, 1.0);
        P.set(2, 5, 1.0); P.set(3, 2*valence+3, 1.0);
        P.set(4, 2, 1.0); P.set(5, 3, 1.0);
        P.set(6, 4, 1.0); P.set(7, 2*valence+2, 1.0);
        P.set(8, 2*valence+7, 1.0); P.set(9, 2*valence+6, 1.0);
        P.set(10, 2*valence+5, 1.0); P.set(11, 2*valence+1, 1.0);
        P.set(12, 2*valence+16, 1.0); P.set(13, 2*valence+15, 1.0);
        P.set(14, 2*valence+14, 1.0); P.set(15, 2*valence+13, 1.0);
        return P;
    };


};


DEAL_II_NAMESPACE_CLOSE

#endif
