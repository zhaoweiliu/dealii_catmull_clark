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
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_poly_tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_system.h>


#include "polynomials_CubicBSpline.hpp"
#include "polynomials_Catmull_Clark.hpp"

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
class CatmullClark;

template <int dim, int spacedim>
class FE_Catmull_Clark : public FiniteElement<dim, spacedim>
{
public:
//  FE_Catmull_Clark(const unsigned int valence,
//                   const unsigned int n_components = 1,
//                   const bool         dominate     = false);

//  FE_Catmull_Clark(const unsigned int                valence,
//                   const std::array<unsigned int, 4> verts_id,
//                   const unsigned int                n_components = 1,
//                   const bool                        dominate     = false);
    
    FE_Catmull_Clark(const unsigned int                valence,
                     const std::array<unsigned int, 4> verts_id,
                     std::shared_ptr<const NonLocalDoFHandler<dim, spacedim>>                    cc_object,
                     const unsigned int                n_components = 1,
                     const bool                        dominate     = false);


    
  virtual std::unique_ptr<FiniteElement<dim, spacedim>> clone() const override;

  /**
   * Return a string that uniquely identifies a finite element. In this case
   * it is <code>FE_Catmull_Clark@<dim@></code>.
   */
  virtual std::string get_name() const override;

  // for documentation, see the FiniteElement base class
  virtual UpdateFlags
  requires_update_flags(const UpdateFlags update_flags) const override;

  virtual double shape_value(const unsigned int i,
                             const Point<dim> & p) const override;

  virtual Tensor<1, dim> shape_grad(const unsigned int i,
                                    const Point<dim> & p) const override;

  virtual Tensor<2, dim> shape_grad_grad(const unsigned int i,
                                         const Point<dim> & p) const override;

  std::vector<double> shape_values(const Point<dim> &p) const;

  std::vector<Tensor<1, dim>> shape_grads(const Point<dim> &p) const;

  std::vector<Tensor<2, dim>> shape_grad_grads(const Point<dim> &p) const;

  void compute(
    const UpdateFlags            update_flags,
    const Point<dim> &           p,
    std::vector<double> &        values,
    std::vector<Tensor<1, dim>> &grads,
    std::vector<Tensor<2, dim>> &grad_grads /*, add more if required*/) const;

  virtual void fill_fe_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const CellSimilarity::Similarity                            cell_similarity,
    const Quadrature<dim> &                                     quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual void fill_fe_face_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual void fill_fe_subface_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const unsigned int                                          sub_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual std::unique_ptr<
    typename FiniteElement<dim, spacedim>::InternalDataBase>
  get_data(
    const UpdateFlags             update_flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim> &       quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual bool
  operator==(const FiniteElement<dim, spacedim> &fe) const override;

  bool is_dominating() const;

  FiniteElementDomination::Domination
  compare_for_domination(const FiniteElement<dim, spacedim> &fe,
                         const unsigned int codim) const override;

  std::vector<std::pair<unsigned int, unsigned int>> hp_vertex_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  class InternalData : public FiniteElement<dim, spacedim>::InternalDataBase
  {
  public:
    using ShapeVector = dealii::Table<2, double>;

    using GradientVector = dealii::Table<2, Tensor<1, dim>>;

    using HessianVector = dealii::Table<2, Tensor<2, dim>>;

    ShapeVector shape_values;

    GradientVector shape_derivatives;

    HessianVector shape_hessian;

  };

  virtual std::shared_ptr<const NonLocalDoFHandler<dim, spacedim>>
  get_non_local_dof_handler() const override
  {
    return non_local_dh;
  }

private:
  std::shared_ptr<const NonLocalDoFHandler<dim, spacedim>> non_local_dh;

  const unsigned int valence;

  const bool dominate;

  // maps ith dof to shape id;
  std::vector<unsigned int> shapes_id_map;

  const typename polynomials_Catmull_Clark<dim>::regular poly_reg;

  const typename polynomials_Catmull_Clark<dim>::one_end_truncated poly_one_end;

  const typename polynomials_Catmull_Clark<dim>::two_ends_truncated
    poly_two_ends;

  FullMatrix<double> compute_subd_matrix(const Point<dim> p,
                                         Point<dim> &     p_mapped,
                                         double &         Jacobian) const;

  constexpr static double mS_12[7][7] = {
    {1. / 64., 3. / 32., 1. / 64., 0., 3. / 32., 1. / 64., 0.},
    {0., 1. / 16., 1. / 16., 0., 0., 0., 0.},
    {0., 1. / 64., 3. / 32., 1. / 64., 0., 0., 0.},
    {0., 0., 1. / 16., 1. / 16., 0., 0., 0.},
    {0., 0., 0., 0., 1. / 16., 1. / 16., 0.},
    {0., 0., 0., 0., 1. / 64., 3. / 32., 1. / 64.},
    {0., 0., 0., 0., 0., 1. / 16., 1. / 16.}};
  // size of matrix S_11 is not fixed but no-zero entries are fixed

  constexpr static double mS_11[7][8] = {
    {1. / 64., 0., 0., 3. / 32., 9. / 16., 3. / 32., 0., 0.},
    {1. / 16., 0., 0., 1. / 16., 3. / 8., 3. / 8., 0., 0.},
    {3. / 32., 0., 0., 1. / 64., 3. / 32., 9. / 16., 3. / 32., 1. / 64.},
    {1. / 16., 0., 0., 0., 0., 3. / 8., 3. / 8., 1. / 16.},
    {1. / 16., 0., 0., 3. / 8., 3. / 8., 1. / 16., 0., 0.},
    {3. / 32., 1. / 64., 3. / 32., 9. / 16., 3. / 32., 1. / 64., 0., 0.},
    {1. / 16., 1. / 16., 3. / 8., 3. / 8., 0., 0., 0., 0.}};

  constexpr static double mS_21[9][7] = {
    {0., 0., 0., 0., 1. / 4., 0., 0.},
    {0., 0., 0., 0., 3. / 8., 1. / 16., 0.},
    {0., 0., 0., 0., 1. / 4., 1. / 4., 0.},
    {0., 0., 0., 0., 1. / 16., 3. / 8., 1. / 16.},
    {0., 0., 0., 0., 0., 1. / 4., 1. / 4.},
    {0., 0., 0., 1. / 16., 3. / 8., 0., 0.},
    {0., 0., 0., 1. / 4., 1. / 4., 0., 0.},
    {0., 0., 1. / 16., 3. / 8., 1. / 16., 0., 0.},
    {0., 0., 1. / 4., 1. / 4., 0., 0., 0.}};

  constexpr static double mS_22[9][7] = {
    {1. / 4., 1. / 4., 0., 0., 1. / 4., 0., 0.},
    {1. / 16., 3. / 8., 1. / 16., 0., 1. / 16., 0., 0.},
    {0., 1. / 4., 1. / 4., 0., 0., 0., 0.},
    {0., 1. / 16., 3. / 8., 1. / 16., 0., 0., 0.},
    {0., 0., 1. / 4., 1. / 4., 0., 0., 0.},
    {1. / 16., 1. / 16., 0., 0., 3. / 8., 1. / 16., 0.},
    {0., 0., 0., 0., 1. / 4., 1. / 4., 0.},
    {0., 0., 0., 0., 1. / 16., 3. / 8., 1. / 16.},
    {0., 0., 0., 0., 0., 1. / 4., 1. / 4.}};

  FullMatrix<double> S_matrix() const
  {
    FullMatrix<double> S(2 * valence + 1);
    double             a_N = 1. - (7.) / (4. * valence);
    double             b_N = 3. / (2. * valence * valence);
    double             c_N = 1. / (4. * valence * valence);
    double             d   = 3. / 8.;
    double             e   = 1. / 16.;
    double             f   = 1. / 4.;
    S.set(0, 0, a_N);
    S.set(0, 1, b_N);
    S.set(0, 2, c_N);
    S.set(1, 0, d);
    S.set(2, 0, f);
    S.set(1, 1, d);
    S.set(1, 2, e);
    S.set(1, 3, e);
    S.set(1, 2 * valence - 1, e);
    S.set(1, 2 * valence, e);
    S.set(2, 1, f);
    S.set(2, 2, f);
    S.set(2, 3, f);
    for (unsigned int iv = 1; iv < valence - 1; ++iv)
      {
        S.set(0, 2 * iv + 1, b_N);
        S.set(0, 2 * iv + 2, c_N);
        S.set(2 * iv + 1, 0, d);
        S.set(2 * iv + 2, 0, f);
        S.set(2 * iv + 1, 2 * iv + 1, d);
        S.set(2 * iv + 1, 2 * iv, e);
        S.set(2 * iv + 1, 2 * iv - 1, e);
        S.set(2 * iv + 1, 2 * iv + 2, e);
        S.set(2 * iv + 1, 2 * iv + 3, e);
        S.set(2 * iv + 2, 2 * iv + 2, f);
        S.set(2 * iv + 2, 2 * iv + 1, f);
        S.set(2 * iv + 2, 2 * iv + 3, f);
      }
    S.set(0, 2 * valence - 1, b_N);
    S.set(0, 2 * valence, c_N);
    S.set(2 * valence - 1, 0, d);
    S.set(2 * valence, 0, f);
    S.set(2 * valence - 1, 1, e);
    S.set(2 * valence, 1, f);
    S.set(2 * valence - 1, 2 * valence - 1, d);
    S.set(2 * valence - 1, 2 * valence - 3, e);
    S.set(2 * valence - 1, 2 * valence - 2, e);
    S.set(2 * valence - 1, 2 * valence, e);
    S.set(2 * valence, 2 * valence - 1, f);
    S.set(2 * valence, 2 * valence, f);
    return S;
  };

  FullMatrix<double> A_matrix() const
  {
    constexpr static double mS_12[7][7] = {
      {1. / 64., 3. / 32., 1. / 64., 0., 3. / 32., 1. / 64., 0.},
      {0., 1. / 16., 1. / 16., 0., 0., 0., 0.},
      {0., 1. / 64., 3. / 32., 1. / 64., 0., 0., 0.},
      {0., 0., 1. / 16., 1. / 16., 0., 0., 0.},
      {0., 0., 0., 0., 1. / 16., 1. / 16., 0.},
      {0., 0., 0., 0., 1. / 64., 3. / 32., 1. / 64.},
      {0., 0., 0., 0., 0., 1. / 16., 1. / 16.}};
    // size of matrix S_11 is not fixed but no-zero entries are fixed

    constexpr static double mS_11[7][8] = {
      {1. / 64., 0., 0., 3. / 32., 9. / 16., 3. / 32., 0., 0.},
      {1. / 16., 0., 0., 1. / 16., 3. / 8., 3. / 8., 0., 0.},
      {3. / 32., 0., 0., 1. / 64., 3. / 32., 9. / 16., 3. / 32., 1. / 64.},
      {1. / 16., 0., 0., 0., 0., 3. / 8., 3. / 8., 1. / 16.},
      {1. / 16., 0., 0., 3. / 8., 3. / 8., 1. / 16., 0., 0.},
      {3. / 32., 1. / 64., 3. / 32., 9. / 16., 3. / 32., 1. / 64., 0., 0.},
      {1. / 16., 1. / 16., 3. / 8., 3. / 8., 0., 0., 0., 0.}};

    FullMatrix<double> A(2 * valence + 8);
    FullMatrix<double> S = S_matrix();
    // S has size 2*val+1 x 2*val+1
    A.fill(S);
    if (valence != 3)
      {
        for (int i = 0; i < 7; ++i)
          {
            for (int j = 0; j < 8; ++j)
              {
                A.set(2 * valence + 1 + i, j, mS_11[i][j]);
              }
          }
      }
    else
      {
        for (int i = 0; i < 7; ++i)
          {
            for (int j = 0; j < 7; ++j)
              {
                A.set(2 * valence + 1 + i, j, mS_11[i][j]);
              }
            A.set(2 * valence + 3, 1, 1. / 64.);
            A.set(2 * valence + 4, 1, 1. / 16.);
          }
      }

    for (int i = 0; i < 7; ++i)
      {
        for (int j = 0; j < 7; ++j)
          {
            A.set(2 * valence + 1 + i, 2 * valence + 1 + j, mS_12[i][j]);
          }
      }
    return A;
  };

  FullMatrix<double> A_bar_matrix() const
  {
    // size of matrix S_11 is not fixed but no-zero entries are fixed

    constexpr static double mS_21[9][7] = {
      {0., 0., 0., 0., 1. / 4., 0., 0.},
      {0., 0., 0., 0., 3. / 8., 1. / 16., 0.},
      {0., 0., 0., 0., 1. / 4., 1. / 4., 0.},
      {0., 0., 0., 0., 1. / 16., 3. / 8., 1. / 16.},
      {0., 0., 0., 0., 0., 1. / 4., 1. / 4.},
      {0., 0., 0., 1. / 16., 3. / 8., 0., 0.},
      {0., 0., 0., 1. / 4., 1. / 4., 0., 0.},
      {0., 0., 1. / 16., 3. / 8., 1. / 16., 0., 0.},
      {0., 0., 1. / 4., 1. / 4., 0., 0., 0.}};

    constexpr static double mS_22[9][7] = {
      {1. / 4., 1. / 4., 0., 0., 1. / 4., 0., 0.},
      {1. / 16., 3. / 8., 1. / 16., 0., 1. / 16., 0., 0.},
      {0., 1. / 4., 1. / 4., 0., 0., 0., 0.},
      {0., 1. / 16., 3. / 8., 1. / 16., 0., 0., 0.},
      {0., 0., 1. / 4., 1. / 4., 0., 0., 0.},
      {1. / 16., 1. / 16., 0., 0., 3. / 8., 1. / 16., 0.},
      {0., 0., 0., 0., 1. / 4., 1. / 4., 0.},
      {0., 0., 0., 0., 1. / 16., 3. / 8., 1. / 16.},
      {0., 0., 0., 0., 0., 1. / 4., 1. / 4.}};

    FullMatrix<double> A_bar(2 * valence + 17, 2 * valence + 8);
    FullMatrix<double> A = A_matrix();
    A_bar.fill(A);
    for (int i = 0; i < 9; ++i)
      {
        for (int j = 0; j < 7; ++j)
          {
            A_bar.set(2 * valence + 8 + i, j, mS_21[i][j]);
          }
      }
    for (int i = 0; i < 9; ++i)
      {
        for (int j = 0; j < 7; ++j)
          {
            A_bar.set(2 * valence + 8 + i, 2 * valence + 1 + j, mS_22[i][j]);
          }
      }
    return A_bar;
  };

  FullMatrix<double> pickmtrx1() const
  {
    FullMatrix<double> P(16, 2 * valence + 17);
    if (valence == 3)
      {
        P.set(0, 1, 1.0);
      }
    else
      {
        P.set(0, 7, 1.0);
      }
    P.set(1, 6, 1.0);
    P.set(2, 2 * valence + 4, 1.0);
    P.set(3, 2 * valence + 12, 1.0);
    P.set(4, 0, 1.0);
    P.set(5, 5, 1.0);
    P.set(6, 2 * valence + 3, 1.0);
    P.set(7, 2 * valence + 11, 1.0);
    P.set(8, 3, 1.0);
    P.set(9, 4, 1.0);
    P.set(10, 2 * valence + 2, 1.0);
    P.set(11, 2 * valence + 10, 1.0);
    P.set(12, 2 * valence + 6, 1.0);
    P.set(13, 2 * valence + 5, 1.0);
    P.set(14, 2 * valence + 1, 1.0);
    P.set(15, 2 * valence + 9, 1.0);
    return P;
  };

  FullMatrix<double> pickmtrx2() const
  {
    FullMatrix<double> P(16, 2 * valence + 17);
    P.set(0, 0, 1.0);
    P.set(1, 5, 1.0);
    P.set(2, 2 * valence + 3, 1.0);
    P.set(3, 2 * valence + 11, 1.0);
    P.set(4, 3, 1.0);
    P.set(5, 4, 1.0);
    P.set(6, 2 * valence + 2, 1.0);
    P.set(7, 2 * valence + 10, 1.0);
    P.set(8, 2 * valence + 6, 1.0);
    P.set(9, 2 * valence + 5, 1.0);
    P.set(10, 2 * valence + 1, 1.0);
    P.set(11, 2 * valence + 9, 1.0);
    P.set(12, 2 * valence + 15, 1.0);
    P.set(13, 2 * valence + 14, 1.0);
    P.set(14, 2 * valence + 13, 1.0);
    P.set(15, 2 * valence + 8, 1.0);
    return P;
  };

  FullMatrix<double> pickmtrx3() const
  {
    FullMatrix<double> P(16, 2 * valence + 17);
    P.set(0, 1, 1.0);
    P.set(1, 0, 1.0);
    P.set(2, 5, 1.0);
    P.set(3, 2 * valence + 3, 1.0);
    P.set(4, 2, 1.0);
    P.set(5, 3, 1.0);
    P.set(6, 4, 1.0);
    P.set(7, 2 * valence + 2, 1.0);
    P.set(8, 2 * valence + 7, 1.0);
    P.set(9, 2 * valence + 6, 1.0);
    P.set(10, 2 * valence + 5, 1.0);
    P.set(11, 2 * valence + 1, 1.0);
    P.set(12, 2 * valence + 16, 1.0);
    P.set(13, 2 * valence + 15, 1.0);
    P.set(14, 2 * valence + 14, 1.0);
    P.set(15, 2 * valence + 13, 1.0);
    return P;
  };
};

DEAL_II_NAMESPACE_CLOSE

#endif
