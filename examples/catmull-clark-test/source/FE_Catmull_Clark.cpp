//
//  FE_Catmull_Clark.cpp
//  step-4
//
//  Created by zhaowei Liu on 05/11/2019.
//

#include "FE_Catmull_Clark.hpp"

#include <deal.II/base/std_cxx14/memory.h>

#include "polynomials_Catmull_Clark.hpp"


DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim>
FE_Catmull_Clark<dim,spacedim>::FE_Catmull_Clark(const unsigned int val, const unsigned int n_components, const bool dominate)
: FiniteElement<dim,spacedim> (
    FiniteElementData<dim>({1,0,0,(val == 1? 5:2*val+4)},
                            n_components,
                            3,
                            FiniteElementData<dim>::H2),
    std::vector<bool>(2*val+8,true),
    std::vector<ComponentMask>(2*val+8,std::vector<bool>(1,true))),
    valence(val),
    dominate(dominate)
{}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_Catmull_Clark<dim, spacedim>::clone() const
{
  return std_cxx14::make_unique<FE_Catmull_Clark<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_Catmull_Clark<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_Catmull_Clark<" << dim << ">(";
  if (this->n_components() > 1)
    {
      namebuf << this->n_components();
      if (dominate)
        namebuf << ", dominating";
    }
  else if (dominate)
    namebuf << "dominating";
  namebuf << ")";
  return namebuf.str();
}



template <int dim, int spacedim>
UpdateFlags
FE_Catmull_Clark<dim, spacedim>::requires_update_flags(const UpdateFlags flags) const
{
    /* Require implementation later*/
  return flags;
}



template<int dim, int spacedim>
double FE_Catmull_Clark<dim, spacedim>::shape_value (const unsigned int i, const Point< dim > &p) const
{
    if (valence == 4){
        // i in [0,15];
        return poly_reg.value(i,p);
    }else if(valence == 2){
        // i in [0,11];
        return poly_one_end.value(i,p);
    }else if (valence == 1){
        // i in [0,8];
        return poly_two_ends.value(i,p);
    }else{
        // i in [0, 2*valence + 7];
//        throw std::runtime_error("please use FE_Catmull_Clark<dim, spacedim>::shape_values instead.");
        std::cout << "\n warning: inefficiently compute shape functions in irregular patch.\n";
        return this->shape_values(p)[i];
    }
}



template<int dim, int spacedim>
Tensor<1,dim> FE_Catmull_Clark<dim, spacedim>::shape_grad (const unsigned int i, const Point< dim > &p) const
{
    if (valence == 4){
        // i in [0,15];
        return poly_reg.grads(i,p);
    }else if(valence == 2){
        // i in [0,11];
        return poly_one_end.grads(i,p);
    }else if (valence == 1){
        // i in [0,8];
        return poly_two_ends.grads(i,p);
    }else{
        // i in [0, 2*valence + 7];
        std::cout << "\n warning: inefficiently compute shape functions in irregular patch.\n";
        return this->shape_grads(p)[i];
    }
}



template<int dim, int spacedim>
std::vector<double> FE_Catmull_Clark<dim, spacedim>::shape_values (const Point< dim > &p) const
{
    if (valence == 1) {
        std::vector<double> shape_vectors(9);
        for (unsigned int i = 0; i < 9; ++i) {
            shape_vectors[i] = poly_two_ends.value(i,p);
        }
        return shape_vectors;
    }else{
        std::vector<double> shape_vectors(2*valence + 8);
        if (valence == 4){
            for (unsigned int i = 0; i < 16; ++i)
            {
                shape_vectors[i] = poly_reg.value(i,p);
            }
        }
        else if(valence == 2){
            for (unsigned int i = 0; i < 12; ++i)
            {
                shape_vectors[i] = poly_one_end.value(i,p);
            }
        }
        else {
            Vector<double> shape_vectors_reg(16);
            Vector<double> shape_vectors_result(2*valence+8);
            Point<dim> p_mapped;
            double jac;
            FullMatrix<double> Subd_matrix = compute_subd_matrix(p, p_mapped, jac);
            for (unsigned int i = 0; i < 16; ++i)
            {
                shape_vectors_reg[i] = poly_reg.value(i,p_mapped);
            }
            Subd_matrix.Tvmult(shape_vectors_result,shape_vectors_reg);
            for (unsigned int i = 0; i < 2*valence+8; ++i) {
                shape_vectors[i] = shape_vectors_result[i];
            }
        }
        return shape_vectors;
    }
}



template<int dim, int spacedim>
std::vector<Tensor<1, dim>> FE_Catmull_Clark<dim, spacedim>::shape_grads (const Point< dim > &p) const
{
    if (valence == 1) {
        std::vector<Tensor<1, dim>> shape_grad_vectors(9);
        for (unsigned int i = 0; i < 9; ++i) {
            shape_grad_vectors[i] = poly_two_ends.grads(i,p);
        }
        return shape_grad_vectors;
    }else{
        std::vector<Tensor<1, dim>> shape_grad_vectors(2*valence + 8);
        if (valence == 4){
            for (unsigned int i = 0; i < 16; ++i)
            {
                shape_grad_vectors[i] = poly_reg.grads(i,p);
            }
        }
        else if(valence == 2){
            for (unsigned int i = 0; i < 12; ++i)
            {
                shape_grad_vectors[i] = poly_one_end.grads(i,p);
            }
        }
        else {
            Vector<double> grad1_reg(16);
            Vector<double> grad2_reg(16);
            
            Vector<double> grad1(2*valence+8);
            Vector<double> grad2(2*valence+8);

            Point<dim> p_mapped;
            double jac;
            FullMatrix<double> Subd_matrix = compute_subd_matrix(p, p_mapped, jac);
            for (unsigned int i = 0; i < 16; ++i)
            {
                grad1_reg[i] = poly_reg.grads(i,p_mapped)[0];
                grad2_reg[i] = poly_reg.grads(i,p_mapped)[1];
            }
            Subd_matrix.Tvmult(grad1,grad1_reg);
            Subd_matrix.Tvmult(grad2,grad2_reg);
            for (unsigned int i = 0; i < 2*valence+8; ++i) {
                shape_grad_vectors[i][0] = grad1[i] * jac;
                shape_grad_vectors[i][1] = grad2[i] * jac;
            }
        }
        return shape_grad_vectors;
    }
}



template<int dim, int spacedim>
void FE_Catmull_Clark<dim, spacedim>::compute(const UpdateFlags update_flags, const Point< dim > &p, std::vector<double> &values,  std::vector<Tensor<1,dim>> &grads /*, add more if required*/) const{
    if (update_flags & update_values){
        if (valence == 1) {
            values.resize(9);
            for (unsigned int i = 0; i < 9; ++i) {
                values[i] = poly_two_ends.value(i,p);
            }
        }else{
            values.resize(2*valence + 8);
            if (valence == 4){
                for (unsigned int i = 0; i < 16; ++i)
                {
                    values[i] = poly_reg.value(i,p);
                }
            }
            else if(valence == 2){
                for (unsigned int i = 0; i < 12; ++i)
                {
                    values[i] = poly_one_end.value(i,p);
                }
            }
            else {
                Vector<double> shape_vectors_reg(16);
                Vector<double> shape_vectors_result(2*valence+8);
                
                Point<dim> p_mapped;
                double jac;
                FullMatrix<double> Subd_matrix = compute_subd_matrix(p, p_mapped, jac);
                for (unsigned int i = 0; i < 16; ++i)
                {
                    shape_vectors_reg[i] = poly_reg.value(i,p_mapped);
                }
                Subd_matrix.Tvmult(shape_vectors_result,shape_vectors_reg);
                for (unsigned int i = 0; i < 2*valence+8; ++i) {
                    values[i] = shape_vectors_result[i];
                }
            }
        }
    }
    if (update_flags & update_gradients){
        if (valence == 1) {
            grads.resize(9);
            for (unsigned int i = 0; i < 9; ++i) {
                grads[i] = poly_two_ends.grads(i,p);
            }
        }else{
            grads.resize(2*valence + 8);
            if (valence == 4){
                for (unsigned int i = 0; i < 16; ++i)
                {
                    grads[i] = poly_reg.grads(i,p);
                }
            }
            else if(valence == 2){
                for (unsigned int i = 0; i < 12; ++i)
                {
                    grads[i] = poly_one_end.grads(i,p);
                }
            }
            else {
                Vector<double> grad1_reg(16);
                Vector<double> grad2_reg(16);
                
                Vector<double> grad1(2*valence+8);
                Vector<double> grad2(2*valence+8);
                Point<dim> p_mapped;
                double jac;
                FullMatrix<double> Subd_matrix = compute_subd_matrix(p, p_mapped, jac);
                for (unsigned int i = 0; i < 16; ++i)
                {
                    grad1_reg[i] = poly_reg.grads(i,p_mapped)[0];
                    grad2_reg[i] = poly_reg.grads(i,p_mapped)[1];
                }
                Subd_matrix.Tvmult(grad1,grad1_reg);
                Subd_matrix.Tvmult(grad2,grad2_reg);
                for (unsigned int i = 0; i < 2*valence+8; ++i) {
                    grads[i][0] = grad1[i] * jac;
                    grads[i][1] = grad2[i] * jac;
                }
            }
        }
    }
}



template<int dim, int spacedim>
FullMatrix<double> FE_Catmull_Clark<dim, spacedim>::compute_subd_matrix(const Point<dim> p, Point<dim> &p_mapped, double &Jacobian) const {
    double u = p[0], v = p[1];
    double eps = 10e-10;
    if (u < eps && v < eps){
        u += eps;
        v += eps;
    }
    int n = int(std::floor(std::min(-std::log2(u), -std::log2(v))+1));
    double pow2 = pow(2.,n-1.);
    int k = -1;
    u *= pow2;
    v *= pow2;
    if (v < 0.5) {
        k = 0; u = 2. * u - 1.; v = 2. * v;
    }else if(u < 0.5){
        k = 2; u = 2. * u; v = 2. * v - 1.;
    }else{
        k = 1; u = 2. * u - 1; v = 2. * v - 1.;
    }
    // mapping p into the sub parametric domian
    p_mapped = {u,v};
    
    FullMatrix<double> P;
    switch (k) {
        case 0:
             P = pickmtrx1();
            break;
        case 1:
             P = pickmtrx2();
            break;
        case 2:
             P = pickmtrx3();
            break;
        default:
            throw std::runtime_error("no picking matrix returned.");
            break;
    }
    FullMatrix<double> D(16,2*valence+8);
    FullMatrix<double> A_bar = A_bar_matrix();
    FullMatrix<double> A_n = A_bar;
    for(int i = 1;i<n;++i){
        A_bar.mmult(A_n, A_matrix());
        A_bar = A_n;
    }
    P.mmult(D,A_n);
    Jacobian = pow(2,n);
    return D;
};



template <int dim, int spacedim>
std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
FE_Catmull_Clark<dim, spacedim>::
get_data(
         const UpdateFlags update_flags,
         const Mapping<dim, spacedim> & /*mapping*/,
         const Quadrature<dim> & quadrature,
         dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,spacedim>& /*output_data*/
         ) const
{
     //Create a default data object.
    std::unique_ptr<
    typename FiniteElement<dim, spacedim>::InternalDataBase>
        data_ptr   = std_cxx14::make_unique<InternalData>();
    auto &data       = dynamic_cast<InternalData &>(*data_ptr);
    data.update_each = requires_update_flags(update_flags);
    std::vector<Point<dim>> qpts = quadrature.get_points();
    const unsigned int n_q_points = quadrature.size();
    if (data.update_each & update_values| update_quadrature_points){
        data.shape_values.reinit(this->dofs_per_cell, n_q_points);
    }  
    if (data.update_each &
      (update_covariant_transformation | update_contravariant_transformation |
       update_JxW_values | update_boundary_forms | update_normal_vectors |
       update_jacobians | update_jacobian_grads | update_inverse_jacobians))
        data.shape_derivatives.reinit(this->dofs_per_cell, n_q_points);;
    
    for (unsigned int iq = 0; iq < n_q_points; ++iq) {
        Point<dim> p = qpts[iq];
        std::vector<double> values;
        std::vector<Tensor<1,dim>> derivatives;
        this->compute(update_flags, p, values, derivatives);
        if (update_flags & update_values){
            for (unsigned int k = 0; k < this->dofs_per_cell; ++k){
                data.shape_values[k][iq] = values[k];
            }
        }
        if (update_flags & update_gradients){
            for (unsigned int k = 0; k < this->dofs_per_cell; ++k){
                data.shape_derivatives[k][iq] = derivatives[k];
            }
        }
    }
    
    return data_ptr;
}



template<int dim, int spacedim>
void FE_Catmull_Clark<dim,spacedim>::fill_fe_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
               const CellSimilarity::Similarity cell_similarity,
               const Quadrature<dim> &quadrature,
               const Mapping<dim, spacedim> &mapping,
               const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
               const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,spacedim>& mapping_data,
               const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
               dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,spacedim>& output_data) const
{
    Assert(dynamic_cast<const InternalData *>(&fe_internal) != nullptr, ExcInternalError());
    const InternalData &fe_data = static_cast<const InternalData &>(fe_internal);
    const UpdateFlags  flags(fe_data.update_each);
    const unsigned int n_q_points = quadrature.size();
        
    Assert(!(flags & update_values) || fe_data.shape_values.n_rows() == this->dofs_per_cell, ExcDimensionMismatch(fe_data.shape_values.n_rows(), this->dofs_per_cell));
    Assert(!(flags & update_values) || fe_data.shape_values.n_cols() == n_q_points, ExcDimensionMismatch(fe_data.shape_values.n_cols(), n_q_points));
    
    if (flags & update_values){
        output_data.shape_values = fe_data.shape_values;
    }
    if (flags & update_gradients){
        for (unsigned int k = 0; k < this->dofs_per_cell; ++k){
            
        }
    }
}


template<int dim, int spacedim>
void FE_Catmull_Clark<dim,spacedim>::fill_fe_face_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
  const unsigned int face_no,
  const Quadrature<dim - 1> &quadrature,
  const Mapping<dim, spacedim> &mapping,
  const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim, spacedim> &mapping_data,
  const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
                                                        dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,spacedim> &output_data) const
{
    
}



template<int dim, int spacedim>
void FE_Catmull_Clark<dim,spacedim>::fill_fe_subface_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
  const unsigned int face_no,
  const unsigned int sub_no,
  const Quadrature<dim - 1> & quadrature,
  const Mapping<dim, spacedim> & mapping,
  const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,spacedim>& mapping_data,
  const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
  dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,spacedim> &output_data) const
{
    
}



template<int dim, int spacedim> bool
FE_Catmull_Clark<dim,spacedim>::operator==(const FiniteElement<dim, spacedim> &fe) const
{
    
}



template<int dim, int spacedim> bool
FE_Catmull_Clark<dim,spacedim>::is_dominating() const
{
    return dominate;
}



template <int dim, int spacedim>
 FiniteElementDomination::Domination
FE_Catmull_Clark<dim,spacedim>::compare_for_domination(const FiniteElement<dim, spacedim> &fe, const unsigned int codim) const
{
    if(codim == 0){
        if (this->n_dofs_per_cell()> fe.n_dofs_per_cell()){
            return FiniteElementDomination::this_element_dominates;
        }
        else{
            return FiniteElementDomination::no_requirements;
        }
    }
}

template <int dim, int spacedim>
 std::vector<std::pair<unsigned int, unsigned int>>
 FE_Catmull_Clark<dim,spacedim>::hp_vertex_dof_identities(
   const FiniteElement<dim, spacedim> &fe_other) const
 {

   if (dynamic_cast<const  FE_Catmull_Clark<dim,spacedim> *>(
         &fe_other) != nullptr)
     {
       return std::vector<std::pair<unsigned int, unsigned int>>(
         1, std::make_pair(0U, 0U));
     }
   else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
     {
       return std::vector<std::pair<unsigned int, unsigned int>>();
     }
   else if (fe_other.dofs_per_face == 0)
     {
       return std::vector<std::pair<unsigned int, unsigned int>>();
     }
   else
     {
       Assert(false, ExcNotImplemented());
       return std::vector<std::pair<unsigned int, unsigned int>>();
     }
 }



template class FE_Catmull_Clark<2,3>;

DEAL_II_NAMESPACE_CLOSE
