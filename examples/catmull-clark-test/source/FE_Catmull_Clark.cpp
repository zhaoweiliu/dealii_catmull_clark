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


//template <int dim, int spacedim>
//FE_Catmull_Clark<dim,spacedim>::FE_Catmull_Clark(const unsigned int val, const unsigned int n_components, const bool dominate)
//: FiniteElement<dim,spacedim> (
//    FiniteElementData<dim>({1,0,0,(val == 1? 5:2*val+4)},
//                            n_components,
//                            3,
//                            FiniteElementData<dim>::H2),
//    std::vector<bool>(2*val+8,true),
//    std::vector<ComponentMask>(2*val+8,std::vector<bool>(1,true))),
//    valence(val),
//    dominate(dominate)
//{}



//template <int dim, int spacedim>
//FE_Catmull_Clark<dim,spacedim>::FE_Catmull_Clark(const unsigned int val, const std::array<unsigned int, 4> verts_id, const unsigned int n_components, const bool dominate)
//: FiniteElement<dim,spacedim> (
//    FiniteElementData<dim>({1,0,0,(val == 1? 5:2*val+4)},
//                            n_components,
//                            3,
//                            FiniteElementData<dim>::H2),
//    std::vector<bool>((val == 1? 9:2*val+8),true),
//    std::vector<ComponentMask>((val == 1? 9:2*val+8),std::vector<bool>(1,true))),
//    valence(val),
//    dominate(dominate)
////    non_local_dh()
//{
//    shapes_id_map.resize((valence == 1? 9:2*val+8));
//    if (val == 1){
//        /*
//               6-----7-----8
//               |     |     |
//               |     |     |
//               |     |     |
//               3-----4-----5
//               |     |     |
//               |     |     |
//               |     |     |
//               0-----1-----2     */
//
//        shapes_id_map[verts_id[0]] = 0;
//        shapes_id_map[verts_id[1]] = 3;
//        shapes_id_map[verts_id[2]] = 4;
//        shapes_id_map[verts_id[3]] = 1;
//        shapes_id_map[4] = 2;
//        shapes_id_map[5] = 5;
//        shapes_id_map[6] = 8;
//        shapes_id_map[7] = 7;
//        shapes_id_map[8] = 6;
//
//        /* -> u j_shape
//          |    0-----1-----2
//       v \/    |     |     |
//               |     |     |
//               |     |     |
//               3-----4-----5
//               |     |     |
//               |     |     |
//               |     |     |
//               6-----7-----8     */
//
//        /*
//        indices mapping for non-local dofs
//        0(4)-----1(5)----2(6)
//        |        |        |
//        |        |        |
//        |        |        |
//        ?--------?-------3(7)
//        |        |        |
//        |        |        |
//        |        |        |
//        ?--------?-------4(8)     */
//
//
//    }
//    else if (val == 2){
//        /*
//         8-----9----10----11
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         4-----5-----6-----7
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         0-----1-----2-----3     */
//
//        shapes_id_map[verts_id[0]] = 2;
//        shapes_id_map[verts_id[1]] = 1;
//        shapes_id_map[verts_id[2]] = 5;
//        shapes_id_map[verts_id[3]] = 6;
//        shapes_id_map[4] = 10;
//        shapes_id_map[5] = 9;
//        shapes_id_map[6] = 8;
//        shapes_id_map[7] = 4;
//        shapes_id_map[8] = 0;
//        shapes_id_map[9] = 3;
//        shapes_id_map[10] = 7;
//        shapes_id_map[11] = 11;
//
//        /* -> u
//         0-----1-----2-----3 v|
//         |     |     |     |  \/
//         |     |     |     |
//         |     |     |     |
//         4-----5-----6-----7
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         8-----9-----10----11     */
//
//        /*
//        7(11)----0(4)-----1(5)------2(6)
//        |        |         |        |
//        |        |         |        |
//        |        |         |        |
//        6(10)----?---------?--------3(7)
//        |        |         |        |
//        |        |         |        |
//        |        |         |        |
//        5(9)-----?---------?--------4(8)     */
//    }
//    else if (val == 4)
//    {
//        /*
//         12----13----14----15
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         8-----9-----10----11
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         4-----5-----6-----7
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         0-----1-----2-----3
//         */
//
//        shapes_id_map[0] = 5;
//        shapes_id_map[1] = 6;
//        shapes_id_map[2] = 9;
//        shapes_id_map[3] = 10;
//        shapes_id_map[4] = 1;
//        shapes_id_map[5] = 2;
//        shapes_id_map[6] = 3;
//        shapes_id_map[7] = 7;
//        shapes_id_map[8] = 11;
//        shapes_id_map[9] = 15;
//        shapes_id_map[10] = 14;
//        shapes_id_map[11] = 13;
//        shapes_id_map[12] = 12;
//        shapes_id_map[13] = 8;
//        shapes_id_map[14] = 4;
//        shapes_id_map[15] = 0;
//
//        /*
//         0-----1-----2-----3
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         4-----5-----6-----7
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         8-----9-----10----11
//         |     |     |     |
//         |     |     |     |
//         |     |     |     |
//         12---13-----14----15
//         */
//
//        /*
//        indices mapping for non-local dofs
//         11(15)---0(4)-----1(5)-----2(6)
//         |        |        |        |
//         |        |        |        |
//         |        |        |        |
//         10(14)--(0)------(1)-------3(7)
//         |        |        |        |
//         |        |        |        |
//         |        |        |        |
//         9(13)---(2)------(3)-------4(8)
//         |        |        |        |
//         |        |        |        |
//         |        |        |        |
//         8(12)----7(11)----6(10)----5(9)
//         */
//
//    }else
//    {
//        // object_index is the index of vertex;
//        shapes_id_map[verts_id[0]] = 0;
//        shapes_id_map[verts_id[1]] = 5;
//        shapes_id_map[verts_id[2]] = 4;
//        shapes_id_map[verts_id[3]] = 3;
//        shapes_id_map[4] = 1;
//        shapes_id_map[5] = 2;
//        for (unsigned int i = 6; i < 2*valence+8; ++i) {
//            shapes_id_map[i] = i;
//        }
//        /*                   2v
//                           /  |
//                          /   |
//                         /    |
//         2v+7----2------1     *------8
//         |       |      |    /      /
//         |       |      |  ..     /
//         |       |      |/      /
//         2v+6----3------0------7
//         |       |      |      |
//         |       |      |      |
//         |       |      |      |
//         2v+5----4------5------6
//         |       |      |      |
//         |       |      |      |
//         |       |      |      |
//         2v+1---2v+2----2v+3---2v+4         */
//
//        /*                   2v-4
//                           /  |
//                          /   |
//                         /    |
//         2v+3-----1-----0     *------4
//         |        |     |    /     /
//         |        |     |  ..    /
//         |        |     |/     /
//         2v+2-----?-----?-----3
//         |        |     |     |
//         |        |     |     |
//         |        |     |     |
//         2v+1-----?-----?-----2
//         |        |     |     |
//         |        |     |     |
//         |        |     |     |
//         2v-3---2v-2---2v-1---2v         */
//
//    }
//}



template <int dim, int spacedim>
FE_Catmull_Clark<dim,spacedim>::FE_Catmull_Clark(const unsigned int val, const std::array<unsigned int, 4> verts_id, std::shared_ptr<const NonLocalDoFHandler<dim, spacedim>>  cc_object, const unsigned int n_components, const bool dominate)
: FiniteElement<dim,spacedim> (
    FiniteElementData<dim>({1,0,0,(val == 1? 5:2*val+4)},
                            n_components,
                            3,
                            FiniteElementData<dim>::H2),
    std::vector<bool>((val == 1? 9:2*val+8),true),
    std::vector<ComponentMask>((val == 1? 9:2*val+8),std::vector<bool>(1,true))),
    valence(val),
    dominate(dominate)
{
    non_local_dh = cc_object;
    shapes_id_map.resize((valence == 1? 9:2*val+8));
    if (val == 1){
        /*
               6-----7-----8
               |     |     |
               |     |     |
               |     |     |
               3-----4-----5
               |     |     |
               |     |     |
               |     |     |
               0-----1-----2     */
        
        shapes_id_map[verts_id[0]] = 0;
        shapes_id_map[verts_id[1]] = 3;
        shapes_id_map[verts_id[2]] = 4;
        shapes_id_map[verts_id[3]] = 1;
        shapes_id_map[4] = 2;
        shapes_id_map[5] = 5;
        shapes_id_map[6] = 8;
        shapes_id_map[7] = 7;
        shapes_id_map[8] = 6;
        
        /* -> u j_shape
          |    0-----1-----2
       v \/    |     |     |
               |     |     |
               |     |     |
               3-----4-----5
               |     |     |
               |     |     |
               |     |     |
               6-----7-----8     */
        
        /*
        indices mapping for non-local dofs
        0(4)-----1(5)----2(6)
        |        |        |
        |        |        |
        |        |        |
        ?--------?-------3(7)
        |        |        |
        |        |        |
        |        |        |
        ?--------?-------4(8)     */
       
        
    }
    else if (val == 2){
        /*
         8-----9----10----11
         |     |     |     |
         |     |     |     |
         |     |     |     |
         4-----5-----6-----7
         |     |     |     |
         |     |     |     |
         |     |     |     |
         0-----1-----2-----3     */
        
        shapes_id_map[verts_id[0]] = 2;
        shapes_id_map[verts_id[1]] = 1;
        shapes_id_map[verts_id[2]] = 5;
        shapes_id_map[verts_id[3]] = 6;
        shapes_id_map[4] = 10;
        shapes_id_map[5] = 9;
        shapes_id_map[6] = 8;
        shapes_id_map[7] = 4;
        shapes_id_map[8] = 0;
        shapes_id_map[9] = 3;
        shapes_id_map[10] = 7;
        shapes_id_map[11] = 11;
        
        /* -> u
         0-----1-----2-----3 v|
         |     |     |     |  \/
         |     |     |     |
         |     |     |     |
         4-----5-----6-----7
         |     |     |     |
         |     |     |     |
         |     |     |     |
         8-----9-----10----11     */
        
        /*
        7(11)----0(4)-----1(5)------2(6)
        |        |         |        |
        |        |         |        |
        |        |         |        |
        6(10)----?---------?--------3(7)
        |        |         |        |
        |        |         |        |
        |        |         |        |
        5(9)-----?---------?--------4(8)     */
    }
    else if (val == 4)
    {
        /*
         12----13----14----15
         |     |     |     |
         |     |     |     |
         |     |     |     |
         8-----9-----10----11
         |     |     |     |
         |     |     |     |
         |     |     |     |
         4-----5-----6-----7
         |     |     |     |
         |     |     |     |
         |     |     |     |
         0-----1-----2-----3
         */
        
        shapes_id_map[0] = 5;
        shapes_id_map[1] = 6;
        shapes_id_map[2] = 9;
        shapes_id_map[3] = 10;
        shapes_id_map[4] = 1;
        shapes_id_map[5] = 2;
        shapes_id_map[6] = 3;
        shapes_id_map[7] = 7;
        shapes_id_map[8] = 11;
        shapes_id_map[9] = 15;
        shapes_id_map[10] = 14;
        shapes_id_map[11] = 13;
        shapes_id_map[12] = 12;
        shapes_id_map[13] = 8;
        shapes_id_map[14] = 4;
        shapes_id_map[15] = 0;
        
        /*
         0-----1-----2-----3
         |     |     |     |
         |     |     |     |
         |     |     |     |
         4-----5-----6-----7
         |     |     |     |
         |     |     |     |
         |     |     |     |
         8-----9-----10----11
         |     |     |     |
         |     |     |     |
         |     |     |     |
         12---13-----14----15
         */
        
        /*
        indices mapping for non-local dofs
         11(15)---0(4)-----1(5)-----2(6)
         |        |        |        |
         |        |        |        |
         |        |        |        |
         10(14)--(0)------(1)-------3(7)
         |        |        |        |
         |        |        |        |
         |        |        |        |
         9(13)---(2)------(3)-------4(8)
         |        |        |        |
         |        |        |        |
         |        |        |        |
         8(12)----7(11)----6(10)----5(9)
         */
        
    }else
    {
        // object_index is the index of vertex;
        shapes_id_map[verts_id[0]] = 0;
        shapes_id_map[verts_id[1]] = 5;
        shapes_id_map[verts_id[2]] = 4;
        shapes_id_map[verts_id[3]] = 3;
        shapes_id_map[4] = 1;
        shapes_id_map[5] = 2;
        for (unsigned int i = 6; i < 2*valence+8; ++i) {
            shapes_id_map[i] = i;
        }
        /*                   2v
                           /  |
                          /   |
                         /    |
         2v+7----2------1     *------8
         |       |      |    /      /
         |       |      |  ..     /
         |       |      |/      /
         2v+6----3------0------7
         |       |      |      |
         |       |      |      |
         |       |      |      |
         2v+5----4------5------6
         |       |      |      |
         |       |      |      |
         |       |      |      |
         2v+1---2v+2----2v+3---2v+4         */
        
        /*                   2v-4
                           /  |
                          /   |
                         /    |
         2v+3-----1-----0     *------4
         |        |     |    /     /
         |        |     |  ..    /
         |        |     |/     /
         2v+2-----?-----?-----3
         |        |     |     |
         |        |     |     |
         |        |     |     |
         2v+1-----?-----?-----2
         |        |     |     |
         |        |     |     |
         |        |     |     |
         2v-3---2v-2---2v-1---2v         */

    }
}



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
    unsigned int j = shapes_id_map[i];
    if (valence == 4){
        // i in [0,15];
        return poly_reg.value(j,p);
    }else if(valence == 2){
        // i in [0,11];
        return poly_one_end.value(j,p);
    }else if (valence == 1){
        // i in [0,8];
        return poly_two_ends.value(j,p);
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
    unsigned int j = shapes_id_map[i];
    if (valence == 4){
        // i in [0,15];
        return poly_reg.grads(j,p);
    }else if(valence == 2){
        // i in [0,11];
        return poly_one_end.grads(j,p);
    }else if (valence == 1){
        // i in [0,8];
        return poly_two_ends.grads(j,p);
    }else{
        // i in [0, 2*valence + 7];
        std::cout << "\n warning: inefficiently compute shape functions in irregular patch.\n";
        return this->shape_grads(p)[i];
    }
}



template<int dim, int spacedim>
Tensor<2,dim> FE_Catmull_Clark<dim, spacedim>::shape_grad_grad (const unsigned int i, const Point< dim > &p) const
{
    unsigned int j = shapes_id_map[i];
    if (valence == 4){
        // i in [0,15];
        return poly_reg.grad_grads(j,p);
    }else if(valence == 2){
        // i in [0,11];
        return poly_one_end.grad_grads(j,p);
    }else if (valence == 1){
        // i in [0,8];
        return poly_two_ends.grad_grads(j,p);
    }else{
        // i in [0, 2*valence + 7];
        std::cout << "\n warning: inefficiently compute shape functions in irregular patch.\n";
        return this->shape_grad_grads(p)[i];
    }
}



template<int dim, int spacedim>
std::vector<double> FE_Catmull_Clark<dim, spacedim>::shape_values (const Point< dim > &p) const
{
    if (valence == 1) {
        std::vector<double> shape_vectors(9);
        for (unsigned int i = 0; i < 9; ++i) {
            unsigned int j = shapes_id_map[i];
            shape_vectors[i] = poly_two_ends.value(j,p);
        }
        return shape_vectors;
    }else{
        std::vector<double> shape_vectors(2*valence + 8);
        if (valence == 4){
            for (unsigned int i = 0; i < 16; ++i)
            {
                unsigned int j = shapes_id_map[i];
                shape_vectors[i] = poly_reg.value(j,p);
            }
        }
        else if(valence == 2){
            for (unsigned int i = 0; i < 12; ++i)
            {
                unsigned int j = shapes_id_map[i];
                shape_vectors[i] = poly_one_end.value(j,p);
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
                unsigned int j = shapes_id_map[i];
                shape_vectors[i] = shape_vectors_result[j];
            }
        }
        return shape_vectors;
    }
}



template<int dim, int spacedim>
std::vector<Tensor<1, dim>> FE_Catmull_Clark<dim, spacedim>::shape_grads (const Point< dim > &p) const
{
    std::vector<Tensor<1, dim>> shape_grad_vectors;
    if (valence == 1) {
        shape_grad_vectors.resize(9);
        for (unsigned int i = 0; i < 9; ++i) {
            unsigned int j = shapes_id_map[i];
            shape_grad_vectors[i] = poly_two_ends.grads(j,p);
        }
        return shape_grad_vectors;
    }else{
        shape_grad_vectors.resize(2*valence + 8);
        if (valence == 4){
            for (unsigned int i = 0; i < 16; ++i)
            {
                unsigned int j = shapes_id_map[i];
                shape_grad_vectors[i] = poly_reg.grads(j,p);
            }
        }
        else if(valence == 2){
            for (unsigned int i = 0; i < 12; ++i)
            {
                unsigned int j = shapes_id_map[i];
                shape_grad_vectors[i] = poly_one_end.grads(j,p);
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
                unsigned int j = shapes_id_map[i];
                shape_grad_vectors[i][0] = grad1[j] * jac;
                shape_grad_vectors[i][1] = grad2[j] * jac;
            }
        }
        return shape_grad_vectors;
    }
}



template<int dim, int spacedim>
std::vector<Tensor<2, dim>> FE_Catmull_Clark<dim, spacedim>::shape_grad_grads (const Point< dim > &p) const
{
    if (valence == 1) {
        std::vector<Tensor<2, dim>> shape_grad_grad_vectors(9);
        for (unsigned int i = 0; i < 9; ++i) {
            unsigned int j = shapes_id_map[i];
            shape_grad_grad_vectors[i] = poly_two_ends.grad_grads(j,p);
        }
        return shape_grad_grad_vectors;
    }else{
        std::vector<Tensor<2, dim>> shape_grad_grad_vectors(2*valence + 8);
        if (valence == 4){
            for (unsigned int i = 0; i < 16; ++i)
            {
                unsigned int j = shapes_id_map[i];
                shape_grad_grad_vectors[i] = poly_reg.grad_grads(j,p);
            }
        }
        else if(valence == 2){
            for (unsigned int i = 0; i < 12; ++i)
            {
                unsigned int j = shapes_id_map[i];
                shape_grad_grad_vectors[i] = poly_one_end.grad_grads(j,p);
            }
        }
        else {
            Vector<double> grad11_reg(16);
            Vector<double> grad12_reg(16);
            Vector<double> grad21_reg(16);
            Vector<double> grad22_reg(16);
            
            Vector<double> grad11(2*valence+8);
            Vector<double> grad12(2*valence+8);
            Vector<double> grad21(2*valence+8);
            Vector<double> grad22(2*valence+8);

            Point<dim> p_mapped;
            double jac;
            FullMatrix<double> Subd_matrix = compute_subd_matrix(p, p_mapped, jac);
            for (unsigned int i = 0; i < 16; ++i)
            {
                grad11_reg[i] = poly_reg.grad_grads(i,p_mapped)[0][0];
                grad12_reg[i] = poly_reg.grad_grads(i,p_mapped)[0][1];
                grad21_reg[i] = poly_reg.grad_grads(i,p_mapped)[1][0];
                grad22_reg[i] = poly_reg.grad_grads(i,p_mapped)[1][1];
            }
            Subd_matrix.Tvmult(grad11,grad11_reg);
            Subd_matrix.Tvmult(grad12,grad12_reg);
            Subd_matrix.Tvmult(grad21,grad21_reg);
            Subd_matrix.Tvmult(grad22,grad22_reg);
            for (unsigned int i = 0; i < 2*valence+8; ++i) {
                unsigned int j = shapes_id_map[i];
                shape_grad_grad_vectors[i][0][0] = grad11[j] * jac * jac;
                shape_grad_grad_vectors[i][0][1] = grad12[j] * jac * jac;
                shape_grad_grad_vectors[i][1][0] = grad21[j] * jac * jac;
                shape_grad_grad_vectors[i][1][1] = grad22[j] * jac * jac;
            }
        }
        return shape_grad_grad_vectors;
    }
}



template<int dim, int spacedim>
void FE_Catmull_Clark<dim, spacedim>::compute(const UpdateFlags update_flags, const Point< dim > &p, std::vector<double> &values,  std::vector<Tensor<1,dim>> &grads,std::vector<Tensor<2,dim>> &grad_grads /*, add more if required*/) const{
    if (update_flags & update_values){
        if (valence == 1) {
            values.resize(9);
            for (unsigned int i = 0; i < 9; ++i) {
                unsigned int j = shapes_id_map[i];
                values[i] = poly_two_ends.value(j,p);
            }
        }else{
            values.resize(2*valence + 8);
            if (valence == 4){
                for (unsigned int i = 0; i < 16; ++i)
                {
                    unsigned int j = shapes_id_map[i];
                    values[i] = poly_reg.value(j,p);
                }
            }
            else if(valence == 2){
                for (unsigned int i = 0; i < 12; ++i)
                {
                    unsigned int j = shapes_id_map[i];
                    values[i] = poly_one_end.value(j,p);
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
                    unsigned int j = shapes_id_map[i];
                    values[i] = shape_vectors_result[j];
                }
            }
        }
    }
    if (update_flags & update_gradients){
        if (valence == 1) {
            grads.resize(9);
            for (unsigned int i = 0; i < 9; ++i) {
                unsigned int j = shapes_id_map[i];
                grads[i] = poly_two_ends.grads(j,p);
            }
        }else{
            grads.resize(2*valence + 8);
            if (valence == 4){
                for (unsigned int i = 0; i < 16; ++i)
                {
                    unsigned int j = shapes_id_map[i];
                    grads[i] = poly_reg.grads(j,p);
                }
            }
            else if(valence == 2){
                for (unsigned int i = 0; i < 12; ++i)
                {
                    unsigned int j = shapes_id_map[i];
                    grads[i] = poly_one_end.grads(j,p);
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
                    unsigned int j = shapes_id_map[i];
                    grads[i][0] = grad1[j] * jac;
                    grads[i][1] = grad2[j] * jac;
                }
            }
        }
    }
    if (update_flags & update_hessians){
        if (valence == 1) {
            grad_grads.resize(9);
            for (unsigned int i = 0; i < 9; ++i) {
                unsigned int j = shapes_id_map[i];
                grad_grads[i] = poly_two_ends.grad_grads(j,p);
            }
        }else{
            grad_grads.resize(2*valence + 8);
            if (valence == 4){
                for (unsigned int i = 0; i < 16; ++i)
                {
                    unsigned int j = shapes_id_map[i];
                    grad_grads[i] = poly_reg.grad_grads(j,p);
                }
            }
            else if(valence == 2){
                for (unsigned int i = 0; i < 12; ++i)
                {
                    unsigned int j = shapes_id_map[i];
                    grad_grads[i] = poly_one_end.grad_grads(j,p);
                }
            }
            else {
                Vector<double> grad_grads11_reg(16);
                Vector<double> grad_grads22_reg(16);
                Vector<double> grad_grads12_reg(16);

                Vector<double> grad_grads11(2*valence+8);
                Vector<double> grad_grads22(2*valence+8);
                Vector<double> grad_grads12(2*valence+8);
                Point<dim> p_mapped;
                double jac;
                FullMatrix<double> Subd_matrix = compute_subd_matrix(p, p_mapped, jac);
                for (unsigned int i = 0; i < 16; ++i)
                {
                    grad_grads11_reg[i] = poly_reg.grad_grads(i,p_mapped)[0][0];
                    grad_grads22_reg[i] = poly_reg.grad_grads(i,p_mapped)[1][1];
                    grad_grads12_reg[i] =poly_reg.grad_grads(i,p_mapped)[0][1];
                }
                Subd_matrix.Tvmult(grad_grads11,grad_grads11_reg);
                Subd_matrix.Tvmult(grad_grads22,grad_grads22_reg);
                Subd_matrix.Tvmult(grad_grads12,grad_grads12_reg);

                for (unsigned int i = 0; i < 2*valence+8; ++i) {
                    unsigned int j = shapes_id_map[i];
                    grad_grads[i][0][0] = grad_grads11[j] * jac * jac;
                    grad_grads[i][1][1] = grad_grads22[j] * jac * jac;
                    grad_grads[i][0][1] = grad_grads12[j] * jac * jac;
                    grad_grads[i][1][0] = grad_grads12[j] * jac * jac;
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
    
    if (data.update_each & update_hessians)
    data.shape_hessian.reinit(this->dofs_per_cell, n_q_points);

    for (unsigned int iq = 0; iq < n_q_points; ++iq) {
        Point<dim> p = qpts[iq];
        std::vector<double> values;
        std::vector<Tensor<1,dim>> derivatives;
        std::vector<Tensor<2,dim>> second_derivatives;
        this->compute(update_flags, p, values, derivatives,second_derivatives);
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
        if (update_flags & update_hessians){
            for (unsigned int k = 0; k < this->dofs_per_cell; ++k){
                data.shape_hessian[k][iq] = second_derivatives[k];
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
    
    Assert(!(flags & update_gradients) || fe_data.shape_derivatives.n_rows() == this->dofs_per_cell, ExcDimensionMismatch(fe_data.shape_derivatives.n_rows(), this->dofs_per_cell));
    Assert(!(flags & update_gradients) || fe_data.shape_derivatives.n_cols() == n_q_points, ExcDimensionMismatch(fe_data.shape_derivatives.n_cols(), n_q_points));
    
    if (flags & update_values){
        output_data.shape_values = fe_data.shape_values;
    }
    if (flags & update_gradients){
        for (unsigned int q_point = 0; q_point< n_q_points ; ++q_point) {
            for (unsigned int idof = 0; idof < this->dofs_per_cell; ++idof) {
                for (unsigned int i = 0; i< spacedim ; ++i) {
                    output_data.shape_gradients[idof][q_point][i] = 0;
                    for (unsigned int j = 0; j< dim ; ++j) {
                        output_data.shape_gradients[idof][q_point][i] += fe_data.shape_derivatives[idof][q_point][j] * mapping_data.inverse_jacobians[q_point][j][i];
                    }
                }
            }
        }
    }
    if (flags & update_hessians){
        for (unsigned int q_point = 0; q_point< n_q_points ; ++q_point) {
            for (unsigned int idof = 0; idof < this->dofs_per_cell; ++idof) {
                for (unsigned int i = 0; i< spacedim ; ++i) {
                    for (unsigned int j = 0; j< spacedim ; ++j) {
                        output_data.shape_hessians[idof][q_point][i][j] = 0;
                        output_data.shape_hessians[idof][q_point][i][j] = 0;
                        for (unsigned int k = 0; k< dim ; ++k) {
                            for (unsigned int l = 0; l < dim; ++l) {
                                output_data.shape_hessians[idof][q_point][i][j] += fe_data.shape_hessian[idof][q_point][k][l] * mapping_data.inverse_jacobians[q_point][l][j] * mapping_data.inverse_jacobians[q_point][k][i];
                            }
                            for (unsigned int n = 0; n < spacedim; ++n) {
                                output_data.shape_hessians[idof][q_point][i][j] -= fe_data.shape_derivatives[idof][q_point][k] * mapping_data.jacobian_pushed_forward_grads[q_point][n][i][j] * mapping_data.inverse_jacobians[q_point][k][n];
                            }
                        }
                    }
                }
            }
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
//    if(codim == 0){
//        if (this->n_dofs_per_cell()> fe.n_dofs_per_cell()){
//            return FiniteElementDomination::this_element_dominates;
//        }
//        else{
//            return FiniteElementDomination::no_requirements;
//        }
//    }
    return FiniteElementDomination::no_requirements;
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
