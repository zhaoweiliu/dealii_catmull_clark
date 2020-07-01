//
//  hp_mapping_fe_field.cpp
//  step-4
//
//  Created by zhaowei Liu on 22/06/2020.
//

#ifndef dealii_cc_mapping_fe_h
#define dealii_cc_mapping_fe_h


#include <deal.II/base/config.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/vector.h>

#include <deal.II/hp/dof_handler.h>

#include <array>

#include "MappingFEField_hp.hpp"


DEAL_II_NAMESPACE_OPEN

template <int dim,
          int spacedim,
          typename VectorType     = Vector<double>,
          typename DoFHandlerType = hp::DoFHandler<dim, spacedim>>
class MappingFEField_hp : public Mapping<dim, spacedim>
{
public:
 
  MappingFEField_hp(const DoFHandlerType &euler_dof_handler,
                 const VectorType &    euler_vector,
                 const unsigned int fe_id,
                 const ComponentMask & mask = ComponentMask());
    
  /**
   * Copy constructor.
   */
  MappingFEField_hp(
    const MappingFEField_hp<dim, spacedim, VectorType, DoFHandlerType> &mapping);

  /**
   * Return a pointer to a copy of the present object. The caller of this copy
   * then assumes ownership of it.
   */
  virtual std::unique_ptr<Mapping<dim, spacedim>>
  clone() const override;

  /**
   * See the documentation of Mapping::preserves_vertex_locations()
   * for the purpose of this function. The implementation in this
   * class always returns @p false.
   */
  virtual bool
  preserves_vertex_locations() const override;

  /**
   * Return the mapped vertices of a cell.
   *
   * This mapping ignores the vertices of the Triangulation it is associated to,
   * and constructs the position of the vertices according to the @p euler_vector
   * that was passed at construction time.
   */
  virtual std::array<Point<spacedim>, GeometryInfo<dim>::vertices_per_cell>
  get_vertices(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
    const override;

  /**
   * @name Mapping points between reference and real cells
   * @{
   */

  // for documentation, see the Mapping base class
  virtual Point<spacedim>
  transform_unit_to_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const Point<dim> &p) const override;

  // for documentation, see the Mapping base class
  virtual Point<dim>
  transform_real_to_unit_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const Point<spacedim> &p) const override;

  /**
   * @}
   */

  /**
   * @name Functions to transform tensors from reference to real coordinates
   * @{
   */

  // for documentation, see the Mapping base class
  virtual void
  transform(const ArrayView<const Tensor<1, dim>> &                  input,
            const MappingKind                                        kind,
            const typename Mapping<dim, spacedim>::InternalDataBase &internal,
            const ArrayView<Tensor<1, spacedim>> &output) const override;

  // for documentation, see the Mapping base class
  virtual void
  transform(const ArrayView<const DerivativeForm<1, dim, spacedim>> &input,
            const MappingKind                                        kind,
            const typename Mapping<dim, spacedim>::InternalDataBase &internal,
            const ArrayView<Tensor<2, spacedim>> &output) const override;

  // for documentation, see the Mapping base class
  virtual void
  transform(const ArrayView<const Tensor<2, dim>> &                  input,
            const MappingKind                                        kind,
            const typename Mapping<dim, spacedim>::InternalDataBase &internal,
            const ArrayView<Tensor<2, spacedim>> &output) const override;

  // for documentation, see the Mapping base class
  virtual void
  transform(const ArrayView<const DerivativeForm<2, dim, spacedim>> &input,
            const MappingKind                                        kind,
            const typename Mapping<dim, spacedim>::InternalDataBase &internal,
            const ArrayView<Tensor<3, spacedim>> &output) const override;

  // for documentation, see the Mapping base class
  virtual void
  transform(const ArrayView<const Tensor<3, dim>> &                  input,
            const MappingKind                                        kind,
            const typename Mapping<dim, spacedim>::InternalDataBase &internal,
            const ArrayView<Tensor<3, spacedim>> &output) const override;

  /**
   * @}
   */

  /**
   * Return the degree of the mapping, i.e. the value which was passed to the
   * constructor.
   */
  unsigned int
  get_degree() const;

  /**
   * Return the ComponentMask of the mapping, i.e. which components to use for
   * the mapping.
   */
  ComponentMask
  get_component_mask() const;

  /**
   * Exception
   */
  DeclException0(ExcInactiveCell);

private:
  /**
   * @name Interface with FEValues
   * @{
   */

  // documentation can be found in Mapping::requires_update_flags()
  virtual UpdateFlags
  requires_update_flags(const UpdateFlags update_flags) const override;

public:
  /**
   * Storage for internal data of this mapping. See Mapping::InternalDataBase
   * for an extensive description.
   *
   * This includes data that is computed once when the object is created (in
   * get_data()) as well as data the class wants to store from between the
   * call to fill_fe_values(), fill_fe_face_values(), or
   * fill_fe_subface_values() until possible later calls from the finite
   * element to functions such as transform(). The latter class of member
   * variables are marked as 'mutable', along with scratch arrays.
   */
  class InternalData : public Mapping<dim, spacedim>::InternalDataBase
  {
  public:
    /**
     * Constructor.
     */
    InternalData(const FiniteElement<dim, spacedim> &fe,
                 const ComponentMask &               mask);

    /**
     * Shape function at quadrature point. Shape functions are in tensor
     * product order, so vertices must be reordered to obtain transformation.
     */
    const double &
    shape(const unsigned int qpoint, const unsigned int shape_nr) const;

    /**
     * Shape function at quadrature point. See above.
     */
    double &
    shape(const unsigned int qpoint, const unsigned int shape_nr);

    /**
     * Gradient of shape function in quadrature point. See above.
     */
    const Tensor<1, dim> &
    derivative(const unsigned int qpoint, const unsigned int shape_nr) const;

    /**
     * Gradient of shape function in quadrature point. See above.
     */
    Tensor<1, dim> &
    derivative(const unsigned int qpoint, const unsigned int shape_nr);

    /**
     * Second derivative of shape function in quadrature point. See above.
     */
    const Tensor<2, dim> &
    second_derivative(const unsigned int qpoint,
                      const unsigned int shape_nr) const;

    /**
     * Second derivative of shape function in quadrature point. See above.
     */
    Tensor<2, dim> &
    second_derivative(const unsigned int qpoint, const unsigned int shape_nr);

    /**
     * Third derivative of shape function in quadrature point. See above.
     */
    const Tensor<3, dim> &
    third_derivative(const unsigned int qpoint,
                     const unsigned int shape_nr) const;

    /**
     * Fourth derivative of shape function in quadrature point. See above.
     */
    Tensor<3, dim> &
    third_derivative(const unsigned int qpoint, const unsigned int shape_nr);

    /**
     * Fourth derivative of shape function in quadrature point. See above.
     */
    const Tensor<4, dim> &
    fourth_derivative(const unsigned int qpoint,
                      const unsigned int shape_nr) const;

    /**
     * Third derivative of shape function in quadrature point. See above.
     */
    Tensor<4, dim> &
    fourth_derivative(const unsigned int qpoint, const unsigned int shape_nr);

    /**
     * Return an estimate (in bytes) or the memory consumption of this object.
     */
    virtual std::size_t
    memory_consumption() const override;

    /**
     * Values of shape functions. Access by function @p shape.
     *
     * Computed once.
     */
    std::vector<double> shape_values;

    /**
     * Values of shape function derivatives. Access by function @p derivative.
     *
     * Computed once.
     */
    std::vector<Tensor<1, dim>> shape_derivatives;

    /**
     * Values of shape function second derivatives. Access by function @p
     * second_derivative.
     *
     * Computed once.
     */
    std::vector<Tensor<2, dim>> shape_second_derivatives;

    /**
     * Values of shape function third derivatives. Access by function @p
     * third_derivative.
     *
     * Computed once.
     */
    std::vector<Tensor<3, dim>> shape_third_derivatives;

    /**
     * Values of shape function fourth derivatives. Access by function @p
     * fourth_derivative.
     *
     * Computed once.
     */
    std::vector<Tensor<4, dim>> shape_fourth_derivatives;

    /**
     * Unit tangential vectors. Used for the computation of boundary forms and
     * normal vectors.
     *
     * This array has (dim-1)*GeometryInfo::faces_per_cell entries. The first
     * GeometryInfo::faces_per_cell contain the vectors in the first
     * tangential direction for each face; the second set of
     * GeometryInfo::faces_per_cell entries contain the vectors in the second
     * tangential direction (only in 3d, since there we have 2 tangential
     * directions per face), etc.
     *
     * Filled once.
     */
    std::array<std::vector<Tensor<1, dim>>,
               GeometryInfo<dim>::faces_per_cell *(dim - 1)>
      unit_tangentials;

    /**
     * Number of shape functions. If this is a Q1 mapping, then it is simply
     * the number of vertices per cell. However, since also derived classes
     * use this class (e.g. the Mapping_Q() class), the number of shape
     * functions may also be different.
     */
    unsigned int n_shape_functions;

    /**
     * Stores the mask given at construction time. If no mask was specified at
     * construction time, then a default one is used, which makes this class
     * works in the same way of MappingQEulerian(), i.e., the first spacedim
     * components of the FiniteElement are used for the euler_vector and the
     * euler_dh.
     *
     * If a mask is specified, then it has to match the underlying
     * FiniteElement, and it has to have exactly spacedim non-zero elements,
     * indicating the components (in order) of the FiniteElement which will be
     * used for the euler vector and the euler dof handler.
     */
    ComponentMask mask;

    /**
     * Tensors of covariant transformation at each of the quadrature points.
     * The matrix stored is the Jacobian * G^{-1}, where G = Jacobian^{t} *
     * Jacobian, is the first fundamental form of the map; if dim=spacedim
     * then it reduces to the transpose of the inverse of the Jacobian matrix,
     * which itself is stored in the @p contravariant field of this structure.
     *
     * Computed on each cell.
     */
    mutable std::vector<DerivativeForm<1, dim, spacedim>> covariant;

    /**
     * Tensors of contravariant transformation at each of the quadrature
     * points. The contravariant matrix is the Jacobian of the transformation,
     * i.e. $J_{ij}=dx_i/d\hat x_j$.
     *
     * Computed on each cell.
     */
    mutable std::vector<DerivativeForm<1, dim, spacedim>> contravariant;

    /**
     * The determinant of the Jacobian in each quadrature point. Filled if
     * #update_volume_elements.
     */
    mutable std::vector<double> volume_elements;

    /**
     * Auxiliary vectors for internal use.
     */
    mutable std::vector<std::vector<Tensor<1, spacedim>>> aux;

    /**
     * Storage for the indices of the local degrees of freedom.
     */
    mutable std::vector<types::global_dof_index> local_dof_indices;

    /**
     * Storage for local degrees of freedom.
     */
    mutable std::vector<double> local_dof_values;
  };

private:
  // documentation can be found in Mapping::get_data()
  virtual std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
  get_data(const UpdateFlags, const Quadrature<dim> &quadrature) const override;

  // documentation can be found in Mapping::get_face_data()
  virtual std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
  get_face_data(const UpdateFlags          flags,
                const Quadrature<dim - 1> &quadrature) const override;

  // documentation can be found in Mapping::get_subface_data()
  virtual std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
  get_subface_data(const UpdateFlags          flags,
                   const Quadrature<dim - 1> &quadrature) const override;

  // documentation can be found in Mapping::fill_fe_values()
  virtual CellSimilarity::Similarity
  fill_fe_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const CellSimilarity::Similarity                            cell_similarity,
    const Quadrature<dim> &                                     quadrature,
    const typename Mapping<dim, spacedim>::InternalDataBase &   internal_data,
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
      &output_data) const override;

  // documentation can be found in Mapping::fill_fe_face_values()
  virtual void
  fill_fe_face_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const typename Mapping<dim, spacedim>::InternalDataBase &   internal_data,
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
      &output_data) const override;

  // documentation can be found in Mapping::fill_fe_subface_values()
  virtual void
  fill_fe_subface_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const unsigned int                                          subface_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const typename Mapping<dim, spacedim>::InternalDataBase &   internal_data,
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
      &output_data) const override;

  /**
   * @}
   */

  /**
   * Specifies whether we access unknowns on the active dofs (with a single
   * Euler vector) or on the level dofs (via a vector of Euler vectors).
   */
  const bool uses_level_dofs;

  /**
   * Reference to the vector of shifts.
   */
  std::vector<
    SmartPointer<const VectorType,
                 MappingFEField_hp<dim, spacedim, VectorType, DoFHandlerType>>>
//    std::vector<const VectorType>
    euler_vector;

  /**
   * Pointer to the DoFHandler to which the mapping vector is associated.
   */
  SmartPointer<const DoFHandlerType,
               MappingFEField_hp<dim, spacedim, VectorType, DoFHandlerType>>
    euler_dof_handler;
    
    const typename hp::DoFHandler<dim, spacedim>::cell_iterator dof_cell;
    
    const unsigned int fe_index;

    

private:
  /**
   * Transform a point @p p on the unit cell to the point @p p_real on the
   * real cell @p cell and returns @p p_real.
   *
   * This function is called by @p transform_unit_to_real_cell and multiple
   * times (through the Newton iteration) by @p
   * transform_real_to_unit_cell_internal.
   *
   * Takes a reference to an @p InternalData that must already include the
   * shape values at point @p p and the mapping support points of the cell.
   *
   * This @p InternalData argument avoids multiple computations of the shape
   * values at point @p p and especially multiple computations of the mapping
   * support points.
   */
  Point<spacedim>
  do_transform_unit_to_real_cell(const InternalData &mdata) const;


  /**
   * Transform the point @p p on the real cell to the corresponding point on
   * the unit cell @p cell by a Newton iteration.
   *
   * Takes a reference to an @p InternalData that is assumed to be previously
   * created by the @p get_data function with @p UpdateFlags including @p
   * update_transformation_values and @p update_transformation_gradients and a
   * one point Quadrature that includes the given initial guess for the
   * transformation @p initial_p_unit.  Hence this function assumes that @p
   * mdata already includes the transformation shape values and gradients
   * computed at @p initial_p_unit.
   *
   * @p mdata will be changed by this function.
   */
  Point<dim>
  do_transform_real_to_unit_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const Point<spacedim> &                                     p,
    const Point<dim> &                                          initial_p_unit,
    InternalData &                                              mdata) const;

  /**
   * Update internal degrees of freedom.
   */
  void
  update_internal_dofs(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const typename MappingFEField_hp<dim, spacedim, VectorType, DoFHandlerType>::
      InternalData &data) const;

  /**
   * See the documentation of the base class for detailed information.
   */
  virtual void
  compute_shapes_virtual(
    const std::vector<Point<dim>> &unit_points,
    typename MappingFEField_hp
                         <dim, spacedim, VectorType, DoFHandlerType>::
      InternalData &data) const;

  /*
   * Which components to use for the mapping.
   */
  const ComponentMask fe_mask;

  /**
   * Mapping between indices in the FE space and the real space. This vector
   * contains one index for each component of the finite element space. If the
   * index is one for which the ComponentMask which is used to construct this
   * element is false, then numbers::invalid_unsigned_int is returned,
   * otherwise the component in real space is returned. For example, if we
   * construct the mapping using ComponentMask(spacedim, true), then this
   * vector contains {0,1,2} in spacedim = 3.
   */
  std::vector<unsigned int> fe_to_real;

  /**
   * FEValues object used to query the given finite element field at the
   * support points in the reference configuration.
   */
  mutable FEValues<dim, spacedim> fe_values;

  /**
   * A variable to guard access to the fe_values variable.
   */
  mutable Threads::Mutex fe_values_mutex;
    
  void
  compute_data(const UpdateFlags      update_flags,
               const Quadrature<dim> &q,
               const unsigned int     n_original_q_points,
               InternalData &         data) const;

  void
  compute_face_data(const UpdateFlags      update_flags,
                    const Quadrature<dim> &q,
                    const unsigned int     n_original_q_points,
                    InternalData &         data) const;


  // Declare other MappingFEField classes friends.
  template <int, int, class, class>
  friend class MappingFEField;
};

/*@}*/

/* -------------- declaration of explicit specializations ------------- */

#ifndef DOXYGEN

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
