// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------
#ifndef dealii_particles_data_out_h
#define dealii_particles_data_out_h

#include <deal.II/base/array_view.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/property_pool.h>

#include <boost/range/iterator_range.hpp>
#include <boost/serialization/map.hpp>

DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_WITH_P4EST

namespace Particles
{
  /**
   * This class manages the DataOut of a Particle Handler
   * From a particle handler, it generates patches which can then be used to
   * write traditional output files. This class currently only supports witing
   * the particle position and their ID and does not allow to write the
   * properties attached to the particles
   *
   * @ingroup Particle
   *
   * @author Bruno Blais, Luca Heltai 2019
   */

  template <int dim, int spacedim>
  class DataOut : public dealii::DataOutInterface<0, spacedim>
  {
  public:
    DataOut() = default;

    ~DataOut() = default;


    /**
     * Build the particles for a given partcle handler
     *
     * @param [in] particle_handler A particle handler for which the patches will be build
     * A dim=0 patch is build for each particle. The position of the particle is
     * used to build the node position and the ID of the particle is added as a
     * single data element
     *
     *
     * @author Bruno Blais, Luca Heltai 2019
     */
    void
    build_patches(const Particles::ParticleHandler<dim, spacedim> &particles);

  protected:
    /**
     * Returns the patches built by the data_out class which was previously
     * built using a particle handler
     */
    virtual const std::vector<DataOutBase::Patch<0, spacedim>> &
    get_patches() const override;
    //    {
    //      return patches;
    //    }

    /**
     * Returns the name of the data sets associated with the patches. In the
     * current implementation the particles only contain the ID
     */
    virtual std::vector<std::string>
    get_dataset_names() const override;
    //    {
    //      return dataset_names;
    //    }

  private:
    std::vector<DataOutBase::Patch<0, spacedim>> patches;

    /**
     * A list of field names for all data components stored in patches.
     */
    std::vector<std::string> dataset_names;
  };

} // namespace Particles

#endif // DEAL_II_WITH_P4EST

DEAL_II_NAMESPACE_CLOSE

#endif
