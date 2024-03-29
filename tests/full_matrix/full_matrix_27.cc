// ---------------------------------------------------------------------
//
// Copyright (C) 2007 - 2018 by the deal.II authors
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



// check FullMatrix::add


#include "../tests.h"
#include "full_matrix_common.h"



template <typename number>
void
check()
{
  FullMatrix<number> m(10, 10), n;
  make_matrix(n);
  m.add(n, 2., 3, 4, 2, 3);
  print_matrix(m);
}
