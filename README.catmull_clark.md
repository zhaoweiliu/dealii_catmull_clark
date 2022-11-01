<<<<<<< HEAD
What is deal.II?
================

deal.II is a C++ program library targeted at the computational solution
of partial differential equations using adaptive finite elements. It uses
state-of-the-art programming techniques to offer you a modern interface
to the complex data structures and algorithms required.

For the impatient:
------------------

Let's say you've unpacked the .tar.gz file into a directory /path/to/dealii/sources. 
Then configure, compile, and install the deal.II library with:

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_INSTALL_PREFIX=/path/where/dealii/should/be/installed/to /path/to/dealii/sources
    $ make install    (alternatively $ make -j<N> install)
    $ make test

To build from the repository, execute the following commands first:

    $ git clone https://github.com/dealii/dealii
    $ cd dealii

Then continue as before.

A detailed *ReadME* can be found at [./doc/readme.html](https://dealii.org/developer/readme.html),
[./doc/users/cmake_user.html](https://dealii.org/developer/users/cmake_user.html),
or at https://www.dealii.org/.

Getting started:
----------------

The tutorial steps are located under examples/ of the installation.
Information about the tutorial steps can be found at
[./doc/doxygen/tutorial/index.html](https://dealii.org/developer/doxygen/deal.II/Tutorial.html)
or at https://www.dealii.org/.

deal.II includes support for pretty-printing deal.II objects inside GDB.
See [`contrib/utilities/dotgdbinit.py`](contrib/utilities/dotgdbinit.py) or
the new documentation page (under 'information for users') for instructions
on how to set this up.

License:
--------

Please see the file [./LICENSE.md](LICENSE.md) for details

Further information:
--------------------

For further information have a look at
[./doc/index.html](https://dealii.org/developer/index.html) or at
https://www.dealii.org.

Continuous Integration Status:
------------------------

| System | Status | More information |
| --- | --- | --- |
| Indent | ![indent](https://github.com/dealii/dealii/workflows/indent/badge.svg) | using GitHub actions |
| Linux | [![Build Status](https://jenkins.tjhei.info/job/dealii/job/master/badge/icon)](https://jenkins.tjhei.info/job/dealii/job/master/) | See https://jenkins.tjhei.info |
| MacOS | [![Build Status](https://jenkins.tjhei.info/job/dealii-OSX/job/master/badge/icon)](https://jenkins.tjhei.info/job/dealii-OSX/job/master/) | See https://jenkins.tjhei.info |
| MacOS | [![Build Status](https://github.com/dealii/dealii/workflows/github-CI/badge.svg)](https://github.com/dealii/dealii/actions?query=workflow%3Agithub-CI) | See https://github.com/dealii/dealii/actions |
| MSVC | [![Build status](https://github.com/dealii/dealii/workflows/github-windows/badge.svg)](https://github.com/dealii/dealii/actions?query=workflow%3Agithub-windows) | See https://github.com/dealii/dealii/actions |
| CDash | [![cdash](https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=CDash&up_color=green&up_message=up&url=https%3A%2F%2Fcdash.43-1.org%2Findex.php%3Fproject%3Ddeal.II)](https://cdash.43-1.org/index.php?project=deal.II) | Various builds and configurations on https://cdash.43-1.org/index.php?project=deal.II |

=======
FE_Catmull_Clark<dim, spacedim> is constructed with a valence. Valence of the cell are defined as the number of the elements connected at the vertex. The valence of the cell is defined as: 
	If the four vertices of the cell all have valence = 4, then the valence of the cell is 4. 
	If the one of the vertices has a valence N≠ 4, the valence of the cell is N. 
	If the cell has an edge on the physical boundary the valence of cell is 2. 
	If the cell has two edges on the physical boundary, this cell has a valence = 1. 

If the valence is 4, the basis functions are tensor-products of two cubic b-splines. If the valence is one or two, basis functions are truncated at boundary. If valence is nonequal to 1, 2 and 4, the scheme in [1] is adopted. The FiniteElementData in FE_Catmull_Clark is defined as:
FiniteElementData<dim>({1,0,0,(val==1?5:2*val+4)}, n_components,3,FiniteElementData<dim>::H2)

The cell has 1 dof per vertex and 2N+4 nonlocal dofs, totally 2N+8 dofs per cell, except the cell has two edges on the physical boundary,  which has 9 dofs in total (1 per vertex and 5 nonlocal dofs). The reason to set 1 dof per vertex is the DoFHandler can use it to numerate the dofs.

The Catmull_Clark<dim,spacedim> class constructed with a Triangulation<dim,spacedim>. It first constructs a hp::DoFHandler using the triangulation and it calls a private function cell_patches to loops over all cells in the triangulation and for each cell, it finds all the cells which share the same vertices with the target cell (cell patch). This function return a vector of n_active_cells number sets cell_patch_vector, the ith set associate with ith active cell and contains cell iterators in the cell patch: 
std::vector<std::set<typename Triangulation<dim,spacedim>::active_cell_iterator>>
With the information in cell_patch_vector, one can have the valences of those cells. 
A FE_Collection is constructed based on how many difference valences there are for all the cells. Each cell is allocated a fe index using the following algorithm:
std::map<int, int> map_valence_to_fe_indices;
int i_fe = 0;
	for (auto cell = dof_handler.begin_active(); cell!= dof_handler.end(); ++cell) {
        		int valence;
       		switch (int ncell_in_patch = cell_patch_vector[cell->active_cell_index()].size()) {
            			case 4:
               			      valence = 1;
                	    	      break;
            			case 6:
               			      valence = 2;
                	                      break;
            			default:
                		valence = ncell_in_patch - 5;
                		      break;
        		}
       	 	std::map<int, int>::iterator iter_fe_valence;
       		 iter_fe_valence= map_valence_to_fe_indices.find(valence);
        		if (iter_fe_valence != map_valence_to_fe_indices.end()) {
           		         cell->set_active_fe_index(iter_fe_valence->second);
        		}else{
            		        map_valence_to_fe_indices.insert(std::pair<int,int>(valence,i_fe));
            		        fe_collection.push_back(FE_Catmull_Clark<dim, spacedim>(valence));
            		       cell->set_active_fe_index(i_fe);
           	   	      ++i_fe;
        	   	 }
  	}

Then the cells are ordered in a specific sequence according to different valences.

    Valence = 4:

             *-----*-----*-----*
             |     |     |     |
             |  5  |  3  |  6  |
             |     |     |     |
             *-----*-----*-----*
             |     |     |     |
             |  1  |  0  |  2  |
             |     |     |     |
             *-----*-----*-----*
             |     |     |     |
             |  7  |  4  |  8  |
             |     |     |     |
             *-----*-----*-----*
	Valence = 2:

             *-----*-----*-----*
             |     |     |     |
             |  5  |  2  |  4  |
             |     |     |     |
             *-----*-----*-----*
             |     |     |     |
             |  3  |  0  |  1  |
             |     |     |     |
             *-----*-----*-----*     

    Valence = 1:



                 *-----*-----*
                 |     |     |
                 |  2  |  3  |
                 |     |     |
                 *-----*-----*
                 |     |     |
                 |  0  |  1  |
                 |     |     |
                 *-----*-----*    
	Valence ≠1, 2, 4:
                                *
                              / |
                            /   |
                          /     |
             *-----*-----* N-2  *-----*
             |     |     |     /     /
             | N+4 | N-1 |   .. 2  /
             |     |     | /     /
             *-----*-----*-----*
             |     |     |     |
             | N+3 |  0  |  1  |
             |     |     |     |
             *-----*-----*-----*
             |     |     |     |
             |  N  | N+1 | N+2 |
             |     |     |     |
             *-----*-----*-----*        


hp::DoFHandler.distribute() will numerate dofs for each vertex, such as called with FE_Q(1). The get_dof_indices function now will give a vector with size 2N+8 but only the first 4 entries has global dof indices and the rest are 0. We need to set the nonlocal dofs with our own scheme by asking from the cell iterator s in cell_patch shown above and reordered the local dof indices as following:

    Valence = 4:
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
	Valence = 2:


                 8-----9----10----11
                 |     |     |     |
                 |     |     |     |
                 |     |     |     |
                 4-----5-----6-----7
                 |     |     |     |
                 |     |     |     |
                 |     |     |     |
                 0-----1-----2-----3     

    Valence = 1:



                 6-----7-----8
                 |     |     |
                 |     |     |
                 |     |     |
                 3-----4-----5
                 |     |     |
                 |     |     |
                 |     |     |
                 0-----1-----2     
	Valence ≠1, 2, 4:
                                      2v
                                     / |
                                   /   |
                                 /     |
                 2v+7-----2-----1      *------8
                    |     |     |     /     /
                    |     |     |  ..     /
                    |     |     | /     /
                 2v+6-----3-----0-----7
                    |     |     |     |
                    |     |     |     |
                    |     |     |     |
                 2v+5-----4-----5-----6
                    |     |     |     |
                    |     |     |     |
                    |     |     |     |
                 2v+1---2v+2---2v+3---2v+4         


One try to set the global dof index vector using the function: cell->set_dof_indices(ordered_dof_indices), but error occurs in function DoFLevel::set_dof_index(…).
dof_indices[dof_offsets[obj_index] + local_index] = global_index;
dof_indices has no size. EXC_BAD_ACCESS

[1] Stam, Jos. "Exact evaluation of Catmull-Clark subdivision surfaces at arbitrary parameter values." Siggraph. Vol. 98. 1998.
>>>>>>> b1c17583d9... Update README.md
