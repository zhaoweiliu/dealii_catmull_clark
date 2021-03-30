# include "CatmullClark_subd.hpp"

//template <int dim, int spacedim>
void Catmull_Clark_subdivision(Triangulation<2,3> &mesh){
    int ncells = mesh.n_active_cells();
    int nfaces = mesh.n_faces();
    int nvertices = mesh.n_vertices();
    // initialise cell data
    std::vector<CellData<2> >                  new_cells;
    SubCellData                                subcelldata;
    // initialise topology data vectors
    std::vector<Point<3>> vertices(nvertices+ncells+nfaces);
    std::vector<Point<3>> cvs(ncells);
    std::vector<Point<3>> fvs(nfaces);
    std::vector<Point<3>> fvs_mid(nfaces);
    std::vector<Point<3>> vvs(nvertices);
    std::vector<std::vector<int>> v_in_cells(nvertices);
    std::vector<std::vector<int>> v_in_faces(nvertices);
    std::vector<bool> v_at_boundary(nvertices,false);
    std::vector<bool> f_at_boundary(nfaces,false);
    
    //Loop over cells to compute new cell and face vertices
    for (Triangulation<2,3>::active_cell_iterator cell=mesh.begin_active();cell!= mesh.end(); ++cell){
        //initialise a point
        Point<3> v{0,0,0};
        //loop over vertices in cell iterator
        for(unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i){
            v_in_cells[cell->vertex_index(i)].push_back(cell->active_cell_index());
            v += cell->vertex(i);
        }
        // compute face point
        v= v/4.0;
        // same it into vector
        cvs[cell->active_cell_index()] = v;
        
        //loop over faces in cell iterator
        for(unsigned int i = 0; i<GeometryInfo<2>::faces_per_cell;++i){
            
            // avoid duplicated faces
            if(cell->neighbor_index(i)<int(cell->active_cell_index())){
                //loop over vertices in each face
                for(unsigned int j=0; j<GeometryInfo<2>::vertices_per_face; ++j){
                    //vertices id for each face
                    v_in_faces[cell->face(i)->vertex_index(j)].push_back(cell->face_index(i));
                }
                fvs_mid[cell->face_index(i)] = cell->face(i)->center();
                // determine whether face on boundary
                if(cell->face(i)->at_boundary()==false){
                    // compute new face points and face midpoints
                    fvs[cell->face_index(i)] = (cvs[cell->neighbor_index(i)] + cvs[cell->active_cell_index()])/4.0 + cell->face(i)->vertex(0)/4.0 + cell->face(i)->vertex(1)/4.0;
                }
                else{
                    f_at_boundary[cell->face_index(i)] = true;
                    // if on the boundary, the new face points are face midpoints
                    for(unsigned int j=0; j<GeometryInfo<2>::vertices_per_face; ++j){
                        v_at_boundary[cell->face(i)->vertex_index(j)] = true;
                    }
                    fvs[cell->face_index(i)] = cell->face(i)->center();
                }
            }
        }
        
        // compute cell data (connectivity)
        std::vector<std::vector<unsigned int>> connectivity(4);
        connectivity[0] = {cell->vertex_index(0),nvertices+cell->face_index(2),nfaces+nvertices+cell->active_cell_index(),nvertices+cell->face_index(0)};
        connectivity[1] = { nvertices + cell->face_index(2),cell->vertex_index(1), nvertices+cell->face_index(1), nfaces+nvertices+cell->active_cell_index()};
        connectivity[2] = {nvertices + cell->face_index(0),nfaces+nvertices+cell->active_cell_index(),nvertices+cell->face_index(3),cell->vertex_index(2)};
        connectivity[3] = {nfaces+nvertices+cell->active_cell_index(),nvertices+ cell->face_index(1),cell->vertex_index(3),nvertices+cell->face_index(3)};
        
        for(unsigned int j = 0; j<GeometryInfo<2>::vertices_per_cell;++j){
            new_cells.emplace_back();
            for (unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i){
                new_cells.back().vertices[i] = connectivity[j][i];
            }
            new_cells.back().material_id = cell->material_id();
        }
    }
    
    //Loop over old vertices and modify them
    for(Triangulation<2,3>::active_vertex_iterator vertex = mesh.begin_active_vertex();vertex!= mesh.end_vertex();++vertex){
        Point<3> cv{0,0,0};
        Point<3> fv(0,0,0);
        double nc = 0.0, nf = 0.0;
        for (unsigned int i = 0; i < v_in_cells[vertex->index()].size(); ++i) {
            cv +=cvs[v_in_cells[vertex->index()][i]];
            nc+=1;
        }
        cv=cv/nc;
        
        for (unsigned int i = 0; i < v_in_faces[vertex->index()].size(); ++i){
            fv += fvs_mid[v_in_faces[vertex->index()][i]];
            nf += 1;
        }
        fv = fv/nf;
        // averaging
        if(v_at_boundary[vertex->index()]==false){
            if (nf == nc){
                vvs[vertex->index()] = vertex->center() + 1./nc * (cv - vertex->center()) + 2./nf * (fv-vertex->center());
                //            vvs[vertex->index()] = (nf-3.)/nf * vertex->center() + 1./nc * cv + 2./nf * fv;
            }else{
                throw std::runtime_error("number of faces and cells are not equal.");
            }
        }
        else{
            Point<3>f_b{0,0,0};
            for (unsigned int i = 0; i < v_in_faces[vertex->index()].size(); ++i){
                if (f_at_boundary[v_in_faces[vertex->index()][i]]==true){
                    f_b += fvs_mid[v_in_faces[vertex->index()][i]];
                }
            }
//            vvs[vertex->index()] = 1./2.*vertex->center() + 1./4.*f_b;
            vvs[vertex->index()] = vertex->center();
        }
    }
    
    for(int i = 0; i<nvertices;++i)
        vertices[i] = vvs[i];
    for(int i = 0; i<nfaces;++i)
        vertices[i+nvertices] = fvs[i];
    for(int i = 0; i<ncells;++i)
        vertices[i+nvertices+nfaces] = cvs[i];
    
    GridTools::delete_unused_vertices(vertices,new_cells,subcelldata);
    Triangulation<2,3> new_mesh;
    GridReordering<2,3>::reorder_cells (new_cells);
    new_mesh.create_triangulation_compatibility(vertices, new_cells, subcelldata);
    mesh.clear();
    mesh.copy_triangulation(new_mesh);
}
