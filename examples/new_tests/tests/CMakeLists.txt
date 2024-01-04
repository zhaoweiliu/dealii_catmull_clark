# look for VTK package
# FIND_PACKAGE(VTK)
# INCLUDE(${VTK_USE_FILE})

SET(VTK_DIR "/usr/local/include/vtk-8.2" CACHE PATH "VTK directory override" FORCE)
file(GLOB LIBRARIES "/Users/zhaoweiliu/Documents/codes/VTK-8.2.0/build/lib/*.dylib")
message("LIBRARIES = ${LIBRARIES}")

#FIND_PACKAGE(VTK REQUIRED)
INCLUDE_DIRECTORIES(${VTK_DIR})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/addition_lib/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/plot/include)

ADD_EXECUTABLE(electro_plate_inflation_ac electro_plate_inflation_ac.cc)
DEAL_II_SETUP_TARGET(electro_plate_inflation_ac)
TARGET_LINK_LIBRARIES(electro_plate_inflation_ac addition_lib ${LIBRARIES})

ADD_EXECUTABLE(incompressible_electroelastic_shell incompressible_electroelastic_shell.cc)
DEAL_II_SETUP_TARGET(incompressible_electroelastic_shell)
TARGET_LINK_LIBRARIES(incompressible_electroelastic_shell addition_lib ${LIBRARIES})

ADD_EXECUTABLE(elec_sphere_arclength   elec_sphere_arclength.cc)
DEAL_II_SETUP_TARGET(elec_sphere_arclength)
TARGET_LINK_LIBRARIES(elec_sphere_arclength addition_lib ${LIBRARIES})

ADD_EXECUTABLE(electroelastic_torus   electroelastic_torus.cc)
DEAL_II_SETUP_TARGET(electroelastic_torus)
TARGET_LINK_LIBRARIES(electroelastic_torus addition_lib ${LIBRARIES})

ADD_EXECUTABLE(electroelastic_torus_phi   electroelastic_torus_phi.cc)
DEAL_II_SETUP_TARGET(electroelastic_torus_phi)
TARGET_LINK_LIBRARIES(electroelastic_torus_phi addition_lib ${LIBRARIES})