# look for VTK package
# FIND_PACKAGE(VTK)
# INCLUDE(${VTK_USE_FILE})

SET(VTK_DIR "/usr/local/include/vtk-8.2" CACHE PATH "VTK directory override" FORCE)
file(GLOB LIBRARIES "/Users/zhaoweiliu/Documents/codes/VTK-8.2.0/build/lib/*.dylib")
message("LIBRARIES = ${LIBRARIES}")

#hlibpro directory
set(HLIBPRODIR "NOT-DEFINED" CACHE PATH "/opt")
if(DEFINED HLIBPRODIR)
    include_directories(${HLIBPRODIR}/include)
#    link_directories(${HLIBPRODIR}/lib)
    file(GLOB HLIB "/Users/zhaoweiliu/Documents/codes/hlibpro-2.9/lib/*dylib")
else()
    message(FATAL_ERROR "HLibPro directory not defined")
endif()

#FIND_PACKAGE(VTK REQUIRED)
INCLUDE_DIRECTORIES(${VTK_DIR})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/addition_lib/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/plot/include)

ADD_EXECUTABLE(shell coupled_shell.cc)
DEAL_II_SETUP_TARGET(shell)
#TARGET_LINK_LIBRARIES(shell addition_lib)
TARGET_LINK_LIBRARIES(shell addition_lib ${LIBRARIES})

ADD_EXECUTABLE(shell_og shell_origin.cc)
DEAL_II_SETUP_TARGET(shell_og)
TARGET_LINK_LIBRARIES(shell_og addition_lib)

ADD_EXECUTABLE(unit_test unit_test.cc)
DEAL_II_SETUP_TARGET(unit_test)
TARGET_LINK_LIBRARIES(unit_test addition_lib ${LIBRARIES})

ADD_EXECUTABLE(nonlinear_shell nonlinear_shell_GL_strain.cc)
DEAL_II_SETUP_TARGET(nonlinear_shell)
TARGET_LINK_LIBRARIES(nonlinear_shell addition_lib ${LIBRARIES})

ADD_EXECUTABLE(cooks_membrane cooks_membrane.cc)
DEAL_II_SETUP_TARGET(cooks_membrane)
TARGET_LINK_LIBRARIES(cooks_membrane addition_lib ${LIBRARIES})

ADD_EXECUTABLE(sphere_inflation sphere_inflation.cc)
DEAL_II_SETUP_TARGET(sphere_inflation)
TARGET_LINK_LIBRARIES(sphere_inflation addition_lib ${LIBRARIES})

ADD_EXECUTABLE(sphere_inflation_ac sphere_inflation_ac.cc)
DEAL_II_SETUP_TARGET(sphere_inflation_ac)
TARGET_LINK_LIBRARIES(sphere_inflation_ac addition_lib ${LIBRARIES})

# ADD_EXECUTABLE(sphere_inflation_u sphere_inflation_u.cc)
# DEAL_II_SETUP_TARGET(sphere_inflation_u)
# TARGET_LINK_LIBRARIES(sphere_inflation_u addition_lib ${LIBRARIES})

ADD_EXECUTABLE(sphere_arclength sphere_arclength_1.cc)
DEAL_II_SETUP_TARGET(sphere_arclength)
TARGET_LINK_LIBRARIES(sphere_arclength addition_lib ${LIBRARIES})

ADD_EXECUTABLE(shell_flexo coupled_shell_flexo.cc)
DEAL_II_SETUP_TARGET(shell_flexo)
#TARGET_LINK_LIBRARIES(shell_flexo addition_lib)
TARGET_LINK_LIBRARIES(shell_flexo addition_lib ${LIBRARIES})

ADD_EXECUTABLE(unit_test_verification unit_test_verification.cc)
DEAL_II_SETUP_TARGET(unit_test_verification)
TARGET_LINK_LIBRARIES(unit_test_verification addition_lib ${LIBRARIES})

ADD_EXECUTABLE(plate_inflation_ac plate_inflation_ac.cc)
DEAL_II_SETUP_TARGET(plate_inflation_ac)
TARGET_LINK_LIBRARIES(plate_inflation_ac addition_lib ${LIBRARIES})

ADD_EXECUTABLE(torus_inflation torus_inflation.cc)
DEAL_II_SETUP_TARGET(torus_inflation)
TARGET_LINK_LIBRARIES(torus_inflation addition_lib ${LIBRARIES})

# ADD_EXECUTABLE(spot_inflation spot_inflation.cc)
# DEAL_II_SETUP_TARGET(spot_inflation)
# TARGET_LINK_LIBRARIES(spot_inflation addition_lib ${LIBRARIES})

ADD_EXECUTABLE(full_sphere full_sphere.cc)
DEAL_II_SETUP_TARGET(full_sphere)
TARGET_LINK_LIBRARIES(full_sphere addition_lib ${LIBRARIES})

ADD_EXECUTABLE(shell_elastic_nonlinear shell_elastic_nonlinear.cc)
DEAL_II_SETUP_TARGET(shell_elastic_nonlinear)
TARGET_LINK_LIBRARIES(shell_elastic_nonlinear addition_lib ${LIBRARIES})

ADD_EXECUTABLE(shell_elastic_inflation shell_elastic_inflation.cc)
DEAL_II_SETUP_TARGET(shell_elastic_inflation)
TARGET_LINK_LIBRARIES(shell_elastic_inflation addition_lib ${LIBRARIES})


ADD_EXECUTABLE(cylinder cylinder.cc)
DEAL_II_SETUP_TARGET(cylinder)
TARGET_LINK_LIBRARIES(cylinder addition_lib ${LIBRARIES})

ADD_EXECUTABLE(shell_elastic_nonlinear_tension shell_elastic_nonlinear_tension.cc)
DEAL_II_SETUP_TARGET(shell_elastic_nonlinear_tension)
TARGET_LINK_LIBRARIES(shell_elastic_nonlinear_tension addition_lib ${LIBRARIES})

ADD_EXECUTABLE(half_torus half_torus.cc)
DEAL_II_SETUP_TARGET(half_torus)
TARGET_LINK_LIBRARIES(half_torus addition_lib ${LIBRARIES})

ADD_EXECUTABLE(elastic_plate elastic_plate.cc)
DEAL_II_SETUP_TARGET(elastic_plate)
TARGET_LINK_LIBRARIES(elastic_plate addition_lib ${LIBRARIES})

ADD_EXECUTABLE(mag_shell_nonlinear mag_shell_nonlinear.cc)
DEAL_II_SETUP_TARGET(mag_shell_nonlinear)
TARGET_LINK_LIBRARIES(mag_shell_nonlinear addition_lib ${LIBRARIES})

ADD_EXECUTABLE(nonlinear_elastic_shell_validation nonlinear_elastic_shell_validation.cc)
DEAL_II_SETUP_TARGET(nonlinear_elastic_shell_validation)
TARGET_LINK_LIBRARIES(nonlinear_elastic_shell_validation addition_lib ${LIBRARIES})

ADD_EXECUTABLE(mag_shell_nonlinear2 mag_shell_nonlinear2.cc)
DEAL_II_SETUP_TARGET(mag_shell_nonlinear2)
TARGET_LINK_LIBRARIES(mag_shell_nonlinear2 addition_lib ${LIBRARIES})

ADD_EXECUTABLE(incompressible_electroelastic_shell incompressible_electroelastic_shell.cc)
DEAL_II_SETUP_TARGET(incompressible_electroelastic_shell)
TARGET_LINK_LIBRARIES(incompressible_electroelastic_shell addition_lib ${LIBRARIES})

ADD_EXECUTABLE(electro_plate_inflation_ac electro_plate_inflation_ac.cc)
DEAL_II_SETUP_TARGET(electro_plate_inflation_ac)
TARGET_LINK_LIBRARIES(electro_plate_inflation_ac addition_lib ${LIBRARIES})
