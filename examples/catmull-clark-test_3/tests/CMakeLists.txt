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
