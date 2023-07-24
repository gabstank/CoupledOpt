##
#  CMake script
##

# Set the name of the project and target:
SET(TARGET "stop")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.

add_definitions("-Wno-deprecated -std=c++17")

INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/BVP
        ${CMAKE_SOURCE_DIR}/include/BVP/Standard
        ${CMAKE_SOURCE_DIR}/include/BVP/EDD
        ${CMAKE_SOURCE_DIR}/include/Material
        ${CMAKE_SOURCE_DIR}/include/Triangulation
        ${CMAKE_SOURCE_DIR}/include/Responses
        ${CMAKE_SOURCE_DIR}/include/Optimizer
        ${CMAKE_SOURCE_DIR}/include/Utils
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/BVP
        ${CMAKE_SOURCE_DIR}/source/BVP/Standard
        ${CMAKE_SOURCE_DIR}/source/BVP/EDD
        ${CMAKE_SOURCE_DIR}/source/Material
        ${CMAKE_SOURCE_DIR}/source/Triangulation
        ${CMAKE_SOURCE_DIR}/source/Regularization
        ${CMAKE_SOURCE_DIR}/source/Responses
        ${CMAKE_SOURCE_DIR}/source/Optimizer
)

FILE(GLOB_RECURSE TARGET_SRC "source/*.cc" "*.cc" "${CMAKE_SOURCE_DIR}/main.cc")
FILE(GLOB_RECURSE TARGET_INC "include/*.h")
SET(TARGET_SRC ${TARGET_SRC} ${TARGET_INC})

SET(CMAKE_CXX_FLAGS "-Wno-deprecated")

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.1.1 QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )
IF (NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
ENDIF ()

#
# Are all dependencies fulfilled?
#
IF (NOT (DEAL_II_WITH_PETSC OR DEAL_II_WITH_TRILINOS) OR NOT DEAL_II_WITH_P4EST OR DEAL_II_PETSC_WITH_COMPLEX) # keep in one line
    MESSAGE(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_PETSC = ON
    DEAL_II_PETSC_WITH_COMPLEX = OFF
    DEAL_II_WITH_P4EST = ON
or
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_P4EST = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
    DEAL_II_PETSC_WITH_COMPLEX = ${DEAL_II_PETSC_WITH_COMPLEX}
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
which conflict with the requirements.
One or both of the aforementioned combinations of prerequisites are not met by your installation, but at least one is required for this tutorial step."
            )
ENDIF ()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# if petsc = off => Trilinos is on.
option(WITH_PETSC "PETSC for Linear algebra, if OFF then Trilinos will be used." ON)
#if(WITH_PETSC)
#    SET(STOP_WITH_PETSC ON)
#endif(WITH_PETSC)

IF(WITH_PETSC)
    MESSAGE(STATUS "Configured for use of PETSC for LA")
    ADD_DEFINITIONS(-DSTOP_USE_PETSC)
ELSE()
    MESSAGE(STATUS "Configured for use of Trilinos for LA")
    ADD_DEFINITIONS(-DSTOP_USE_TRILINOS)
ENDIF()


SET(CLEAN_UP_FILES *.log *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2 *.vtu *.pvtu *.pvd ${CMAKE_SOURCE_DIR}/working_dir)
PROJECT(${TARGET})

# setup target for documentation using doxygen
FIND_PACKAGE(Doxygen)
IF(NOT DOXYGEN_FOUND)
    MESSAGE("-- Doxygen was not found. Documentation will not be available.")
ENDIF()
if(DOXYGEN_FOUND)
  SET( DOXYGEN_IN           ${CMAKE_CURRENT_SOURCE_DIR}/doc/doc-config.dox   )
  SET( DOXYGEN_OUT          ${CMAKE_CURRENT_BINARY_DIR}/options.dox           )

  #-- Configure the Template Doxyfile for our specific project
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  #-- Add a custom target to run Doxygen when ever the project is built
  #-- First command creates a directory in current build folder.
  #   Second command runs Doxygen.
  add_custom_target (doc ${CMAKE_COMMAND} -E make_directory docs
                     COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     COMMENT "Generating API documentation with Doxygen" VERBATIM
                     )
endif(DOXYGEN_FOUND)

DEAL_II_INVOKE_AUTOPILOT()