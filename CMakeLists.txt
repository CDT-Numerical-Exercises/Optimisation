# Setup project
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(BaseProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set build type variable in preprocessor
if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _upper_build_type)
    add_compile_definitions(BUILD_${_upper_build_type}=1)
endif()

add_compile_definitions(HAVE_INLINE=1)

OPTION (VERBOSE_LINESEARCH "Print during iterations of the linesearch algorithm (mostly for debugging purposes)" OFF)
IF(VERBOSE_LINESEARCH)
  add_compile_definitions(VERBOSE_LINESEARCH=1)
ENDIF()

# add include dir
include_directories(${PROJECT_SOURCE_DIR}/include)

# add libraries
link_directories(${PROJECT_SOURCE_DIR}/lib)

# add sources and headers
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.h)

# add executable
add_executable(quadratic_example quadratic_example.cpp ${sources} ${headers})
add_executable(problem1 problem1.cpp ${sources} ${headers})
add_executable(problem2 problem2.cpp ${sources} ${headers})
add_executable(problem3 problem3.cpp ${sources} ${headers})

# add libraries
find_package(GSL REQUIRED)

# add Boost libraries
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.71.0 COMPONENTS iostreams system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(quadratic_example GSL::gsl GSL::gslcblas ${Boost_LIBRARIES})
target_link_libraries(problem1 GSL::gsl GSL::gslcblas ${Boost_LIBRARIES})
target_link_libraries(problem2 GSL::gsl GSL::gslcblas ${Boost_LIBRARIES})
target_link_libraries(problem3 GSL::gsl GSL::gslcblas ${Boost_LIBRARIES})
