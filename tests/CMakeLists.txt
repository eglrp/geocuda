enable_testing()

file(GLOB TEST_SRC "*.hpp" "*.h" "*.cpp" "*.cu")
file(GLOB GEOCUDA_SRC "../geocuda/*")

include_directories(${GeographicLib_INCLUDE_DIRS})
cuda_add_executable(geocuda_test ${TEST_SRC} ${GEOCUDA_SRC})
target_link_libraries(geocuda_test ${GeographicLib_STATIC_LIBRARIES})

source_group(geocuda FILES ${GEOCUDA_SRC})
source_group(test FILES ${TEST_SRC})