enable_testing()

file(GLOB TEST_SRC "*.hpp" "*.h" "*.cpp" "*.cu")

include_directories(${GeographicLib_INCLUDE_DIRS})
cuda_add_executable(geocuda_test ${TEST_SRC})
target_link_libraries(geocuda_test geocuda ${GeographicLib_STATIC_LIBRARIES})
