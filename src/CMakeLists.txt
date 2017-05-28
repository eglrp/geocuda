include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(geocuda)

if(GEOCUDA_BUILD_TESTS)
    add_subdirectory(tests)    
endif(GEOCUDA_BUILD_TESTS)
