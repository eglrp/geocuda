file(GLOB GEOCUDA_SRC "*.h" "*.cu" "*.cpp")

cuda_add_library(geocuda ${GEOCUDA_SRC})