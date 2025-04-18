cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(examples)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(xTorch REQUIRED)
#find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} ${TRANSFORM_FILES} main.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
#target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

#target_link_libraries(examples PRIVATE xTorch::xTorch)
#target_include_directories(your_target PRIVATE ${XTORCH_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
#        OpenSSL::SSL
#        OpenSSL::Crypto
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${TORCH_LIBRARIES}
#        xTorch::xTorch
)


#add_executable(${PROJECT_NAME} main.cpp)
#
#target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
#set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
#message(STATUS "CMAKE_PROJECT_NAME:${CMAKE_PROJECT_NAME}")
#message(STATUS "PROJECT_NAME:${PROJECT_NAME}")
