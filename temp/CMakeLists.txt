cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(temp)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)


#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBZIP REQUIRED libzip)
#
#include_directories(${LIBZIP_INCLUDE_DIRS})
#link_directories(${LIBZIP_LIBRARY_DIRS})


find_package(Torch REQUIRED)
find_package(xTorch REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SndFile REQUIRED)
#find_package(onnxruntime REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} ${TRANSFORM_FILES})

set(TEST_LIBRARIES ${CURL_LIBRARIES}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${XTORCH_LIBRARIES}
        xTorch::xTorch
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
        #        ${ONNXRUNTIME_LIBRARY}
        #        onnxruntime
        ONNX::Runtime
        ${LIBZIP_LIBRARIES}  # Add this line
)

#========================================= celeb =========================================

add_executable(celeb ${TEST_SOURCE_FILES} celeb_test.cpp)
target_link_libraries(celeb "${TORCH_LIBRARIES}")
set_property(TARGET celeb PROPERTY CXX_STANDARD 17)
target_include_directories(celeb PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(celeb ${TEST_LIBRARIES})
target_include_directories(celeb PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})
target_link_libraries(celeb ${TEST_LIBRARIES})



#========================================= classifying_handwritten_digits_with_lenet_on_mnist =========================================

add_executable(py_test ${TEST_SOURCE_FILES} py_exec_test_3.cpp)
target_link_libraries(py_test "${TORCH_LIBRARIES}")
set_property(TARGET py_test PROPERTY CXX_STANDARD 17)
target_include_directories(py_test PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(py_test ${TEST_LIBRARIES})
target_include_directories(py_test PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})
target_link_libraries(py_test ${TEST_LIBRARIES})


#
##========================================= fine_tuning_resnet_on_cifar10 =========================================
#
#add_executable(fine_tuning_resnet_on_cifar10 ${TEST_SOURCE_FILES} fine_tuning_resnet_on_cifar10.cpp)
#target_link_libraries(fine_tuning_resnet_on_cifar10 "${TORCH_LIBRARIES}")
#set_property(TARGET fine_tuning_resnet_on_cifar10 PROPERTY CXX_STANDARD 17)
#target_include_directories(fine_tuning_resnet_on_cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
#target_link_libraries(fine_tuning_resnet_on_cifar10 ${TEST_LIBRARIES})
#target_include_directories(fine_tuning_resnet_on_cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})
#target_link_libraries(fine_tuning_resnet_on_cifar10 ${TEST_LIBRARIES})
#
#
#
#
##