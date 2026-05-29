cmake_minimum_required(VERSION 3.20)
project(MLIR_Educational_Example LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

set(MLIR_EDU_SOURCES
  lib/Dialect/MyHW/MyHWDialect.cpp
  lib/Conversion/StableHloLiteToMyHW.cpp
  lib/Conversion/MyHWToLLVM.cpp
)

add_library(MLIREducationalExample STATIC ${MLIR_EDU_SOURCES})
llvm_update_compile_flags(MLIREducationalExample)

target_link_libraries(MLIREducationalExample
  PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRPass
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
)

add_executable(mlir-educational-example
  tools/mlir-educational-example.cpp
)
llvm_update_compile_flags(mlir-educational-example)
target_link_libraries(mlir-educational-example PRIVATE MLIREducationalExample)

set_target_properties(mlir-educational-example PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
)

install(TARGETS mlir-educational-example
  RUNTIME DESTINATION bin
)

add_custom_target(run-smoke
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/compile_pipeline.py
          --input ${PROJECT_SOURCE_DIR}/Tiny_NN_Recent.mlir
          --work-dir ${PROJECT_SOURCE_DIR}/build/smoke
          --emit-executable
  DEPENDS mlir-educational-example
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

enable_testing()
add_test(NAME compile-smoke
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/compile_pipeline.py
          --input ${PROJECT_SOURCE_DIR}/Tiny_NN_Recent.mlir
          --work-dir ${PROJECT_SOURCE_DIR}/build/ctest-smoke
)
