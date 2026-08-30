cmake_minimum_required(VERSION 3.20)
project(moge2_native VERSION 0.1.1 LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../native_support/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)
option(MOGE2_WITH_METAL "Build the Apple Metal/MPSGraph executor" ${APPLE})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
find_package(Vulkan REQUIRED COMPONENTS glslc)
set(MOGE2_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MOGE2_GENERATED_DIR}")
function(moge2_shader NAME)
  set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
  set(SPIRV "${MOGE2_GENERATED_DIR}/${NAME}.spv")
  set(HEADER "${MOGE2_GENERATED_DIR}/${NAME}_spv.h")
  add_custom_command(OUTPUT "${SPIRV}"
    COMMAND Vulkan::glslc -fshader-stage=compute --target-env=vulkan1.3
      --target-spv=spv1.6 -Werror "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
      -o "${SPIRV}" "${SOURCE}"
    DEPENDS "${SOURCE}" VERBATIM)
  add_custom_command(OUTPUT "${HEADER}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${SPIRV}" "-DOUTPUT=${HEADER}"
      "-DVARIABLE=da3_${NAME}_spv"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
    DEPENDS "${SPIRV}" VERBATIM)
  set(MOGE2_SHADER_HEADERS ${MOGE2_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
endfunction()
foreach(SHADER add_scaled add add_position bilinear_align_true
    bilinear_align_true_image bilinear_align_false bmm bmm_score_half bmm_value_half conv2d conv2d8
    conv2d_half conv2d8_half conv_transpose_nonoverlap
    conv2d_replicate concat_uv remap_points_mask solve_focal_shift final_depth
    final_depth_image preprocess_moge_texture
    conv_transpose_nonoverlap_half elementwise gelu layer_norm linear linear16
    linear_vec8 linear_half linear16_half prepare_tokens position_bicubic
    project_tokens project_tokens_half relu softmax_lastdim softmax_lastdim_half
    replace_token qk_rope capture_concat add_uv exponential tokens_to_nchw)
  moge2_shader(${SHADER})
endforeach()
foreach(SHARED_SHADER quantize_rows_int8 linear_int8_tiled)
  inferbridge_compile_native_support_shader(
    NAME "${SHARED_SHADER}" PREFIX da3
    GENERATED_DIR "${MOGE2_GENERATED_DIR}"
    EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
    OUT_VAR MOGE2_SHADER_HEADERS)
endforeach()

add_executable(moge2_gpu_encoder_probe
  tools/gpu_encoder_probe.cpp src/encoder_gpu.cpp src/model_config.cpp src/gpu_model.cpp
  src/operators.cpp src/safetensors.cpp src/vulkan.cpp ${MOGE2_SHADER_HEADERS})
target_include_directories(moge2_gpu_encoder_probe PRIVATE
  src "${MOGE2_GENERATED_DIR}")
target_link_libraries(moge2_gpu_encoder_probe PRIVATE Vulkan::Vulkan)
if(WIN32)
  target_link_libraries(moge2_gpu_encoder_probe PRIVATE d3d12 dxgi)
endif()

set(MOGE2_NATIVE_SOURCES
  src/inferbridge_harness.cpp src/external_gpu.cpp src/gpu_preprocess.cpp
  src/graph_gpu.cpp src/encoder_gpu.cpp src/model_config.cpp src/moge_operators.cpp
  src/gpu_model.cpp src/operators.cpp src/safetensors.cpp src/vulkan.cpp
  ${MOGE2_SHADER_HEADERS})
if(MOGE2_WITH_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "MOGE2_WITH_METAL requires an Apple platform")
  endif()
  list(APPEND MOGE2_NATIVE_SOURCES src/metal_external_gpu.mm)
endif()
function(moge2_native_target TARGET_NAME HARNESS_ID)
  add_library(${TARGET_NAME} SHARED ${MOGE2_NATIVE_SOURCES})
  target_include_directories(${TARGET_NAME} PUBLIC include PRIVATE
    src "${MOGE2_GENERATED_DIR}")
  target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
  if(MOGE2_WITH_METAL)
    target_link_libraries(${TARGET_NAME} PRIVATE
      "-framework Foundation" "-framework Metal"
      "-framework MetalPerformanceShadersGraph")
    target_compile_definitions(${TARGET_NAME} PRIVATE MOGE2_WITH_METAL=1)
  endif()
  target_compile_definitions(${TARGET_NAME} PRIVATE
    INFERBRIDGE_HARNESS_BUILDING_LIBRARY MOGE2_BUILD_DLL
    "MOGE2_HARNESS_ID=\"${HARNESS_ID}\"")
  if(WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE d3d12 dxgi)
  endif()
  if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /W4 /permissive-)
  endif()
endfunction()
moge2_native_target(moge2_native "inferbridge.moge-2.native")
moge2_native_target(metricanything_native "inferbridge.metricanything.native")

include(CTest)
if(BUILD_TESTING AND APPLE AND MOGE2_WITH_METAL)
  add_executable(moge2_metal_full_graph tests/metal_full_graph.cpp)
  target_include_directories(moge2_metal_full_graph PRIVATE include src)
  target_link_libraries(moge2_metal_full_graph PRIVATE moge2_native)
  add_test(NAME moge2_metal_full_graph COMMAND moge2_metal_full_graph)
  set_tests_properties(moge2_metal_full_graph PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(WIN32)
  add_executable(moge2_d3d12_full_graph tests/d3d12_full_graph.cpp)
  target_include_directories(moge2_d3d12_full_graph PRIVATE include)
  target_link_libraries(moge2_d3d12_full_graph PRIVATE
    moge2_native d3d12 dxgi)
  if(MSVC)
    target_compile_options(moge2_d3d12_full_graph PRIVATE /W4 /permissive-)
  endif()
  add_executable(metricanything_d3d12_full_graph tests/d3d12_full_graph.cpp)
  target_include_directories(metricanything_d3d12_full_graph PRIVATE include)
  target_link_libraries(metricanything_d3d12_full_graph PRIVATE
    metricanything_native d3d12 dxgi)
  if(MSVC)
    target_compile_options(metricanything_d3d12_full_graph PRIVATE /W4 /permissive-)
  endif()
endif()

add_executable(moge2_gpu_full_graph_probe
  tools/gpu_full_graph_probe.cpp src/graph_gpu.cpp src/encoder_gpu.cpp src/model_config.cpp
  src/moge_operators.cpp src/gpu_model.cpp src/operators.cpp
  src/safetensors.cpp src/vulkan.cpp ${MOGE2_SHADER_HEADERS})
target_include_directories(moge2_gpu_full_graph_probe PRIVATE
  src "${MOGE2_GENERATED_DIR}")
target_link_libraries(moge2_gpu_full_graph_probe PRIVATE Vulkan::Vulkan)
if(WIN32)
  target_link_libraries(moge2_gpu_full_graph_probe PRIVATE d3d12 dxgi)
endif()
if(MSVC)
  target_compile_options(moge2_gpu_full_graph_probe PRIVATE /W4 /permissive-)
endif()
if(MSVC)
  target_compile_options(moge2_gpu_encoder_probe PRIVATE /W4 /permissive-)
endif()
