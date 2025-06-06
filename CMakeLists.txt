cmake_minimum_required(VERSION 3.8)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(CUDA_neural_net LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(CUDA_neural_net main.cpp src/kernel.cu src/headers/structs.cuh src/headers/vectorcu.cuh)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDA_neural_net PROPERTY CXX_STANDARD 20)
  set_property(TARGET CUDA_neural_net PROPERTY CUDA_ARCHITECTURES 75)
endif()

# ensure proper CUDA linking and compilation settings
set_target_properties(CUDA_neural_net PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(CUDA_neural_net CUDA::cudart)

target_compile_options(CUDA_neural_net PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --use_fast_math>
)