cmake_minimum_required(VERSION 3.8)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(CUDACmakeTemplate LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(CUDACmakeTemplate main.cpp kernel.cu)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDACmakeTemplate PROPERTY CXX_STANDARD 20)
  set_property(TARGET CUDACmakeTemplate PROPERTY CUDA_ARCHITECTURES 75)
endif()

# ensure proper CUDA linking and compilation settings
set_target_properties(CUDACmakeTemplate PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(CUDACmakeTemplate CUDA::cudart)

target_compile_options(CUDACmakeTemplate PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --use_fast_math>
)