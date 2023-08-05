cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(aplusb)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} libclew libutils)
