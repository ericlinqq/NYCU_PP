cmake_minimum_required(VERSION 2.8)
project( DisplayImage LANGUAGES C CXX)
# This is set separately because nvcc doesn't understand these flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++17")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage BM_pthread.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(DisplayImage PRIVATE Threads::Threads)