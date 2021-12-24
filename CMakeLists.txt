project(sideSurface)
find_package( OpenCV REQUIRED )
add_executable( sideSurface demo.cpp side_box_surface_blur.cpp)
target_link_libraries( sideSurface ${OpenCV_LIBS} )