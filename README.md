# fast_mit_apriltags

This is a library for detecting AprilTags in images. It is a (faster) variant of the library originally written by Michael Kaess:
http://people.csail.mit.edu/kaess/apriltags/

It differs in the fact that some image processing operations, that were implemented "by hand" in the original library, have been replaced by OpenCV for improved performance. Additionally, a part of the computations was multithreaded.

The changes improve the performance from around 5 fps to around 15 fps for a 640x480 camera image on my machine.

# Publication

Ed Olson, AprilTag: A robust and flexible visual fiducial system, Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2011
