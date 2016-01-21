# fast_mit_apriltags

This is a C++11 library for detecting AprilTags in images. It is a (faster) variant of the excellent library originally written by Michael Kaess:
[http://people.csail.mit.edu/kaess/apriltags/](http://people.csail.mit.edu/kaess/apriltags/).

It only differs in the fact that some image processing operations, that were implemented "by hand" in the original library, have been replaced by OpenCV for improved performance. Additionally, a part of the computations was multithreaded using `std::thread`.

The changes improve the performance from around 5 fps to around 15 fps for a 640x480 camera image on my machine.

# Copyright and License

This library is based on the implementaton by [Michael Kaess](http://people.csail.mit.edu/kaess/apriltags/). Just like his library, this library is available as open source under GNU LGPL version 2.1.

# Publication

Ed Olson, [AprilTag: A robust and flexible visual fiducial system](http://april.eecs.umich.edu/papers/details.php?name=olson2011tags), Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2011
