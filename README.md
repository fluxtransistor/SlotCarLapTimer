# SlotCars

This is a Python program with OpenCV that grabs a webcam image and uses it to count lap times on a slot car racetrack.

The camera needs to be positioned above the finish line in stable lighting conditions. 
The program will detect the car based on its colour and count lap times for it.
If the car detection does not work as expected, for example detecting the same car as a different one,
the single-car branch offers basic functionality while only supporting one car at a time.
