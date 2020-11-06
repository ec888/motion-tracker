motion-tracker
==============
Track ant movement. Input is a video; output is a list of ant tracks and a output video with the tracks added to the original video. I use the ant tracker code for an independent research project on ant behaviors.

Motion tracking of moving objects in video using OpenCV 3.0.0

Ported the original repo from OpenCV 3.0.0 and python 2 to python 3, and cleaned-up + tuned the code and added documentation. 

The algorithm: 

| Step | Description                                                                                                                                                                                                            | OpenCV code                            |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| 1    | Read image from video                                                                                                                                                                                                  |                                        |
| 2    | Change to grayscale                                                                                                                                                                                                    | cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)  |
| 3    | Feed frames into background/motion detector. The detector calculates the “background” from multiple frames, and uses it to identify which “foreground” objects are moving. The output is the mask of changed pixels. | cv2.createBackgroundSubtractorMOG2     |
| 4    | Blur and threshold to reduce noise                                                                                                                                                                                     | cv2.filter2D(..., kenel) cv2.threshold |
| 5    | Detect contours and the center of the contours                                                                                                                                                                         | cv2.findContours                       |
| 6    | Calc path:  From accumulated paths in past iterations, if the new center-point is a distance of less than r away, add the closest center point to the paths list. Paths that are not continued are archived.                    |                                        |

To use: 
1. install open CV. 
2. run:
python ProcVideo.py ant_example.mov out.avi
