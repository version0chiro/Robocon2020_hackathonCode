# SpOO_python
This is an implementation of the paper "Non-contact estimation of heart rate and oxygen saturation using ambient light" doi number :10.1364/BOE.6.000086 in python using openCV

# Installation:
  -dlib
  -openCV
  -Scipy
  -Numpy
  -Imutils

# Instruction:
To run the code, make a a video of targets face in frame and save it inside videos folder. Inside code spoo.py change the video capture input to the name of the video you have.
The frames will be loaded and go through fuctions in process:
  - The first frame will detect the face and make a mask to be used for rest of the frame.
  - The hsv range of skin colour has been provided so that the code will extract skin from the mask
  - The RGB mean is converted into a signal which is sent to SPOO estimation function
  - I_R and I_B is calucated using the Red and Blue channel
  - This is fed to the formula provided in the paper to esimated the spo2 level inside body
  
The file face_detection.py is called as a class to to find the mask for the first frame but can be used as a standalone file if someone wants to experiment.
 
# Tip: To get a video with desired frame number:
Use the record_vid.py script with inputting your desired number of frames in it. This will save a .avi file in the folder which can later be used by the main spoo.py file
