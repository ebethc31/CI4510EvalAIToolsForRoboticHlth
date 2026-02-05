YOLO


YOLO stream old -  version of code which takes in a video input and returns an output file with bounding boxes around object

YOLO stream - adjusted version of YOLO stream old which takes in feed from the computer webcam and HYPOTHETICALLY continues to track objects and create an output video until the button q is pressed. 
        * This does not yet work - it is able to connect to the webcam and continue to track, but pressing the q button to terminate does not work. 
        * IDEA TO FIX THIS ERROR - The OpenCV pop-up is not popping up for some reason (fix that!) 
                                 - Create new terminal and call the code using 'python YOLO stream old.py' to run the program? Apparently that may work instead? 


Conda installation instructions

1. Create a new conda env
Open a terminal and change directory to the Yolo folder. 
`conda create -n yolo python=3.10 pip`

2. Activate the environment
`conda activate yolo`

3. Install the required packages
`pip install -r requirements.txt`

4. Select the conda environment in vscode by opening a python file then clicking the python version on the bottom right of the interface and selecting the "yolo" environment we just created. 