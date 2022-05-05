This is a tool to visualize bounding boxes on videos.

Input is a json format with bounding boxes (predictions and ground truth) and file names. The Flask server will generate images with boxes overlay on the images and from a browser you can switch the images to display. 

Ground truth boxes are in white colors, and prediction boxes are displayed in blue (least confident) to red (most confident)

To install, run

`conda create --name flask python==3.7`

`conda activate flask`

`pip install flask requests matplotlib pydicom pandas imageio opencv-python`

To start server, run:

`conda activate flask`

`export FLASK_APP=/local/home/li/video_checker/app.py`

`flask run --host=0.0.0.0 --port=6001`

From browser enter your ip:6001, for example,

http://10.228.91.54:6001/

`