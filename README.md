# CARLA-Speed-Traffic-Sign-Detection-Using-Yolo
Final Master Thesis on how to detect CARLA Simulator's Speed Traffic Sign using YOLO v3 Neural Network for object detection

###### Thesis
Attached PDF: "Speed Traffic Sign Detection on the CARLA simulator using YOLO"

###### CARLA
Compiled version (0.8.2) for Windows (https://carla.readthedocs.io/en/stable/getting_started/)

Download modified manual_control.py script in order to detect the CARLA speed traffic signs.

###### Dataset
32.052 CARLA images (2.950 labelled and 29.102 unlabelled) mixing different sessions in different scenarios (Town01/Town02) and weather conditions (Sunny/Rainy/Cloudy). 

Download at https://drive.google.com/drive/folders/17x4_53WLIbxRN_6y2oBJ_ODnGYF8weWV?usp=sharing

###### Network
YOLOv3-tiny version pretrained to detect CARLA speed traffic signs.

Download the generated weights yolov3-tiny-obj_5000.weights file at the same Drive link (https://drive.google.com/drive/folders/17x4_53WLIbxRN_6y2oBJ_ODnGYF8weWV?usp=sharing).

Download the yolov3-tiny configured network at yolov3-tiny-obj.cfg file.

More details: http://pjreddie.com/darknet/yolo/.
