"""

# NOTE: https://github.com/alexattia/SimpsonRecognition/tree/fa65cc3124ed606e0ad6456ae49c734a2685db52
# NOTE: https://github.com/ngduyanhece/object_localization/blob/master/label_pointer.py

Files description
label_data.py : tools functions for notebooks + script to name characters from frames from .avi videos
label_pointer.py : point with mouse clicks to save bounding box coordinates on annotations text file (from already labeled pictures)
train.py : training simple convnet
train_frcnn.py -p annotation.txt : training Faster R-CNN with data from the annotation text file
test_frcnn.py -p path/test_data/ : testing Faster R-CNN

"""