# 11.5

`python cv.py` in local virtual env: (deep-learning).
'./checkpoints/weights_11_2.h5': 512*512 image size: 【best】
from data_keras_512 import load_train_data, load_test_data
from main_keras_512 import get_unet
img[mask > 0.6] = (0, 0, 255)

## demo
- put down the webcam angle. stand further. 
- be sure to roll out my sleeves during live demo: segmentation won't cover my skins. 

# 11.3

- display only the masked area of the streaming frames. 
- stream from webcam to video on local.
- and pick the most representative 3 frames.

presentation:
- show original dataset, the effect we want to achieve.
- show neural network architecture, the u-net paper.


- our app is your virual closet.
- you can easily swap in any clothes you like. 


our main goal is: improve online shopping experience in fashion industry.
we want to recognize and save your favourite cloths into your virtual closet, with one snap. 