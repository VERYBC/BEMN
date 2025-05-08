# <div align="center">BEMN</div>
This repository contains the dataset and the code for our paper **BEMN: Balanced Bias Enhanced Multi-Branch Networkfor Cross-View Geo-Localization**. Thank you for your kindly attention.
## Test on Xian-37
1. Download the model weights trained on University-1652 and place them in the **checkpoint** folder
2. run eval_xian.py to test the performance of the model on Xian-37
## About Xian-37
Xian-37 dataset comprises drone, ground and satellite images collected from 37 prominent landmarks in Xi’an, China. The drone and ground images are captured in real-world settings, with variations in viewing angles, acquisition times, imaging devices, and the resolution of the images, making the dataset highly challenging. The satellite images are extracted from Google Maps and captured at different times. For each scene, the dataset provides five images from drone or ground perspectives, with drone views being more prevalent, along with one corresponding satellite reference image.

The structure of this dataset is similar to University-1652:
<pre>├── Xian-37/
|   ├── drone/                /* drone-view images
|       ├── 0001
|       ├── 0002
|       ...
|       ├── 0037
|   ├── satellite/            /* satellite-view images
|       ├── 0001
|       ├── 0002
|       ...
|       ├── 0037</pre>
## Download pre-training weights
You can download the pre-training weights for the first training stage on University-1652 from the following link:

[Google Driver](https://drive.google.com/file/d/1D5wBxH0No2I8KePcxfZFhsF8Gr5PAN8u/view?usp=sharing)
