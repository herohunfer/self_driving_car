# Image Based Object Detection System for Self-driving Cars Application
This project is based on **YOLO V1 & YOLO V2**  
This repository used code from [YOLO V2](https://pjreddie.com/darknet/yolo/)


## Introduction
[Competition](https://www.bittiger.io/competition/evaluation)This project is to perform detection on 4 classes of objects (Vehicle, Pedestrian, Cyclist and Traffict Lights) on the road based on 10,000 training image. The model uses YOLO V1 MXNET and YOLO V2 Darknet framework.


## Result
For YOLO V1 please check [demo_test.ipynb]().  I ran 600 epochs and here is [detect_full_scale-0600.params]().  
For YOLO V2 please run following command on test image:
```
./darknet detector test cfg/obj.data cfg/capstone.cfg backup/capstone_40000.weights testing/70495.jpg
```
alternatively, you can run this command to convert all validation set and store results in results/ folder:  
```
./darknet detector valid cfg/obj.data cfg/capstone.cfg backup/capstone_4
0000.weights -i 0
```
This can be used to get final output and calculate mean AP.  

Thanks to John, we have two great test videos:  
[Downtown Video](https://www.youtube.com/watch?v=50Uf_T12OGY)   
[Highway Video](https://www.youtube.com/watch?v=GMtusG5tuC8&t=2s)  
And run the following commands to check:
```
cd darknet_alex/darknet
./darknet detector demo cfg/obj.data cfg/capstone.cfg ../../yolo\ v2/darknet/backup/capstone_40000.weights ~/Downloads/Driving\ Downtown\ -\ Toronto’s\ Main\ Street\ -\ Toronto\ Canada.mp4 -out_filename downtown.avi

./darknet detector demo cfg/obj.data cfg/capstone.cfg ../../yolo\ v2/darknet/backup/capstone_40000.weights ~/Downloads/Highway\ 401\ Through\ Toronto\ Worlds\ Busiest\ Freeway.mp4 -out_filename highway.avi
```


## Installation

1. For YOLO V2 please build darknet:
```
cd darknet
make
```

## Citing  
```
@article{redmon2016yolo9000,
  title={YOLO9000: Better, Faster, Stronger},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1612.08242},
  year={2016}
}
```
