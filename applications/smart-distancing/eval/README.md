# Smart Social Distancing - Performance Metrics for Object Detection

This module provides easy-to-use functions which enables you to evaluate available exported results from detectors.

### How to use
**Step 1: Build Image**
```
cd neuralet/applications/smart-distancing/eval
# Build Docker image
sudo docker build -f Dockerfile-eval -t detector-eval .
```
**Step 2: Create The Input Files**

* Create a separate ground truth text file for each image in the directory `neuralet/applications/smart-distancing/eval/eval_files/groundtruths/`.
* In these files each line should be in the format: `<class_name> <left> <top> <width> <height>`.
* E.g. A sample of ground truth bounding boxes of an image:
```
bottle 6 234 39 128
person 1 156 102 180
person 36 111 162 305
person 91 42 247 458
```
* Create a separate detection text file for each image in the directory `neuralet/applications/smart-distancing/eval/eval_files/detresults/`
* The names of the detection files must match their correspond ground truth (e.g. "detections/2008_000182.txt" represents the detections of the ground truth: "groundtruths/2008_000182.txt").
* In these files each line should be in the following format: `<class_name> <confidence> <left> <top> <width> <height>`

**Step 3: Run The Docker Image**
```
cd neuralet/applications/smart-distancing/eval
# Run Docker image
sudo docker run -it -v /PATH_TO_CLONED_REPO_ROOT/:/repo detector-eval
```
You can also run the docker image with optional arguments
```
sudo docker run -it -v $(pwd)/../../../:/repo detector-eval -gt 'eval_files/groundtruths_1' -det eval_files/detresults_1 -t 0.75
```


| Argument | Description | Defauls |
| -------- | -------- | -------- |
| `-gt` | folder that contains the ground truth bounding boxes files (Must be located at /eval directory)     | `/eval/eval_files/groundtruths`     |
| `-det` | folder that contains your detected bounding boxes files (Must be located at /eval directory)     | `/eval/eval_files/detresults`     |
| `-t` | IOU thershold that tells if a detection is TP or FP     | `0.50`     |

[More details of how mAP is calculated](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/README.md).[[1]](#1)

## References
<a id="1">[1]</a>
Rafael Padilla, Sergio Lima Netto and Eduardo A. B. da Silva  (2020). 
Survey on Performance Metrics for Object-Detection Algorithms. 
International Conference on Systems, Signals and Image Processing (IWSSIP)
 
