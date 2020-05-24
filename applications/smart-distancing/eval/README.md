# Smart Social Distancing - Performance Metrics for Object Detection

This module provides easy-to-use functions which enables you to evaluate available exported results from detectors.

### How to use
**Step 1: Build Image**
```
cd neuralet/applications/smart-distancing/eval
# Build Docker image
docker build -f Dockerfile-eval -t detector-eval .
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
docker run -it -v /PATH_TO_CLONED_REPO_ROOT/:/repo detector-eval
```
You can also run the docker image with optional arguments
```
docker run -it -v /PATH_TO_CLONED_REPO_ROOT/:/repo detector-eval -gt 'eval_files/groundtruths_1' -det eval_files/detresults_1 -t 0.75
```


| Argument | Description | Default |
| -------- | -------- | -------- |
| `-gt` | folder that contains the ground truth bounding boxes files (Must be located at /eval directory)     | `/eval_files/groundtruths`     |
| `-det` | folder that contains your detected bounding boxes files (Must be located at /eval directory)     | `/eval_files/detresults`     |
| `-t` | IOU thershold that tells if a detection is TP or FP     | `0.50`     |



### Run on Edgetpu Devices
It's also possible to evaluate a quantized tflite model on edgetpu, below is the instruction for evaluting models on edgetpu devices.

#### Run on AMD64 node with a connected Coral USB Accelerator
Create a separate ground truth text file for each image in the directory `neuralet/applications/smart-distancing/eval/eval_files/groundtruths/`.

Create a .txt file and add the classId and its name to that file.
E.g.
```
# PATH/classes.txt
0:face
1:face-mask
```
Build and Run Docker Image
```
cd neuralet/applications/smart-distancing/eval
# Build Docker image
docker build -f Dockerfile-amd64-usbtpu-eval -t detector-eval-usbtpu .
docker run -it -v /PATH_TO_CLONED_REPO_ROOT/:/repo detector-eval-usbtpu --model_path 'PATH/model.tflite' --classes 'PATH/classes.txt' --minscore '0.25' --img_path 'PATH/test_imgs' --img_size '300,300,3' --result_dir 'PATH/detresults/' -gt 'PATH/groundtruths' -t '0.5'
```
| Argument | Description | Default |
| -------- | -------- | -------- |
| `--model_path` | the path of tflite model    | `edgetpu/data/mobilenet_v1_face_mask_edgetpu.tflite`     | 
| `--classes` | the path of .txt files contains class Ids and its name     | `eval_files/sample_classes.txt`     |
| `--minscore` | the minimum confidence score for detecting objects     | `0.25`     |
| `--img_path` | folder that contains the validation set data (images)     | `val_images`     |
| `--img_size` | the input size of the model  | `300,300,3`     |
| `--result_dir` | folder that the detector results will be    | `/eval_files/detresults`     |
| `-gt` | folder that contains the ground truth bounding boxes files     | `/eval_files/groundtruths`     |
| `-t` | IOU thershold that tells if a detection is TP or FP     | `0.50`     |

[More details of how mAP is calculated](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/README.md).[[1]](#1)
## References
<a id="1">[1]</a>
Rafael Padilla, Sergio Lima Netto and Eduardo A. B. da Silva  (2020). 
Survey on Performance Metrics for Object-Detection Algorithms. 
International Conference on Systems, Signals and Image Processing (IWSSIP)
