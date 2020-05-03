
# Smart Social Distancing - Performance Metrics for Object Detection

This module provides easy-to-use functions which enables you to evaluate available models on **Smart Social Distancing** project.

### How to use
**Step 1: Download the Required Files**
```
cd neuralet/applications/smart-distancing/

# Download a sample video file from https://megapixels.cc/oxford_town_centre/
./download_sample_video.sh
pip install -r eval/requirements.txt
```
**Step 2: Extract Test Images**

To evaluate the town center dataset, you need to have access to test images first. We provide a python script named ```extract_towncenter.py``` that automatically reads the downloaded video (`TownCentreXVID.avi`) and creates a folder named ```test_images``` inside ```neuralet/applications/smart-distancing/data/``` directory and places the input images in this folder. Run the script below to extract test images:

```
cd neuralet/applications/smart-distancing/eval
python extract_towncentre.py
```
**Step 3: Export Xmls Files**

We provide a ground truth ```xmls.tar``` file that matches the extracted test images at the previous step. Follow the instructions below to extract the xmls ground truth file:

```
cd neuralet/applications/smart-distancing/eval
# Extract xmls.tar to the folder named 'xmls'
./extract_xmls.sh
```
**Step 4: Create the Ground Truth Files**

In order to perform the evaluation, you need to create a separate ground truth text file for each image. The command below creates the ground truth files with the correct format in a folder named `groundtruths/`.
```
cd neuralet/applications/smart-distancing/eval
python export_gt_from_xmls.py --xml_dir xmls/ --output_dir groundtruths
```
**Step 5: Set the Smart Social Distancing App to 'EVAL' Mode**

To evaluate **Smart Social Distancing** models, you should change the application mode to evaluation mode. The application will then read the images that are in ```neuralet/applications/smart-distancing/data/test_images/.``` directory and export the detection results into the directory which is set at the config file.
```
# Open applications/smart-distancing/[CONFIG.ini] ('config-skeleton.ini' for edgetpu device, 'config-jetson.ini' for jetson device) and set the parameters as follows
[APP]
InferenceMode: IMG_EVAL
[Evaluation]
ImagesPath: /repo/applications/smart-distancing/data/test_images
GtImageSize: 960,540,3
ResultDir: /repo/applications/smart-distancing/eval/detections_results
```
**Step 6: Run the Application**
Follow the instructions [here](https://github.com/neuralet/neuralet/blob/master/applications/smart-distancing/README.md) to run the **Smart Social Distancing App** based on the device you are using. After running the application, the detection results will be exported to ```neuralet/applications/smart-distancing/eval/detections_results/[DeviceName]-[ModeName]/``` directory.

**Step 7: Calculate mAP**
```
cd neuralet/applications/smart-distancing/eval
python pascal_evaluator.py -gt groundtruth/ --det detections_results/[DeviceName]-[ModeName]/ -t 0.5 -gtcoords abs -detcoords rel -imgsize (960,540)
#E.g. python pascal_evaluator.py -gt groundtruths/ --det detections_results/EdgeTPU-pedestrian_ssd_mobilenet_v2/ -t 0.5

```

## Results

The table below shows the performance of **Smart Social Distancing App** with several models on different devices.


| Model            | Device | IOU  | mAP  | FPS  |
| ---------------- | ------ | ---- | ---- | ---- |
| Mobilenet_SSD_V2 | AMD64-Coral USB Accelerator   | 0.5 | 34.91% | 94 |
| Pedestrian_SSD_MobileNet_V2 | AMD64-Coral USB Accelerator   | 0.5 | 84.01% | 151 |
| Pedestrian_SSDLite_MobileNet_V2 | AMD64-Coral USB Accelerator   | 0.5 | 84.29% | 141 |
| Mobilenet_SSD_V2 |  Coral Dev Board   | 0.5 | 34.91% | ~70 |
| Pedestrian_SSD_MobileNet_V2 |  Coral Dev Board   | 0.5 | 84.01% | ~180 |
| Pedestrian_SSDLite_MobileNet_V2 |  Coral Dev Board   | 0.5 | 84.29% | ~170 |
[More details of how mAP is calculated](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/README.md).[[1]](#1)

## References
<a id="1">[1]</a>
Rafael Padilla, Sergio Lima Netto and Eduardo A. B. da Silva  (2020). 
Survey on Performance Metrics for Object-Detection Algorithms. 
International Conference on Systems, Signals and Image Processing (IWSSIP)
 
