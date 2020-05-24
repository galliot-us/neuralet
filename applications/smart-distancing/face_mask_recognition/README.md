# Smart Social Distancing - Face Mask Classifier
Face-Mask classifier is simple classifier which is implemented using Tensorflow and Keras to classifiey face and face-mask images.
## Usage

### Install requirements
```
# cd neuralet/applications/smart-distancing/face_mask_recognition/
pip install -r requirements.txt
```
### Train The Model
The train and validation folders should contain 'n' folders each containing images of respective classes.
For example:
```
# train/
# |_face
# |    |__face1.jpg
# |    |__face2.jpg
# |_face-mask
#      |__face-mask1.jpg
#      |__face-mask2.jpg
```

```
# cd neuralet/applications/smart-distancing/face_mask_recognition/
python face_mask_classifier.py --train_dir PATH_TO_TRAIN_IMG_FOLDER/ --validation_dir PATH_TO_VAL_IMG_FOLDER/
```
| Argument | Description | Default |
| -------- | -------- | -------- |
| `--train_dir` | folder that contains the train images  | None     |
| `--validation_dir` | folder that contains the validation images   |  None    |
| `--input_size` | the size of model's input     | 224     |
| `--no_channels` | number of channel for single each image     | 3     |
| `--no_classes` | number of classes     | 2     |
| `--epoch` | number of epochs    | 25     |
| `--save_dir` | the path of exported model    | `saved_model`     |
| `--model_file_name` | keras exported .h5 file name     | `model.h5`     |
| `--learning_rate` | the optimizer learning rate     | 0.005    |
| `--batch_size` | batch size     | 64     |
| `--result_dir` | the path of exporting confusion matrix and evalution resuls png files    | `/results`     |
| `--classes` | list of class names   | `["face", "face-mask"]`     |
| `--pretrained` | if you set it to Ture a pretrained model will be downloaded     | True     |

### Confusion Matrix

Below is the confusion matrix exported after training the face-mask classifier
![Confusion Matrix](https://github.com/mrn-mln/neuralet/blob/face-mask-classifier/applications/smart-distancing/face_mask_recognition/results/confusion_matrix.png?raw=true)

### Evalution Results
Below is the sample result of evalution images. The text under each images correponds the Label/Prediction
![image alt](https://github.com/mrn-mln/neuralet/blob/face-mask-classifier/applications/smart-distancing/face_mask_recognition/results/eval_results.png?raw=true)

### Export Inference Graph
In order to run inference on the trained model, the model should be frozen as follows:
```
# cd neuralet/applications/smart-distancing/face_mask_recognition/
python export_inference_graph.py --export_path inference_model/ --model_path saved_model/model.h5
```

### Inference
After exporting a frozen graph you can easly run inference as follows:
```
# cd neuralet/applications/smart-distancing/face_mask_recognition/
python inference.py --model_dir inference_model/ --input_size 224 --imgs_dir TEST_IMAGES_DIR/ --classes ["face", "face-mask"] --result_dir results/
```
Below is a sample of results after running inference on test images:
![](https://github.com/mrn-mln/neuralet/blob/face-mask-classifier/applications/smart-distancing/face_mask_recognition/results/inference_result.png?raw=true)
