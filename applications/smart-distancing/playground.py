import cv2 as cv
from tqdm import tqdm

# video_uri = 'data/TownCentreXVID.avi'
video_uri = '/repo/applications/smart-distancing/data/TownCentreXVID.avi'
input_cap = cv.VideoCapture(video_uri)

out = cv.VideoWriter(
    'appsrc ! videoconvert ! '
    'vp8enc threads=4 deadline=1 ! webmmux streamable=true ! '
    'tcpserversink host=0.0.0.0 port=8080',
    0, 10, (640, 480)
)
print("isOpened=", out.isOpened())
if not out.isOpened():
    print(cv.getBuildInformation())
    exit(1)

t = tqdm()
while input_cap.isOpened():
    _, cv_image = input_cap.read()
    cv_image = cv.resize(cv_image, (640, 480))
    out.write(cv_image)
    t.update()

    # print(cv_image.shape)
    # break

