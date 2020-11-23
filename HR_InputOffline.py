from keras.models import load_model
from keras.models import model_from_json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize


with open('model.json', 'r') as f:
    model = model_from_json(f.read())
    print('Model successfully loaded')

# Load weights into the new model

#model.load_weights('model.h5')

model.load_weights('model.h5')

characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
                      'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                      'v', 'w', 'x', 'y', 'z']

#input citra uji
image = cv2.imread('b29.jpg')
height, width, depth = image.shape

# resizing the image to find spaces better
image = cv2.resize(image, dsize=(width * 5, height * 4), interpolation=cv2.INTER_CUBIC)
cv2.imshow('resize', image)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# binary
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# dilation
kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilation', img_dilation)

# adding GaussianBlur
gsblur = cv2.GaussianBlur(img_dilation, (5, 5), 0)
cv2.imshow('blur', gsblur)

# find contours
ctrs, hier = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

m = list()

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctrs: cv2.boundingRect(ctrs)[0])
pchl = list()
dp = image.copy()
# for i, ctr in enumerate(sorted_ctrs):
#     # Get bounding box
#     x, y, w, h = cv2.boundingRect(ctr)
#     cv2.rectangle(dp, (x - 10, y - 10), (x + w + 10, y + h + 10), (90, 0, 255), 9)

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
    roi = img_dilation[y - 10:y + h + 10, x - 10:x + w + 10]
    roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    roi = cv2.bitwise_not(roi)

    roi = np.array(roi)
    t = np.copy(roi)
    t = t / 255.0
    t = 1 - t
    t = t.reshape(1, 784)
    m.append(roi)
    pred = model.predict_classes(t)
    pchl.append(pred)

pcw = list()
interp = 'bilinear'
fig, axs = plt.subplots(nrows=len(sorted_ctrs), sharex=True, figsize=(1, len(sorted_ctrs)))
for i in range(len(sorted_ctrs)):
    # print (pchl[i][0])
    pcw.append(characters[pchl[i][0]])
    axs[i].set_title('-------> predicted letter: ' + characters[pchl[i][0]], x=2.5, y=0.24)
    axs[i].imshow(m[i], interpolation=interp)
plt.show()

predstring = ''.join(pcw)
print('Predicted String: ' + predstring)
