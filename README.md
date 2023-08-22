# YOLOv7 Inference and Comparison

This repository contains code to perform inference using the YOLOv7 model and compare the outputs between the original PyTorch model (.pt) and the ONNX converted model (.onnx).
Instead of linking a Google Colab Notebook, I've used different fenced code blocks for different code blocks in a notebook. Pre-trained weights (yolov7.pt, not yolov7-tiny.pt that is used in the [yolov7](https://github.com/WongKinYiu/yolov7) repository) has been used. Done on GPU environment

## Prerequisites

- Install necessary libraries using the following commands:
```
!pip install onnx
!pip install onnxruntime
#!pip install --ignore-installed PyYAML
#!pip install Pillow
!pip install protobuf<4.21.3
!pip install onnxruntime-gpu
!pip install onnx>=1.9.0
!pip install onnx-simplifier>=0.3.6 --user
```
- Check the version of your Python and PyTorch if required
```
import sys
import torch
print(f"Python version: {sys.version}, {sys.version_info} ")
print(f"Pytorch version: {torch.__version__} ")
```
- To check nvidia system management interface(monitor performance and usage of gpu)
```
!nvidia-smi
```
## Getting Started

### 1. Clone the YOLOv7 repository:
```
!git clone https://github.com/WongKinYiu/yolov7
```
### 2. Download the pre-trained weights:
```
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```
### 3. Perform inference using the YOLOv7 model:
```
!python detect.py --weights ./yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```
### 4. Viewing the image:
```
from PIL import Image
Image.open('/content/yolov7/runs/detect/exp/horses.jpg')
```

## Exporting to ONNX

### 1. Export the YOLOv7 model to ONNX format:
```
%cd /content/yolov7/
!python export.py --weights ./yolov7.pt
--grid --end2end --simplify
--topk-all 100 --iou-thres 0.65 --conf-thres 0.35
--img-size 640 640 --max-wh 640
```
## Inference with ONNX Model

### 1. Inference for ONNX model
```
import cv2
cuda = True
w = "/content/yolov7/yolov7.onnx"
img = cv2.imread('/content/yolov7/inference/images/horses.jpg')
```
### 2. Run the provided code to perform inference using the ONNX model and visualize the results.
```
import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image = img.copy()
image, ratio, dwdh = letterbox(image, auto=False)
image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)

im = image.astype(np.float32)
im /= 255
im.shape

outname = [i.name for i in session.get_outputs()]
outname

inname = [i.name for i in session.get_inputs()]
inname

inp = {inname[0]:im}
```
### 3. Viewing the image:
```
ori_images = [img.copy()]

for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
    image = ori_images[int(batch_id)]
    box = np.array([x0,y0,x1,y1])
    box -= np.array(dwdh*2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()
    cls_id = int(cls_id)
    score = round(float(score),3)
    name = names[cls_id]
    color = colors[name]
    name += ' '+str(score)
    cv2.rectangle(image,box[:2],box[2:],color,2)
    cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)

Image.fromarray(ori_images[0])
result_image = Image.fromarray(ori_images[0])
output_image_path = '/content/yolov7/onnx_images/horses.jpg'

result_image.save(output_image_path)
```

## Comparing Outputs

### 1. The raw data of these images is saved as binary files in the `runs/raw_data` directory.
```
/*new folders raw_data and pt have been created for convenience, make sure to create them or remove them from path.*/
pt_image_path = "/content/yolov7/runs/detect/exp/horses.jpg"
pt_inference_image = cv2.imread(pt_image_path)
pt_raw_data = pt_inference_image.tobytes()
# Save the raw data to a binary file
pt_raw_data_path = "/content/yolov7/runs/raw_data/pt/pt.bin"
with open(pt_raw_data_path,"wb") as file:
  file.write(pt_raw_data)
```

```
output_inference_image = cv2.imread(output_image_path)
output_raw_data = output_inference_image.tobytes()

# Save the raw data to a binary file
output_raw_data_path = "/content/yolov7/runs/raw_data/onnx/onnx.bin"
with open(output_raw_data_path, "wb") as file:
    file.write(output_raw_data)
```


### 2. The following script will print whether the raw data from the .pt and .onnx models are the same or different.
```
with open(pt_raw_data_path,"rb") as file:
  pt_raw_data = file.read()

with open(output_raw_data_path, "rb") as file:
    onnx_raw_data = file.read()

# Compare raw data
if pt_raw_data == onnx_raw_data:
    print("Raw data from .pt and .onnx models are the same.")
else:
    print("Raw data from .pt and .onnx models are different.")
```
### 3. Converting raw data to grayscale image:
```
def rd_to_img(rd,width,height):
  # Convert raw data to numpy array
  array = np.frombuffer(rd,dtype=np.uint8)
  # Reshape to image dimensions
  image_array = array.reshape((height, width, 3))
  # Create PIL Image
  image = Image.fromarray(image_array)
  return image

pt_image = rd_to_img(pt_raw_data,512,773)
onnx_image = rd_to_img(onnx_raw_data,512,773)
```
## Viewing Differences

### 1. The script will also visualize the difference between the grayscale versions of the .pt and .onnx inference images using the Pillow library.
```
from PIL import Image, ImageChops
import numpy as np

# Convert images to grayscale and compute the absolute difference
diff_image = ImageChops.difference(pt_image.convert("L"), onnx_image.convert("L"))
diff_image = diff_image.convert("RGB")

# Save and display the difference image
diff_image.save("difference_image.png")
diff_image.show()

```

### 2. The difference image will be saved as `difference_image.png` and displayed.

## Conclusion

By following the steps in this README, you can perform inference using the YOLOv7 model, export it to ONNX, and compare the outputs of both models. There are differences in inference image obtained and I've not been able to find out the exact reasons for them. The clearly visible differences in the inference images is floating-point precision, hence leading to small numerical variations in predictions. Since the input images for both the models are the same, the only other reasons I assume might be the cause of these differences are 
- issues while converting .pt to .onnx (unable to see any issues, will come back to it)
- different input preprocessing between .pt and .onnx model (working on it)
- PyTorch and ONNX Runtime might handle certain operations differently due to variations in backend libraries(will look into it soon)

The grayscale images only show the parts of the image that have variation in pixels
- White: Maximum difference possible
- Black: No difference at all(Identical)
- Greyscale levels between black and white:
  - Closer to Black being minimum difference
  - Closer to White being more difference
Ideally we should be getting a completely black image (will need to make sure bounding box, class id, etc have the same colours/fonts)

## Authors

Siddhartha A K
@siddhark1303@gmail.com

Siddhartha A K

## License

This project is licensed under the MIT License
    
