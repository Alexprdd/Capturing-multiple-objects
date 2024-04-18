import cv2
import numpy

config_file="ssd_mobile.pbtxt"
frozen_model="frozen_inference_graph.pb"
model=cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels=[]
yolo="yolo3.txt"
file=open(yolo,"rt")
classLabels=file.read().rstrip("\n").split("\n")

print(classLabels)

#model training

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

image=cv2.imread("Image2.jpg")
img_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB,)
cv2.imshow("image", img_rgb)
cv2.waitKey(0)

print(model.detect(img_rgb,confThreshold=0.5))

index, confidence, box=model.detect(img_rgb,confThreshold=0.5)

for i, confidence, boxes in zip(index.flatten(),confidence.flatten(),box):
    if i<80:
        cv2.rectangle(image,boxes,color=(45,45,45),thickness=2)
        cv2.putText(image,classLabels[i-1],(boxes[0]+10,boxes[1]+40), cv2.FONT_ITALIC,fontScale=1,color=(100,35,233),thickness=2)

cv2.imshow("final",image)
cv2.waitKey(0)