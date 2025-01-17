from ultralytics import YOLO

# model = YOLO("yolov8n-pose.pt")
# results = model.train(data="mypose.yaml",epochs=200,imgsz=640,batch=4,workers=0)

model = YOLO("/home/gpu/zhanghongxing/spaper/v8/yamlfile/se9.yaml").load("/home/gpu/zhanghongxing/spaper/v8/yolov8n-pose.pt")
model.train(data="/home/gpu/zhanghongxing/spaper/v8/mypose.yaml",pretrained=True,epochs=1000,task='position',name='se9',batch=16,workers=0)


#model = YOLO("/home/gpu/zhanghongxing/spaper/v8/yolov8n-pose.pt")
#model.train(data="/home/gpu/zhanghongxing/spaper/v8/mypose.yaml",pretrained=True,epochs=800,name='noelanoloss',batch=16,workers=0)

