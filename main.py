from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8m.pt")
    results = model.train(data="Brain-Tumor-Detection-1/data.yaml",
                          epochs=50,
                          imgsz=640,
                          batch=8,
                          device=0
                          )


