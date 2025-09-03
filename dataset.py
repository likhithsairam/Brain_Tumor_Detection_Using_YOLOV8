from roboflow import Roboflow

rf = Roboflow(api_key="zqgLAVkf1ebJFkJVOYVh")
project = rf.workspace("likhithworkspace").project("brain-tumor-detection-gos2k")
version = project.version(1)
dataset = version.download("yolov8")
