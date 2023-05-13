import os
import argparse
from dotenv import load_dotenv
from tracker import Tracker

if __name__ == '__main__':
    load_dotenv()
    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="path to yolo trained weights")
    ap.add_argument("--conf", type=float, default=0.2, help="minimum probability to filter weak detections")
    ap.add_argument("--sample", required=True, help="video sample filename")
    ap.add_argument("--oname", required=True, help="object to detect")
    ap.add_argument("--owidth", type=float, required=True, help="object real width in meters")
    ap.add_argument("--oheight", type=float, required=True, help="object real height in meters")
    ap.add_argument("--cfocal", type=float, required=True, help="camera focal length")
    ap.add_argument("--cwidth", type=float, required=True, help="camera sensor width in millimeters")
    ap.add_argument("--cheight", type=float, required=True, help="camera sensor height in millimeters")
    ap.add_argument("--ratiodev", type=float, default=0.2, help="maximum box ratio deviance")
    ap.add_argument("--smooth", type=float, default=1., help="apply single exponential smoothing to bounding box coordinates")
    ap.add_argument("--async", action="store_true", help="async mode with prediction")
    ap.add_argument("--video", action="store_true", help="save video")
    args = vars(ap.parse_args())

    Tracker(args, PROJ_DIR, os.getenv("MLFLOW_TRACKING_URI")).run()

