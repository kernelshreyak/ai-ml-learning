import pixellib
from pixellib.semantic import semantic_segmentation
import cv2


capture = cv2.VideoCapture(0)

segment_video = semantic_segmentation()
segment_video.load_ade20k_model("models/deeplabv3_xception65_ade20k.h5")
segment_video.process_camera_ade20k(capture, overlay=True, frames_per_second= 15, output_video_name="output_video.mp4", show_frames= True,
frame_name= "frame")