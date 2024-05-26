## Realtime computer vision on web using Tensorflow.js, MediaPipe and ml5.js

Based on https://www.hackersrealm.net/post/realtime-human-pose-estimation-using-python 


### Example in root: Pose detection

Uses mediapipe to detect pose on incoming video stream and stream results back to browser

1. Start server
```bash
uvicorn server:app --reload 
```

2. Open index.html in a browser

### Other examples
 - Image classification on video and static image using MobileNet and ml5.js [here](./image-classification/)
 - Object detection using COCO-SSD and Tensorflow.js [here](./object-detection/)
