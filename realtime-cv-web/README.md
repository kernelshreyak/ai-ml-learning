## Realtime pose detection on web with server-side processing

Based on https://www.hackersrealm.net/post/realtime-human-pose-estimation-using-python 


Uses mediapipe to detect pose on incoming video stream and stream results back to browser

1. Start server
```bash
uvicorn server:app --reload 
```

2. Open index.html in a browser
