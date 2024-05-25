import '@tensorflow/tfjs-backend-core';
import '@tensorflow/tfjs-backend-webgl';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';

// Uncomment the line below if you want to use TensorFlow.js runtime.
// import '@tensorflow/tfjs-converter';

// Uncomment the line below if you want to use MediaPipe runtime.
import '@mediapipe/selfie_segmentation';



// Add an event listener to the button
document.getElementById('startWebcam').addEventListener('click', startWebcam);


const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation; // or 'BodyPix'

const segmenterConfig = {
runtime: 'mediapipe', // or 'tfjs'
modelType: 'general' // or 'landscape'
};

segmenter = await bodySegmentation.createSegmenter(model, segmenterConfig);

const video = document.getElementById('webcam');
const people = await segmenter.segmentPeople(video);
console.log(people);
