# Browser demo
Upload demo.html, demo.js, weights.json and asl_weights.bin to the root of the original GitHub repository. Enable GitHub Pages from main / (root). Visit https://tanindrag.github.io/Sign-Language-Detection-Project/demo.html after deployment.

The demo reconstructs the original Keras CNN using TensorFlow.js and the original float32 weights extracted from the uploaded Keras archive. It uses MediaPipe Hands, browser camera APIs and speech synthesis. No retraining. The browser preprocessing approximates OpenCV INTER_AREA resizing with canvas resampling; real-world predictions may differ. Camera requires HTTPS or localhost. External CDN libraries must load. Static letters only, J and Z excluded. Webcam and image data stay in the browser. Test with your own hand photos before sharing.

Local test: run `python -m http.server 8000` from this folder and visit http://localhost:8000/demo.html. Do not open via file:// because model fetches may be blocked.
