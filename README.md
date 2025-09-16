# Eye Controlled Mouse Cursor

Control your mouse cursor and perform clicks using only your eye movements and winks, powered by computer vision and MediaPipe Face Mesh.

## Features

- Move the mouse cursor by tracking your eye position.
- Click by winking your left eye.
- Real-time webcam-based control.
- Visual feedback for detected eye landmarks and Eye Aspect Ratio (EAR).

## Requirements

- Python 3.7+
- Webcam

## Installation

Clone this repository and install the required dependencies:

```sh
git clone https://github.com/ER-Harshita-15/eye_controlled_mouse.git
cd eye_controlled_mouse
pip install -r requirements.txt
```

## Usage

Run the main script to start controlling your mouse with your eyes:

```sh
python eye_controlled_mouse.py
```

- Move your head/eyes to move the mouse cursor.
- Wink your left eye to perform a mouse click.

## How It Works

- Uses [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) to detect facial landmarks.
- Tracks iris position for cursor movement.
- Calculates Eye Aspect Ratio (EAR) to detect winks for clicking.
- Provides real-time visual feedback via webcam window.

## Troubleshooting

- Make sure your webcam is connected and accessible.
- If the cursor movement or click detection is inaccurate, adjust the `wink_threshold` value in [`eye_controlled_mouse.py`](eye_controlled_mouse.py).
- For best results, use in a well-lit environment.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe)
- [OpenCV](https://opencv.org/)
-