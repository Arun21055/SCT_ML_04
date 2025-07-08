# Hand Gesture Recognition with CNN and Webcam

## Project Structure

```
├── leapGestRecog/
│   └── 00/ to 09/
│       └── 01_palm to 10_down/
├── gesture_model.h5
├── gesture_recognition.py
├── README.md
```

## Features

- CNN-based classifier for 10 hand gestures
- Real-time webcam inference
- Model auto-saves after training
- Loads saved model on next runs

## Model Classes

- 01_palm
- 02_l
- 03_fist
- 04_fist_moved
- 05_thumb
- 06_index
- 07_ok
- 08_palm_moved
- 09_c
- 10_down

## Requirements

```bash
pip install numpy opencv-python tensorflow scikit-learn matplotlib
```

## Dataset

Download: https://www.kaggle.com/datasets/gti-upm/leapgestrecog

Extract folder as:

```
leapGestRecog/
    ├── 00/
    ├── 01/
    └── ...
```

Each subfolder contains gesture classes.

## How to Run

```bash
python gesture_recognition.py
```

- If `gesture_model.h5` exists, it loads and starts webcam.
- If not, it trains model and then starts webcam.

## Live Prediction

- Shows green box (ROI) where hand should be placed
- Displays gesture class and confidence score
- Press `q` to quit webcam window

## Map Classes to Digits (Optional)

```python
label_map = {
    '01_palm': '0', '02_l': '1', '03_fist': '2',
    '04_fist_moved': '3', '05_thumb': '4',
    '06_index': '5', '07_ok': '6',
    '08_palm_moved': '7', '09_c': '8', '10_down': '9'
}
```

## Sample Output

- Prediction format:  
  `03_fist (98.7%)`
- Region of interest is outlined by green rectangle
