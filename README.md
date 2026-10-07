# Steps for preparation

## Python venv
Note these are the commands for Linux, ask your favorite AI for the steps on Windows, it's slightly different

python -m venv .venv <br />
source .venv/bin/activate

## Install dependencies
pip install -r requirements.txt <br />
curl -L -o yolov11s-face.pt https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11s-face.pt

# Run the application
python main.py

## Hats
Extra hats can be added by placing a .png with transparent background in the hats folder, no code changes required

## Usage
Press Q to quit
