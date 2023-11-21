# rPPG Toolbox - EfficientPhys Prediction
This repository is a fork of the Ubicomp Lab's rPPG-Toolbox, specifically tailored for utilizing a pretrained version of the EfficientPhys model. The primary purpose of this modification is to apply the pre-trained model on video inputs and overlay the model's predictions onto the video stream.

## Purpose
The core objective of this repository is to leverage the power of the EfficientPhys model, which has been trained on the PURE dataset, for real-time or offline prediction of vital signs from video inputs. By incorporating this modified code, users can seamlessly visualize the model's predictions overlaid on top of the provided video input.

## Usage
To utilize this repository:

Clone or download the repository to your local machine.

Ensure you have the necessary dependencies installed:
```bash
pip install -r reqs.txt
```

Navigate to the configs/UlysseConfig.yaml file.

Specify the path to your video in the TEST.DATA.DATA_PATH attribute.

Set the PREPROCESS attribute to true if it is the first time running the code on the video.

Run the following command in your terminal:
```bash
python main.py --config_file ./configs/UlysseConfig.yaml
```

![GIF of a Pleth Prediction using EfficientPhys trained on the PURE Dataset](figures/Pleth_prediction.gif)

Above is an example of a video processed using this repository's code, showcasing the model's predictions overlaid on the input video.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork and submit pull requests to enhance the functionality of this repository.

