# rPPG Toolbox - EfficientPhys Prediction
This repository is a fork of the Ubicomp Lab's rPPG-Toolbox, specifically tailored for utilizing a pretrained version of the EfficientPhys model. The primary purpose of this modification is to apply the pre-trained model on video inputs and overlay the model's predictions onto the video stream.

"""EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023)
Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff
"""

## Purpose
The core objective of this repository is to leverage the power of the EfficientPhys model, which has been trained on the PURE dataset, for real-time or offline prediction of vital signs from video inputs. By incorporating this modified code, users can seamlessly visualize the model's predictions overlaid on top of the provided video input.  

This repository marks the culmination of my internship journey at **Norbert Health**, showcasing the practical application of skills acquired during this tenure. It underscores the seamless integration and enhancement of a pretrained *EfficientPhys* model, initially trained on the PURE dataset, within the **rPPG-Toolbox** derived repository. Furthermore, I extended the model's capabilities by training it on a proprietary dataset, prioritizing data privacy. This project stands as a testament to my adeptness in handling and improving rPPG deep learning models with custom data while maintaining utmost confidentiality. Through this endeavor, I aim to underscore my proficiency in signal processing techniques and the adept integration of advanced rPPG models into the realm of healthcare technology.

## Usage
To use this repository:

Clone or download the repository to your local machine.

Ensure you have the necessary dependencies installed:
```bash
pip install -r reqs.txt
```

Navigate to the configs/UlysseConfig.yaml file.

Specify the path to your video in the TEST.DATA.DATA_PATH attribute.

Set the **PREPROCESS** attribute to *True* if it is the first time running the code on the video.

Run the following command in your terminal:
```bash
python main.py --config_file ./configs/UlysseConfig.yaml
```

![GIF of a Pleth Prediction using EfficientPhys trained on the PURE Dataset](figures/Pleth_prediction.gif)

Above is an example of a video processed using this repository's code, showcasing the model's predictions overlaid on the input video.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork and submit pull requests to enhance the functionality of this repository.

