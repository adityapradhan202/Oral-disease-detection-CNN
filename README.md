# Oral-disease-detection-CNN

Using CNN model to Oral diseases like Gingitivitis, Hypodontia, Tooth discoloration, Mouth Ulcer, and caries.

**Requirements:**
Python 3.12 and Cuda 11.8 with cudnn v8.9.0

### Setup and installation:

First clone this repository using gitbash:
```
git clone https://github.com/adityapradhan202/Oral-disease-detection-CNN.git
```

Open the terminal in the root level of this project folder.
Then write this command to create a virtual environment:
```
python -m venv venv
```

Then activate the virtual environment.
```
venv\Scripts\activate
```

Then install all the required dependecies using:
```
pip install -r requirements.txt
```

### CLI command:
Use this command to use the model:-

```
python cli.py image_path
# For example
python cli.py .\sample_images\ulcer_m.jpg
```

### Project structure:
```
model/
|  |_eff2_model97.pth
sample_images/
torchscript/
|  |_data_setup.py
|  |_engine.py
|  |_model_builder.py
|  |_train.py
|  |_utils.py
.gitignore
cli.py
LICENSE
model_train_evaluate.py
predict.py
README.md
requirements.txt
```


### About model training:
For model training, efficientnet_b0 model has been used, which is a pretrained model. In other words, transfer learning has been used. **Why transfer learning?** Pretrained models have good general knowledge because these models are trained on popular and big dataset and indentify a large number of classes. When we train these pretrained models on our custom datasets, then the weights are adjusted accordingly, to gain good knowledge of our custom images.

Instead of saving the whole model, the state dictionary has been saved which contains weights and biases because it's more efficient.

**Model is provided in the [model](./model/) directory.**

**Note:** The model has been trained images with JPG format.