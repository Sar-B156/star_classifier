# Star Classifier
This project is a machine learning application for classifying stars based on their physical characteristics.
The model uses a neural network trained on astrophysical data to predict the type of a star given its temperature, luminosity, radius, and spectral properties.
The model will predict various star types, ranging from type 0 to type 6. You can see the specifics like this:
| Star Type Code | Star Classification |
|---------------|---------------------|
| 0 | Brown Dwarf |
| 1 | Red Dwarf |
| 2 | White Dwarf |
| 3 | Main Sequence |
| 4 | Supergiant |
| 5 | Hypergiant |


## 1. Dataset
This dataset is an astronomical dataset containing information on star types based on temperature and color. Each combination of features represents a different type of star value. Each feature in the dataset has a different meaning but is correlated with one another.
### Dataset Features

| Feature Name | Description |
|------------|-------------|
| Temperature (K) | Surface temperature of the star measured in Kelvin |
| Luminosity (L/Lo) | Luminosity of the star relative to the Sun |
| Radius (R/Ro) | Radius of the star relative to the Sun |
| Absolute magnitude (Mv) | Absolute visual magnitude of the star |
| Star color | Observed color of the star (categorical feature) |
| Spectral Class | Spectral classification of the star (O, B, A, F, G, K, M) |
| Star Type | Target label representing the class of the star |

In a stellar system, each star has a distinct motion and behavior. Furthermore, each star we discover has its own unique type. You can read more [here](https://www.kaggle.com/datasets/deepu1109/star-dataset/data)


## 2. Clone & Setup
The first step is to clone my repository
```bash
https://github.com/Sar-B156/star_classifier.git
```
Move the location to the repository that has been cloned to `star-classifier`
To prevent something undesirable, I suggest if you use an environment like `python -m venv venv`


Install all required dependencies listed in the requirements.txt file
```bash
pip install -r requirements.txt
```

## 3. Model Usage
The trained neural network model is provided in this repository and can be used directly for inference.
Users are required to prepare the input features in the same format as the training data.
The input data will be automatically preprocessed before being passed to the model, and the model will return a predicted star type based on the given parameters.

To use the trained model, follow these steps:
1. Prepare the star parameters required by the model, such as temperature, luminosity, radius, and spectral information
2. Run the inference script to load the trained model and preprocessing objects
3. The model will output the predicted class of the star

```python
from src.predict import predict_star

input_data = {
    "Temperature (K)": 5778,
    "Luminosity(L/Lo)": 1.0,
    "Radius(R/Ro)": 1.0,
    "Absolute magnitude (Mv)": 4.83,
    "Star color": "Yellow",
    "Spectral Class": "G"
}

prediction = predict_star(input_data)
print(prediction)
```

To create the neuron structure, you can use the structure I have provided in `src\model.py` or use the code below:
```python
import torch
import torch.nn as nn

class StarClassifier(nn.Module):
  def __init__(self, input_dim=6, num_classes=6):
    super().__init__()

    self.fc1 = nn.Linear(input_dim, 16)
    self.fc2 = nn.Linear(16, 16)
    self.fc3 = nn.Linear(16, num_classes)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return (x)
```

## 4. Demonstration
To visualize and try out the live test of the model I have created, you can see the demo [here](https://starclassifier-obama.streamlit.app/)
