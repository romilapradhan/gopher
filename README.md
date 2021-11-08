# Debugging bias: identify tuples responsible for fairness criteria violation.

## Getting Started

Intergration/faster_explanation_german.ipynb contains latest explanation generation pipeline.

Intergration/load_dataset.py contains loading and preprocessing of three biased datasets (**Adult, German, SQF**).

Intergration/classifier.py contains implementation of classfiers including Logistic Regression, Support Vector Machine, and Neural Network.

## Classifier Hyperparameters

### Logistic Regression

| Dataset | Learning Rate | Weight Decay | Epoch Num |
| :-----: | :-----------: | :----------: | :-------: |
| German  |     0.05      |     0.03     |    100    |
|  Adult  |     0.05      |     0.03     |    100    |
|   SQF   |     0.05      |     0.03     |    300    |

### Neural Network

1 hidden layer with 10 nodes

| Dataset | Learning Rate | Weight Decay | Epoch Num | Batch Size |
| :-----: | :-----------: | :----------: | :-------: | :--------: |
| German  |     0.05      |    0.001     |    100    |     80     |
|  Adult  |     0.05      |    0.001     |    500    |    1024    |
|   SQF   |     0.05      |    0.001     |    300    |    1024    |

### SVM 

smooth-hinge with beta=1, linear kernal

| Dataset | Learning Rate | Weight Decay | Epoch Num |
| :-----: | :-----------: | :----------: | :-------: |
| German  |     0.05      |     0.1      |    100    |
|  Adult  |     0.05      |     0.1      |    100    |
|   SQF   |     0.05      |     0.1      |    100    |

### 
