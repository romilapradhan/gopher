# Gopher: Interpretable Data-Based Explanations for Fairness Debugging

## Experiment Environment (also included in requirements.txt)
```
numpy==1.23.1
pandas==1.3.2
scikit_learn==1.0.2
scipy==1.5.4
torch==1.9.0
tqdm==4.62.2
```

## Getting Started

### Notebooks

**Below are complete notebooks showing how we generate explanations:**

explanations/removal_expl.ipynb contains removal-based explanation generation pipeline.

explanations/update_expl.ipynb contains update-based explanation generation pipeline.

**For simplicity, we also prepare 2 corresponding tutorial notebooks that briefly go through the necessarily steps to generate explanations:**

explanations/removal_expl_tutorial.ipynb contains necessary steps of generating removal-based explanations:
* Choose the classifier, dataset.
* Precompute the hessian and first-order derivatives.
* Generate removal-based explanation by invoking ```explanation_candidate_generation``` and then filter the results based on containments by invoking ```get_top_k_expl```.

explanations/update_expl_tutorial.ipynb contains necessary steps of generating update-based explanations:
* Choose the classifier, dataset.
* Precompute the first-order derivatives.
* Generate update-based explanation by invoking ```get_update_expl```.

**(Running notebooks would be sufficient to obtain explainations presented in the paper. In order to conduct scalability experiments, setting duplicates provided insied the notebooks it able to help test runtime vs dataset size with the same feature number and relationships, and it is recommended to use %%time inside cells to test the runtime.)**

### Functions

explanations/load_dataset.py contains functions of loading and preprocessing of biased datasets (**Adult, German, SQF**).

explanations/classifier.py contains implementation of classfiers including Logistic Regression, Support Vector Machine, and Neural Network.

explanations/influence.py contains functions of computing first-order derivatives and hessian.

explanations/expl.py contains functions of generating removal-based and update-based explanation.

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
