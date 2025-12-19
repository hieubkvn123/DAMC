# Experiments for "Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information" 

This project, "Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information", is focused on building and testing recommendation systems that leverage implicit feedback data. The task is the learn a predictor:

$$
\mathbb{R}^{m\times n}\ni \widehat Z\approx G,
$$ 

where $G$ is a ground-truth matrix with partially-observed entries (e.g., ratings), given:

- 1. **Unlabeled data**: A large set of user-item pairs $(i, j)$ drawn from an unknown sampling distribution $P$ without observing the ratings. These correspond to _implicit feedback_.
- 2. **Labeled data**: A much smaller set of pairs $(i, j)$ with observed noisy labels $\widetilde G_{i,j}$. These correspond to _explicit feedback_.


## Directory Structure

- `autoencoders/`: Contains the autoencoder classes and functions used within the models.
- `data/`: Data files and datasets used for training and testing the models.
  - `douban_movie(u3022m6977)/`: Data related to the Douban Movie dataset.
  - `ml-100k/`: The MovieLens 100K dataset.
  - `yelp_data(u14085b14037)/`: Yelp dataset.
- `models/`: Contains the model definitions.
  - `joint_model.py`: The main model of the project.
- `tests/`: Unit tests for different components of the project.
- `utils/`: Utility functions and classes.
  - `data_utils.py`: Functions for data preprocessing and manipulation.

## Core Model

The `JointModel` is a key part of this project, designed to work with implicit feedback datasets. It utilizes autoencoders to handle sparse and high-dimensional data typically found in such datasets.

### JointModel Features

- Handles various types of implicit feedback.
- Implements autoencoder-based architecture.
- Configurable for different datasets and requirements.

## Getting Started

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd DAMC 
pip install -r requirements.txt
```

### Running the Model

To run the `JointModel` or other models in this project, you can use the `main.py` script. This script accepts various command-line arguments to configure the model run. Below is an example of how to run the model:

```bash
python main.py --dataset "ml-100k" --p_value 0.2 --step 1000 --latent_side_info 100 --latent_M 50 --lambda_ 0.01
```



### Notes

- **Running the Model**: The command-line example assumes that users are familiar with basic Python and command-line operations. Adjust the provided example and descriptions based on the actual implementation of your `main.py` script.
- **Testing**: The testing instructions are fairly standard for Python projects using pytest. If your project has specific requirements or setups for testing, include those details in this section.

This expanded README provides clear instructions for users on how to run models and execute tests, enhancing the usability and accessibility of your project.
