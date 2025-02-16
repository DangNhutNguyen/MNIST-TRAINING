# MNIST Neural Network

This repository contains a simple implementation of a Neural Network trained to classify handwritten digits from the MNIST dataset. The neural network is built using TensorFlow, and the model uses a multi-layer perceptron architecture with three hidden layers.

## Project Description

The project aims to train a neural network to recognize digits (0-9) from the MNIST dataset. The dataset consists of 28x28 pixel grayscale images of handwritten digits, along with their corresponding labels.

In this project, the following steps are implemented:

1. **Data Loading**: The MNIST dataset is loaded and preprocessed.
2. **Model Building**: A neural network with three hidden layers is defined.
3. **Training**: The neural network is trained using the Adam optimizer to minimize the loss using the `softmax_cross_entropy_with_logits` loss function.
4. **Testing**: After training, the model's accuracy is evaluated using the test dataset.

The final model achieves an accuracy of approximately 94%.

## Requirements

- Python 3.x (Preferably 3.5 or later)
- TensorFlow
- NumPy
- Matplotlib

### Install Dependencies

To install the required dependencies, create a virtual environment and install the necessary packages:

```bash
# Create a virtual environment (optional but recommended)
python -m venv myenv
# Activate the virtual environment
# On Windows
myenv\Scripts\activate
# On Mac/Linux
source myenv/bin/activate

# Install required packages
pip install tensorflow numpy matplotlib
```

## Files Overview

- `mnist_neural_network.ipynb`: The Jupyter notebook containing the code for training the neural network on the MNIST dataset.
- `README.md`: This file containing the project overview, setup instructions, and usage details.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/dangnhutnguyen/MNIST-TRAINING.git
   cd mnist-neural-network
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook training.ipynb
   ```

3. Run all cells to train and test the model. The notebook will download the MNIST dataset, train the neural network, and output the final accuracy.

## Explanation of Code

### 1. **Data Loading**:

```python
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```
The `input_data.read_data_sets` function loads the MNIST dataset into training and test sets, with one-hot encoding for the labels.

### 2. **Neural Network Model**:

The model consists of three hidden layers, each with 500 nodes, followed by an output layer with 10 nodes (one for each class).

```python
hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
```

### 3. **Training**:

The neural network is trained using the Adam optimizer with a batch size of 100 and 5 epochs.

```python
optimizer = tf.train.AdamOptimizer().minimize(cost)
```

### 4. **Evaluation**:

After training, the model is evaluated on the test dataset, and accuracy is printed:

```python
accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
```

## Results

The model achieves an accuracy of approximately 94% on the MNIST test dataset.

### Sample Output:

```
Epoch 0 completed out of 5  loss: 1593870.59964
Epoch 1 completed out of 5  loss: 375178.777066
Epoch 2 completed out of 5  loss: 204155.209455
Epoch 3 completed out of 5  loss: 122060.331427
Epoch 4 completed out of 5  loss: 73701.5975397
Accuracy:  0.9423
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

