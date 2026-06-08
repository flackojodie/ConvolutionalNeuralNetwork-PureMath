# Convolutional Neural Network Puremath
Neural network from scratch using cupy with experiments on a ton of things, including my own optimizer and activation function. This is part 3 of the puremath notebooks where I develop AI models using math libraries only in python, in this specific notebook I didnt use autograd but differentiated each equation by hand and use that for backpropagation.

The data being used is of dog images of 10 different breeds, its a regular multiclass classifcation problem.

In this notebook, I have created a convolutional neural network using cupy. a lot of the math involving the concepts is discussed here but obviously not all of it, again to reiterate from my earlier notebook, this notebook is more of a journal and is an accumulation of everything that I have learnt in the fields of math, ai theory and programming, and it may contain errors. Matter of fact, it is really easy to see the progression of my ideas from the initial part of the notebook right down to the final phase. I have tried to write most of the things I have learnt throughtout the journey of creating the CNN down in this notebook and the other book dedicated for the optimizer idea I have `CurveAndGradientBasedOptimizer` that modulates the learning rate with respect to the curvature data and not time.

Some of the code is made with help from LLMs such as chatgpt, but the ideas are all originally mine and I have tried to restrict myself on the usage of AI in order to develop the intuition required to program such complex algorithms. for functions such as convolution and maxpool, I have relied on AI but I have still tried to code it out by myself.

This notebook uses cupy as an alternative for numpy, hence to run the notebook for yourself, you need to connect to a GPU, if you run on colab make sure to change the runtime type to GPU

I had to migrate to kaggle in the middle of making the notebook hence the paths to the dataset and the evaluation images are inconsistent, in order to run the notebook you have to manually change the paths (I will figure out a way to resolve this issue in future notebooks)

In many places of the notebook, I have written down some ideas of mine which I will implement in the upcoming notebooks.

I have not yet completed the custom optimizer, as my mathematical knowledge does not stretch that far but I will make it and optimize it in the next notebooks, right now it uses Rayleigh Quotient to estimate the eigenvalues (curvature) and causes a unidirectional descent into gradients but utilizing Lanczos iteration and Krylov subspaces may resolve this issue, its an idea I will look into.

(also on an undocumented run of the notebook, `trained_model_12` and `trained_model_13` both evaluated to give a score of 5 out of 9)

## Features of the notebook:
- Full neural network implementation in cupy only
- Complete backpropgation by calculating the derivatives by hand
- Multiple activation functions (including a custom built one `ownAF`) + Parametric (learnable) activation functions
  - ReLU
  - Swish
  - Parametric ReLU
  - Parametric Swish
  - Custom activation function `ownAF`
- He/Xavier initialization
- Preprocess pipeline
- Custom layers
  - Fast-fourier transform based convolution
  - An even faster im2col + general matrix multiplication convolution
  - Pooling layers (max, global)
  - Batchnormalization + an even faster batchnormalization algorithm
  - Strided convolutions
  - Squeeze-and-Excitation blocks (`LightChannelAttention`)
  - Dropout
  - DropConnect (not implemented but explored)
  - Non-learnable softmax (a learnable counterpart is when the temperature of the softmax is trainable, but im not sure if its a good idea)
- One-Hot Encoding of labels
- Label Smoothing
- Optimizers:
  - AdamW + Nesterov AdamW
  - Adam + Nesterov Adam
  - SGD + Nesterov SGD
- Crossentropy
- Vicinal-Risk Minimization such as MixUp augmentation
- image augmentations using albumentations
- K-Fold Training instead of train-test split
- and the hessian and gradient based optimizer `CurveAndGradientBasedOptimizer`
- Evalutaion function that runs inference on already downloaded images off of the internet, images can be seen in the folder `testimages_for_CNN_puremath`

Most of the math that goes into these topics are discussed in the notebook itself.

## Experiments performed and the best model:

| Model    | Convolution Type       | SE Blocks | Activation Function    | Optimizer                              | LR Schedule      | Label Smoothing | Augmentation | MixUp | Epochs | Evaluation |
|----------|------------------------|------------|------------------------|----------------------------------------|------------------|-----------------|--------------|-------|--------|------------|
| Model 1  | Standard Conv          | No         | ReLU               | None                                   | None             | No              | No           | No    | 0      |     -       |
| Model 2  | im2col Strided Conv    | No         | ReLU               | None                                   | None             | No              | No           | No    | 10     |     1       |
| Model 3  | im2col All-Strided Conv| Yes        | ReLU               | None                                   | None             | No              | No           | No    | 20     |     2       |
| Model 4  | im2col All-Strided Conv| Yes        | ReLU               | AdamW + Nesterov                       | None             | No              | No           | No    | 2      |     3       |
| Model 5  | im2col All-Strided Conv| Yes        | Swish                  | AdamW + Nesterov                       | None             | No              | No           | No    | 7      |     3       |
| Model 6  | im2col All-Strided Conv| Yes        | OwnAF                  | AdamW + Nesterov                       | None             | Yes             | Yes          | No    | 1      |      4      |
| Model 7  | im2col All-Strided Conv| Yes        | OwnAF                  | AdamW + Nesterov                       | None             | Yes             | Yes          | No    | 1      |      1      |
| Model 8  | im2col All-Strided Conv| Yes        | PReLU                  | AdamW + Nesterov                       | None             | Yes             | Yes          | No    | 1      |     2       |
| Model 9  | im2col All-Strided Conv| Yes        | Parametric Swish       | AdamW + Nesterov                       | None             | No              | Yes          | No    | 1      |      2      |
| Model 10 | im2col All-Strided Conv| Yes        | PReLU                  | AdamW + Nesterov                       | Cosine Annealing | Yes             | Yes          | No    | 10      |      5      |
| Model 11 | im2col All-Strided Conv| Yes        | PReLU                  | AdamW + Nesterov                       | Cosine Annealing | Yes             | Yes          | Yes   | 10      |     6       |
| Model 12 | im2col All-Strided Conv| Yes        | ReLU               | AdamW + CurveAndGradientBasedOptimizer       | None             | Yes              | Yes           | Yes   | 5      |       5     |
| Model 13 | im2col All-Strided Conv| Yes        | OwnAF                  | AdamW + Nesterov + CurveAndGradientBasedOptimizer | None             | Yes             | Yes           | Yes   | 5      |      5      |

Hence the best model out of all of them is model 11 followed by model 12, 13, 10

## Future improvements:
- Implementing the ideas written in the notebook
- Fixing the `CurveAndGradientBasedOptimizer` so that its no longer unidirectional
- Training for longer
- A more aggressive augmentation pipeline
- using a parametric `ownAF`
- implementing super-expressive activation function (EUAF and PEUAF, No idea how it would affect the result)
- Tuning the hyperparameters a bit

## Final Note:
This project is part of an ongoing learning journey in machine learning and mathematical AI foundations. The focus is on understanding, experimentation, andimprovement rather than achieving state-of-the-art performance.


