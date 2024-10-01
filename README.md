# NeuralWorksCustom Framework
NeuralWorksCustom (NWCustom) Framework is a lightweight, customisable neural network toolkit inspired by PyTorch. It offers a simple yet powerful API for building, training, and running custom neural network models. The framework is designed to be beginner-friendly while providing flexibility for more advanced users to experiment with their own architectures and blocks.

The ultimate goal, like NeuralWorks, is to create a versatile and robust system that allows users, even those without extensive machine learning backgrounds, to design and implement their own neural network architectures. Future plans include expanding support for more advanced model types, including blocks that may be used in LLMs (Transformers, Pre-made/ready Attention blocks).

Enjoy!

# Features
- Custom Tensor class
- Basic neural network layers (Linear, Conv2d, ReLU, Sigmoid).
- Loss functions (MSELoss, CrossEntropyLoss).
- Optimizers (SGD, Adam).
- DataLoader for batch processing.
- Model class for easy network composition.
- Save and load functionality for trained models.
- Visualisation tools for model weights and predictions.

# Requirements
- Python >= 3.8
- NumPy >= 1.23.1
- MatPlotLib >= 3.7.1

# Getting Started
To use the NWCustom Framework, simply copy the provided code into your project or import it as a module.

```python
from nwcustom import *

Define your model
model = Model([
    Linear(1, 50),
    Sigmoid(),
    Linear(50, 1)
])

# Create a loss function and optimizer
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Prepare your data
x = np.random.randn(1000, 1)
y = your_target_function(x)
dataset = Dataset(x, y)
data_loader = DataLoader(dataset, batch_size=48, shuffle=True)

# Train your model
model.train(data_loader, loss_fn, optimizer, epochs=100)

# Save and load your model
model.save('model.json')
model.load('model.json')

# Make predictions and visualize results
# (See the provided code for examples)
```

# Future Updates
Version 1.0.1:

- CUDA acceleration support via full CuPy integration.
- Expanded layer types (e.g., BatchNorm, Dropout).
- Additional activation functions.
- More advanced optimisers (e.g., RMSprop, AdaGrad).
- Enhanced visualisation tools.
- Improved error handling and debugging features.

Release Date: N/A

Version 1.0.2:

- Support for recurrent neural networks (RNN, LSTM, GRU).
- Model ensembling capabilities.
- Transfer learning support.
- Integration with popular datasets.
- Performance optimizations.
- Custom pre-made, drop-in Attention blocks.
- Shifted Window (Swin) Transformers /w MLP repacing KANs.
- DCGAN support.

More planned for the future.

Release Date: N/A

# Contributing
Contributions to the NWCustom Framework are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

Feedback? motions.07-busses@icloud.com

# License
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">NeuralWorksCustom (NWCustom) Framework Version 1.0.0</span> by <span property="cc:attributionName">Oguzhan Cagirir</span> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
