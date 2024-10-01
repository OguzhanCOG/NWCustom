# NeuralWorksCustom Framework. Inspired by PyTorch, simpler and with planned modifications to support my own architectures/experimental blocks.

# Need an AD(ifferentiation) system for deeper nets.

import json
import matplotlib as pyplot
import matplotlib.pyplot as pyplot

# ALPHA: Drop-in CuPy (CUDA) implementation.
try:
    raise ImportError # Temporary skip.
    import cupy as np
    print("CuPy is installed. Using CUDA acceleration!")
    # use_cupy = True
except ImportError:
    import numpy as np
    # print("CuPy is not installed. Using NumPy/CPU!")
    # use_cupy = False

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad # Does this tensor need to compute gradients? Usually, Yes.
        self.grad_fn = None

    def __add__(self, other):
        return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, other):
        return Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

    def __matmul__(self, other):
        return Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

    # Calculating gradients for backprop. If no external grad provided, default to ones
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad # Accumulate. 
        
        # If there is a function to propagate backwards, do it!
        if self.grad_fn:
            self.grad_fn(grad)

    def zero_grad(self):
        # Reset the grads after each optim step to avoid grad accum!
        self.grad = None

    def __repr__(self):
        return f"Tensor({self.data})" # Fancy.

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x): # x @ W + b
        self.x = x
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def backward(self, grad):
        self.weight.grad = self.x.data.T @ grad
        if self.bias is not None:
            self.bias.grad = np.sum(grad, axis=0)
        return grad @ self.weight.data.T

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        # Slide kernels over the input to compute the output iteratively; standard implementation.
        self.x = x
        batch_size, in_channels, height, width = x.data.shape
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # Apply the conv. Summing over the kernel window with respect to the filter.
                        out[b, c, h, w] = np.sum(x_padded[b, :, h * self.stride:h * self.stride + self.kernel_size, w * self.stride:w * self.stride + self.kernel_size] * self.weight.data[c]) + self.bias.data[c]

        return Tensor(out, requires_grad=True)

    def backward(self, grad):
        batch_size, out_channels, out_height, out_width = grad.data.shape
        batch_size, in_channels, height, width = self.x.data.shape
        x_padded_grad = np.zeros_like(self.x.data)
        x_padded = np.pad(self.x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for b in range(batch_size):
            for c in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        # Find grads based on the slice of the input that was first used in fwd pass.
                        x_slice = x_padded[b, :, h_start:h_end, w_start:w_end]
                        x_padded_grad[b, :, h_start:h_end, w_start:w_end] += grad.data[b, c, h, w] * self.weight.data[c]
                        self.weight.grad[c] += grad.data[b, c, h, w] * x_slice
                        self.bias.grad[c] += grad.data[b, c, h, w]

        # Remove padding from grads if applicable.
        if self.padding != 0:
            x_padded_grad = x_padded_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # Feed the gradient backwards to the prev layer.
        self.x.backward(x_padded_grad)

class ReLU:
    def forward(self, x):
        self.x = x
        self.output = np.maximum(0, x.data)
        return Tensor(self.output, requires_grad=True)

    def backward(self, grad):
        relu_grad = self.output > 0
        return grad * relu_grad

class Sigmoid:
    def forward(self, x):
        self.x = x
        self.output = 1 / (1 + np.exp(-x.data))
        return Tensor(self.output, requires_grad=True)

    # Use the Chen Lu.
    def backward(self, grad):
        sg = self.output * (1 - self.output)
        return grad * sg

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.mean((y_pred.data - y_true.data) ** 2)
        return Tensor(loss, requires_grad=True)

    def backward(self):
        grad = 2 * (self.y_pred.data - self.y_true.data) / np.prod(self.y_true.data.shape)
        return grad

class CrossEntropyLoss:
    # Only ever useful for classification at the moment...
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.sum(y_true.data * np.log(y_pred.data + 1e-9)) / y_true.data.shape[0]
        return Tensor(loss, requires_grad=True)

    def backward(self):
        grad = -self.y_true.data / (self.y_pred.data + 1e-9) / self.y_true.data.shape[0]
        self.y_pred.backward(Tensor(grad))

class SGD:
    # Stochastic Gradient Descent
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
                param.zero_grad()

class Adam:
    # Standard Adam implementation.
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = {param: np.zeros_like(param.data) for param in params}
        self.v = {param: np.zeros_like(param.data) for param in params}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                self.m[param] = self.betas[0] * self.m[param] + (1 - self.betas[0]) * param.grad
                self.v[param] = self.betas[1] * self.v[param] + (1 - self.betas[1]) * (param.grad ** 2)
                m_hat = self.m[param] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[param] / (1 - self.betas[1] ** self.t)
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                param.zero_grad()

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        # Mod to support CuPy
        if shuffle:
            use_cupy = False # Temporary skip.
            if use_cupy:
                self.indices = np.array(self.indices)
                np.random.shuffle(self.indices)
            else:
                np.random.shuffle(self.indices)

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_x = [self.dataset[j][0] for j in batch_indices]
            batch_y = [self.dataset[j][1] for j in batch_indices]
            yield np.array(batch_x), np.array(batch_y) # No need for any mod here...

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                params.append(layer.weight)
            if hasattr(layer, 'bias'):
                params.append(layer.bias)
        return params

    def train(self, data_loader, loss_fn, optimizer, epochs):
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            for batch in data_loader:
                x, y = batch
                x = Tensor(x, requires_grad=True)
                y = Tensor(y, requires_grad=False)
                y_pred = self.forward(x)
                loss = loss_fn.forward(y_pred, y)
                grad = loss_fn.backward()
                self.backward(grad)
                optimizer.step()
                total_loss += loss.data
                num_batches += 1
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    def save(self, path):
        state_dict = {}
        for name, param in self.named_parameters():
            state_dict[name] = param.data.tolist()
        with open(path, 'w') as f:
            json.dump(state_dict, f)

    def load(self, path):
        with open(path, 'r') as f:
            state_dict = json.load(f)
        for name, param in self.named_parameters():
            param.data = np.array(state_dict[name])

    def named_parameters(self):
        params = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                params.append((f'layer_{i}_weight', layer.weight))
            if hasattr(layer, 'bias'):
                params.append((f'layer_{i}_bias', layer.bias))
        return params

# Define your model in the array of objects here.
# Really simple model; works well to fit graphs, or any sort of data.
model = Model([Linear(1, 50), Sigmoid(), Linear(50, 1)])

loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define function to attempt to fit here. Cartesian only!
def test_function(x):
    return -4 * x**3 + 2 * x**2 + 1 - np.exp(x - 5)

x = np.random.randn(1000, 1)
y = test_function(x)
dataset = Dataset(x, y)
data_loader = DataLoader(dataset, batch_size=48, shuffle=True)

model.train(data_loader, loss_fn, optimizer, epochs=100)

# Test save & load mechanism here...
model.save('model.json')
model.load('model.json')

# Test points
test_x = np.linspace(-2, 2, 100).reshape(-1, 1)
test_y = test_function(test_x)

model_predictions = []
for x in test_x:
    input_tensor = Tensor(x, requires_grad=False)
    output = model.forward(input_tensor)
    model_predictions.append(output.data)

model_predictions = np.array(model_predictions)

pyplot.figure(figsize=(10, 6))
pyplot.scatter(test_x, model_predictions, color='red', label='Predictions')
pyplot.plot(test_x, test_y, color='blue', label='Actual')
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.title('Model Predictions vs Actual Function')
pyplot.legend()
pyplot.grid(True)
pyplot.show()
pyplot.savefig('test.png')

def visualize_model_weights(model):
    layers = model.layers
    
    for i, layer in enumerate(layers):
        if isinstance(layer, Linear):
            weights = layer.weight.data
            pyplot.subplot(len(layers), 1, i + 1)
            pyplot.title(f'Linear - Layer {i + 1}')
            pyplot.imshow(weights, cmap='coolwarm', aspect='auto')
            pyplot.colorbar(label='Weights')
        
        elif isinstance(layer, Conv2d):
            weights = layer.weight.data.reshape(layer.out_channels, -1)
            pyplot.subplot(len(layers), 1, i + 1)
            pyplot.title(f'Convolution (2D) - Layer {i + 1}')
            pyplot.imshow(weights, cmap='coolwarm', aspect='auto')
            pyplot.colorbar(label='Weights')

    pyplot.tight_layout()
    pyplot.show()

visualize_model_weights(model)
