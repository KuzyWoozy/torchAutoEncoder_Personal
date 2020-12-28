import torch as t

class Autoencoder(t.nn.Module):
    def __init__(self, input_shape=784):
        super().__init__()

        self.layer1 = t.nn.Linear(input_shape, 50)
        t.nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity="relu")
        t.nn.init.zeros_(self.layer1.bias)

        self.layer2 = t.nn.Linear(50, input_shape)
        t.nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity="relu")
        t.nn.init.zeros_(self.layer2.bias)

        self.loss = t.nn.BCELoss(reduction="sum")
        self.opti = t.optim.Adam(self.parameters(), lr=0.001)

    # Returns the output of the final layer, of the forward propogation
    def forward(self, x):
        self.train()

        x = t.relu(self.layer1(x))
        x = t.sigmoid(self.layer2(x))
        return x

    # Uses loss of the forward propogation to train network
    def backward(self, pred, y):
        # Calculate the loss and calculate grads
        loss = self.loss(pred, y)
        loss.backward()
        # Apply grads to weights
        self.opti.step()
        self.opti.zero_grad()

        return loss.item()

    # Forward propogate without autograd
    def predict(self, x):
        self.eval()
        with t.no_grad():
            x = self(x)
            return x

    def evaluate(self, x, y):
        return self.loss(self.predict(x), y).item()

    # Save model params
    def save(self, path):
        t.save(self.state_dict(), path)

    # Load saved model params
    def load(self, path):
        self.load_state_dict(t.load(path, map_location=t.device('cpu')))
