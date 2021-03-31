import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load the features and targets from the dataset
data = load_breast_cancer()
X, Y = data.data, data.target

print('Features: ')
print(data.feature_names)
print('Targets: ')
print(data.target_names)

def create_model(layer_dims):

    '''
    creates a neural network with RELU hidden layers and Sigmoid output layer
    :param layer_dims: number of neurons in each layer
    :return: the neural network
    '''
    model = torch.nn.Sequential()
    for idx, dim in enumerate(layer_dims):
        if (idx < len(layer_dims) - 1):
            module = torch.nn.Linear(dim, layer_dims[idx + 1])
            init.xavier_normal(module.weight)
            model.add_module("linear" + str(idx), module)
        else:
            model.add_module("sig" + str(idx), torch.nn.Sigmoid())
        if (idx < len(layer_dims) - 2):
            model.add_module("relu" + str(idx), torch.nn.ReLU())

    return model


""" format data"""
# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# scale the data for the neural network
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


"""create the neural network """
dim_in = X_train.shape[1]
dim_out = 1
layer_dims = [dim_in, 20, 10, dim_out]
input_dimension = X_train.shape
#model = torch.nn.Linear(dim_in, dim_out)
model = create_model(layer_dims)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)


"""train the model"""
# set up loss and optimiser
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 2000
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

#training loop
for it in range(n_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)

    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()

    if (it + 1) % 50 == 0:
        print(
            f'In this epoch {it + 1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')



"""evaluate model"""
with torch.no_grad():
    p_train = model(X_train)
    p_train = (p_train.numpy() > 0.5)

    train_acc = np.mean(y_train.numpy() == p_train)

    p_test = model(X_test)
    p_test = (p_test.numpy() > 0.5)

    test_acc = np.mean(y_test.numpy() == p_test)

print('training accuracy:', train_acc)
print('testing accuracy: ', test_acc)

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()