import numpy as np
import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import helpers

covariates_rnn = np.load('data/covariates_rnn.npy')
outcomes_rnn = np.load('data/outcomes_rnn.npy')
seq_length = np.load('data/seq_length.npy').astype(np.int16)
X = torch.tensor(covariates_rnn).float()
y = torch.tensor(outcomes_rnn).float()
rnn_dataset = helpers.Dataset(X, y, seq_length)
rnn_dataloader = torch.utils.data.DataLoader(rnn_dataset, batch_size = 32, shuffle = True)
y_masked = y[rnn_dataset.seq_mask_y == 1]

device = "cuda"
input_size = covariates_rnn.shape[2]
output_size = outcomes_rnn.shape[2]
hidden_size = 64
num_stacked_layers = 2
lr = 1e-4

random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

model = helpers.rnn(input_size, output_size, hidden_size, num_stacked_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
num_epochs = 2000
loss_list = np.zeros(num_epochs)
for epoch in range(num_epochs):
    for (idx, (X_batch, y_batch, seq_mask_X_batch, seq_mask_y_batch)) in enumerate(rnn_dataloader):
        batch_size = X_batch.shape[0]
        h0 = torch.zeros(num_stacked_layers, batch_size, hidden_size).float().to(device)
        c0 = torch.zeros(num_stacked_layers, batch_size, hidden_size).float().to(device)
        X_batch = X_batch.float().to(device)
        y_batch = y_batch.float().to(device)
        y_batch = y_batch[seq_mask_y_batch == 1]
        out = model.forward(X_batch, h0, c0)
        out = out[seq_mask_y_batch == 1]
        loss = torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(out), y_batch, reduction = "mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_list[epoch] = loss.item()
    '''
    with torch.no_grad():
        h0 = torch.zeros(num_stacked_layers, X.shape[0], hidden_size).to(device)
        c0 = torch.zeros(num_stacked_layers, y.shape[0], hidden_size).to(device)
        _, pred = model.predict(X.to(device), h0, c0)
        pred_masked = pred[rnn_dataset.seq_mask_y == 1].to("cpu")
        acc = (torch.sum(pred_masked == y_masked) / torch.numel(y_masked))
        cm = confusion_matrix(y_masked, pred_masked.numpy())
        tp = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        fp = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    '''
    if device == "cuda":
        torch.cuda.empty_cache()
torch.save(model.state_dict(), "rnn/rnn_weights_{}_{}_{}_{:.0e}.pth".format(num_stacked_layers, hidden_size, num_epochs, lr))
np.save("rnn/loss_list_{}_{}_{}_{:.0e}.npy".format(num_stacked_layers, hidden_size, num_epochs, lr), loss_list)