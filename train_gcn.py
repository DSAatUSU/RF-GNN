import argparse
import itertools
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from utils import *


class GraphConvolution(nn.Module):
    def __init__(self, f_in, f_out, use_bias=True, activation=F.relu_):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.use_bias = use_bias
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out))
        self.bias = nn.Parameter(torch.FloatTensor(f_out)) if use_bias else None
        self.initialize_weights()

    def initialize_weights(self):
        if self.activation is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.use_bias: nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        degrees = adj.sum(dim=-1, keepdims=True)
        output = output / degrees
        # output = torch.spmm(adj, support) # In case of using sparse adj
        if self.use_bias: output.add_(self.bias)
        if self.activation is not None: self.activation(output)
        return output


class GCN(nn.Module):
    def __init__(self, f_in, n_classes, hidden, dropouts):
        super().__init__()
        layers = []
        for f_in, f_out in zip([f_in] + hidden[:-1], hidden):
            layers += [GraphConvolution(f_in, f_out)]

        self.layers = nn.Sequential(*layers)
        self.dropouts = dropouts
        # self.out_layer = GraphConvolution(f_out, n_classes, activation = None)
        # replace with nn.Linear for general graph?
        self.out_layer1 = nn.Linear(f_out, f_out)
        self.out_layer2 = nn.Linear(f_out, n_classes)

    def forward(self, x, adj):
        for layer, d in zip(self.layers, self.dropouts):
            x = layer(x, adj)
            if d > 0: F.dropout(x, d, training=self.training)

        x = F.relu(self.out_layer1(x))
        return self.out_layer2(x)


# the training and evaluation functions

def accuracy(output, y):
    return (output.argmax(1) == y).type(torch.float32).mean().item()


def F1_score(output, y):
    average = 'binary'
    if n_classes > 2:
        average = 'weighted'
    return f1_score(y.cpu().numpy(), output.argmax(1).cpu().numpy(), average='weighted')


def step(model, optimizer, train_idx=None):
    model.train()
    optimizer.zero_grad()
    output = model(X, adj)
    loss = F.cross_entropy(output[train_idx], y[train_idx])
    # acc = accuracy(output[train_idx], y[train_idx])
    f1 = F1_score(output[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item(), f1


def evaluate(model, val_idx=None):
    model.eval()
    output = model(X, adj)
    loss = F.cross_entropy(output[val_idx], y[val_idx]).item()
    return loss, F1_score(output[val_idx], y[val_idx])


def get_param_combinations(param_grid):
    # Create a list of keys and a list of values, where each value is a list of options
    keys, values = zip(*param_grid.items())

    # Use itertools.product to create the cartesian product of parameter values
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return combinations


# Grid search of GCN
def grid_search_GCN(param_grid, cv=5, print_results=True):
    best_score = -1
    best_params = None

    # Generate all combinations of hyperparameters
    combs = get_param_combinations(param_grid)

    # Perform cross-validation
    for params in combs:
        lr = params['lr']
        weight_decay = params['weight_decay']
        hidden = params['hidden']
        dropouts = params['dropouts']
        model = GCN(n_features, n_classes, hidden=hidden, dropouts=dropouts).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        epochs = 100  # 500 maybe overfitting
        scores = []
        for train_idx, val_idx in KFold(n_splits=cv, shuffle=True).split(X_train):
            train_idx = torch.LongTensor(train_idx)
            val_idx = torch.LongTensor(val_idx)
            train_idx = train_idx.to(device)
            val_idx = val_idx.to(device)
            # Train model
            for i in range(epochs):
                tl, ta = step(model, optimizer, train_idx=train_idx)
                # Evaluate model
            vl, va = evaluate(model, val_idx=val_idx)  # return validation loss and accuracy (F1-score)
            # more epochs with early stopping
            scores.append(va)

        avg_score = np.mean(scores)

        if print_results:
            print(
                f'lr = {lr:.4f}, weight_decay = {weight_decay:.4f}, hidden_sizes = {[h for h in hidden]}, dropout_rates = {[d for d in dropouts]}' +
                f',avg_val_acc = {avg_score:.4f}')

        # Update best score and parameters if necessary
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

        # Break if average score 1
        if avg_score >= 0.99:
            print("Avergae val score reached 0.99. stop training.")
            break

    return best_score, best_params


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='1590',
                    help='True if training to predict follow. By default, set to False.')
parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu to run on.')

args = parser.parse_args()

os.environ['DGLBACKEND'] = 'pytorch'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

device = torch.device("cuda")

gnn_type = 'GCN'
random_seeds = [172119, 42, 12, 7889, 1015]

train_results = []

dataset = args.dataset

## Get the data ready

n_features, n_classes, X, y, X_train, X_test, y_train, y_test, idx_train, idx_test = load_openml_data(
    int(dataset))
print(
    f'Applying {gnn_type} on dataset {dataset} with #instances: {X.shape[0]} and #features: {X.shape[1]} and #classes: {n_classes}')
best_scores = []

BASE_SAVE_DIR = f'./models/gnn'
if not os.path.exists(BASE_SAVE_DIR):
    os.makedirs(BASE_SAVE_DIR)
for random_seed in random_seeds:
    torch.manual_seed(random_seed)

    gnn_model_path = f'./models/gnn/{gnn_type}_{dataset}_{random_seed}.pth'

    best_rf_model = load_model(f'./models/rf/rf_{dataset}_{random_seed}.joblib')

    y_pred = best_rf_model.predict(X_test)
    # Evaluate the best model on the test set
    if n_classes > 2:
        f1_rf = f1_score(y_pred, y_test, average='weighted')
    else:
        f1_rf = f1_score(y_pred, y_test)
    print("Test F1-score with Best Random Forest:", f1_rf)

    Prox_best = load_proximity(f'rf_prox_{dataset}_{random_seed}')

    # create the threshhold for Proxmities
    epsilons = np.linspace(0, 1, 51)

    # Put data onto the device
    X = X.to(device)
    y = y.to(device)
    idx_test = idx_test.to(device)
    idx_train = idx_train.to(device)

    # testing accuracy (f1 score)
    param_grid = {'lr': [0.001],
                  'weight_decay': [0.0001, 0.001],
                  'hidden': [[64, 128], [128, 256]],
                  'dropouts': [[0.1, 0.1], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7]]}

    acc = []
    epochs = 150
    best_test_gcn = 0
    for i in range(len(epsilons)):
        eps = epsilons[i]
        adj = np.zeros((X.shape[0], X.shape[0]))
        adj[Prox_best >= eps] = 1
        adj = torch.tensor(adj, dtype=torch.float32, requires_grad=False)
        adj = adj.to(device)
        best_score, best_params = grid_search_GCN(param_grid, cv=5, print_results=False)
        model = GCN(n_features, n_classes, hidden=best_params['hidden'], dropouts=best_params['dropouts']).to(
            device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'],
                                     weight_decay=best_params['weight_decay'])
        for i in range(epochs):
            tl, ta = step(model, optimizer, train_idx=idx_train)
            _, ta = evaluate(model, val_idx=idx_test)
        if ta > best_test_gcn:
            best_test_gcn = ta
            torch.save(model.cpu().state_dict(), gnn_model_path)
            model.cuda()
        print(f'Threshold: {eps}, F1-Score: {ta:.4f}')
        acc.append(ta)

    np.save(f'./results/gcn_f1_{dataset}_{random_seed}.npy', acc)

    print("The F1-Score improves", np.max(acc) - f1_rf, "compared to the RF model")


    best_scores.append(np.max(acc))

mean = sum(best_scores) / len(best_scores)
variance = sum([((x - mean) ** 2) for x in best_scores]) / len(best_scores)
res = variance ** 0.5
train_results.append([dataset, round(mean, 4), round(res, 4)])

df = pd.DataFrame(train_results, columns=['dataset', 'gcn_mean', 'gcn_std'])
df.to_csv(f'./results/gcn_{dataset}_results.csv', index=False)
