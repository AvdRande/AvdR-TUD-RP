import math
import torch.nn as nn
import torch
import torch.utils.data

from helper_fun import *

def train(train_feature_vector, train_labels, hierarchy, label_names, epochs):
    train_labels_with_parents = np.array([add_parents_to_labels(train_label, hierarchy, label_names) for train_label in train_labels])
    h_m = torch.from_numpy(make_hier_matrix(hierarchy, len(train_labels_with_parents[0])).T) # may have to transpose
    
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

    hyperparams = {
        'batch_size': 4,
        'num_layers': 3,
        'dropout': 0.7,
        'non_lin': 'relu',
        'hidden_dim': math.floor(len(train_feature_vector[0])/2),
        'lr': 0.0001,
        'weight_decay': 1e-05
    }

    train_loader = torch.utils.data.DataLoader(dataset=list(zip(train_feature_vector, train_labels_with_parents)),
                                        batch_size=hyperparams['batch_size'], 
                                        shuffle=True)
    
    model = ConstrainedFFNNModel(
        len(train_feature_vector[0]),
        hyperparams['hidden_dim'],
        len(train_labels_with_parents[0]),
        hyperparams,
        h_m 
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
    criterion = nn.BCELoss()

    model.to(device)

    for epoch in range(epochs):
        print("Epoch,", epoch)
        model.train()

        for i, (x, labels) in enumerate(train_loader):

            x = x.to(device)
            labels = labels.to(device)
        
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            output = model(x.float())

            #MCLoss
            constr_output = get_constr_out(output, h_m)
            train_output = labels*output.double()
            train_output = get_constr_out(train_output, h_m)
            train_output = (1-labels)*constr_output.double() + labels*train_output

            loss = criterion(train_output, labels.type(torch.DoubleTensor)) 

            predicted = constr_output.data > 0.5

            # Total number of labels
            total_train = labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train = (predicted == labels.byte()).sum()

            loss.backward()
            optimizer.step()
    return model

def predict(model, test_feature_vector):  
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    predictions = []
    for i, x in enumerate(torch.from_numpy(test_feature_vector)):
        model.eval()
                
        x = x.to(device)

        predictions.append(model(x.float()).detach().numpy()[0])
    print("debug")
    return np.array([row[-220:] for row in predictions])

# code kindly used from https://github.com/EGiunchiglia/C-HMCNN/blob/master/main.py
class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()
        
        self.nb_layers = hyperparams['num_layers']
        self.R = R
        
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        
        self.drop = nn.Dropout(hyperparams['dropout'])
        
        
        self.sigmoid = nn.Sigmoid()
        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out