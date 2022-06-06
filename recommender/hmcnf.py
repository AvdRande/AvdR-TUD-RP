import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from helper_fun import binlabels_to_text, target_labels_at_level, get_lvlsizes_from_tree

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, Pg, Pl1, Pl2, Pl3, Pl4):
        "Initialization. All items are in parallel."
        # self.data = torch.from_numpy(np.array(data))
        # self.Pg = torch.from_numpy(np.array(Pg))
        # self.Pl1 = torch.from_numpy(np.array(Pl1))
        # self.Pl2 = torch.from_numpy(np.array(Pl2))
        # self.Pl3 = torch.from_numpy(np.array(Pl3))
        self.data = torch.FloatTensor(data)
        self.Pg = torch.FloatTensor(Pg)
        self.Pl1 = torch.FloatTensor(Pl1)
        self.Pl2 = torch.FloatTensor(Pl2)
        self.Pl3 = torch.FloatTensor(Pl3)
        self.Pl4 = torch.FloatTensor(Pl4)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.data[index]
        Pg = self.Pg[index]
        Pl1 = self.Pl1[index]
        Pl2 = self.Pl2[index]
        Pl3 = self.Pl3[index]
        Pl4 = self.Pl4[index]
        target = (Pg, Pl1, Pl2, Pl3, Pl4)

        return data, target

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), [target[0].to(device), target[1].to(device), target[2].to(device), target[3].to(device), target[4].to(device)]
        optimizer.zero_grad()
        Pg, Pl1, Pl2, Pl3, Pl4 = model(data.float(), training=True)
        loss = custom_loss(Pg, Pl1, Pl2, Pl3, Pl4, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

class Net(nn.Module):

    def __init__(self, C, Cl1, Cl2, Cl3, Cl4, dropout, h_size, n_dim, reduced=False):
        super(Net, self).__init__()
        self.C = C     # total number of classes, including hierachical labels
        self.Cl1 = Cl1
        self.Cl2 = Cl2
        self.Cl3 = Cl3
        self.Cl4 = Cl4
        self.reduced = reduced
        self.dropout = dropout
        self.h_size = h_size
        self.x_size = n_dim
        
        self.global1 = nn.Linear(self.x_size, h_size)
        self.batch_norm1 = nn.BatchNorm1d(h_size)
        self.global2 = nn.Linear(h_size + self.x_size, h_size)
        self.batch_norm2 = nn.BatchNorm1d(h_size)
        self.global3 = nn.Linear(h_size + self.x_size, h_size)
        self.batch_norm3 = nn.BatchNorm1d(h_size)
        self.global4 = nn.Linear(h_size + self.x_size, h_size)
        self.batch_norm4 = nn.BatchNorm1d(h_size)
        self.globalOut = nn.Linear(h_size, self.C)
        
        self.local1 = nn.Linear(h_size, h_size)
        self.batch_normL1 = nn.BatchNorm1d(h_size)
        self.localOut1 = nn.Linear(h_size, self.Cl1)
        self.local2 = nn.Linear(h_size, h_size)
        self.batch_normL2 = nn.BatchNorm1d(h_size)
        self.localOut2 = nn.Linear(h_size, self.Cl2)
        self.local3 = nn.Linear(h_size, h_size)
        self.batch_normL3 = nn.BatchNorm1d(h_size)
        self.localOut3 = nn.Linear(h_size, self.Cl3)
        self.local4 = nn.Linear(h_size, h_size)
        self.batch_normL4 = nn.BatchNorm1d(h_size)
        self.localOut4 = nn.Linear(h_size, self.Cl4)

    def forward(self, x, training=True):
        Ag1 = F.dropout(self.batch_norm1(F.relu(self.global1(x))), p=self.dropout, training=training)
        Al1 = F.dropout(self.batch_normL1(F.relu(self.local1(Ag1))), p=self.dropout, training=training)
        Pl1 = torch.sigmoid(self.localOut1(Al1))

        Ag2 = F.dropout(self.batch_norm2(F.relu(self.global2(torch.cat([Ag1, x], dim=1)))), p=self.dropout, training=training)
        Al2 = F.dropout(self.batch_normL2(F.relu(self.local2(Ag2))), p=self.dropout, training=training)
        Pl2 = torch.sigmoid(self.localOut2(Al2))
    
        Ag3 = F.dropout(self.batch_norm3(F.relu(self.global3(torch.cat([Ag2, x], dim=1)))), p=self.dropout, training=training)
        Al3 = F.dropout(self.batch_normL3(F.relu(self.local3(Ag3))), p=self.dropout, training=training)
        Pl3 = torch.sigmoid(self.localOut3(Al3))
    
        Ag4 = F.dropout(self.batch_norm4(F.relu(self.global4(torch.cat([Ag3, x], dim=1)))), p=self.dropout, training=training)
        Al4 = F.dropout(self.batch_normL4(F.relu(self.local4(Ag4))), p=self.dropout, training=training)
        Pl4 = torch.sigmoid(self.localOut4(Al4))
        
        Pg = torch.sigmoid(self.globalOut(Ag4))
        
        return Pg, Pl1, Pl2, Pl3, Pl4    # return all outputs to compute loss

def train(train_feature_vector, train_labels, hierarchy, label_names, epochs):
    tree_sizes = get_lvlsizes_from_tree(hierarchy) + [len(train_labels[0])]
    model = Net(
        sum(tree_sizes),
        tree_sizes[0],
        tree_sizes[1],
        tree_sizes[2],
        tree_sizes[3],
        0.2,
        # np.random.randint(128, 384),
        sum(tree_sizes),
        len(train_feature_vector[0])
    )
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    labels_at_lvls = [
        [target_labels_at_level(binlabels_to_text(label, label_names), hierarchy, i) for label in train_labels] 
        for i in range(4)]

    training_set = Dataset(
        train_feature_vector,
        [labels_at_lvls[3][i] + labels_at_lvls[2][i] + labels_at_lvls[1][i] + labels_at_lvls[0][i] for i in range(len(labels_at_lvls[0]))],
        labels_at_lvls[0],
        labels_at_lvls[1],
        labels_at_lvls[2],
        labels_at_lvls[3]
    )

    for epoch in range(epochs):
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=10)
        train_model(model, device, train_loader, optimizer, epoch)

    return model

def predict(model, test_feature_vector, depth):
    predictions = []
    model.eval()
    beta = 0.5

    test_feature_tensor = torch.from_numpy(test_feature_vector).type(torch.FloatTensor)

    full_pred, predat0, predat1, predat2, predat3 = model.forward(test_feature_tensor, training=False)

    test_set = Dataset(
        test_feature_tensor,
        full_pred,
        predat0,
        predat1,
        predat2,
        predat3
    )

    test_loader = torch.utils.data.DataLoader(test_set)

    with torch.no_grad():
        for data, target in test_loader:
            Pg, Pl1, Pl2, Pl3, Pl4 = model(data.float(), training=False)
            predictions.append((beta*(torch.from_numpy(np.concatenate((Pl4[0], Pl3[0], Pl2[0], Pl1[0])))) + (1 - beta)*Pg[0]).tolist()[:220])

    return predictions

def get_name():
    return "HMCN-F"


def custom_loss(Pg, Pl1, Pl2, Pl3, Pl4, target):
    return F.binary_cross_entropy(
        Pg, 
        target[0].float(), 
    ) + F.binary_cross_entropy(
        Pl1, 
        target[1].float(), 
    ) + F.binary_cross_entropy(
        Pl2, 
        target[2].float(), 
    ) + F.binary_cross_entropy(
        Pl3, 
        target[3].float(), 
    ) + F.binary_cross_entropy(
        Pl4, 
        target[4].float(), 
    )

