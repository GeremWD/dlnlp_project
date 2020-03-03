from transformer import TransformerEncoder
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def train_model(model, epochs, train_data, train_masks, train_targets, dev_data, dev_masks, dev_targets, lr, batch_size):
    max_len = train_data.shape[1]
    embedding_dim = train_data.shape[2]
    dev_data = torch.Tensor(dev_data).float().cuda()
    dev_masks = torch.Tensor(dev_masks).cuda()
    dev_targets = torch.LongTensor(dev_targets).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(ignore_index=-1)

    # Shuffle train data 
    n_examples = len(train_data)-(len(train_data)%batch_size)
    # Shuffle index
    p = np.random.permutation(n_examples)
    # Get all batches in array form
    batches_x = torch.from_numpy(train_data[:n_examples][p].reshape(-1, batch_size, max_len, embedding_dim)).float().cuda()
    batches_mask = torch.from_numpy(train_masks[:n_examples][p].reshape(-1, batch_size, max_len)).cuda()
    batches_y = torch.from_numpy(train_targets[:n_examples][p].reshape(-1, batch_size, max_len)).cuda()

    # Set model in training mode
    model.train()
    x = []
    y = []
    for i in range(epochs):
        train_loss = 0
        j = 0
        for batch_x, batch_mask, batch_y in zip(batches_x, batches_mask, batches_y):
            j += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            #prediction and loss
            pred = model(batch_x, batch_mask)
            loss_val = loss(pred.view((-1, 5)), batch_y.view((-1,)))
            train_loss += loss_val

            #backward and updates
            loss_val.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.)
            optimizer.step()
            
        dev_pred = model(dev_data, dev_masks)
        dev_loss = loss(dev_pred.view((-1, 5)), dev_targets.view(-1,)).item()
        _, predicted = torch.max(dev_pred, 2)
        total = dev_targets.view((-1,)).size(0)
        dev_accuracy = (predicted == dev_targets).sum().item() / total
        precision = ((predicted > 0) * (predicted == dev_targets)).sum().item()
        precision_total = (predicted > 0).sum().item()
        if precision_total > 0:
            precision /= precision_total
        recall = ((dev_targets > 0) * (predicted == dev_targets)).sum().item()
        recall_total = (dev_targets > 0).sum().item()
        if recall_total > 0:
            recall /= recall_total
        dev_f1 = (2 * precision * recall)
        if recall + precision > 0:
            dev_f1 /= recall + precision

        train_loss =  train_loss.item() / len(train_data)
        x.append(i+1)
        y.append(dev_f1)
        print("Epoch", i+1, " train_loss :", train_loss, " dev_loss :", dev_loss, " dev_accuracy :", dev_accuracy, " dev_f1 :", dev_f1)

    plt.xlabel("Epochs")
    plt.ylabel("F1 score")
    plt.plot(x, y)


if __name__=='__main__':
    train_data = np.load("data/train_embeddings.npy")
    train_masks = np.array(np.load("data/train_masks.npy"), dtype=np.uint8)
    train_targets = np.load("data/train_targets.npy")
    dev_data = np.load("data/testa_embeddings.npy")
    dev_masks = np.array(np.load("data/testa_masks.npy"), dtype=np.uint8)
    dev_targets = np.load("data/testa_targets.npy")
    embedding_dim = train_data.shape[2]
    model = TransformerEncoder(2, embedding_dim, 8, 2*embedding_dim, 0.15, 5).float().cuda()
    train_model(model, 20, train_data, train_masks, train_targets, dev_data, dev_masks, dev_targets, 0.01, 16)
    