from transformer import TransformerEncoder
import torch
from torch import nn
import numpy as np


def train_model(model, epochs, train_data, train_masks, train_targets, dev_data, dev_masks, dev_targets, lr, batch_size):
    max_len = train_data.shape[1]
    embedding_dim = train_data.shape[2]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # Set model in training mode
    model.train()

    for i in range(epochs):
        # Shuffle train data 
        n_examples = len(train_data)-(len(train_data)%batch_size)
        # Shuffle index
        p = np.random.permutation(n_examples)
        # Get all batches in array form
        
        batches_x = torch.from_numpy(train_data[:n_examples][p].reshape(-1, batch_size, max_len, embedding_dim)).float()
        batches_mask = torch.from_numpy(train_masks[:n_examples][p].reshape(-1, batch_size, max_len))
        batches_y = torch.from_numpy(train_targets[:n_examples][p].reshape(-1, batch_size, max_len))
        
        train_loss = 0
        i = 0
        for batch_x, batch_mask, batch_y in zip(batches_x, batches_mask, batches_y):
            i += 1
            print(i / len(batches_x))
            # zero the parameter gradients
            optimizer.zero_grad()

            #prediction and loss
            pred = model(batch_x, batch_mask)
            loss_val = loss(pred.view((-1, 5)), batch_y.view((-1,)))
            #print(loss_val)
            #loss_val = loss_val[:, :]

            train_loss += loss_val

            #backward and updates
            loss_val.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.)
            optimizer.step()

        dev_pred = model(torch.Tensor(dev_data).float(), torch.Tensor(dev_masks))
        dev_loss = loss(dev_pred.view((-1, 5)), torch.LongTensor(dev_targets).view((-1,))).item()
        train_loss =  train_loss.item() / len(train_data)
        print("train_loss : ", train_loss, "dev_loss : ", dev_loss)


if __name__=='__main__':
    train_data = np.load("data/train_embeddings.npy")
    train_masks = np.array(np.load("data/train_masks.npy"), dtype=np.uint8)
    train_targets = np.load("data/train_targets.npy")
    dev_data = np.load("data/testa_embeddings.npy")
    dev_masks = np.array(np.load("data/testa_masks.npy"), dtype=np.uint8)
    dev_targets = np.load("data/testa_targets.npy")
    embedding_dim = train_data.shape[2]
    model = TransformerEncoder(2, embedding_dim, 8, 2*embedding_dim, 0.15, 5).float()

    #test_pred = model(torch.Tensor(train_data[:1]).float(), torch.Tensor(dev_masks[:1]))
    #print(dev_masks[0])
    #print(test_pred)

    train_model(model, 10, train_data, train_masks, train_targets, dev_data, dev_masks, dev_targets, 0.001, 16)