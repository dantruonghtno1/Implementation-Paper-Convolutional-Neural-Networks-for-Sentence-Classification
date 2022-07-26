import random
import time
import torch.nn as nn 
import numpy as np 
import torch 
from utils import make_word2vec_vector_cnn

loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(args, model, optimizer, train_dataloader, val_dataloader=None, word2vec=None, padding_token=None, unk_token=None):

    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

    for epoch_i in range(args.epochs):
        t0_epoch = time.time()
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):          
            batch = make_word2vec_vector_cnn(args, batch, word2vec, padding_token, unk_token)
            b_input_ids, b_labels = batch[0], batch[1]
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to('cuda:0')
            model.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_dataloader)
        if val_dataloader is not None:
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

def evaluate(args, model, val_dataloader, word2vec, padding_token, unk_token):
    model.eval()

    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:
        batch = make_word2vec_vector_cnn(args, batch, word2vec, padding_token, unk_token)
        b_input_ids, b_labels = batch[0], batch[1]
        b_labels = b_labels.type(torch.LongTensor)
        b_labels = b_labels.to('cuda:0')

        with torch.no_grad():
            logits = model(b_input_ids)
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy