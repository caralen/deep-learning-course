import numpy as np
import torch
import torch.nn.utils as utils


def train(model, data, optimizer, criterion, args):
    model.train()
    model.float()
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x, y, _ = batch
        logits = model.forward(x)
        loss = criterion(logits, y.float())
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def evaluate(model, data, criterion, name):
    model.eval()
    with torch.no_grad():
        Y, Y_ = ([], [])
        for batch_num, batch in enumerate(data):
            x, y, _ = batch
            logits = model.forward(x)
            loss = criterion(logits, y.float())
            yp = torch.round(torch.sigmoid(logits)).int()

            Y += yp.detach().numpy().tolist()
            Y_ += y.detach().numpy().tolist()

        accuracy, f1, confusion_matrix = eval_perf_binary(np.array(Y), np.array(Y_))
        accuracy, f1 = [round(x*100, 3) for x in (accuracy, f1)]
        print(f'[{name}] accuracy: {accuracy}, f1: {f1}, confusion_matrix: {confusion_matrix}')
        return accuracy, f1


def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y == Y_, Y_ == True))
    fn = sum(np.logical_and(Y != Y_, Y_ == True))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))
    fp = sum(np.logical_and(Y != Y_, Y_ == False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp+fn + tn+fp)
    f1 = 2 * (recall*precision) / (recall + precision)
    confusion_matrix = [[tp, fp], [fn, tn]]
    return accuracy, f1, confusion_matrix