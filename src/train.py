import torch
from torch import nn, optim
from tqdm.auto import tqdm
from timeit import default_timer as timer


def accuracy_func(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def print_train_time(start, end, device=None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_func(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y)
            test_acc += accuracy_func(y, y_pred.argmax(dim=1))

    return test_loss / len(dataloader), test_acc / len(dataloader)


def train_loop(model, train_dataloader, test_dataloader, device, epochs=10, lr=0.0001):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    start = timer()
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        train_losses.append(train_loss.detach().cpu().numpy())
        test_losses.append(test_loss.detach().cpu().numpy())
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    total_time = print_train_time(start, timer(), device)
    return {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "test_loss": test_losses,
        "test_acc": test_accuracies,
        "train_time": total_time
    }
