
import torch

from torch import nn

from tqdm.auto import tqdm

from torchmetrics import Accuracy

from torch.utils.tensorboard import SummaryWriter


def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device):
    
    accuracy = Accuracy(task="multiclass", num_classes=len(dataloader)).to(device)
    model.train()
    train_accTorchMetric = 0
    train_loss = 0

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_accTorchMetric += accuracy(y_pred.argmax(dim=1), y)
        train_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()
    train_accTorchMetric /= len(dataloader)
    print (f"TrainAccuracy:: {train_accTorchMetric *100:.2f}%")
    return train_loss, train_accTorchMetric

def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device):
    accuracy = Accuracy(task="multiclass", num_classes=len(dataloader)).to(device)
    model.eval()
    test_accTorchMetric = 0
    test_loss = 0
    
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_accTorchMetric+= accuracy(y_pred.argmax(dim=1), y)

        test_accTorchMetric /= len(dataloader)
        print (f"TestAccuracy:: {test_accTorchMetric *100:.2f}%")
        return test_loss, test_accTorchMetric
    
def train_model_epochs(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, writer: torch.utils.tensorboard.SummaryWriter, device, epochs: int=10) -> dict[str, list[float]]:

    accuracy = Accuracy(task="multiclass", num_classes=len(train_dataloader)).to(device)
    results = {"train_loss": [],
               "train_accuracy": [],
               "test_loss": [],
               "test_accuracy": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_acc)

        #Experiment Tracking
        if writer:
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss}, global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc}, global_step=epoch)
            writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))
            writer.close()
        else:
            pass
    
    writer.close()

    return results
