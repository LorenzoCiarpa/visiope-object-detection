from tqdm import tqdm
from hyperparams import hypers

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, position= 0, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(hypers.DEVICE), y.to(hypers.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())


    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    
    