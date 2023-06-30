import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def get_incorrect_predictions(
    device:torch.device,
    loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.functional,
) -> list:
    """Get all incorrect predictions.
    https://github.com/parrotletml/era_session_seven/blob/main/mnist/utils.py#L111-L135
    """
    model.eval()
    incorrects = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrects.append([
                        d.cpu(),
                        t.cpu(),
                        p.cpu(),
                        o[p.item()].cpu(),
                    ])
    return incorrects


def find_learning_rates(
    device:torch.device,
    train_loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.functional,
    optimizer:torch.optim.Optimizer,
    init_value:float=1e-8,
    final_value:float=10.,
    beta: float=0.98,
) -> tuple[list[float], list[float]]:
    """Compute loss values for each learning rate value.
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    trn_loader = train_loader
    net = model
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    lrs = []
    for data in trn_loader:
        batch_num += 1
        # Get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # Compute the smoothed loss
        avg_loss = beta*avg_loss+(1-beta)*loss.item()
        smoothed_loss = avg_loss/(1-beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4*best_loss:
            return lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        lrs.append(lr)
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return lrs, losses


def plot_batch_samples(
    dataloader: torch.utils.data.DataLoader,
) -> None:
    """Plot sample images from a batch."""
    batch_data, batch_label = next(iter(dataloader))
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0).permute(1, 2, 0))
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def plot_learning_rates(
    lrs:list[float],
    losses:list[float],
) -> None:
    """Plot loss vs learning rate for the learning rate finder.
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    x = lrs
    y = losses
    plt.plot(x, y)
    plt.xscale('log')
    plt.xticks([10**-exponent for exponent in range(-1, 9)])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    ymin = min(y)
    xmin = x[y.index(ymin)]
    plt.title(f"Minimum loss = {ymin:.02f} at learning rate = {xmin:.02f}")
    plt.axhline(y=ymin, color='red', linestyle='dotted', alpha=0.5)
    plt.axvline(x=xmin, color='red', linestyle='dotted', alpha=0.5)
    plt.show()


def plot_learning_curves(
    results: dict[str, list[float]],
    epoch:int,
) -> None:
    """Plot training and test losses and accuracies."""
    epochs = range(1, epoch+1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(epochs, results['train_loss'], label='Train', marker='.')
    axs[0].plot(epochs, results['test_loss'], label='Test', marker='.')
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Average loss")
    axs[0].set_ylim(bottom=0, top=None)
    axs[0].grid()
    axs[1].plot(epochs, results['train_acc'], label='Train', marker='.')
    axs[1].plot(epochs, results['test_acc'], label='Test', marker='.')
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_ylim(bottom=None, top=100)
    axs[1].grid()
    plt.setp(axs, xticks=range(5, epoch+1, 5))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_incorrect_predictions(
    predictions:list,
    classes:list[str],
    count:int=10,
) -> None:
    """Plot incorrect predictions.
    https://github.com/parrotletml/era_session_seven/blob/main/mnist/utils.py#L111-L135
    """
    print(f'Total Incorrect Predictions {len(predictions)}')
    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return
    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'Guess: {classes[p.item()]}\nRight: {classes[t.item()]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break