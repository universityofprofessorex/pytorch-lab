"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch.profiler

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from icecream import ic
import pyfiglet
from rich import print
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from torchmetrics import Accuracy

# SOURCE: https://colab.research.google.com/drive/1nCj54XryHcoMARS4cSxivn3Ci1I6OtvO?usp=sharing#scrollTo=i4a9YMBCToGc
def calculate_IoU(bb1, bb2):
    # calculate IoU(Intersection over Union) of 2 boxes
    # **IoU = Area of Overlap / Area of Union
    # https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb

    (
                bb1_xmin,
                bb1_ymin,
                bb1_xmax,
                bb1_ymax,
    ) = bb1

    (
                bb2_xmin,
                bb2_ymin,
                bb2_xmax,
                bb2_ymax,
    ) = bb2

    x_left = max(bb1_xmin, bb2_xmin)
    y_top = max(bb1_ymin, bb2_ymin)
    x_right = min(bb1_xmax, bb2_xmax)
    y_bottom = min(bb1_ymax, bb2_ymax)
    # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # calculate Overlapping area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1_xmax - bb1_xmin) * (bb1_ymax - bb1_ymin)
    bb2_area = (bb2_xmax - bb2_xmin) * (bb2_ymax - bb2_ymin)
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area


def display_ascii_text(txt: str, font: str = "stop"):
    title = pyfiglet.figlet_format(txt, font=font)
    print(f"[magenta]{title}[/magenta]")


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # display_ascii_text("train_step")
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./runs/profiler", worker_name="worker0"
        ),
        # save information about operator's input shapes.
        record_shapes=True,
        #  track tensor memory allocation/deallocation.
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        # record source information (file and line number) for the ops.
        with_stack=True,
    ) as prof:
        # Loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            # X, y = X.to(device), y.to(device)
            # TODO: Might have to remove non_blocking=True

            # Here the .to() method not only takes the device, but also sets non_blocking=True, which enables asynchronous data copies to GPU from pinned memory, hence allowing the CPU to keep operating during the transfer; non_blocking=True is simply a no-op otherwise.
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
            prof.step()

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # display_ascii_text("test_step")
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            # X, y = X.to(device), y.to(device)
            # TODO: Might have to remove non_blocking=True
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# 1. Take in various parameters required for training and test steps
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: SummaryWriter,  # new parameter to take in a writer
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    writer: A SummaryWriter() instance to log model results to.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """

    # display_ascii_text("train")
    ic(
        f"[INFO] Training model {model.__class__.__name__} on device '{device}' for {epochs} epochs..."
    )
    # ic(model)
    # ic(train_dataloader)
    # ic(test_dataloader)
    # ic(optimizer)
    # ic(loss_fn)
    # ic(epochs)
    # ic(device)

    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Make sure model on target device
    model.to(device)

    # ############## TENSORBOARD ########################
    # if writer:
    #     images_test_dataloader, labels_test_data_loader = next(iter(test_dataloader))
    #     # example_data, example_targets = examples.next()
    #     grid = torchvision.utils.make_grid(images_test_dataloader)
    #     writer.add_image('twitter_facebook_tiktok_images', grid, 0)
    #     # writer.add_graph(model, images_test_dataloader)
    # ###################################################

    # ############## TENSORBOARD ########################
    # if writer:
    #     writer.add_graph(model)
    # ###################################################

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(
            f"[INFO] train_step for model {model.__class__.__name__} on device '{device}' epoch={epoch}..."
        )
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"[INFO] test_step for model {model.__class__.__name__} on device '{device}' epoch={epoch}..."
        )
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )

            # Close the writer
            writer.close()
        else:
            pass
    ### End new ###

    # Return the filled results at the end of the epochs
    return results


def train_localization_fn(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    # writer: SummaryWriter = None,  # new parameter to take in a writer
):

    total_loss = 0.0

    # accuracy_counter = Accuracy()

    # Put model in train mode
    model.train()  # Dropout On


    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./runs/profiler", worker_name="cropworker0"
        ),
        # save information about operator's input shapes.
        record_shapes=True,
        #  track tensor memory allocation/deallocation.
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        # record source information (file and line number) for the ops.
        with_stack=True,
    ) as prof:

        for data in tqdm(dataloader):

            # Send data to target device
            images, gt_bboxes = data
            images, gt_bboxes = (
                images.to(device, non_blocking=True),
                gt_bboxes.to(device, non_blocking=True),
            )

            # 1. Forward pass
            bboxes, loss = model(images, gt_bboxes)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            total_loss += loss.item()
            prof.step()

    return total_loss / len(dataloader)


def eval_localization_fn(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):

    total_loss = 0.0

    # Put model in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()  # detected bounding boxes
    det_scores = list()  # detected scores
    true_boxes = list()  # ground_truth bounding boxes

    # with torch.no_grad():
    with torch.inference_mode():
        for data in tqdm(dataloader):

            images, gt_bboxes = data
            images, gt_bboxes = (
                images.to(device, non_blocking=True),
                gt_bboxes.to(device, non_blocking=True),
            )

            bboxes, loss = model(images, gt_bboxes)

            # import bpdb

            # bpdb.set_trace()

            # loss is a predicted score
            # predicted_scores

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_localization(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    validloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    # loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    # writer: SummaryWriter,
    writer: SummaryWriter = None,  # new parameter to take in a writer
):

    best_valid_loss = np.Inf

    ic(
        f"[INFO] Training model {model.__class__.__name__} on device '{device}' for {epochs} epochs..."
    )

    # Make sure model on target device
    model.to(device)

    for epoch in tqdm(range(epochs)):

        print(
            f"[INFO] train_step for model {model.__class__.__name__} on device '{device}' epoch={epoch}..."
        )
        train_loss = train_localization_fn(model, trainloader, optimizer, device)
        print(
            f"[INFO] test_step for model {model.__class__.__name__} on device '{device}' epoch={epoch}..."
        )
        valid_loss = eval_localization_fn(model, validloader, device)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), "screencropnet_best_model.pt")
            print("WEIGHTS-ARE-SAVED")
            best_valid_loss = valid_loss

        print(f"Epoch : {epoch + 1} train loss : {train_loss} valid loss : {valid_loss}")



        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": valid_loss},
                global_step=epoch,
            )
            # writer.add_scalars(
            #     main_tag="Accuracy",
            #     tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
            #     global_step=epoch,
            # )

            # Close the writer
            writer.close()
        else:
            pass
    ### End new ###
