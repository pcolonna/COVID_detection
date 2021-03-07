import logging

import model
import torch
import utils
from early_stopping import EarlyStopping

logging.basicConfig(format="%(asctime)s     %(levelname)s   %(message)s", level=logging.INFO)


def main(dl_train, dl_test, test_dataset, epochs):
    logging.info("Starting training...")

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=2)

    modified_resnet, loss_fn, optimizer = model.create()

    for e in range(0, epochs):
        print("\n" + "=" * 20)
        print(f"Starting epoch {e + 1} / {epochs}")
        print("=" * 20)

        train_loss = 0

        # Set the model to training again
        modified_resnet.train()

        # We enumerate over the data loader, so over eveything
        for train_step, (images, labels) in enumerate(dl_train):
            # we reinitialize the optimizer and set the grads to 0
            optimizer.zero_grad()

            # Now get the ouptus
            outputs = modified_resnet(images)

            loss = loss_fn(outputs, labels)

            # Now we take a gradient step
            loss.backward()
            optimizer.step()  # Will update the parameters values

            train_loss += loss.item()  # loss is a tensor, so append loss.item

            logging.info(f"Training loss: {train_loss / (train_step + 1)}")

        logging.info(f"Evaluating at end of epoch {e + 1}")

        acc = 0.0
        val_loss = 0.0
        modified_resnet.eval()

        for val_step, (images, labels) in enumerate(dl_test):
            outputs = modified_resnet(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            # we add to accuracy the number of correct predictions
            # True is 1 and False 0
            acc += sum((preds == labels).numpy())

        val_loss /= val_step + 1
        acc = acc / len(test_dataset)

        logging.info(f"Validation loss: {val_loss:.4f}, Accuracy: {acc:.4f}")
        # utils.show_preds(modified_resnet, dl_test)

        modified_resnet.train()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, acc, modified_resnet)

        if early_stopping.early_stop:
            break
