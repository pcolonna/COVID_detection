def train(epochs):
    print("Starting training...")

    for e in range(0, epochs):
        print("=" * 20)
        print(f"Starting epoch {e + 1} / {epochs}")
        print("=" * 20)

        train_loss = 0

        # Set the model to training again
        resnet18.train()

        # We enumerate over the data loader, so over eveything
        for train_step, (images, labels) in enumerate(dl_train):
            # we reinitialize the optimizer and set the grads to 0
            optimizer.zero_grad()

            # Now get the ouptus
            outputs = resnet18(images)

            loss = loss_fn(outputs, labels)

            # Now we take a gradient step
            loss.backward()
            optimizer.step()  # Will update the parameters values

            train_loss += loss.item()  # loss is a tensor, so append loss.item

            if train_step % 20 == 0:
                # Every twenty steps, evaluate the model
                print("Evaluating at step", train_step)
                acc = 0.0
                val_loss = 0.0
                resnet18.eval()

                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)

                    # we add to accuracy the number of correct predictions
                    # True is 1 and False 0
                    acc += sum((preds == labels).numpy())

                val_loss /= val_step + 1
                acc = acc / len(test_dataset)

                print(f"val loss: {val_loss:.4f}, Acc: {acc:.4f}")
                show_preds()

                resnet18.train()

                # We set a stop condition
                if acc > 0.95:
                    print("Performance condition satisfied....")
                    return

        train_loss /= train_test + 1

        print(f"Training loss: {train_loss:.4f}")
