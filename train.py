import time
import torch
import os

#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, writer, run_directory, grad_clip_norm=0):
    best_acc = 0.0
    model_param_fname = os.path.join(run_directory, "model_params.pt")

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                writer.add_scalar(tag="general/lr", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writer.add_scalar(tag=phase + "/loss", scalar_value=epoch_loss, global_step=epoch)
            writer.add_scalar(tag=phase + "/acc", scalar_value=epoch_acc, global_step=epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), model_param_fname)
        
        writer.add_scalar(tag="general/time", scalar_value=time.time()-epoch_start_time, global_step=epoch)

        model.load_state_dict(torch.load(model_param_fname))
        model.eval()

    return model, best_acc