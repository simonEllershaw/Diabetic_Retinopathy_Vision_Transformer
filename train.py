import time
import torch
import os
import metrics
import visualisation

#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, nb_classes, writer, run_directory, warmup_steps, num_epochs_to_converge, grad_clip_norm=0):
    best_kappa = float('-inf')
    model_param_fname = os.path.join(run_directory, "model_params.pt")
    num_epochs_no_improvement = 0

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            confusion_matrix = torch.zeros(nb_classes, nb_classes)

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
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
                confusion_matrix = update_conf_matrix(confusion_matrix, labels, preds)
            if phase == 'train':
                writer.add_scalar(tag="general/lr", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = metrics.calc_accuracy(confusion_matrix)
            epoch_f1_macro = metrics.calc_macro_f1_score(confusion_matrix)
            epoch_kappa = metrics.calc_weighted_quadratic_kappa(confusion_matrix)
            confusion_matrix_vis = visualisation.plot_confusion_matrix(confusion_matrix, list(range(outputs.size(1))))

            writer.add_figure(tag="Confusion Matrix/" + phase, figure=confusion_matrix_vis, global_step=epoch)
            writer.add_scalar(tag=phase + "/loss", scalar_value=epoch_loss, global_step=epoch)
            writer.add_scalar(tag=phase + "/acc", scalar_value=epoch_acc, global_step=epoch)
            writer.add_scalar(tag=phase + "/f1_macro", scalar_value=epoch_f1_macro, global_step=epoch)
            writer.add_scalar(tag=phase + "/kappa", scalar_value=epoch_kappa, global_step=epoch)

            # deep copy the model
            if phase == 'val':
                if epoch_kappa > best_kappa:
                    best_kappa = epoch_kappa
                    torch.save(model.state_dict(), model_param_fname)
                    num_epochs_no_improvement = 0
                elif epoch > warmup_steps:
                    num_epochs_no_improvement += 1
            
        writer.add_scalar(tag="general/time", scalar_value=time.time()-epoch_start_time, global_step=epoch)
        if num_epochs_no_improvement == num_epochs_to_converge:
            break

    model.load_state_dict(torch.load(model_param_fname))
    model.eval()

    return model, best_kappa

def update_conf_matrix(confusion_matrix, labels, preds):
    for l, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[l, p] += 1
    return confusion_matrix



