import time
import torch
import os
import metrics
import visualisation

#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, nb_classes, writer, run_directory, warmup_steps, num_epochs_to_converge, accumulation_steps, grad_clip_norm=0):
    best_loss = float('inf')
    model_param_fname = os.path.join(run_directory, "model_params.pt")
    num_epochs_no_improvement = 0

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                mini_batch_num = 1
            else:
                model.eval()   
            running_loss = 0.0
            confusion_matrix = torch.zeros(nb_classes, nb_classes)
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        if (mini_batch_num) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            mini_batch_num = 0
                            scheduler.step()
                        mini_batch_num += 1
                # statistics
                running_loss += loss.item() * inputs.size(0) * accumulation_steps
                confusion_matrix = metrics.update_conf_matrix(confusion_matrix, labels, preds)
            epoch_loss = running_loss / dataset_sizes[phase]
            print(epoch, phase, epoch_loss)  
            if phase == 'train':
                # Update LR
                writer.add_scalar(tag="general/lr", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)
            elif phase == 'val':
                # Check if model performance has improved if so save model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), model_param_fname)
                    num_epochs_no_improvement = 0
                    best_conf_matrix = confusion_matrix
                    best_epoch = epoch
                elif epoch > warmup_steps:
                    num_epochs_no_improvement += 1
            # Log epoch statistics
            class_labels = list(range(outputs.size(1)))
            write_epoch_statistics_to_tensorboard(writer, phase, epoch, epoch_loss, confusion_matrix, class_labels)
        writer.add_scalar(tag="general/time", scalar_value=time.time()-epoch_start_time, global_step=epoch)
        if num_epochs_no_improvement == num_epochs_to_converge:
            break
    # Return best model and perf metric at end of training
    confusion_matrix_vis = visualisation.plot_confusion_matrix(best_conf_matrix, class_labels)
    writer.add_figure(tag="Confusion Matrix/" + phase, figure=confusion_matrix_vis, global_step=100+best_epoch)
    model.load_state_dict(torch.load(model_param_fname))
    model.eval()
    return model, best_loss  

def write_epoch_statistics_to_tensorboard(writer, phase, epoch, epoch_loss, confusion_matrix, class_labels):
    # Calc statistics
    confusion_matrix_vis = visualisation.plot_confusion_matrix(confusion_matrix, class_labels)
    #Write to tensorboard
    writer.add_figure(tag="Confusion Matrix/" + phase, figure=confusion_matrix_vis, global_step=epoch)
    writer.add_scalar(tag=phase + "/loss", scalar_value=epoch_loss, global_step=epoch)



