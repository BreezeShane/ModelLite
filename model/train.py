"""
    Model loading and training(Fine-tuning) defitions.
"""
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from analysis.benchmark import evaluate_model
from schemas.context_classes import TrainingContextConfig

def _setup_fine_tuning(cfg: TrainingContextConfig, model):
    """ Define optim, scheduler and loss. """
    optimizer_cfg = dict(cfg.optimizer)
    optimizer_class = getattr(torch.optim, optimizer_cfg.pop("name"))
    param_groups_cfg = optimizer_cfg.pop("param_groups")
    param_groups = []
    for group in param_groups_cfg:
        group_dict = dict(group)
        param_names = group_dict.pop("param_names")
        params = [
            param
            for name, param in model.named_parameters()
            if any(param_name in name for param_name in param_names)
        ]
        param_groups.append({'params': params, **group_dict})

    optimizer = optimizer_class(param_groups, **optimizer_cfg)

    scheduler = None
    if cfg.scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)
        scheduler_kwargs = {**cfg.scheduler.required_params, **cfg.scheduler.optional_params}
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, scheduler, criterion

def fine_tune_the_model(cfg: TrainingContextConfig, model, train_loader, val_loader=None):
    """ Fine-tuning the model. """
    writer = SummaryWriter(log_dir=cfg.log_dir)

    optimizer, scheduler, criterion = _setup_fine_tuning(cfg, model=model)
    model.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.num_epochs}'):
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clip
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        if val_loader:
            val_accuracy = evaluate_model(model, val_loader, device=cfg.device)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        else:
            print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f"Epoch {epoch+1}/{cfg.num_epochs} completed. Learning rate: "
              f"{scheduler.get_last_lr()}. Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    writer.add_histogram(f'Weights/{name}', param, epoch)

    example_input, _ = next(iter(train_loader))
    example_input = example_input.to(cfg.device)
    writer.add_graph(model, example_input)

    writer.close()
    return model
