"""One train/validation pass using the existing YOLO detection loss."""

from itertools import islice

import torch
from tqdm import tqdm


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              ema=None, max_steps=None):
    training = optimizer is not None
    model.train(training)
    total_loss, steps = 0.0, 0
    total = min(len(loader), max_steps) if max_steps else len(loader)
    batches = islice(loader, max_steps) if max_steps else loader
    progress = tqdm(batches, total=total, desc="train" if training else "val")
    for images, targets, y_true in progress:
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        y_true = [target.to(device) for target in y_true]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                outputs = model(images)[:3]
                loss = sum(criterion(level, output, targets, y_true[level])
                           for level, output in enumerate(outputs))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite detection loss: {loss.item()}")
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                if ema is not None:
                    ema.update(model)
        total_loss += loss.item()
        steps += 1
        progress.set_postfix(loss=total_loss / steps)
    if steps == 0:
        raise ValueError("No batches available; check annotation lists and batch size")
    return total_loss / steps
