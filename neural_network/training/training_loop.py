import torch
import torch.nn as nn
from ground_segmentation import NewGroundSegmentation
import dataloader as dl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in uso:", device)

model = NewGroundSegmentation().to(device)
loss_calc = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 40

best_val_loss = float("inf")

patience = 10
epochs_without_improvement = 0

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0

    for range_batch, ground_batch in dl.training_loader:

        range_batch = range_batch.to(device)
        ground_batch = ground_batch.to(device)

        output = model(range_batch)
        loss = loss_calc(output, ground_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    avg_loss = train_loss / len(dl.training_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

    model.eval()
    val_loss = 0.0

    true_positive_ground = 0
    false_positive_ground = 0
    false_negative_ground = 0
    true_positive_not_ground = 0
    false_positive_not_ground = 0
    false_negative_not_ground = 0

    with torch.no_grad():
        for range_batch, ground_batch in dl.validation_loader:

            range_batch = range_batch.to(device)
            ground_batch = ground_batch.to(device)

            output = model(range_batch)
            loss = loss_calc(output, ground_batch)
            val_loss += loss.item()

            probabilities = torch.sigmoid(output)
            predictions = (probabilities >= 0.5).float()

            true_positive_ground += ((predictions == 1) & (ground_batch == 1)).sum().item()
            false_positive_ground += ((predictions == 1) & (ground_batch == 0)).sum().item()
            false_negative_ground += ((predictions == 0) & (ground_batch == 1)).sum().item()
            true_positive_not_ground += ((predictions == 0) & (ground_batch == 0)).sum().item()
            false_positive_not_ground += ((predictions == 0) & (ground_batch == 1)).sum().item()
            false_negative_not_ground += ((predictions == 1) & (ground_batch == 0)).sum().item()

    avg_val_loss = val_loss / len(dl.validation_loader)

    denominator_ground = true_positive_ground + false_positive_ground + false_negative_ground
    denominator_not_ground = true_positive_not_ground + false_positive_not_ground + false_negative_not_ground

    if denominator_ground > 0:
        ground_iou = true_positive_ground / denominator_ground
    else:
        ground_iou = 1.0

    if denominator_not_ground > 0:
        not_ground_iou = true_positive_not_ground / denominator_not_ground
    else:
        not_ground_iou = 1.0

    miou = (ground_iou + not_ground_iou) / 2

    print(f"Epoch {epoch+1} / {num_epochs} - Validation Loss: {avg_val_loss:.4f} - Ground IoU: {ground_iou:.4f} - Not Ground IoU: {not_ground_iou:.4f} - mIoU: {miou:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("Salvataggio del miglior modello.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping: la validation loss non migliora.")
        break