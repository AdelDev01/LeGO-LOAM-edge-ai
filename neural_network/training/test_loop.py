import torch
import torch.nn as nn
from ground_segmentation import NewGroundSegmentation
import dataloader as dl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in uso:", device)

model = NewGroundSegmentation().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

loss_calc = nn.BCEWithLogitsLoss()
test_loss = 0.0

tp_ground = fp_ground = fn_ground = 0
tp_not_ground = fp_not_ground = fn_not_ground = 0

with torch.no_grad():
    for range_batch, ground_batch in dl.test_loader:

        range_batch = range_batch.to(device)
        ground_batch = ground_batch.to(device)

        output = model(range_batch)
        test_loss += loss_calc(output, ground_batch).item()

        predictions = (torch.sigmoid(output) >= 0.5).float()

        tp_ground += ((predictions == 1) & (ground_batch == 1)).sum().item()
        fp_ground += ((predictions == 1) & (ground_batch == 0)).sum().item()
        fn_ground += ((predictions == 0) & (ground_batch == 1)).sum().item()
        tp_not_ground += ((predictions == 0) & (ground_batch == 0)).sum().item()
        fp_not_ground += ((predictions == 0) & (ground_batch == 1)).sum().item()
        fn_not_ground += ((predictions == 1) & (ground_batch == 0)).sum().item()

avg_test_loss = test_loss / len(dl.test_loader)

den_ground = tp_ground + fp_ground + fn_ground
den_not_ground = tp_not_ground + fp_not_ground + fn_not_ground

ground_iou = tp_ground / den_ground if den_ground > 0 else 1.0
not_ground_iou = tp_not_ground / den_not_ground if den_not_ground > 0 else 1.0
miou = (ground_iou + not_ground_iou) / 2

# Precision, Recall e F1-score per entrambe le classi.
# La classe positiva è valutata separatamente: ground (1) e non-ground (0).

ground_precision = (
    tp_ground / (tp_ground + fp_ground)
    if (tp_ground + fp_ground) > 0 else 0.0
)
ground_recall = (
    tp_ground / (tp_ground + fn_ground)
    if (tp_ground + fn_ground) > 0 else 0.0
)
ground_f1 = (
    2 * ground_precision * ground_recall /
    (ground_precision + ground_recall)
    if (ground_precision + ground_recall) > 0 else 0.0
)

not_ground_precision = (
    tp_not_ground / (tp_not_ground + fp_not_ground)
    if (tp_not_ground + fp_not_ground) > 0 else 0.0
)
not_ground_recall = (
    tp_not_ground / (tp_not_ground + fn_not_ground)
    if (tp_not_ground + fn_not_ground) > 0 else 0.0
)
not_ground_f1 = (
    2 * not_ground_precision * not_ground_recall /
    (not_ground_precision + not_ground_recall)
    if (not_ground_precision + not_ground_recall) > 0 else 0.0
)

macro_f1 = (ground_f1 + not_ground_f1) / 2
macro_precision = (ground_precision + not_ground_precision) / 2
macro_recall = (ground_recall + not_ground_recall) / 2

print(f"Test Loss: {avg_test_loss:.4f} - Ground IoU: {ground_iou:.4f} "
      f"- Not Ground IoU: {not_ground_iou:.4f} - mIoU: {miou:.4f}")

print(
    f"Ground      | Precision: {ground_precision:.4f} "
    f"| Recall: {ground_recall:.4f} | F1: {ground_f1:.4f}"
)
print(
    f"Non-ground  | Precision: {not_ground_precision:.4f} "
    f"| Recall: {not_ground_recall:.4f} | F1: {not_ground_f1:.4f}"
)
print(
    f"Macro       | Precision: {macro_precision:.4f} "
    f"| Recall: {macro_recall:.4f} | F1: {macro_f1:.4f}"
) 