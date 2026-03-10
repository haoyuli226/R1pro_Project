import torch
import numpy as np
import random

from bc_model import BCModel

DEVICE = torch.device("cpu")
MODEL_PATH = "bc_model_best.pt"
DATA_PATH = "./processed_data/bc_dataset.pt"

NUM_PRINT_SAMPLES = 5


def main():

    print("\n===== Loading Dataset =====")
    data = torch.load(DATA_PATH, map_location=DEVICE)

    images = data["images"]
    actions = data["actions"]

    print("Total samples:", images.shape[0])
    print("Image shape:", images.shape)
    print("Action shape:", actions.shape)

    joint_gt = actions[:, :7]
    grip_gt = actions[:, 7]

    print("\n===== Building Model =====")
    model = BCModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    print("\n===== Sample Inference =====")

    indices = random.sample(range(images.shape[0]), NUM_PRINT_SAMPLES)

    with torch.no_grad():
        for i, idx in enumerate(indices):

            img = images[idx].unsqueeze(0).to(DEVICE)

            output = model(img)   # ← 关键修改

            joint_pred = output[:, :7].cpu().squeeze(0)
            grip_pred = output[:, 7].cpu().item()

            print(f"\n--- Sample {i} (Index {idx}) ---")

            print("GT Joint:   ", joint_gt[idx].numpy())
            print("Pred Joint: ", joint_pred.numpy())

            print("GT Grip:    ", grip_gt[idx].item())
            print("Pred Grip:  ", grip_pred)

            joint_mse = torch.mean((joint_pred - joint_gt[idx])**2).item()
            grip_mse = (grip_pred - grip_gt[idx].item())**2

            print("Joint MSE:", joint_mse)
            print("Grip  MSE:", grip_mse)

    print("\n===== Full Dataset Evaluation =====")

    total_joint_mse = 0
    total_grip_mse = 0

    with torch.no_grad():
        for i in range(images.shape[0]):

            img = images[i].unsqueeze(0).to(DEVICE)

            output = model(img)

            joint_pred = output[:, :7].squeeze(0)
            grip_pred = output[:, 7].item()

            total_joint_mse += torch.mean(
                (joint_pred - joint_gt[i])**2
            ).item()

            total_grip_mse += (grip_pred - grip_gt[i].item())**2

    print("Average Joint MSE:", total_joint_mse / images.shape[0])
    print("Average Grip  MSE:", total_grip_mse / images.shape[0])

    print("\n===== Validation Finished =====")


if __name__ == "__main__":
    main()