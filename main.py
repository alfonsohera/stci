#from google.colab import drive
import sys
import os
from zipfile import ZipFile
import torch

# <local imports>
import config
import data
import myModel
import plots
import functions
import myModel
# </local imports>
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader


def collate_fn(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, processor, _ = myModel.getModelDefinitions()
    input_values = processor(
        [item['audio']['array'] for item in batch],  # Access the 'array' key
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values

    prosodic_features = torch.stack([
        torch.tensor(item["prosodic_features"]) for item in batch
    ])

    labels = torch.tensor([item["label"] for item in batch])

    return {
        "input_values": input_values.to(device),
        "prosodic_features": prosodic_features.to(device),
        "labels": labels.to(device)
    }


def testModel(model, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(dataset["test"], batch_size=8, collate_fn=collate_fn)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            logits = model(
                input_values=batch["input_values"],
                prosodic_features=batch["prosodic_features"]
            ).logits

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Calculate metrics 
    test_accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["Healthy", "MCI", "AD"]
    )
    # Print results 
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(report)


def main_fn():
    model_name, processor, base_model = myModel.getModelDefinitions()
    # Data extraction and feature engineering
    data.DownloadAndExtract()
    data_df = functions.createDataframe()
    data_df = functions.featureEngineering(data_df)
    _, weights_tensor = functions.setWeightedCELoss()
    # Plots
    #plots.plotAgeDistribution(data_df)
    #functions.createAgeSexStats(data_df)
    #plots.plotProsodicFeatures(data_df)
    #plots.histogramProsodicFeatures(data_df)
    # Data splits
    train_df, val_df, test_df = data.datasetSplit(data_df, 0.12)
    # Apply standard scaling to the splits
    train_df, val_df, test_df = data.ScaleDatasets(train_df, val_df, test_df)
    # Create HF's dataset
    data.createHFDatasets(train_df, val_df, test_df)
    # Load HF's dataset
    dataset = data.loadHFDataset()
    # Load model
    model, optimizer = myModel.loadModel()
    # Create trainer
    trainer = myModel.createTrainer(model, optimizer, dataset, weights_tensor)
    """ trainer.train()
    torch.save(model.state_dict(), "./wav2vec2_classification/model.pth")
    processor.save_pretrained("./wav2vec2_classification")
    if config.training_from_scratch:
        model.config.save_pretrained(config.checkpoint_dir)
    print("Training complete! Model saved to ./wav2vec2_classification")  """


def test():
    model, _ = myModel.loadModel()
    dataset = data.loadDataset()
    testModel(model, dataset)


args = sys.argv[1:]
if len(args) > 0:
    mode = args[0]
    if mode == 'offline':
        config.running_offline = True
    else:
        config.running_offline = False
else:
    config.running_offline = True

main_fn()
