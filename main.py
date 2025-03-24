#from google.colab import drive
import sys
import os
from zipfile import ZipFile
import torch
import pandas as pd
# <local imports>
import myConfig
import myData
import myModel
import myFunctions
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
    # Ensure paths are configured correctly
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    model_name, processor, base_model = myModel.getModelDefinitions()
    # Data extraction and feature engineering
    myData.DownloadAndExtract()    
    
    # Check if dataframe.csv exists in the Data directory
    data_file_path = os.path.join(myConfig.DATA_DIR, "dataframe.csv")   
    if os.path.exists(data_file_path):
        # Load existing dataframe
        data_df = pd.read_csv(data_file_path)
        print(f"Loaded existing dataframe from {data_file_path}")
        
        # Check if paths are absolute and convert if needed
        if '/' in data_df['file_path'].iloc[0] and not data_df['file_path'].iloc[0].startswith(('Healthy', 'MCI', 'AD')):
            print("Converting absolute paths to relative paths...")
            data_df = myFunctions.convert_absolute_to_relative_paths(data_df)
            # Save the updated dataframe
            data_df.to_csv(data_file_path, index=False)
    else:
        # Create dataframe and save it
        data_df = myFunctions.createDataframe()        
        data_df = myFunctions.featureEngineering(data_df)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_file_path), exist_ok=True)        
        # Save dataframe
        data_df.to_csv(data_file_path, index=False)
        print(f"Created and saved dataframe to {data_file_path}")
    _, weights_tensor = myFunctions.setWeightedCELoss()
    # Feature engineering    
    
    
    if not os.path.exists(myConfig.OUTPUT_PATH):
        # Data splits
        train_df, val_df, test_df = myData.datasetSplit(data_df)
        # Apply standard scaling to the splits
        train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
        # Create HF's dataset
        myData.createHFDatasets(train_df, val_df, test_df)    
    # Load HF's dataset
    dataset = myData.loadHFDataset()
    # Load model
    model, optimizer = myModel.loadModel(model_name)
    # Create trainer
    trainer = myModel.createTrainer(model, optimizer, dataset, weights_tensor)
    trainer.train()
    torch.save(model.state_dict(), "./wav2vec2_classification/model.pth")
    processor.save_pretrained("./wav2vec2_classification")
    if myConfig.training_from_scratch:
        model.config.save_pretrained(myConfig.checkpoint_dir)
    print("Training complete! Model saved to ./wav2vec2_classification")


def test():
    model, _ = myModel.loadModel()
    dataset = myData.loadDataset()
    testModel(model, dataset)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        mode = args[0]
        if mode == 'offline':
            myConfig.running_offline = True
        else:
            myConfig.running_offline = False
    else:
        myConfig.running_offline = True
    main_fn()
