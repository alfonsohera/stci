## Screening Tool for Cognitive Impairment

A deep learning system for detecting cognitive impairment stages (Healthy, MCI, AD) from speech audio samples using two different approaches: Wav2Vec2-based transformers and a CNN+RNN dual-path architecture.

## Project Structure

```
Repository/
├── main.py                      # Main entry point with CLI interface and wav2vec2 pipeline
├── myConfig.py                  # Configuration and settings
├── myData.py                    # Data loading and processing utilities
├── myModel.py                   # Wav2Vec2 model definition and training utilities
├── myFunctions.py               # General utility functions
├── myAudio.py                   # Audio processing and feature extraction
├── myPlots.py                   # Visualization and analysis tools
├── myThresholdOptimization.py   # Threshold optimization for classification
├── feature_analysis.py          # Feature analysis utilities
├── cnn_rnn_model.py             # CNN+RNN model architecture
├── cnn_rnn_train.py             # CNN+RNN training and evaluation functions
├── cnn_rnn_data.py              # Data utilities specific to CNN+RNN
└── environment.yml              # Conda environment definition
```

### Key Components

- **main.py**: Entry point with argument parsing, training/testing pipelines
- **myModel.py**: Wav2Vec2-based model, training, and evaluation
- **myData.py**: Data loading, preprocessing, and feature extraction
- **cnn_rnn_model.py**: CNN+RNN model architecture for audio classification
- **cnn_rnn_train.py**: Training and evaluation functions for CNN+RNN models
- **myThresholdOptimization.py**: Optimizing decision thresholds for better classification

## CLI Command Reference

The system provides a command-line interface with different operation modes and pipeline options.

### Basic Command Structure

```bash
python main.py <mode> [--pipeline <pipeline>] [--online] [--no_manual]
```

### Arguments

- **mode**: Operation mode
  - `train`: Train a model from scratch
  - `finetune`: Fine-tune an existing model
  - `test`: Evaluate a trained model
  - `optimize`: Perform threshold optimization
  - `test_thresholds`: Evaluate with optimized thresholds
  - `optimize_hpo`: Perform hyperparameter optimization

- **--pipeline**: Model pipeline to use
  - `wav2vec2`: Transformer-based pipeline (default)
  - `cnn_rnn`: CNN+RNN dual-path architecture

- **--online**: Run with online services enabled (WandB logging)
  - If not specified, runs in offline mode

- **--no_manual**: Disable manual features for cnn_rnn pipeline
  - Only applicable to the CNN+RNN pipeline
  - If not specified, manual prosodic features are used

- **--trials**: Number of trials for hyperparameter optimization
  - Only applicable with the `optimize_hpo` mode

### Example Commands

#### Wav2Vec2 Pipeline

1. **Train a model from scratch (offline)**
   ```bash
   python main.py train --pipeline wav2vec2
   ```

2. **Fine-tune an existing model with WandB logging**
   ```bash
   python main.py finetune --pipeline wav2vec2 --online
   ```

3. **Evaluate a trained model**
   ```bash
   python main.py test --pipeline wav2vec2
   ```

4. **Optimize classification thresholds**
   ```bash
   python main.py optimize --pipeline wav2vec2
   ```

5. **Test with optimized thresholds**
   ```bash
   python main.py test_thresholds --pipeline wav2vec2
   ```

6. **Perform hyperparameter optimization**
   ```bash
   python main.py optimize_hpo --pipeline wav2vec2 --trials 50 --online
   ```

#### CNN+RNN Pipeline

1. **Train a CNN+RNN model with manual features**
   ```bash
   python main.py train --pipeline cnn_rnn
   ```

2. **Train a CNN+RNN model without manual features**
   ```bash
   python main.py train --pipeline cnn_rnn --no_manual
   ```

3. **Fine-tune CNN+RNN model with WandB logging**
   ```bash
   python main.py finetune --pipeline cnn_rnn --online
   ```

4. **Evaluate CNN+RNN model**
   ```bash
   python main.py test --pipeline cnn_rnn
   ```

5. **Optimize thresholds for CNN+RNN model**
   ```bash
   python main.py optimize --pipeline cnn_rnn
   ```

6. **Test CNN+RNN model with optimized thresholds**
   ```bash
   python main.py test_thresholds --pipeline cnn_rnn
   ```

## Model Pipelines

### Wav2Vec2 Pipeline

The Wav2Vec2 pipeline uses a pre-trained transformer-based model fine-tuned on speech audio for cognitive impairment detection. It processes raw audio waveforms and can optionally incorporate extracted prosodic features.

Key characteristics:
- Transformer-based architecture
- Pre-trained on large speech datasets
- Fine-tuned for cognitive impairment classification
- Takes raw audio as input

### CNN+RNN Pipeline

The CNN+RNN pipeline uses a dual-path architecture combining convolutional layers for feature extraction and recurrent layers for temporal processing. It can optionally incorporate manual prosodic features.

Key characteristics:
- Convolutional layers process spectral features
- Recurrent layers capture temporal patterns
- Optional manual features pathway
- Balanced augmented dataset for training

## Data Augmentation

The system uses data augmentation via SpecAugment to improve model generalization and address class imbalance:

### Augmentation Techniques

- **SpecAugment**: For spectrograms in the CNN+RNN pipeline
  - Time masking
  - Frequency masking


### Augmentation Strategy

- Applied during training to underrepresented classes
- Balanced dataset creation through stratified augmentation

## Threshold Optimization

Both pipelines support threshold optimization to improve classification performance:

1. Standard classification uses argmax to select the class with the highest probability
2. Optimized thresholds adjust decision boundaries based on validation data
3. Two optimization methods are supported:
   - Youden's J statistic (balances sensitivity and specificity)
   - F1-score optimization (balances precision and recall)

## Hyperparameter Optimization

The project implements automated hyperparameter optimization to find the most effective model configurations:

### HPO Approach

- **Optimization Framework**: Uses Optuna for efficient hyperparameter search
- **Search Strategy**: Implements Bayesian optimization with Tree-structured Parzen Estimator
- **Objective Function**: Maximizes validation set performance (macro F1-score)
- **Cross-Validation**: Employs stratified k-fold cross-validation for robust parameter selection


### Running HPO

```bash
python main.py optimize_hpo --pipeline wav2vec2 --trials 50
python main.py optimize_hpo --pipeline cnn_rnn --trials 50
```

## Requirements

The project requires Python 3.8+ and several libraries listed in the environment.yml file. Set up the environment with:

```bash
conda env create -f environment.yml
conda activate stci
```

The models have been train using an NVIDIA A40 GPU with 48GB of RAM, comparable HW is recommended to avoid running out of memory.

## Data Structure

The system extracts and organizes the downloaded files in class directories:
```
Data/
├── Healthy/
│   ├── Healthy-W-01-001.wav
│   └── ...
├── MCI/
│   ├── MCI-W-01-001.wav
│   └── ...
└── AD/
    ├── AD-W-01-001.wav
    └── ...
```

## Results and Performance

### Performance Metrics

The system evaluates models using multiple metrics:
- Accuracy
- Precision, Recall, F1-score (per class and macro average)
- Specificity and Sensitivity
- Confusion matrices
- ROC-AUC and PR-AUC curves

### Visualization

Performance visualization tools are available in myPlots.py:
- Confusion matrices
- ROC curves
- PR curves
- Feature importance plots
- Attention weight visualization (for Wav2Vec2)

