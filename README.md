# Screening Tool for Cognitive Impairment

A deep learning system for detecting cognitive impairment stages (Healthy, MCI, AD) from speech audio samples using two different approaches: Wav2Vec2-based transformers and a CNN dual-path architecture.

## Project Structure


```
Repo/
├── src/
│   ├── Common/             # Common utilities and shared code
│   │   ├── Config.py       # Configuration settings 
│   │   ├── Data.py         # Data loading and preparation 
│   │   ├── Functions.py    # Shared utility functions 
│   │   ├── ThresholdOptimization.py  # Threshold optimization utilities
│   │   ├── Audio.py        # Audio processing utilities 
│   │   ├── Plots.py        # Visualization utilities 
│   │   ├── FeatureAnalysis.py # Feature analysis utilities 
│   │   └── Speech2text.py  # Speech-to-text conversion utilities 
│   ├── Wav2Vec2/           # Wav2Vec2 transformer model pipeline
│   │   └── Model.py        # Wav2Vec2 model and training utilities 
│   └── Cnn/                # CNN model pipeline
│       ├── cnn_data.py     # CNN-specific data preparation
│       ├── cnn_model.py    # CNN model architecture
│       └── cnn_train.py    # CNN training and evaluation
├── main.py                 # Main script for running training and evaluation
├── cam_utils.py            # Class Activation Mapping visualization utilities
├── analyze_chunking.py     # Audio chunking analysis tools
├── compare_models.py       # Model comparison utilities
├── feature_analysis.py     # Feature analysis utilities
├── environment.yml         # Conda environment definition
└── README.md               # This file
```

## Components

### Common
Contains shared utilities and code used across different model pipelines:
- **Config.py**: Configuration settings and hyperparameters
- **Data.py**: Data loading, preprocessing, and dataset creation
- **Functions.py**: Shared utility functions
- **ThresholdOptimization.py**: Threshold optimization for model predictions
- **Audio.py**: Audio processing and feature extraction
- **Plots.py**: Visualization and analysis tools
- **FeatureAnalysis.py**: Extended feature analysis utilities
- **Speech2text.py**: Speech-to-text conversion utilities

### Wav2Vec2
Contains the transformer-based audio classification pipeline:
- **Model.py**: Wav2Vec2 model definition, fine-tuning and evaluation

### CNN
Contains the CNN-based audio classification pipeline:
- **cnn_data.py**: CNN-specific data preparation
- **cnn_model.py**: CNN model architecture definitions
- **cnn_train.py**: Training and evaluation functions

## CLI Command Reference

The system provides a command-line interface with different operation modes and pipeline options.

### Basic Command Structure

```bash
python main.py <mode> [--pipeline <pipeline>] [--no_prosodic] [--multi_class]
```

### Arguments

- **mode**: Operation mode
  - `train`: Train a model from scratch
  - `finetune`: Fine-tune an existing model
  - `test`: Evaluate a trained model
  - `optimize`: Perform threshold optimization
  - `test_thresholds`: Evaluate with optimized thresholds
  - `cv`: Run cross-validation
  - `hpo`: Perform hyperparameter optimization

- **--pipeline**: Model pipeline to use
  - `wav2vec2`: Transformer-based pipeline
  - `cnn`: CNN dual-path architecture (default)

- **--no_prosodic**: Disable prosodic features for CNN pipeline
  - Only applicable to the CNN pipeline
  - If not specified, prosodic features are used

- **--multi_class**: Use multi-class classification (Healthy vs MCI vs AD)
  - By default, binary classification (Healthy vs Non-Healthy) is used
  - Especially useful for the CNN pipeline

- **--folds**: Number of folds for cross-validation (default: 5)
  - Only applicable with the `cv` mode

- **--trials**: Number of trials for hyperparameter optimization (default: 50)
  - Only applicable with the `hpo` mode

- **--resume**: Resume previous hyperparameter optimization study
  - Only applicable with the `hpo` mode

### Example Commands

#### Wav2Vec2 Pipeline

1. **Train a model from scratch**
   ```bash
   python main.py train --pipeline wav2vec2
   ```

2. **Fine-tune an existing model**
   ```bash
   python main.py finetune --pipeline wav2vec2
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

#### CNN Pipeline (Default)

1. **Train a CNN model with prosodic features**
   ```bash
   python main.py train
   ```

2. **Train a CNN model without prosodic features**
   ```bash
   python main.py train --no_prosodic
   ```

3. **Fine-tune CNN model**
   ```bash
   python main.py finetune
   ```

4. **Evaluate CNN model**
   ```bash
   python main.py test
   ```

5. **Optimize thresholds for CNN model**
   ```bash
   python main.py optimize
   ```

6. **Test CNN model with optimized thresholds**
   ```bash
   python main.py test_thresholds
   ```

7. **Run cross-validation with CNN model**
   ```bash
   python main.py cv --folds 5
   ```

8. **Perform hyperparameter optimization**
   ```bash
   python main.py hpo --trials 50
   ```

9. **Train a CNN model with multi-class classification**
   ```bash
   python main.py train --multi_class
   ```

10. **Test a CNN model with multi-class classification**
   ```bash
   python main.py test --multi_class
   ```

11. **Run cross-validation with multi-class classification**
   ```bash
   python main.py cv --folds 5 --multi_class
   ```

## Model Pipelines

### Classification Modes

The project supports two classification approaches:

- **Multiclass Classification**: The default mode with three classes (Healthy, MCI, AD)
- **Binary Classification**: Simplified mode with two classes (Healthy vs. Non-Healthy), where MCI and AD samples are combined into a single "Non-Healthy" class

Both classification modes are available for all model pipelines. Binary classification can be useful when the goal is to screen for any cognitive impairment rather than distinguishing between different impairment stages.

### Wav2Vec2 Pipeline

The Wav2Vec2 pipeline uses a pre-trained transformer-based model fine-tuned on speech audio for cognitive impairment detection. It processes raw audio waveforms and can optionally incorporate extracted prosodic features.

Key characteristics:
- Transformer-based architecture
- Pre-trained on large speech datasets
- Fine-tuned for cognitive impairment classification
- Takes raw audio as input

### CNN Pipeline (Default)

The CNN pipeline uses a dual-path architecture combining convolutional layers for feature extraction and recurrent layers for temporal processing. It can optionally incorporate manual prosodic features.

Key characteristics:
- Convolutional layers process spectral features
- Recurrent layers capture temporal patterns
- Optional manual features pathway
- Balanced augmented dataset for training
- Chunking approach for handling variable-length inputs
- Class Activation Mapping (CAM) visualization

## Data Augmentation

The system uses data augmentation via SpecAugment to improve model generalization and address class imbalance:

### Augmentation Techniques

- **SpecAugment**: For spectrograms in the CNN pipeline
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
python main.py hpo --trials 50
```

## Requirements

The project requires Python 3.8+ and several libraries listed in the environment.yml file. Set up the environment with:

```bash
conda env create -f environment.yml
conda activate stci
```

The models have been trained using an NVIDIA A40 GPU with 48GB of RAM, comparable hardware is recommended to avoid running out of memory.

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

Performance visualization tools are available in Common/Plots.py:
- Confusion matrices
- ROC curves
- PR curves
- Feature importance plots

## Third-Party Libraries and Credits

This project relies on several open-source libraries and pre-trained models. I'd like to acknowledge and thank the developers and researchers behind these tools:

### Core Audio Processing

- **[Pyannote Audio](https://github.com/pyannote/pyannote-audio)** - Used for voice activity detection and speaker diarization. Developed by Hervé Bredin and the pyannote team.

- **[Demucs](https://github.com/facebookresearch/demucs)** - Used for high-quality voice separation from background noise. Developed by Facebook Research.

- **[PANN (CNN14)](https://github.com/qiuqiangkong/audioset_tagging_cnn)** - Pre-trained audio neural network used as a feature extractor. Developed by Qiuqiang Kong et al.

- **[Librosa](https://librosa.org/)** - Core audio processing library used for feature extraction and audio manipulation.

- **[Praat Parselmouth](https://github.com/YannickJadoul/Parselmouth)** - Python interface to the Praat software used for extracting prosodic features.

### Data Sources

- **Original Speech Corpus**: The audio dataset used in this project is from the research paper ["Discriminating speech traits of Alzheimer's disease assessed through a corpus of reading task for Spanish language"](https://doi.org/10.1016/j.csl.2021.101341) by Ivanova et al. (2021), published in Speech Communication. The corpus contains Spanish language speech recordings of elderly adults with varying degrees of cognitive impairment.

### Machine Learning and Deep Learning

- **[PyTorch](https://pytorch.org/)** and **[TorchAudio](https://pytorch.org/audio)** - Core deep learning frameworks used for model development.

- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - Used for the Wav2Vec2 transformer models and processing.

- **[Scikit-learn](https://scikit-learn.org/)** - Used for machine learning algorithms, metrics, and data preprocessing.

### Visualization and Analysis

- **[Matplotlib](https://matplotlib.org/)** and **[Seaborn](https://seaborn.pydata.org/)** - Used for data visualization and result plotting.

- **[Plotly](https://plotly.com/)** - Used for interactive visualizations.

- **[Weights & Biases](https://wandb.ai/)** - Used for experiment tracking and visualization.

Please make sure to respect the licenses of these libraries when using this code.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


