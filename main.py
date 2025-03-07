def manageImports():
    from google.colab import drive
    import os
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
    import requests
    import shutil
    from zipfile import ZipFile
    import torch
    import librosa
    import datasets
    import evaluate
    import pandas as pd
    import scipy.signal
    import torch.nn as nn
    import parselmouth
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    from parselmouth.praat import call
    from tqdm import tqdm
    from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
    from transformers import (
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2Processor,
        TrainingArguments,
        Trainer,
        get_scheduler
    )

    from torch.nn.utils.rnn import pad_sequence
    from bitsandbytes.optim import Adam8bit
    from sklearn.metrics import classification_report, confusion_matrix
    from pydub import AudioSegment
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

