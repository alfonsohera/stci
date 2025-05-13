# Root module exports
from .src import Common
from .src import Wav2Vec2
from .src import Cnn

# Export key functions for convenience
from .src.Common.Config import configure_paths
from .src.Common.Data import DownloadAndExtract, loadHFDataset
from .src.Wav2Vec2.Model import getModelDefinitions, loadModel, createTrainer
from .src.Cnn.cnn_train import main_cnn, test_cnn

# Version info
__version__ = "1.0.0"