import torchaudio
import os
import torch
import librosa
import numpy as np
import torch
import torchvision
import random
import torchvision.transforms as transforms
from audio_datasets  import audioMNIST
from audio_datasets  import speech_commands
from settings import train_batch_size, test_batch_size

from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def load_datasets(dataset_pointer :str,pipeline:str):
    global labels
    if pipeline == 'mel':
        pipeline_on_wav = WavToMel()
    elif pipeline =='spec':
        pipeline_on_wav = WavToSpec()
    if not os.path.exists(dataset_pointer):
            print(f"Downloading: {dataset_pointer}")
    if dataset_pointer == 'SpeechCommands':
        train_set,test_set = speech_commands.create_speechcommands(pipeline,pipeline_on_wav,dataset_pointer)
    elif dataset_pointer == 'audioMNIST':
        train_set, test_set = audioMNIST.create_audioMNIST(pipeline,pipeline_on_wav,dataset_pointer)
    # elif dataset_pointer == 'Ravdess':
    #     train_set, test_set = ravdess.create_ravdess(pipeline,pipeline_on_wav,dataset_pointer)
    #     labels = np.load('./labels/ravdess_label.npy')
    else:
        raise Exception("Enter correct dataset pointer")
        
    if dataset_pointer == 'SpeechCommands' or dataset_pointer == 'audioMNIST':
        train_set = DatasetProcessor(train_set)
        test_set = DatasetProcessor(test_set)

    train_loader = DataLoader(train_set, batch_size=train_batch_size,shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size,shuffle=False)
        
    return train_loader,test_loader

class DatasetProcessor(Dataset):
  def __init__(self, annotations):
    self.audio_files = annotations
    self.features = [] 
    self.labels = [] 
    for idx, path in enumerate(self.audio_files):
       d = torch.load(path)
       d["feature"] = d["feature"][None,:,:]
       self.features.append(d["feature"])
       self.labels.append(d["label"])

  def __len__(self):
    return len(self.audio_files)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

class DatasetProcessor_randl(Dataset):
  def __init__(self, annotations,device,num_classes):
    self.audio_files = annotations
    self.features = []
    self.labels = [] 
    for idx, path in enumerate(self.audio_files):
       d = torch.load(path)
       d["feature"] = d["feature"][None,:,:]
       self.features.append(d["feature"].to(device))
       new_label = d["label"] 
       while new_label == d["label"]:
            new_label = random.randint(0, (num_classes-1))
       new_label = torch.tensor(new_label).to(device)
       self.labels.append(new_label)

  def __len__(self):
    return len(self.audio_files)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx] 

class WavToMel(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        n_fft=1024,
        n_mel=32
    ):
        super().__init__()

        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mel, sample_rate=input_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform)

        mel = self.mel_scale(spec)

        return mel
    
class WavToSpec(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        n_fft=1024,
        n_mel=32
    ):
        super().__init__()

        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mel, sample_rate=input_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform)
        spec = torch.from_numpy(librosa.power_to_db(spec))
        return spec
