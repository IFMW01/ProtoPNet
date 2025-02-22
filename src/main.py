import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from audio_datasets import load_datasets
# from audiomentations import Compose, AddGaussianNoise, Shift, Gain
import argparse
import re
import numpy as np
from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
import torchaudio
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from torchaudio.datasets import SPEECHCOMMANDS
import math
import librosa


parser = argparse.ArgumentParser()
args = parser.parse_args()

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set

# *************************** CIFAR 10 **********************************

# train_dataset = datasets.CIFAR10(
#             ".", train=True, download=True, transform=transforms.Compose([transforms.AugMix(),transforms.ToTensor(), transforms.Normalize(0 ,1)])
#         )

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=train_batch_size, shuffle=True,
#     num_workers=2, pin_memory=False)
# # push set
# train_push_dataset = datasets.CIFAR10(
#             "", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0 ,1 )])
#         )
# train_push_loader = torch.utils.data.DataLoader(
#     train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
#     num_workers=2, pin_memory=False)
# # test set
# test_dataset = datasets.CIFAR10(
#             ".", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0 ,1)])
#         )
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=test_batch_size, shuffle=False,
#     num_workers=2, pin_memory=False)

# *************************** CIFAR 10 **********************************

# *************************** AUDIO DATASET **********************************


# class SubsetSC(SPEECHCOMMANDS):
#     def __init__(self, subset: str = None):
#         super().__init__("./", download=True)

#         def load_list(filename):
#             filepath = os.path.join(self._path, filename)
#             with open(filepath) as fileobj:
#                 return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

#         if subset == "validation":
#             self._walker = load_list("validation_list.txt")
#         elif subset == "testing":
#             self._walker = load_list("testing_list.txt")
#         elif subset == "training":
#             excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
#             excludes = set(excludes)
#             self._walker = [w for w in self._walker if w not in excludes]

# train_set = SubsetSC("training")
# test_set = SubsetSC("testing")

# class MelSpectogram(torch.nn.Module):
#     def __init__(
#         self,
#         input_freq=16000,
#         resample_freq=16000,
#         n_fft=1024,
#         n_mel=32,
#         stretch_factor=0.8,
#     ):
#         super().__init__()

#         self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)

#         self.mel_scale = torchaudio.transforms.MelScale(
#             n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         spec = self.spec(waveform)

#         mel = self.mel_scale(spec)

#         return mel
    
# class WavToSpec(torch.nn.Module):
#     def __init__(
#         self,
#         input_freq=16000,
#         resample_freq=16000,
#         n_fft=1024,
#         n_mel=32,
#         stretch_factor=0.8,
#     ):
#         super().__init__()

#         self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)

#         self.mel_scale = torchaudio.transforms.MelScale(
#             n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         spec = self.spec(waveform)
#         # spec = torch.from_numpy(librosa.power_to_db(spec))
#         # mel = self.mel_scale(spec)
#         spec = torch.from_numpy(librosa.power_to_db(spec))

#         return spec

# agument = Compose([
#     Gain(p=0.25),
#     AddGaussianNoise(
#         min_amplitude=1,
#         max_amplitude=2,
#         p=0.25
#     ),
#     Shift(p=0.25),
# ])

# def pad_sequence_aug(batch):
#     # Make all tensor in a batch the same length by padding with zeros
#     batch = [item.t() for item in batch]
#     batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
#     batch = agument(samples=batch, sample_rate=16000)
#     batch = torch.Tensor(batch)
#     return batch.permute(0, 2, 1)

# def aug_collate_fn(batch):

#     # A data tuple has the form:
#     # waveform, sample_rate, label, speaker_id, utterance_number

#     tensors, targets = [], []

#     # Gather in lists, and encode labels as indices
#     for waveform, _, label, *_ in batch:
#         tensors += [waveform]
#         targets += [label_to_index(label)]
#         # targets += torch.eye(35)[label_to_index(label)]
#     # tensors = np.array(tensors)
#     # Group the list of tensors into a batched tensor
#     tensors = pad_sequence_aug(tensors)
#     tensors = pipeline_to_spec(tensors)
#     targets = torch.stack(targets)

#     return tensors, targets

# labels = np.load('./lables.npy')
# labels = labels.tolist()
# # transform = torchaudio.transforms.MelSpectrogram(new_sample_rate,n_fft = 1024, hop_length=512,n_mels =32)

# pipeline_to_spec = WavToSpec()
# pipeline_to_spec.to(dtype=torch.float32)

# def label_to_index(word):
#     # Return the position of the word in labels
#     return torch.tensor(labels.index(word))


# def index_to_label(index):
#     # Return the word corresponding to the index in labels
#     # This is the inverse of label_to_index
#     return labels[index]

# def pad_sequence(batch):
#     # Make all tensor in a batch the same length by padding with zeros
#     batch = [item.t() for item in batch]
#     batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
#     return batch.permute(0, 2, 1)


# def collate_fn(batch):

#     # A data tuple has the form:
#     # waveform, sample_rate, label, speaker_id, utterance_number

#     tensors, targets = [], []

#     # Gather in lists, and encode labels as indices
#     for waveform, _, label, *_ in batch:
#         tensors += [waveform]
#         targets += [label_to_index(label)]
#         # targets += torch.eye(35)[label_to_index(label)]

#     # Group the list of tensors into a batched tensor
#     tensors = pad_sequence(tensors)
#     tensors = pipeline_to_spec(tensors)
#     targets = torch.stack(targets)

#     return tensors, targets



# train_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=train_batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=False,
#     collate_fn = aug_collate_fn
# )
# train_push_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=train_push_batch_size,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=False,
#     collate_fn = collate_fn
# )
# test_loader = torch.utils.data.DataLoader(
#     test_set,
#     batch_size=test_batch_size,
#     shuffle=False,
#     drop_last=False,
#     num_workers=4,
#     pin_memory=False,
#     collate_fn = collate_fn
# )

# *************************** AUDIO DATASET **********************************

train_loader,test_loader = load_datasets.load_datasets('SpeechCommands', 'mel')
train_push_loader = train_loader
# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=False, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.80, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be normalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.80, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(10):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.80, log=log)
   
logclose()

