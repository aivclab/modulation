#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

import time
from pathlib import Path

import torch
import torchaudio
from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import (
    find_n_misclassified,
    trainable_parameters,
)
from draugr.visualisation import progress_bar
from modulation.classification.procedure import (
    single_epoch_evaluation,
    single_epoch_fitting,
)
from modulation.data.audio.speech.recognition import SpeechCommands

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

from torch import optim
from torch.utils.data import DataLoader

from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    get_model_hash,
    get_num_parameters,
    global_torch_device,
    load_model,
    save_model,
)
from draugr.random_utilities import seed_stack
from modulation import PROJECT_APP_PATH
from modulation.classification import M5
from modulation.torch_utilities.collation import collate_transform_wrapped


def main(
    n_epoch=60,
    batch_size=256,
    new_sample_rate=8000,
    train: bool = True,
    load_previous_model: bool = True,
    model_name="m5_speech_command",
    model_storage_path=PROJECT_APP_PATH.user_data / "speech_command" / "models",
    data_path=Path.home() / "Data" / "Audio" / "SpeechCommands",
    drop_last_train=False,
):
    """

    :param n_epoch:
    :type n_epoch:
    :param batch_size:
    :type batch_size:
    :param new_sample_rate:
    :type new_sample_rate:
    :param train:
    :type train:
    :param load_previous_model:
    :type load_previous_model:
    :param model_name:
    :type model_name:
    :param model_storage_path:
    :type model_storage_path:
    :param data_path:
    :type data_path:
    :param drop_last_train:
    :type drop_last_train:
    """
    device = global_torch_device(True)

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print(f"using {device}")
    seed_stack(0)

    train_set = SpeechCommands(data_path, split=SplitEnum.training)

    waveform, sample_rate, category, speaker_id, utterance_number = train_set[0]
    transform = torch.nn.Sequential(
        torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    )

    model = M5(n_input=transform(waveform).shape[0], n_output=len(train_set.categories))

    persistence_model_name = (
        f"{model_name}_{get_model_hash(transform)}_{get_model_hash(model)}"
    )
    if load_previous_model:
        candidate = load_model(
            model_name=persistence_model_name,
            model_directory=model_storage_path,
            raise_on_failure=False,
        )
        if candidate:
            model = candidate
            print(f"model loaded")

    transform.to(device)
    model.to(device)

    collate_fn = collate_transform_wrapped(train_set.label_to_index, transform)

    # print(f"Trainable parameters: {named_trainable_parameters(model)}")
    print(
        f"Number of trainable parameters: {get_num_parameters(model, only_trainable = True)}"
    )

    if train:
        train_proc(
            model,
            collate_fn,
            train_set,
            n_epoch,
            persistence_model_name,
            drop_last_train,
            model_storage_path,
            batch_size,
            model_name,
            num_workers,
            pin_memory,
            device,
            data_path,
        )
    else:
        stest_proc(
            model,
            collate_fn,
            train_set,
            model_name,
            num_workers,
            pin_memory,
            device,
            data_path,
        )


def train_proc(
    model,
    collate_fn,
    train_set,
    n_epoch,
    persistence_model_name,
    drop_last_train,
    model_storage_path,
    batch_size,
    model_name,
    num_workers,
    pin_memory,
    device,
    data_path,
):
    """

    :param model:
    :type model:
    :param collate_fn:
    :type collate_fn:
    :param train_set:
    :type train_set:
    :param n_epoch:
    :type n_epoch:
    :param persistence_model_name:
    :type persistence_model_name:
    :param drop_last_train:
    :type drop_last_train:
    :param model_storage_path:
    :type model_storage_path:
    :param batch_size:
    :type batch_size:
    :param model_name:
    :type model_name:
    :param num_workers:
    :type num_workers:
    :param pin_memory:
    :type pin_memory:
    :param device:
    :type device:
    :param data_path:
    :type data_path:
    """
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        SpeechCommands(data_path, split=SplitEnum.validation),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    optimiser = optim.Adam(trainable_parameters(model), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(
        optimiser, step_size=20, gamma=0.1
    )  # reduce the learning after 20 epochs by a factor of 10

    with TensorBoardPytorchWriter(
        PROJECT_APP_PATH.user_log / model_name / str(time.time())
    ) as writer:
        best_valid_acc = 0
        for ith_epoch in progress_bar(
            range(1, n_epoch + 1), total=n_epoch, description="Epoch #"
        ):
            single_epoch_fitting(
                model,
                optimiser,
                train_loader,
                epoch=ith_epoch,
                writer=writer,
                device_=device,
            )
            scheduler.step()
            accuracy = single_epoch_evaluation(
                model,
                valid_loader,
                subset=SplitEnum.validation,
                epoch=ith_epoch,
                writer=writer,
                device=device,
            )
            if accuracy > best_valid_acc:
                best_valid_acc = accuracy
                save_model(
                    model,
                    model_name=persistence_model_name,
                    save_directory=model_storage_path,
                )
                writer.blip("new_best_model", ith_epoch)


def stest_proc(
    model, collate_fn, train_set, model_name, num_workers, pin_memory, device, data_path
):
    """

    :param model:
    :type model:
    :param collate_fn:
    :type collate_fn:
    :param train_set:
    :type train_set:
    :param model_name:
    :type model_name:
    :param num_workers:
    :type num_workers:
    :param pin_memory:
    :type pin_memory:
    :param device:
    :type device:
    :param data_path:
    :type data_path:
    """
    seed_stack(0)
    with TensorBoardPytorchWriter(
        PROJECT_APP_PATH.user_log / model_name / str(time.time())
    ) as writer:
        test_loader = DataLoader(
            SpeechCommands(data_path, split=SplitEnum.testing),
            batch_size=1,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        acc = single_epoch_evaluation(
            model,
            test_loader,
            subset=SplitEnum.testing,
            epoch=0,
            writer=writer,
            device=device,
        )

        find_n_misclassified(model, test_loader, mapper=train_set.index_to_label)
        print(acc)


if __name__ == "__main__":
    main(train=False)
