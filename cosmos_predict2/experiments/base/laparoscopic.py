# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


# Laparoscopic dataset and dataloader
example_video_dataset_laparoscopic = L(VideoDataset)(
    dataset_dir="/home/xum35/datasets/laparoscopic_partial_excision_of_kidney_using_robotic_assistance",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_laparoscopic = L(get_generic_dataloader)(
    dataset=example_video_dataset_laparoscopic,
    sampler=L(get_sampler)(dataset=example_video_dataset_laparoscopic),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


# Video2World training configuration for laparoscopic dataset
predict2_video2world_training_2b_cosmos_laparoscopic_partial_excision_of_kidney_using_robotic_assistance = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_laparoscopic_partial_excision_of_kidney_using_robotic_assistance",
    ),
    # Disable uploading reproducible setup to S3 for local runs
    upload_reproducible_setup=False,
    dataloader_train=dataloader_train_laparoscopic,
    checkpoint=dict(
        save_iter=2,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[2_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=1000,
        callbacks=dict(),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
)


cs = ConfigStore.instance()

# Register the configuration with Hydra ConfigStore
for _item in [
    predict2_video2world_training_2b_cosmos_laparoscopic_partial_excision_of_kidney_using_robotic_assistance,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
