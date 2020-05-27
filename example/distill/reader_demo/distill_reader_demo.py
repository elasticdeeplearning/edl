# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from paddle_edl.distill.distill_reader import DistillReader

BATCH_NUM = 10
BATCH_SIZE = 16
EPOCH_NUM = 4

# data format of data source user provides, can be:
# 'sample_generator', 'sample_list_generator', 'batch_generator'
DATA_FORMAT = 'batch_generator'


def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label


# If the data generator yields one sample each time,
# use DistillReader.set_sample_generator to set the data source.
def sample_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM * BATCH_SIZE):
            image, label = get_random_images_and_labels([1, 28, 28], [1])
            yield image, label

    return __reader__


# If the data generator yield list of samples each time,
# use DistillReader.set_sample_list_generator to set the data source.
def sample_list_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            sample_list = []
            for _ in range(BATCH_SIZE):
                image, label = get_random_images_and_labels([1, 28, 28], [1])
                sample_list.append([image, label])
            yield sample_list

    return __reader__


# If the data generator yields a batch each time,
# use DistillReader.set_batch_generator to set the data source.
def batch_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = get_random_images_and_labels(
                [BATCH_SIZE, 1, 28, 28], [BATCH_SIZE, 1])
            yield batch_image, batch_label

    return __reader__


def set_data_source(distill_reader):
    if DATA_FORMAT == 'sample_generator':
        distill_reader.set_sample_generator(sample_generator_creator())
    elif DATA_FORMAT == 'sample_list_generator':
        distill_reader.set_sample_list_generator(sample_list_generator_creator(
        ))
    elif DATA_FORMAT == 'batch_generator':
        distill_reader.set_batch_generator(batch_generator_creator())
    else:
        raise ValueError('Unsupported data format')


# Define DistillReader
distill_reader = DistillReader(ins=['img', None], predicts=['prediction'])
set_data_source(distill_reader)

if DATA_FORMAT == 'sample_generator':
    step = 0
    for img, label, prediction in distill_reader():
        assert img.shape == (1, 28, 28)
        assert label.shape == (1, )
        assert prediction.shape == (10, )
        if step % BATCH_SIZE == 0:
            print('one sample prediction={}'.format(prediction))
        step += 1
elif DATA_FORMAT == 'sample_list_generator':
    for sample_list in distill_reader():
        assert len(sample_list) == BATCH_SIZE
        for img, label, prediction in sample_list:
            assert img.shape == (1, 28, 28)
            assert label.shape == (1, )
            assert prediction.shape == (10, )
        print('one sample prediction={}'.format(sample_list[0][2]))
elif DATA_FORMAT == 'batch_generator':
    for img, label, prediction in distill_reader():
        assert img.shape == (BATCH_SIZE, 1, 28, 28)
        assert label.shape == (BATCH_SIZE, 1)
        assert prediction.shape == (BATCH_SIZE, 10)
        print('one sample prediction={}'.format(prediction[0]))
