import itertools
import math
import random

import cv2
import numpy as np

dataDir = "../dataset/"
seqNames = ("ape", "benchvise", "cam", "cat", "duck")


def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH | -1)


def load_poses_and_image(parent_dir, label):
    with open(parent_dir + '/poses.txt', 'r') as f:
        count = 0
        samples = []
        sample = []
        for line in f:
            count += 1
            if count % 2 == 1:  # this is the remainder operator
                if not line.strip():
                    print('Empty Line in %s' % (parent_dir + '/poses.txt'))
                else:
                    sample = []
                    sample.append(load_image(parent_dir + '/' + line.split()[1]))
            else:
                sample.append(label)
                sample.extend([float(i) for i in line.split()])
                samples.append(sample)
        return samples


def load_training_indices(training_split_file):
    with open(training_split_file, 'r') as f:
        for line in f:
            training_indices = [int(i) for i in line.split(',')]
            return training_indices
    return None


def load_coarse_db():
    coarse_db = []
    for i in range(5):
        seq_db = load_poses_and_image(dataDir + 'coarse/' + seqNames[i], i)
        coarse_db.extend(seq_db)
        print(len(seq_db))
    print(len(coarse_db))
    return coarse_db


def load_fine_db():
    fine_db = []
    for i in range(5):
        seq_db = load_poses_and_image(dataDir + 'fine/' + seqNames[i], i)
        fine_db.extend(seq_db)
        print(len(seq_db))
    print(len(fine_db))
    return fine_db


def load_real_db():
    real_db_train = []
    real_db_test = []
    training_indices = load_training_indices(dataDir + 'real/training_split.txt')

    for i in range(5):
        seq_db = load_poses_and_image(dataDir + 'real/' + seqNames[i], i)
        for i in range(0, len(seq_db)):
            if i in training_indices:
                real_db_train.append(seq_db[i])
            else:
                real_db_test.append(seq_db[i])

    print(len(real_db_train))
    print(len(real_db_test))
    return real_db_train, real_db_test


def load_all_dataset():
    coarse_db = load_coarse_db()
    fine_db = load_fine_db()
    real_db_train, real_db_test = load_real_db()
    train_db = fine_db
    train_db.extend(real_db_train)
    return coarse_db, train_db, real_db_test


def calculate_global_training_mean(images):
    src = images[0]
    n_channels = src.shape[-1]
    mean = np.zeros(n_channels)
    n_images = len(images)

    for image in images:
        image = np.asarray(image, dtype=np.float32)
        mean += image.mean(axis=(0, 1)) / n_images

    return mean


def calculate_global_std(images, mean=None):
    """ Calculates the training standard deviation.
        Args:
            filenames: A list of image files for which standard deviation is calculated.
            mean: pre-calculated mean for the dataset. If None, the function would calculate
                the mean first.
        Returns: numpy array of standard deviation along each channel
        """
    if mean is None:
        mean = calculate_global_training_mean(images)

    src = images[0]
    n_channels = src.shape[-1]
    std = np.zeros(n_channels)
    n_images = len(images)

    for image in images:
        image = np.asarray(image, dtype=np.float32)
        std_img = np.sum(np.sum(np.square(image - mean), axis=0), axis=0) / (n_images * src.shape[0] * src.shape[1] - 1)
        std += std_img

    return np.sqrt(std)


def compute_mean_and_deviation():
    db, train, test = load_all_dataset()
    images = [i[0] for i in train]
    print([len(i) for i in [db, train, test]])
    mean = calculate_global_training_mean(images)
    dev = calculate_global_std(images)
    return mean, dev


def compute_quaternion_angle_diff(quat1, quat2):
    sum = 0
    for i in range(4):
        sum = sum + quat1[i] * quat2[i]
    sum = np.clip(sum, -1, 1)
    angle = 2 * math.acos(abs(sum)) * 180 / np.pi
    return angle


class DatasetGenerator:
    db, train, test = load_all_dataset()
    images = [i[0] for i in itertools.chain(db, train)]
    mean = calculate_global_training_mean(images)
    dev = calculate_global_std(images, mean)

    def __init__(self, iter_per_epoch, batch_size, is_train):
        self.iter_per_epoch = iter_per_epoch
        self.batch_size = batch_size
        self.is_train = is_train
        self.normalize_images()

    def normalize_images(self):
        for sample in self.db:
            sample[0] = (sample[0] - self.mean) / self.dev

        for sample in self.train:
            sample[0] = (sample[0] - self.mean) / self.dev

        for sample in self.test:
            sample[0] = (sample[0] - self.mean) / self.dev

    def get_closest_puller(self, train_sample):
        puller = None
        minimum_quaternion_distance = 10000
        for i in range(0, len(self.db)):
            if train_sample[1] == self.db[i][1]:
                quaternion_distance = compute_quaternion_angle_diff(train_sample[2:6], self.db[i][2:6])
                if quaternion_distance < minimum_quaternion_distance:
                    minimum_quaternion_distance = quaternion_distance
                    # print(self.db[i][2:6], " ", train_sample[2:6])
                    puller = self.db[i]
                    # print(train_sample[2:6], '\n', self.db[i][2:6], '\n', quaternion_distance)
        # print(puller[2:6], " ", train_sample[2:6])
        return puller

    def get_next_batch(self):
        anchor_index = random.sample(range(0, len(self.train)), self.batch_size)
        anchors = [self.train[i] for i in anchor_index]

        pullers = [self.get_closest_puller(anchor) for anchor in anchors]

        pusher_index = random.sample(range(0, len(self.db)), self.batch_size)  # Randomly chosen
        pushers = [self.db[i] for i in pusher_index]
        return [(anchor, puller, pusher) for anchor, puller, pusher in zip(anchors, pullers, pushers)]

    def __iter__(self):
        for i in range(0, self.iter_per_epoch):
            yield self.get_next_batch()


if __name__ == '__main__':
    # gen = DatasetGenerator(5, 4)
    # for batch in iter(gen):
    #     print(len(batch), len(batch[0]), batch[0][0][0].shape)
    pass

