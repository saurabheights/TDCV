import os

import numpy as np
import tensorflow as tf
import torch

tf.enable_eager_execution()
from sklearn.neighbors import NearestNeighbors
from tensorboardX import SummaryWriter
import models
from loss import Triplet_And_Pair_Loss
from data_loader import DatasetGenerator, compute_quaternion_angle_diff
import matplotlib.pyplot as plt


num_epochs = 100
iter_per_epoch = 1000
batch_size = 32
margin = 0.01
checkpoint_dir = None  # "model_checkpoint_epoch_003_Loss:_0.003.ckpt"
output_dir = 'outputPair1Triplet1-2/'


def grad(model, images):
    with tf.GradientTape() as tape:
        loss_value = Triplet_And_Pair_Loss(model, images)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


gen = DatasetGenerator(iter_per_epoch, batch_size, True)

# Create model and optimizer
model = models.TdcvModel()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.99)
global_step = tf.train.get_or_create_global_step()

# Load a pre-trained model
if checkpoint_dir is not None:
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Tensorboard - Does not works with Eager Execution since there is no graph. Using TensorboardX
writer = SummaryWriter(output_dir)

train_loss_results = []
itr = 0
for epoch in range(num_epochs):
    # Training loop - using batches of 32
    epoch_loss_sum = 0.0
    for batch in iter(gen):
        images_np = []
        for sample in batch:
            images_np.append(sample[0][0])  # Anchor
            images_np.append(sample[1][0])  # Puller
            images_np.append(sample[2][0])  # Pusher
        dataset = tf.data.Dataset.from_tensor_slices(images_np).batch(batch_size * 3)
        for batch_input in dataset:
            # Optimize the model
            loss_value, grads = grad(model, batch_input)
            optimizer.apply_gradients(zip(grads, model.variables), global_step)
            itr += 1

            if itr % 10 == 0:
                print('%s, %s' % (itr, tf.reduce_sum(loss_value).numpy()))
                writer.add_scalar('train/loss', tf.reduce_sum(loss_value).numpy(), itr)

        # Track progress
        # epoch_loss_avg(loss_value)  # add current batch loss
        train_loss_results.append(tf.reduce_sum(loss_value).numpy())
        epoch_loss_sum = epoch_loss_sum + tf.reduce_sum(loss_value).numpy()

    # end epoch
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_sum / iter_per_epoch))
    writer.add_scalar('train/epoch_loss', epoch_loss_sum / iter_per_epoch, itr)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(output_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(file_prefix=checkpoint_prefix)

    # Compute the DB descriptors
    db = gen.db
    db_images = [db_sample[0] for db_sample in db]
    dataset = tf.data.Dataset.from_tensor_slices(db_images).batch(1)
    db_data_for_knn = np.ndarray([len(db), 1 + 4 + 16], dtype=np.float64)  # Save Embedding Features and 4 quaternions
    db_image_index = 0
    embeddings = np.ndarray([0,16], dtype=np.float64)
    label_imgs = torch.IntTensor()
    metadata = []
    for image in dataset:
        # Compute the descriptors
        embedding = model(image)
        db_data_for_knn[db_image_index, :] = np.append(np.array(db[db_image_index][1:6]), embedding.numpy()[0, :])
        embeddings = np.vstack((embeddings, embedding.numpy()))
        metadata.append(db[db_image_index][1])
        label_img = torch.from_numpy(image.numpy() * gen.dev + gen.mean).int().permute(0,3,1,2).squeeze()
        label_imgs = torch.cat((label_imgs, label_img.view(-1, *label_img.shape)))
        db_image_index += 1

    writer.add_embedding(embeddings,
                         metadata=metadata,
                         label_img=label_imgs,
                         tag='test_epoch_'+str(epoch))

    # Save the trained descriptors
    filename = os.path.join(output_dir, "TrainedDBFeatures-epoch_%d.bin" % epoch)
    with open(filename, 'w') as f:
        db_data_for_knn.tofile(f)

    # Compute NN Index
    db_data_for_knn_label = db_data_for_knn[:, 0]
    db_data_for_knn_quat = db_data_for_knn[:, 1:5]
    db_data_for_knn_features = db_data_for_knn[:, 5:21]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(db_data_for_knn_features)

    # Compute accuracy over each test images
    test_dataset = gen.test
    test_images = [test_sample[0] for test_sample in gen.test]
    dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(1)
    test_image_index = 0
    accuracy = 0
    histogram_of_angular_difference = np.ndarray([0], dtype=np.float64)
    angular_differences = []
    for test_image_index, image in enumerate(dataset):
        # Compute the descriptors
        embedding = model(image)
        test_label = test_dataset[test_image_index][1]
        prediction_distance, prediction_index = nbrs.kneighbors(embedding.numpy())
        prediction_label = int(np.round(db_data_for_knn_label[prediction_index[0][0]]))
        if prediction_label == test_label:
            accuracy += 1
            angular_difference = compute_quaternion_angle_diff(db_data_for_knn_quat[prediction_index[0][0]],
                                                               test_dataset[test_image_index][2:6])
            angular_differences.append(angular_difference)

    bins = list(range(0, 181, 1))
    hist, bin_edges = np.histogram(angular_differences, bins=bins)
    print('Histogram for each degree', hist)
    bins = [0, 10, 20, 40, 180]
    hist, bin_edges = np.histogram(angular_differences, bins=bins)
    hist = hist / len(test_images)
    print(hist)
    hist = np.cumsum(hist)
    print(hist)
    fig, ax = plt.subplots()
    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)), hist, width=1)
    # Set the ticks to the middle of the bars
    ax.set_xticks([i for i, j in enumerate(hist)])
    # Set the xticklabels to a string that tells us what the bin edges were
    ax.set_xticklabels(['{} - {}'.format(bins[i], bins[i + 1]) for i, j in enumerate(hist)])
    plt.savefig(output_dir + 'Test_Histogram_Plot_%3d.png'%epoch)
    plt.show()

    accuracy = accuracy / len(test_images)
    print("The accuracy(%% of correct predictions) is: %f" % accuracy)

# export scalar data to JSON for external processing
writer.export_scalars_to_json(os.path.join(output_dir, "./all_scalars.json"))
writer.close()
