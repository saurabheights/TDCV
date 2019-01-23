import tensorflow as tf
import numpy as np

import models
from data_loader import DatasetGenerator
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from data_loader import compute_quaternion_angle_diff

tf.enable_eager_execution()

epoch=14
db_descriptors_file='outputPair3Triplet1/TrainedDBFeatures-epoch_%d.bin'% epoch

# Load the model
model = models.TdcvModel()
checkpoint_dir = 'output2'
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Load the trained descriptors
with open(db_descriptors_file, 'r') as f:
    db_data_for_knn = np.fromfile(f)
    db_data_for_knn=db_data_for_knn.reshape(-1, 21)

# Create Neighbor KNN Index of db descriptors
db_data_for_knn_label = db_data_for_knn[:,0]
db_data_for_knn_quat = db_data_for_knn[:,1:5]
db_data_for_knn_features = db_data_for_knn[:,5:21]
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(db_data_for_knn_features)


# Load the dataset
iter_per_epoch = 200  # Not needed
batch_size = 16  # Not needed
gen = DatasetGenerator(iter_per_epoch, batch_size, False)

# Compute accuracy over each test images
test_dataset = gen.test
test_images = [test_sample[0] for test_sample in gen.test]
dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(1)
accuracy = 0

angular_differences = []
for test_image_index, image in enumerate(dataset):
    # Compute the descriptors
    embedding = model(image)
    test_label = test_dataset[test_image_index][1]
    prediction_distance, prediction_index = nbrs.kneighbors(embedding.numpy(), 5)
    prediction_label = int(np.round(db_data_for_knn_label[prediction_index[0][0]]))
    if prediction_label == test_label:
        accuracy +=1
        angular_difference = compute_quaternion_angle_diff(db_data_for_knn_quat[prediction_index[0][0]],
                                                           test_dataset[test_image_index][2:6])
        angular_differences.append(angular_difference)

bins = list(range(0,181,10))
hist, bin_edges = np.histogram(angular_differences, bins=bins)
hist= hist/len(test_images)
print(hist)
fig,ax = plt.subplots()
# Plot the histogram heights against integers on the x axis
ax.bar(range(len(hist)),hist,width=1)
# Set the ticks to the middle of the bars
ax.set_xticks([i for i,j in enumerate(hist)])
# Set the xticklabels to a string that tells us what the bin edges were
ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])
plt.show()
plt.savefig(checkpoint_dir + '/Test_Histogram_Plot_DuringTesting.png')
accuracy = accuracy/len(test_images)
print("The accuracy(%% of correct predictions) is: %f" % accuracy)
