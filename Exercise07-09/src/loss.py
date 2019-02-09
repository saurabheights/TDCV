import tensorflow as tf


def Triplet_And_Pair_Loss(model, images, batch_size=16, margin=0.01, pair_loss_weight_factor=1):
    embeddings = model(images)
    anchor_output = embeddings[0:batch_size * 3:3]
    positive_output = embeddings[1:batch_size * 3:3]
    negative_output = embeddings[2:batch_size * 3:3]
    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)
    triplet_loss = tf.maximum(0.0, 1 - d_neg / (d_pos + margin))
    pair_loss = d_pos

    # Reduce again and apply weight
    triplet_loss = tf.reduce_sum(triplet_loss)
    pair_loss = pair_loss_weight_factor*tf.reduce_sum(pair_loss)
    return triplet_loss, pair_loss