import tensorflow as tf


def Triplet_And_Pair_Loss(model, images, batch_size=16, margin=0.01):
    embeddings = model(images)
    anchor_output = embeddings[0:batch_size * 3:3]
    positive_output = embeddings[1:batch_size * 3:3]
    negative_output = embeddings[2:batch_size * 3:3]
    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)
    triple_loss = tf.maximum(0.0, 1 - d_neg / (d_pos + margin))
    pair_loss = d_pos
    # Reduce again
    loss = tf.reduce_sum(triple_loss) + 2*tf.reduce_sum(pair_loss)
    return loss