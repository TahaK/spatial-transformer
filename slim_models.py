import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from spatial_transformer import transformer

def localization(inputs):
    with tf.variable_scope('localization'):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding='VALID',
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.max_pool2d(inputs, [2, 2], scope='pool1')
            net = slim.conv2d(net, 20, [5, 5], stride=2, scope='conv1')
            net = slim.conv2d(net, 20, [5, 5], scope='conv2')
            net = slim.conv2d(net, 50, [9, 9], scope='conv3')
            net = slim.conv2d(net, 6, [1, 1], scope='conv4', activation_fn = None, biases_initializer = tf.constant_initializer(np.array([[1., 0, 0, 0, 1., 0]])))
            return net

def classification(transformed):
    with tf.variable_scope('classification'):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding='VALID',
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.avg_pool2d(transformed, [2,2], scope='pool1')
            net = slim.conv2d(net, 32, [7, 7], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 48, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.conv2d(net, 256, [4, 4], scope='conv3')
            embedding = tf.squeeze(net)
            net = slim.conv2d(net, 10, [1, 1], scope='conv4', activation_fn = None) 
            net = tf.squeeze(net, [1, 2])
            return net, embedding

def triplet_loss(anchor, positive, negative, margin):
    with tf.variable_scope('triplet_loss'):
        eps = 1e-10
        f_p = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), axis=1) + eps)
        f_n = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), axis=1) + eps)
        l = tf.reduce_mean(tf.maximum(0. , f_p - f_n + margin)) 
        return l

def stn_net(x_a, y_a, x_p, y_p, x_n, y_n, keep_prob, learning_rate, margin,values, log_dir, batch_size):
    x = tf.concat((x_a, x_p, x_n),axis=0, name= "imput")
    y = tf.concat((y_a, y_p, y_n),axis=0, name = "labels")

    # x_tensor = tf.reshape(x, [x.shape[0].value, 60, 60, 1])
    x_tensor = tf.reshape(x, [3*batch_size, 60, 60, 1], name='image')
    
    h_fc_loc2 = localization(x_tensor)
    out_size = (60, 60)
    h_trans = transformer(x_tensor, h_fc_loc2, out_size)
    h_trans = tf.reshape(h_trans, [3*batch_size, 60, 60, 1])
    
    y_logits, embedding = classification(h_trans)

    
    #embedding_viz = tf.Variable(embedding, trainable=False, name='viz_embedding')
    embedding_viz = tf.Variable(tf.zeros([3*batch_size, 256]), name="embedding")
    assignment = embedding_viz.assign(embedding)
    
    global_step = tf.Variable(0, trainable=False)
    boundaries = [2000, 4000, 6000, 8000]
    values = values
    alpha = tf.train.piecewise_constant(global_step, boundaries, values, name='alpha')
    # alpha = tf.Variable(0, trainable=False,dtype=tf.float32)

    anchor = embedding[:batch_size]
    positive = embedding[batch_size:(2*batch_size)]
    negative = embedding[(2*batch_size):]
    
    embedding_loss =  triplet_loss(anchor, positive, negative, margin) / margin
    
    classification_loss = tf.losses.softmax_cross_entropy(y, y_logits)
    
    # Loss
    loss =  tf.multiply((1-alpha) , classification_loss)  +  tf.multiply( alpha , embedding_loss)
    
    # Optimizer 
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    # Minimize
    optimizer = opt.minimize(loss, global_step=global_step)

    grads = opt.compute_gradients(loss)

    predictions = tf.argmax(y_logits, 1)
    labels = tf.argmax(y, 1)
    
    tf.summary.scalar("embedding", embedding_loss)
    tf.summary.scalar("xent", classification_loss)

    # Add summary histograms
    #for grad, var in grads:
    #  if grad is not None:
    #      tf.summary.histogram(var.op.name + '/gradients', grad)

    #for var in tf.trainable_variables():
    #  tf.summary.histogram(var.op.name, var)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(labels, predictions)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    summ = tf.summary.merge_all()
    
    input_image = tf.summary.image('input', x_tensor[-10:],10)
    localized_image = tf.summary.image('localized', h_trans[-10:],10)
    
    training_accuracy = tf.summary.scalar('training_accuracy', accuracy)
    validation_accuracy = tf.summary.scalar('validation_accuracy', accuracy)


    return optimizer, predictions, labels, h_trans, h_fc_loc2, summ, training_accuracy, validation_accuracy, input_image, localized_image, assignment

