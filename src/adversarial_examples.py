import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def fgsm(model, x_train, y_desired, epsilon=0.1):
    with tf.GradientTape() as tape:
        x_train = tf.convert_to_tensor(x_train)
        y_desired = tf.convert_to_tensor(y_desired)
        tape.watch(x_train)
        prediction = model(x_train)

        loss_object = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_object(y_desired, prediction)

    gradient = tape.gradient(loss, x_train)

    signed_grad = tf.sign(gradient)

    signed_grad = tf.clip_by_value(signed_grad, 0, 1.0)

    return signed_grad


def fgsm_iterative(model, x_train, y_desired, epsilon=0.05, epoch = 20):
    for iter in range(0, epoch):
        with tf.GradientTape() as tape:
            x_train = tf.convert_to_tensor(x_train)
            y_desired = tf.convert_to_tensor(y_desired)
            tape.watch(x_train)
            prediction = model(x_train)

            loss_object = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_object(y_desired, prediction)

        gradient = tape.gradient(loss, x_train)

        signed_grad = tf.sign(gradient)

        signed_grad = tf.clip_by_value(signed_grad, -1.0, 1.0)

        x_train = x_train - epsilon*signed_grad

    return x_train
