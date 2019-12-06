import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def train_neural_network(x_train, y_train, epochs):
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(47, activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs)

    return model


def train_nn(train_data, val_data, train_temp, load_model=""):
    (x_train, y_train) = train_data
    (x_val, y_val) = val_data

    def cross_entropy_loss(y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred / train_temp)


    if(load_model != ""):
        model = keras.models.load_model(load_model, custom_objects={'cross_entropy_loss': cross_entropy_loss})
        #model = keras.models.load_model('final_1_init.h5', custom_objects={'cross_entropy_loss': cross_entropy_loss})
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss=cross_entropy_loss, optimizer=sgd, metrics=['accuracy'])
        return model

    # Create a model
    model = keras.Sequential()

    # Add 9 layers to the model

    # 2D Convolutional Layer with 32 output filters, 3x3 size of convolutional window and ReLu Activation function

    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Flatten the input
    model.add(keras.layers.Flatten())
    # Densely connected neural network layer with 200 output units and ReLu activation function
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(47))

    sgd = keras.optimizers.SGD(lr=0.0, decay=1e-6, momentum=0.9, nesterov=True)

    # Configure the learning process
    model.compile(loss=cross_entropy_loss, optimizer=sgd, metrics=['accuracy'])

    # Train a model
    history = model.fit(x_train, y_train, batch_size=128, validation_data=(x_val, y_val), nb_epoch=50, shuffle=True)

    # evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_val, y_val, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))


    return model


def train_distillation(train_data, val_data, train_temp, load_model = ""):
    (x_train, y_train) = train_data
    (x_val, y_val) = val_data

    if(load_model != ""):
        normal = train_nn(train_data, val_data, train_temp, load_model)
        return normal
    else:
        normal = train_nn(train_data, val_data, train_temp, "")

    normal_predicted = normal.predict(x_train)

    with tf.compat.v1.Session() as sess:
        prob_pred = sess.run(tf.nn.softmax(normal_predicted / train_temp))
        y_train = prob_pred

    distilled = train_nn((x_train, y_train), val_data, train_temp)
    distilled_predicted = distilled.predict(x_train)
    y_val_classes = distilled.predict_classes(x_val)
    y_val_classes = tf.keras.utils.to_categorical(y_val_classes)
    return distilled
