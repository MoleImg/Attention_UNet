# coding: utf-8
"""
TODO:
    Data import & training & test for the model
"""

__author__ = 'ACM'

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse


from AttResUNet import Attention_ResUNet
from tensorflow.contrib.keras import optimizers, callbacks
import Config as conf

if conf.TRAIN_FLAG:
    # data import
    x_train, y_train = mio.loadBatchData(
        conf.TRAIN_DATA_PATH, conf.TRAINING_SIZE, start_num=conf.TRAINING_START)

    # model constrconftion
    model = Attention_ResUNet(dropout_rate=conf.DROPOUT_RATE, batch_norm=conf.BATCH_NORM_FLAG)
    if conf.MODEL_LOAD_FLAG:
        # model.load_weights(conf.MODEL_LOAD_PATH)
        model.load_weights(conf.MODEL_LOAD_PATH)
    # training setup
    optimizer = optimizers.Adam() # training optimizer
    loss = ['mean_squared_error'] # training loss function
    metrics = ['mae'] # training evaluation metrics

    # model configuration
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Tensorboard visualization
    if conf.TENSORBOARD_FLAG:
        tb = callbacks.TensorBoard(log_dir=conf.LOG_PATH,
                                  histogram_freq=0,
                                  batch_size=conf.BATCH_SIZE,
                                  write_graph=False,
                                  write_images=True)
                                  # embeddings_freq=0,
                                  # embeddings_layer_names=None,
                                  # embeddings_metadata=None)
    else:
        tb = None


    # model training
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=conf.BATCH_SIZE,
                        epochs=conf.EPOCH,
                        verbose=1,
                        callbacks=[tb],
                        validation_split=conf.VALIDATION_SPLIT,
                        validation_data=None,
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0)

    # visualization
    if conf.TRAINING_VISUAL_FLAG:
        print(history.history.keys())

        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='lower left')
        plt.show()

        # results storing
        # fig.savefig('performance.png')
    if conf.MODEL_SAVE_FLAG:
        model.save(conf.MODEL_SAVE_PATH)


