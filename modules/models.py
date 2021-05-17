import random

from tqdm import tqdm

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam


def create_model_encoders(X, y):
    """Create a CNN model and realtive encoders
    """
    inp = Input((X.shape[1], X.shape[2], X.shape[3]))

    conv_1 = Conv2D(
        filters=4,
        kernel_size=5
    )(inp)
    conv_1 = Activation('sigmoid')(conv_1)
    pol_1 = MaxPooling2D(
        pool_size=2,
        strides=(2, 2)
    )(conv_1)

    conv_2 = Conv2D(
        filters=9,
        kernel_size=5
    )(pol_1)
    conv_2 = Activation('sigmoid')(conv_2)
    pol_2 = MaxPooling2D(
        pool_size=2,
        strides=(2, 2)
    )(conv_2)

    flat = Flatten()(pol_2)
    dense = Dense(80)(flat)
    dense = Activation('sigmoid')(dense)
    dense = Dense(40)(dense)
    dense = Activation('sigmoid')(dense)
    dense = Dense(20)(dense)
    dense = Activation('sigmoid')(dense)

    out = Dense(y.max())(dense)
    out = Activation('softmax')(dense)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy'
    )

    conv_1_enc = Model(inp, conv_1)

    conv_2_enc = Model(inp, conv_2)

    dense_enc = Model(inp, dense)

    return model, conv_1_enc, conv_2_enc, dense_enc


def get_representations(model, X_tr, y_tr, X_ts, conv_1_enc, conv_2_enc,
                        dense_enc, epochs):
    """Train model and get batch-to-batch generated representation
    """
    representations = {
        'conv_1': {},
        'conv_2': {},
        'dense': {},
    }
    representations['conv_1'][0] = conv_1_enc.predict(X_ts)
    representations['conv_2'][0] = conv_2_enc.predict(X_ts)
    representations['dense'][0] = dense_enc.predict(X_ts)

    index_batch = 1
    for epoch in range(epochs):

        rnd_batches = [batch for batch in range(X_tr.shape[0])]
        random.shuffle(rnd_batches)
        for rnd_batch in tqdm(rnd_batches):

            model.train_on_batch(
                X_tr[rnd_batch, :, :, :, :],
                y_tr[rnd_batch, :, :]
            )
            representations['conv_1'][index_batch] = conv_1_enc.predict(X_ts)
            representations['conv_2'][index_batch] = conv_2_enc.predict(X_ts)
            representations['dense'][index_batch] = dense_enc.predict(X_ts)

            index_batch += 1

    return representations
