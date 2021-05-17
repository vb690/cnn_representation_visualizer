import numpy as np

from tensorflow.keras.datasets import fashion_mnist


def prepare_alligned_umap_data(embeddings, max_time_index=10):
    """Prepare data for running alligned umap
    """
    list_embeddings = []
    mappers = []
    for time_index in range(max_time_index):

        time_embedding = embeddings[time_index]
        list_embeddings.append(
            time_embedding
        )
        mappers.append(
            {embedding_index: embedding_index for
                embedding_index in range(time_embedding.shape[0])}
        )

    return list_embeddings, mappers[1:]


def load_data(loader=fashion_mnist, channels=1, batch_size=500, sample=1000):
    """Load data from dataset loader
    """
    train, test = loader.load_data()

    X_tr, y_tr = train
    X_ts, y_ts = test

    rows, columns = X_tr.shape[1], X_tr.shape[2]

    X_tr = X_tr.reshape(-1, rows, columns, channels)
    X_tr = X_tr.reshape(-1, batch_size, rows, columns, channels)

    y_tr = y_tr.reshape(-1, batch_size, channels)

    random_ind = np.random.choice(
        [i for i in range(X_ts.shape[0])],
        sample,
        replace=False
    )
    X_ts = X_ts.reshape(-1, rows, columns, channels)

    X_ts = X_ts[random_ind]
    y_ts = y_ts[random_ind]

    return X_tr, y_tr, X_ts, y_ts
