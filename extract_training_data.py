import pickle

import numpy as np

import pandas as pd

from umap.aligned_umap import AlignedUMAP

from modules.models import create_model_encoders, get_representations
from modules.utils.data_utils import load_data, prepare_alligned_umap_data

EPOCHS = 2

X_tr, y_tr, X_ts, y_ts = load_data()

selected_samples = []
for unique_class in range(10):

    index = np.argwhere(y_ts.flatten() == unique_class).flatten()[0]
    selected_samples.append(index)

np.save('results//images//fashion_mnist.npy', X_ts[selected_samples])

model, conv_1_enc, conv_2_enc, dense_enc = create_model_encoders(
    X_tr[0],
    y_tr[0]
)

representations = get_representations(
    model=model,
    X_tr=X_tr,
    y_tr=y_tr,
    X_ts=X_ts,
    conv_1_enc=conv_1_enc,
    conv_2_enc=conv_2_enc,
    dense_enc=dense_enc,
    epochs=EPOCHS
)

for batch_n in range(len(representations['conv_1'])):

    representations['conv_1'][batch_n] =   \
        representations['conv_1'][batch_n][selected_samples]
    representations['conv_2'][batch_n] =    \
        representations['conv_2'][batch_n][selected_samples]

with open('results//filters//fashion_mnist_conv_1.pkl', 'wb') as out_conv_1:
    pickle.dump(representations['conv_1'], out_conv_1, pickle.HIGHEST_PROTOCOL)
with open('results//filters//fashion_mnist_conv_2.pkl', 'wb') as out_conv_2:
    pickle.dump(representations['conv_2'], out_conv_2, pickle.HIGHEST_PROTOCOL)

embeddings, relations = prepare_alligned_umap_data(
    representations['dense'],
    max_time_index=len(representations['dense'])
)

print('Start Alligned UMAP')

mapper = AlignedUMAP(
    metric='cosine',
    n_neighbors=30,
    alignment_regularisation=0.1,
    alignment_window_size=10,
    n_epochs=200,
    random_state=42,
    verbose=True
)
mapper.fit(embeddings, relations=relations)
reductions = mapper.embeddings_

n_embeddings = len(reductions)
embedding_df = pd.DataFrame(
    np.vstack(reductions),
    columns=('UMAP_1', 'UMAP_2')
)
embedding_df['batcn_n'] = np.repeat(
    np.array([i for i in range(n_embeddings)]),
    reductions[0].shape[0]
)
embedding_df['id'] = np.tile(
    np.arange(reductions[0].shape[0]),
    n_embeddings
)
embedding_df['class'] = np.tile(
    y_ts.flatten(),
    n_embeddings
)

###############################################################################

embedding_df.to_csv(
    'results//embeddings//fashion_mnist.csv'
)
