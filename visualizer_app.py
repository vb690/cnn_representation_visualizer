import pickle

import numpy as np
from scipy.interpolate import interp1d

import pandas as pd

from PIL import Image

import streamlit as st

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

CATEGORY_MAPPER = {
    0: 'T-shirt / Top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dresss',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}
INDEX_MAPPER = {category: index for index, category in CATEGORY_MAPPER.items()}


def visualize_filters(embeddings, image_index=0, time_index=0, **kwargs):
    """Visulize learned filters and return figure
    """
    plt.close('all')
    filters = embeddings[time_index][image_index, :, :, :]
    column = int(np.sqrt(filters.shape[-1]))
    fig, axs = plt.subplots(
        column,
        column,
        figsize=(10, 10),
        sharex=True,
        sharey=True
    )

    for filter_index, ax in enumerate(axs.flatten()):

        ax.imshow(
            filters[:, :, filter_index],
            vmin=0,
            vmax=1,
            **kwargs
        )
        # ax.set_title(f'Filter {filter_index} after batch {time_index}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    return fig


def plotly_scatter_3d(traces, width=1000, height=600):
    """Produce a plotly scatter 3d plot
    """
    names = set()
    fig = go.Figure(data=traces)
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name)
    )
    fig.update_layout(
        width=width,
        height=height,
        autosize=True,
        scene=dict(
            aspectratio=dict(x=1.1, y=2.25, z=1.1),
            xaxis=dict(title='UMAP 1'),
            yaxis=dict(title='Batch Number'),
            zaxis=dict(title='UMAP 2'),

        ),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.5, y=0.1, z=-0.1)
        )
    )
    return fig


def plotly_scatter_2d(traces_1, traces_2, width=1000, height=800):
    """Produce a plotly scatter 3d plot
    """
    names = set()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True
    )
    fig.add_traces(
        traces_1,
        rows=[1 for i in range(len(traces_1))],
        cols=[1 for i in range(len(traces_1))]
    )
    fig.add_traces(
        traces_2,
        rows=[2 for i in range(len(traces_2))],
        cols=[1 for i in range(len(traces_2))]
    )

    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name)
    )
    fig.update_yaxes(
        title_text='UMAP 1',
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text='UMAP 2',
        row=2,
        col=1
    )
    fig.update_xaxes(
        title_text='Batch Number',
        row=2,
        col=1
    )
    fig.update_layout(
        width=width,
        height=height,
        autosize=True,
    )
    return fig


def visualize_embedding(embedding_df, time_index, selected_images,
                        selected_emeddings, flatland):
    """Visualize UMAP traces as 3d or 2d Plots
    """
    embedding_df = embedding_df[embedding_df['id'] <= selected_images]
    n_embeddings = embedding_df['batcn_n'].max() + 1
    n_images = embedding_df['id'].max() + 1
    classes = embedding_df[embedding_df['batcn_n'] == 0]['class'].values

    f_umap_1 = interp1d(
        embedding_df['batcn_n'][embedding_df['id'] == 0],
        embedding_df['UMAP_1'].values.reshape(n_embeddings, n_images).T,
        kind='cubic'
    )
    f_umap_2 = interp1d(
        embedding_df['batcn_n'][embedding_df['id'] == 0],
        embedding_df['UMAP_2'].values.reshape(n_embeddings, n_images).T,
        kind='cubic'
    )

    z = np.linspace(0, time_index, time_index)
    palette = px.colors.diverging.Spectral
    interpolated_traces = [f_umap_1(z), f_umap_2(z)]

    if flatland:
        traces_1_umap = []
        traces_2_umap = []
    else:
        traces_3_umap = []

    for image in range(n_images):

        if CATEGORY_MAPPER[classes[image]] in selected_emeddings:
            visible = True
        else:
            visible = 'legendonly'

        if flatland:
            # first dimesnion
            trace_1_umap = go.Scatter(
                x=z,
                y=interpolated_traces[0][image],
                mode='lines',
                line=dict(
                    color=palette[classes[image]],
                    width=1.5
                ),
                opacity=1,
                legendgroup=f'Category {CATEGORY_MAPPER[classes[image]]}',
                name=f'Category {CATEGORY_MAPPER[classes[image]]}',
                visible=visible
            )
            traces_1_umap.append(trace_1_umap)
            # Second dimension
            trace_2_umap = go.Scatter(
                x=z,
                y=interpolated_traces[1][image],
                mode='lines',
                line=dict(
                    color=palette[classes[image]],
                    width=1.5
                ),
                opacity=1,
                legendgroup=f'Category {CATEGORY_MAPPER[classes[image]]}',
                name=f'Category {CATEGORY_MAPPER[classes[image]]}',
                visible=visible
            )
            traces_2_umap.append(trace_2_umap)
        else:
            trace_3_umap = go.Scatter3d(
                x=interpolated_traces[0][image],
                y=z,
                z=interpolated_traces[1][image],
                mode='lines',
                line=dict(
                    color=palette[classes[image]],
                    width=1.5
                ),
                opacity=1,
                legendgroup=f'Category {CATEGORY_MAPPER[classes[image]]}',
                name=f'Category {CATEGORY_MAPPER[classes[image]]}',
                visible=visible
            )
            traces_3_umap.append(trace_3_umap)

    if flatland:
        fig_1_2_umap = plotly_scatter_2d(
            traces_1=traces_1_umap,
            traces_2=traces_2_umap
        )
        return fig_1_2_umap
    else:
        fig_3_umap = plotly_scatter_3d(
            traces=traces_3_umap
        )
        return fig_3_umap


@st.cache(hash_funcs={dict: lambda _: None})
def load_data(dataset_name):
    """Load convolution filters for a given dataset
    """
    images = np.load(f'results//images//{dataset_name}.npy')

    with open(f'results//filters//{dataset_name}_conv_1.pkl', 'rb') as in_conv:
        conv_1 = pickle.load(in_conv)

    with open(f'results//filters//{dataset_name}_conv_2.pkl', 'rb') as in_conv:
        conv_2 = pickle.load(in_conv)

    embedding_df = pd.read_csv(f'results//embeddings//{dataset_name}.csv')

    return images, conv_1, conv_2, embedding_df


@st.cache(hash_funcs={dict: lambda _: None})
def get_figures(images, conv_1, conv_2, embedding_df, image_index,
                batch_number, selected_images, selected_emeddings, flatland):
    """Get all the figures objects
    """
    fig_image, ax_image = plt.subplots(
        figsize=(10, 10)
    )
    ax_image.imshow(
        images[image_index, :, :, 0],
        cmap='binary'
    )
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    fig_conv_1 = visualize_filters(
            conv_1,
            image_index=image_index,
            time_index=batch_number,
            cmap='magma'
        )

    fig_conv_2 = visualize_filters(
            conv_2,
            image_index=image_index,
            time_index=batch_number,
            cmap='magma'
        )

    fig_embeddings = visualize_embedding(
        embedding_df=embedding_df,
        time_index=batch_number,
        selected_images=selected_images,
        selected_emeddings=selected_emeddings,
        flatland=flatland
    )

    figures = {
        'fig_image': fig_image,
        'fig_conv_1': fig_conv_1,
        'fig_conv_2': fig_conv_2
    }
    if flatland:
        figures['fig_embeddings_1_2'] = fig_embeddings
    else:
        figures['fig_embeddings_3'] = fig_embeddings

    return figures


def run_app():
    """Run Streamlit app
    """
    st.set_page_config(
        page_title='CNN Anatomy Visualizer',
        page_icon='🤖',
        layout='wide'
    )

    ###########################################################################

    st.title('Anatomy of a Convolutional Neural Network')

    col1_mnsit, col2_lenet = st.beta_columns(2)
    lenet = Image.open('images//header.png')
    fmnist = Image.open('images//fashion_mnist.png')
    col1_mnsit.header('Fashion-MNSIT Dataset')
    col1_mnsit.markdown(
        """
        Fashion-MNIST is a dataset of Zalando's article images consisting of
        a training set of 60,000 examples and a test set of 10,000 examples.
        Each example is a 28x28 grayscale image, associated with a label
        from 10 classes. Fashion-MNIST is intended to serve as a direct drop-in
        replacement for the original MNIST dataset for overcoming some of its
        limitations (e.g. simplicity and over-use).
        """
    )
    col1_mnsit.image(fmnist, caption='Fashion-MNIST')
    col2_lenet.header('LeNet-5 Convolutional Neural Network')
    col2_lenet.markdown(
        """
        LeNet-5 is one of the earliest convolutional neural network (CNN)
        architecture propsed by Yann LeCun et al. in 1989. It consists of
        seven layers implementing the basic operations of found in CNNs:
        convolution, pooling and fully connected. The lightweight
        architecture employed for this project includes
        """
    )
    col2_lenet.image(lenet, caption='LeNet-5')
    col2_lenet.markdown(
        """
        | Components |  |
        |-|-|
        | Input Image | Grayscale 28x28 images |
        | First Convolution | Four filters from a 5x5 kernel |
        | Max Pooling | Spatial subsampling from a 2x2 window |
        | Second Convolution | Nine filters from a 5x5 kernel |
        | Max Pooling | Spatial subsampling from a 2x2 window |
        | Fully Connected | Three fully connected layers with \
            80, 40 and 20 hidden units |
        | Activation |  Each layer with trainable parameters is \
            followed by a sigmoid function|
        | Output | Fully connected layer with softmax activation function|
        """
    )

    ###########################################################################

    images, conv_1, conv_2, embedding_df = load_data('fashion_mnist')

    st.sidebar.title('Visualizer Parameters')

    st.sidebar.header('Training Stage')
    batch_number = st.sidebar.slider(
        'Select Batch Number',
        min_value=0,
        max_value=len(conv_1) - 1,
        value=len(conv_1) - 1
    )

    st.sidebar.header('Filters Visualization')
    image_index = st.sidebar.selectbox(
        'Select Category',
        [CATEGORY_MAPPER[category] for category in range(10)]
    )
    image_index = INDEX_MAPPER[image_index]

    st.sidebar.header('Embedding Visualization')
    selected_images = st.sidebar.slider(
        'Select N° Images Embedded',
        min_value=1,
        max_value=1000,
        value=200
    )
    selected_emeddings = st.sidebar.multiselect(
        'Select Categories Embedded',
        [CATEGORY_MAPPER[category] for category in range(10)] + ['all'],
        default=['all']
    )
    if 'all' in selected_emeddings:
        selected_emeddings = [
            CATEGORY_MAPPER[category] for category in range(10)
        ]
    flatland = st.sidebar.checkbox(
        'Visualize Embedding in Flatland',
        value=False
    )



    ###########################################################################

    figures = get_figures(
        images=images,
        conv_1=conv_1,
        conv_2=conv_2,
        embedding_df=embedding_df,
        image_index=image_index,
        batch_number=batch_number,
        selected_images=selected_images,
        selected_emeddings=selected_emeddings,
        flatland=flatland
    )

    with st.beta_expander('Convolutional Filters'):
        col1_image, col2_filters, col3_filters = st.beta_columns(3)
        col1_image.header('Input Image')
        col1_image.pyplot(figures['fig_image'])
        col2_filters.header('First Convolution')
        col2_filters.pyplot(figures['fig_conv_1'])
        col3_filters.header('Second Convolution')
        col3_filters.pyplot(figures['fig_conv_2'])

    with st.beta_expander('Learned Embedding'):
        st.header('Temporal Alligned UMAP')
        if flatland:
            st.plotly_chart(figures['fig_embeddings_1_2'])
        else:
            st.plotly_chart(figures['fig_embeddings_3'])


if __name__ == '__main__':
    run_app()
