import pickle

import numpy as np
from scipy.interpolate import interp1d

import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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
                        selected_emeddings, flatland, category_mapper):
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

        if category_mapper[classes[image]] in selected_emeddings:
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
                    width=0.75
                ),
                opacity=1,
                legendgroup=f'Category {category_mapper[classes[image]]}',
                name=f'Category {category_mapper[classes[image]]}',
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
                    width=0.75
                ),
                opacity=1,
                legendgroup=f'Category {category_mapper[classes[image]]}',
                name=f'Category {category_mapper[classes[image]]}',
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
                    width=0.75
                ),
                opacity=1,
                legendgroup=f'Category {category_mapper[classes[image]]}',
                name=f'Category {category_mapper[classes[image]]}',
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


def get_figures(images, conv_1, conv_2, embedding_df, image_index,
                batch_number, selected_images, selected_emeddings, flatland,
                category_mapper):
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
        flatland=flatland,
        category_mapper=category_mapper
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
