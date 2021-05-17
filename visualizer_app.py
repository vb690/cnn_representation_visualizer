from PIL import Image

import streamlit as st

from modules.utils.app_utils import load_data, get_figures

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


def run_app():
    """Run Streamlit app
    """
    st.set_page_config(
        page_title='CNN Anatomy Visualizer',
        page_icon='🤖',
        layout='wide'
    )

    ###########################################################################

    st.title('The Learning Anatomy of a Convolutional Neural Network')

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
        architecture employed for this project has the following structure.
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
        | Output | Fully connected layer with 10 hidden units and \
            softmax activation function|
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
        value=350
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
        flatland=flatland,
        category_mapper=CATEGORY_MAPPER
    )

    with st.beta_expander('Convolutional Filters'):
        col1_image, col2_filters, col3_filters = st.beta_columns(3)
        col1_image.header('Input Image')
        col1_image.pyplot(figures['fig_image'])
        col2_filters.header('First Convolution')
        col2_filters.pyplot(figures['fig_conv_1'])
        col2_filters.markdown(
            """
            The first convolutional filters extrapolate global
            features that mostly pertain the general structure of the image.
            Such as the countour or large details of the represented object.
            """
        )
        col3_filters.header('Second Convolution')
        col3_filters.pyplot(figures['fig_conv_2'])
        col3_filters.markdown(
            """
            The second convolutional filters work more locally extracting
            minute details from the previous representation.
            Such as lines, edges or textutre information.
            """
        )

    with st.beta_expander('Learned Embedding'):
        st.header('Temporal Alligned UMAP')
        st.markdown(
            """
            Traditional Artificial Neural Network applications work under the
            hypothesis that similarities between inputs, with respect to their
            targets, can be represented by learning their spatial location
            in a z-dimensional space (where z is the dimensionality of the
            portion of the network producing the representation) where
            objects that are similar to each other are also closer in space.

            For making this spatial property more evident we will leverage the
            Uniform Manifold Approximation and Projection algorithm (UMAP)
            by McInnes et al. UMAP is a dimension reduction technique that

            1. Constructs a high dimensional graph representation of a given
            dataset.
            2. Then optimizes a low-dimensional version of the same graph to
            be as structurally similar as possible to the original one.
            """
        )
        if flatland:
            st.plotly_chart(figures['fig_embeddings_1_2'])
        else:
            st.plotly_chart(figures['fig_embeddings_3'])
        st.markdown(
            """
            We can see how the representation of each category of the
            Fashion-MNIST learned by the last fully connected layer of
            LeNet-5 (the one just before the final softmax
            classifier) shift in space over training.
            #### Factoids
            * Obejcts like Ankle Boots, Sneakers and Sandals (i.e. footwear)
            form a compact group that progressively moves away from things
            like pullovers.
            * Higly similar objects like
            Shirts and Coats occupy almost the same space.
            * Objects with low variability in shape like trousers have a
            consistent compact representation while variable things like
            dresses progressively spread out in space.
            """
        )

    with st.beta_expander('References and Useful Links'):
        st.markdown(
            """
            1. Xiao, Han, Kashif Rasul, and Roland Vollgraf. "Fashion-mnist: a
            novel image dataset for benchmarking machine learning algorithms."
            arXiv preprint arXiv:1708.07747 (2017).
            2. LeCun, Yann, et al. "Backpropagation applied to handwritten zip
            code recognition." Neural computation 1.4 (1989): 541-551.
            3. McInnes, Leland, John Healy, and James Melville. "Umap: Uniform
            manifold approximation and projection for dimension reduction."
            arXiv preprint arXiv:1802.03426 (2018).
            4. [Understanding UMAP]
            (https://pair-code.github.io/understanding-umap/)
            5. [Website UMAP](https://umap-learn.readthedocs.io/en/latest/)
            6. [Temporal Alligned UMAP](https://umap-learn.readthedocs.io/en/latest/aligned_umap_politics_demo.html)
            """
        )


if __name__ == '__main__':
    run_app()
