#streamlit run main.py

import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex  # Import Annoy

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Model setup
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)

    return normalized_result

def build_annoy_index(feature_list, n_trees=10):
    dim = feature_list.shape[1]  # Dimension of the features
    annoy_index = AnnoyIndex(dim, 'euclidean')

    for i in range(len(feature_list)):
        annoy_index.add_item(i, feature_list[i])

    annoy_index.build(n_trees)
    
    return annoy_index

def recommend(features, annoy_index, n_neighbors=6):
    indices = annoy_index.get_nns_by_vector(features, n_neighbors)
    return indices

# Build Annoy index (only once)
annoy_index = build_annoy_index(feature_list)

# Steps
# File upload -> Save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file) == 1:
        # Display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommendation
        indices = recommend(features, annoy_index)
        # Show recommendations
        col1, col2, col3, col4, col5 = st.columns(5)

        # Display recommended images with an expander for larger view
        for idx, col in zip(indices, [col1, col2, col3, col4, col5]):
            with col:
                img_path = filenames[idx]
                st.image(img_path)
                with st.expander("View larger"):
                    st.image(img_path, use_column_width=True)  # Display larger image within the expander
    else:
        st.header("Some error occurred in file upload")
