import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from scipy.ndimage.interpolation import zoom

from PIL import Image

import keras
from keras import models
from keras.models import Sequential, load_model
from keras.preprocessing.image import img_to_array, load_img, array_to_img

from streamlit_drawable_canvas import st_canvas

st.beta_set_page_config(page_title="Handwritten number recognition", page_icon="✍",
                        layout='centered', initial_sidebar_state="collapsed")


def runPrediction(model, image):
    im = Image.fromarray((image * 255).astype(np.uint8))
    img = img_to_array(im)
    # im.save("currentImage", "png")

    # img = img_to_array(img)
    # # reshape into a single sample with 1 channel
    # img = img.reshape(1, 28, 28, 1)
    # # prepare pixel data
    # img = img.astype('float32')
    # img = img / 255.0
    return img


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def main():
    model = load_model("./final_model.h5")
    # title
    st.markdown("""
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Handwritten number 🔢 </h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.beta_columns([2, 2])

    with col1:
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
           Definition of the project

            """)
        '''
        ## How does it work ❓ 
        You can upload an 80x80 image with a black background and white writing. Then press the prediction button and the result will be displayed at the bottom.
        '''

    with col2:
        uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg'])
        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name,
                            "FileType": uploaded_file.type,
                            "FileSize": uploaded_file.size}
            st.write(file_details)

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=28,
            width=28,
            drawing_mode="freedraw",
            key="canvas",
        )
        if st.button('Predict') and canvas_result.image_data is not None:
            col1.write('''
            ## Results 🔍 
            ''')
            col1.success(
                f"{runPrediction(model, canvas_result.image_data)} are recommended by the A.I for your farm.")
    # code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning(
        "Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
