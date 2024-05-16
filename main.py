import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import streamlit_shadcn_ui as ui
from st_aggrid import AgGrid
from transformers import pipeline
import os
from streamlit_lottie import st_lottie
from IPython.display import JSON
import json
from st_aggrid import AgGrid, GridOptionsBuilder, AgGridTheme
from bs4 import BeautifulSoup
import re
import unicodedata
from llama_cpp import Llama
from transformers import AutoTokenizer
from tqdm import tqdm

import streamlit as st
st.set_page_config(layout="wide")


import section1
import section2
import section3
import section4



            
def main():
    
    st.sidebar.markdown(
        """
        <style>
        .sidebar-image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50%;
        }
        .sidebar-image {
            max-width: 50%;
            max-height: 50%;
        }
        </style>
        <div class="sidebar-image-container">
            <img class="sidebar-image" src="https://tse4.mm.bing.net/th/id/OIG1.vxQ7No7lAOSK9rm5CPdQ?w=270&h=270&c=6&r=0&o=5&dpr=1.5&pid=ImgGn" alt="Centered Image">
            
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("# TinyTextWizard", unsafe_allow_html=True)
    
    selected_tab = ui.tabs(options=['Home', 'Data Summary', 'Feedback Analysis', 'Agent Performance'], default_value='Home', key="kanaries")





    #old_df
    
    if selected_tab == 'Data Summary':
        section1.section1()
    elif selected_tab == 'Feedback Analysis':
        section2.section2()
    elif selected_tab == 'Agent Performance':
        section3.section3()
    elif selected_tab == 'Home':
        section4.section4()

if __name__ == "__main__":
    main()
