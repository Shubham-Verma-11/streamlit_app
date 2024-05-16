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
import os
from streamlit_lottie import st_lottie
from IPython.display import JSON
import json
from st_aggrid import AgGrid, GridOptionsBuilder, AgGridTheme 
import re

def section2():
    # Load data
    filtered_df = pd.read_csv('Tagged_data_category.csv')

    # Remove numbers before text in 'Major Category' column
    filtered_df['Major Category'] = filtered_df['Major Category'].apply(lambda x: re.sub(r'^\d+\.\s*', '', str(x)))

    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%d-%m-%Y')
    
    start_date, end_date = st.sidebar.columns(2)
    
    # Define sidebar layout
    start_date, end_date = st.sidebar.columns(2)
    
    # Create date inputs in sidebar columns
    with start_date:
        start_date_input = st.date_input("Start date", datetime.today())
    
    with end_date:
        end_date_input = st.date_input("End date", datetime.today())
    
    # Convert start_date and end_date to datetime64[ns]
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)
    
    num_agents = st.sidebar.radio('Select Top/Bottom Agents', ['All', 'Top 5'])
    
    tag_options = filtered_df['NPS Type'].unique().tolist()
    tag_options.insert(0, 'All')  # Add 'All' option
    tag_filter = st.sidebar.selectbox("NPS Type", options=tag_options, index=0)

    #st.write(start_date, end_date)
    
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    # Apply filters
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & 
                                  (filtered_df['Date'] <= end_date)]
    if tag_filter != 'All':
        filtered_df = filtered_df[filtered_df['NPS Type'] == tag_filter]
    
    #----------Main Issues Table------------------
    
    unique_topics_contribution = {}
    
    # Calculate % contribution for main issues
    total_non_empty_records = filtered_df['Major Category'].notnull().sum()
    for topic, count in filtered_df['Major Category'].value_counts().items():
        percentage_contribution = (count / total_non_empty_records) * 100
        unique_topics_contribution[topic] = {'% Contribution': round(percentage_contribution, 2)}
    
    # Create DataFrame from the unique_topics_contribution dictionary
    percentage_contributions_df = pd.DataFrame(unique_topics_contribution).T.reset_index().rename(columns={'index': 'Major Category'})

    # Create a dictionary mapping issues to their descriptions
    issue_description_dict = {
        'Positive Interactions and Politeness': 'Positive interactions, politeness, and friendliness towards customers',
        'Service Quality and Satisfaction': 'Overall satisfaction with service quality',
        'Product-Related Satisfaction': 'Satisfaction with product features and performance'}
    
    # Add a Description column based on the Major Category column
    percentage_contributions_df['Description'] = percentage_contributions_df['Major Category'].map(issue_description_dict)

    # Add % Contribution column
    percentage_contributions_df['% Contribution'] = percentage_contributions_df['Major Category'].apply(lambda x: unique_topics_contribution.get(x, {}).get('% Contribution', 0))
    
    #--------------------------------------------------
    # Apply filter based on selection
    if num_agents == 'Top 5':
        percentage_contributions_df = percentage_contributions_df.nlargest(5, '% Contribution')
        
    # Reset index to start from 1
    percentage_contributions_df.index = np.arange(1, len(percentage_contributions_df) + 1)
    
    st.subheader("Main Issues and their Contributions")
    # Add index to DataFrame
    percentage_contributions_df_with_index = percentage_contributions_df.reset_index(drop=True)
    # Filter out rows containing "NaN" as a string
    percentage_contributions_df_with_index = percentage_contributions_df_with_index[percentage_contributions_df_with_index['Major Category'] != 'nan']
    
    # Add Count column
    percentage_contributions_df_with_index['Count'] = percentage_contributions_df_with_index['Major Category'].apply(lambda x: filtered_df['Major Category'].value_counts().get(x, 0))

    # Reorder columns
    percentage_contributions_df_with_index = percentage_contributions_df_with_index[['Major Category', '% Contribution', 'Count', 'Description']]

    # Display total records
    st.text(f"Total records = {total_non_empty_records}")

    AgGrid(percentage_contributions_df_with_index, height=400, theme=AgGridTheme.STREAMLIT, fit_columns_on_grid_load=True, wrap_text=True)
        
    #------------Sub Issue Table------------
    
    # Create a list to store expanders
    expanders = []
    # Iterate over each unique value in the 'Major Category' column
    for i in range(0, len(filtered_df['Major Category'].unique()), 2):
       issue1 = filtered_df['Major Category'].unique()[i]
       #issue1_text = re.sub(r"^\d+\.", "", issue1).strip()
       # Filter the DataFrame for the current issue
       filtered_df_1 = filtered_df[filtered_df['Major Category'] == issue1]
       # Calculate the percentage contribution of each sub-issue within the issue group
       sub_issue_counts = filtered_df_1['Sub Category'].value_counts()
       total_count = sub_issue_counts.sum()
       percentage_contributions = (sub_issue_counts / total_count) * 100
       # Round the percentage contribution to 2 decimals
       percentage_contributions_rounded = percentage_contributions.round(2)
       # Create a DataFrame for the sub-issues and their percentage contributions
       sub_issue_df = pd.DataFrame({
           'Sub issues': percentage_contributions_rounded.index,
           '% contribution': percentage_contributions_rounded.values
       })
    
       # Check if the sub-issue DataFrame is empty
       if not sub_issue_df.empty:
           # Select top 3 sub-issues by percentage contribution
           top3_sub_issues = sub_issue_df.head(3)

            # Calculate subtotal for top 3 sub-issues
           subtotal = top3_sub_issues['% contribution'].sum()
           subtotal_row = pd.DataFrame({'Sub issues': ['Subtotal'], '% contribution': [subtotal]})
           top3_sub_issues = pd.concat([top3_sub_issues, subtotal_row], ignore_index=True)
           # Create expanders with two tables inside each expander
           if i + 1 < len(filtered_df['Major Category'].unique()):
               issue2 = filtered_df['Major Category'].unique()[i + 1]
               filtered_df2 = filtered_df[filtered_df['Major Category'] == issue2]
               sub_issue_counts2 = filtered_df2['Sub Category'].value_counts()
               total_count2 = sub_issue_counts2.sum()
               percentage_contributions2 = (sub_issue_counts2 / total_count2) * 100
               percentage_contributions_rounded2 = percentage_contributions2.round(2)
               sub_issue_df2 = pd.DataFrame({
                   'Sub issues': percentage_contributions_rounded2.index,
                   '% contribution': percentage_contributions_rounded2.values
               })
               # Select top 3 sub-issues by percentage contribution
               top3_sub_issues2 = sub_issue_df2.head(3)

               # Calculate subtotal for top 3 sub-issues
               subtotal2 = top3_sub_issues2['% contribution'].sum()
               subtotal_row2 = pd.DataFrame({'Sub issues': ['Subtotal'], '% contribution': [subtotal2]})
               top3_sub_issues2 = pd.concat([top3_sub_issues2, subtotal_row2], ignore_index=True)

               sub_issue_df2.index = range(1, len(sub_issue_df2)+1)
               #st.subheader("Sub Issues and their Contributions")
               with st.expander(f"{issue1} and {issue2}"):
                   col1, col2 = st.columns(2)
                   with col1:
                       st.write(issue1)
                       st.write(top3_sub_issues.to_html(index=False, justify='center'), unsafe_allow_html=True)
                   with col2:
                       st.write(issue2)
                       st.write(top3_sub_issues2.to_html(index=False, justify='center'), unsafe_allow_html=True)
           else:
               with st.expander(f"Sub issues for {issue1}"):
                   st.write(issue1)
                   st.write(top3_sub_issues.to_html(index=False, justify='center'), unsafe_allow_html=True)
