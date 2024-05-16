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

def custom_week_number(date):
    # Get the year and month of the date
    year = date.year
    month = date.month
    # Find the first day of the month
    first_day_of_month = pd.Timestamp(year, month, 1)
    # Calculate the week number based on the day of the month
    week_number = (date - first_day_of_month).days // 7 + 1
    return week_number

def section1():
    # Load data

    filtered_df = pd.read_csv('Tagged_data_category.csv')
    
    #filtered_df = pd.DataFrame()
    #if 'final_combined_data.xlsx' in os.listdir():
    #    filtered_df = pd.read_excel('final_combined_data.xlsx')
    #else:
    #    if 'data' not in st.session_state:
            #st.session_state.data = load_data() 
    #        filtered_df = old_df
        
    start_date, end_date = st.sidebar.columns(2)
    
     # Filters for Agents
    all_agents = ['ALL'] + list(filtered_df['Agent Name'].unique())
    default_agents = ['All'] if 'All' in all_agents else all_agents[:1]
    selected_agent = st.sidebar.multiselect('Agent', all_agents, default=default_agents)
   
    # selected_agent = st.multiselect("Filter by Agent Name:", ['ALL'] + list(data1['Agent Name_x'].unique()))
    # start_date_value = start_date.date_input("Start Date", filtered_df['Created At Date'].min())
    # end_date_value = end_date.date_input("End Date", filtered_df['Created At Date'].max())
  
    # # Convert Python date objects to Pandas Timestamp objects
    # start_date = pd.Timestamp(start_date_value)
    # end_date = pd.Timestamp(end_date_value)
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%d-%m-%Y')

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
      
    if selected_agent and 'ALL' not in selected_agent:
               filtered_df = filtered_df[filtered_df['Agent Name'].isin(selected_agent)]
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

    # Calculate overall average Voice AHT, average sNPS, average Resolution Score for each month
    filtered_df['Month'] = filtered_df['Date'].dt.to_period('M')
    filtered_df['Voice AHT'] = pd.to_numeric(filtered_df['Voice AHT'], errors='coerce')
    avg_aht  = filtered_df['Voice AHT'].mean()
    avg_aht= round(avg_aht,2)
    avg_sNPS  = filtered_df['Agent sNPS'].mean()
    avg_sNPS= round(avg_sNPS,2)

    # Calculate overall average Resolution Score for each month
    # overall_res_one_count = filtered_df[(filtered_df['Resolution Answer'] == 1) | (filtered_df['Resolution Answer'] == 0.5)].shape[0]
    # # total_tickets = filtered_df.shape[0]
    # total_ticket = len(filtered_df)
    # overall_resolution_score = (overall_res_one_count / total_ticket) * 100
    # overall_resolution_score = round(overall_resolution_score,2)


    # Calculate overall resolution score
    overall_res_one_count = filtered_df[(filtered_df['Resolution Answer'] == 1) | (filtered_df['Resolution Answer'] == 0.5)].shape[0]
    total_ticket = len(filtered_df)
    overall_resolution_score = 0
    if total_ticket > 0:
        overall_resolution_score = (overall_res_one_count / total_ticket) * 100
        overall_resolution_score = round(overall_resolution_score,2)

           
    # Calculate counts of each category
    tag_counts = filtered_df['NPS Type'].value_counts()

    avg_aht  = filtered_df['Voice AHT'].mean()
    avg_aht= round(avg_aht,2)
    avg_sNPS  = filtered_df['Agent sNPS'].mean()
    avg_sNPS= round(avg_sNPS,2)

    
    st.write('### *Overall score*')
    # Calculate percentages
    total_count = len(filtered_df)
    #st.write(total_count)
    if (total_count !=0):
        promoter_percent = (tag_counts.get('Promoter', 0) / total_count) * 100
        promoter_percent = round(promoter_percent,2)
        neutral_percent = (tag_counts.get('Neutral', 0) / total_count) * 100
        neutral_percent = round(neutral_percent,2)
        detractor_percent = (tag_counts.get('Detractor', 0) / total_count) * 100
        detractor_percent= round(detractor_percent,2)
         
        # Round percentages to 2 decimal places
        promoter_percent = round(promoter_percent, 2)
        neutral_percent = round(neutral_percent, 2)
        detractor_percent = round(detractor_percent, 2)
    else:
        st.write("No Records in Given Time")
        promoter_percent = 0
        neutral_percent = 0
        detractor_percent = 0

   # Create a pie chart
    # Sample data for the pie chart
    labels = ['Promoter', 'Neutral', 'Detractor']
    values = [promoter_percent, neutral_percent, detractor_percent]
    
    # Define colors for the pie chart slices
    colors = ['#FF555F', '#00AECF', '#073161']

    # Create a pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, showlegend=True, marker=dict(colors=colors))])
    
    # Set layout options for the pie chart
    fig_pie.update_layout(
        title='',  # Set the title of the chart
        title_x=0.5,  # Position the title in the middle horizontally
        title_y=0.95,  # Adjust the vertical position of the title
        height=450,  # Set the height of the chart
        width=400, margin=dict(t=0) # Set the width of the chart
    )

# Create the main expander for WOW Analysis
    wow_expander = st.expander("Overall Score", expanded=False)
  
    # Add the plots to the WOW Analysis expander in a single column
    with wow_expander:

        # Define custom CSS style for the content inside the expander
        expander_style = """
            <style>
            .streamlit-expanderContent {
                max-width:100px;
                margin-bottom: 0px;
            }
            </style>
        """
        st.markdown(expander_style, unsafe_allow_html=True)

    # col1, col2, col3= st.columns(3)
        cols = st.columns(2)

        # Define the CSS style for the metric cards
        card_style = """
            background: linear-gradient(90deg, #ffffff, #e0e0e0);
            height: 100px;
            text-align: center;
            width: 100%;
            border-radius: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
            border-collapse: collapse;
            margin-bottom: 20px;
        """

        # Apply style to the metric cards
        with cols[0]:
            st.markdown(
                f"""
                <div style="{card_style}">
                    <h4>NPS</h4>
                    <p style="color:red; font-weight:bold; text-align:center">{avg_sNPS}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="{card_style}">
                    <h4>AHT</h4>
                    <p style="color:red; font-weight:bold; text-align:center">{avg_aht}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="{card_style}">
                    <h4>Resolution Score</h4>
                    <p style="color:red; font-weight:bold; text-align:center">{overall_resolution_score}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with cols[1]:
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown(
                """
                <style>
                .stPlot {
                    border: 2px solid black !important;
                    padding: 5px;
                    margin: 5px 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
    # Adding avg_aht, avg_sNPS, overall_resolution_score in same line as box template
    # Display the title
    st.write('### *Week on Week Analysis*')
    
#     # Create the plots
    # Weekly AHT
    # Create a bar chart using Plotly
    # Apply the custom_week_number function to calculate week number
    filtered_df['Week_Number'] = filtered_df['Date'].apply(custom_week_number)
    # filtered_df['Week_Number'] = filtered_df['Created At Date'].dt.isocalendar().week
    weekly_avg_aht = round((filtered_df.groupby('Week_Number')['Voice AHT'].mean().reset_index()),2)
    # Convert Week_Number to integer
    #weekly_avg_aht['Week_Number'] = weekly_avg_aht['Week_Number'].astype(int)
    # Sort the DataFrame by Week_Number
    weekly_avg_aht = weekly_avg_aht.sort_values('Week_Number')
    
    # fig_aht = px.line(weekly_avg_aht, x='Week_Number', y='Voice AHT', labels={'Voice AHT': 'Average Voice AHT'})
    fig_aht = px.line(weekly_avg_aht, x='Voice AHT', y='Week_Number', labels={'Voice AHT': 'Average Voice AHT'})


    fig_aht = go.Figure()

    fig_aht.add_trace(go.Bar(
        x=weekly_avg_aht['Week_Number'].astype(str),
        y=weekly_avg_aht['Voice AHT'],
        # x=weekly_avg_aht['Voice AHT'],
        # y=weekly_avg_aht['Week_Number'].astype(str),  # Convert week number to string to treat it as categorical,
        # orientation='h',
        marker_color='#073161',  # Set the color of the bars
        text=weekly_avg_aht['Voice AHT'],  # Add text labels on bars
        textposition='auto',
        showlegend= False # Automatically position the text labels
    ))

               
    # Update the layout of the bar chart
    fig_aht.update_layout(
        title=dict(
        text='Voice AHT by Week',
        x=0.3,  # Set the title's horizontal position to the center
        y=0.9  # Set the title's vertical position closer to the top
        ),
        xaxis_title='Week Number',
        yaxis_title='Average Voice AHT',
        # xaxis_title='Average Voice AHT',
        # yaxis_title='Week Number',
        height=400,
        width=500,
        # margin=dict(l=30, r=50, t=50, b=50),  # Adjust margins
        plot_bgcolor="white",  # Background color
        margin=dict(l=30, r=50, t=50, b=50)
        
    )


    # Weekly Resolution
    weekly_resolution = round(filtered_df.groupby('Week_Number')['Resolution Answer'].mean().reset_index(),2)
    fig_resolution = px.line(weekly_resolution, y='Week_Number', x='Resolution Answer', labels={'Resolution Answer': 'Resolution Percentage'})

    fig_resolution = go.Figure()

    fig_resolution.add_trace(go.Bar(
        x=weekly_resolution['Week_Number'],
        y=weekly_resolution['Resolution Answer'],
        # orientation='v', 
        marker_color='#00AECF',  # Set the color of the bars
        text=weekly_resolution['Resolution Answer'],  # Add text labels on bars
        textposition='auto', 
        showlegend=False,
        # width=0.8  # Adjust the width of the bars
        # showlegend=False # Automatically position the text labels
    ))
    
    # Update the layout of the bar chart
    fig_resolution.update_layout(
        title=dict(
        text='Resolution Score by Week',
        x=0.2,  # Set the title's horizontal position to the center
        y=0.9  # Set the title's vertical position closer to the top
    ),
        xaxis_title='Week Number',
         yaxis_title='Resolution Score',
        height=400,
        width=500,
        margin=dict(l=30, r=50, t=50, b=50), 
        # Adjust margins
        plot_bgcolor="white",  # Background color
    )
   
    # Weekly NPS
    weekly_avg_sNPS = round(filtered_df.groupby('Week_Number')['Agent sNPS'].mean().reset_index(),2)
    # fig_nps = px.line(weekly_avg_sNPS, x='Week_Number', y='Agent sNPS', labels={'Agent sNPS': 'Average Agent sNPS'})
    fig_nps = px.line(weekly_avg_sNPS, x='Agent sNPS', y='Week_Number', labels={'Agent sNPS': 'Average Agent sNPS'})


    fig_nps = go.Figure()

    fig_nps.add_trace(go.Bar(
        x=weekly_avg_sNPS['Week_Number'],
        y=weekly_avg_sNPS['Agent sNPS'],
        marker_color= '#FF555F',  # Set the color of the bars
        text=weekly_avg_sNPS['Agent sNPS'],  # Add text labels on bars
        textposition='auto',  # Automatically position the text labels
        textfont_color='white',
        showlegend=False
    ))
    
    # Update the layout of the bar chart
    fig_nps.update_layout(
        title=dict(
        text='NPS by Week',
        x=0.3,  # Set the title's horizontal position to the center
        y=0.9  # Set the title's vertical position closer to the top
    ),
        yaxis_title= 'NPS',
        xaxis_title= 'Week Number',
        height=400,
        width=500,
        margin=dict(l=30, r=50, t=50, b=50),  # Adjust margins
        plot_bgcolor="white",  # Background color
    )

        # Create the main expander for WOW Analysis
    wow_expander = st.expander("Week Over Week Analysis")
    
    # Add the plots to the WOW Analysis expander in a single column
    with wow_expander:
        # Create a layout with three columns for the graphs
        col1, col2, col3 = st.columns(3)
    
    # Add the plots to each column with borders and margin
        with col1:
            # st.write('### Weekly AHT')
            st.plotly_chart(fig_nps, use_container_width=True)  # Use_container_width ensures the chart fits in the container
            st.markdown(
                """
                <style>
                .stPlot {
                    border: 2px solid black !important;
                    padding: 10px;
                    margin: 10px 0;
                    
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
        with col2:
            # st.write('### Resolution Score')
            st.plotly_chart(fig_resolution, use_container_width=True)
            st.markdown(
                """
                <style>
                .stPlot {
                    border: 2px solid black !important;
                    padding: 10px;
                    margin: 10px 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            # st.write('### sNPS')
            st.plotly_chart(fig_aht, use_container_width=True)
            st.markdown(
                """
                <style>
                .stPlot {
                    border: 2px solid black !important;
                    padding: 10px;
                    margin: 10px 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )   
    
    # Sort agents based on sNPS scores
    new_avg_snps = filtered_df.groupby('Agent Name')['Agent sNPS'].mean().reset_index(name='sNPS')
    new_avg_snps = new_avg_snps.round(2)
    new_avg_aht = filtered_df.groupby('Agent Name')['Voice AHT'].mean().reset_index(name='AHT')
    new_avg_aht = new_avg_aht.round(2)
    
    # Filter rows with Resolution Answer equal to 1 or 0.5
    res_one_count = filtered_df[(filtered_df['Resolution Answer'] == 1) | (filtered_df['Resolution Answer'] == 0.5)].groupby('Agent Name').size().reset_index(name='ResOneCount')
    
    total_tickets = filtered_df.groupby('Agent Name').size().reset_index(name='TotalTickets')

    # Convert columns to numeric types if needed
    res_one_count['ResOneCount'] = res_one_count['ResOneCount'].astype(int)
    total_tickets['TotalTickets'] = total_tickets['TotalTickets'].astype(int)
    
    # Calculate the resolution scores
    resolution_scores = (res_one_count['ResOneCount'] / total_tickets['TotalTickets'])
    resolution_score = round(resolution_scores, 2)
   
    # Merge sNPS, AHT, and Resolution Score dataframes
    merged_df = pd.merge(new_avg_snps.round(2), new_avg_aht.round(2), on='Agent Name')
    merged_df['Resolution Score'] = resolution_score
   
    # Reset index after merging
    merged_df = merged_df.reset_index(drop=True)

    # Sort merged dataframe based on sNPS scores (for demonstration)
    sorted_agents = merged_df.sort_values(by='sNPS', ascending=False)
    
    # Display agents with sNPS, AHT, and Resolution Score
    st.write('### *Agents with sNPS, AHT, and Resolution Score*')
    
    # # Display top 5 and bottom 5 agents
    top_agents = sorted_agents.head(5).reset_index(drop=True)
    bottom_agents = sorted_agents.tail(5).reset_index(drop=True)
    
    # # Reset index to start from 1
    top_agents.index = np.arange(1, len(top_agents) + 1)
    bottom_agents.index = np.arange(1, len(bottom_agents) + 1)

    # Set Pandas options to display the entire contents of the Sub_Issues_Percentage column
    pd.set_option('display.max_colwidth', None)

    wow_expander = st.expander("Agents", expanded=False)
    
    with wow_expander:
        st.write('*Top Agents*')
        #st.write(top_agents)
        AgGrid(top_agents, height=210, theme=AgGridTheme.STREAMLIT, fit_columns_on_grid_load=True,key = "top")
        st.write('*Bottom Agents*')
        AgGrid(bottom_agents, height=210, theme=AgGridTheme.STREAMLIT, fit_columns_on_grid_load=True,key = "bottom")
