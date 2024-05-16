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

from bs4 import BeautifulSoup
import re
import unicodedata
from llama_cpp import Llama
from transformers import AutoTokenizer
from tqdm import tqdm



def generate_bar_chart_all_agents(weekly_data, y_column, title, color):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weekly_data['Week_Number'],
        y=weekly_data[y_column],
        marker_color=color,  # Set the color of the bars
        text=weekly_data[y_column],  # Add text labels on bars
        textposition='auto',  # Automatically position the text labels
        width=0.75
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_y=1,
        xaxis_title='Week Number',
        yaxis_title=title,
        height=300,
        width=999,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
        xaxis=dict(
        tickmode='array',  # Set the tick mode
        tickvals=weekly_data['Week_Number'].unique(),  # Set the tick values to the unique weeks
        tickformat='d'  # Set the tick format to display integers
        )
    )

    return fig

def generate_stacked_bar_chart_selected_agents(weekly_data, y_column, title, colors, selected_agents):
    fig = go.Figure()
    colors = ['#FF555F', '#073161', '#00AECF', '#FFD700', '#008000']

    for i, agent in enumerate(weekly_data['Agent Name'].unique()):
        agent_data = weekly_data[weekly_data['Agent Name'] == agent]
        fig.add_trace(go.Bar(
            x=agent_data['Week_Number'],
            y=agent_data[y_column],
#             orientation = 'h',
            name=agent if selected_agents and 'All' not in selected_agents else None,
            showlegend=False if selected_agents and 'All' in selected_agents else True,
            marker_color=colors[i] if selected_agents and 'All' not in selected_agents else colors[i % len(colors)],
#             text=agent_data[y_column],
            textposition='auto',
        ))

    fig.update_layout(
        barmode='stack',  # Set barmode to 'stack' for stacked bar chart
        title=title,
        title_x=0.5,
        title_y=1,
        xaxis_title='Week Number',
        yaxis_title=title,
        height=400,
        width=999,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=20, r=20, t=30, b=30),
        plot_bgcolor="white",
        xaxis=dict(
        tickmode='array',  # Set the tick mode
        tickvals=weekly_data['Week_Number'].unique(),  # Set the tick values to the unique weeks
        tickformat='d'  # Set the tick format to display integers
        )
    )
    
    return fig


def wow_aht_snps_resolution_agent_training(data1, selected_agents, selected_topics, date_range):
    filtered_df = data1.copy()

    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= np.datetime64(start_date)) & (filtered_df['Date'] <= np.datetime64(end_date))]

    if selected_agents and 'All' not in selected_agents:
        filtered_df = filtered_df[filtered_df['Agent Name'].isin(selected_agents)]
        
    if selected_topics and 'All' not in selected_topics:
        filtered_df = filtered_df[filtered_df['Major Category'].isin(selected_topics)]

    if selected_agents and 'All' in selected_agents:
        # Weekly NPS
        filtered_df['Agent sNPS'] = pd.to_numeric(filtered_df['Agent sNPS'], errors='coerce')
        weekly_avg_sNPS = round(filtered_df.groupby(['Week_Number'])['Agent sNPS'].mean().reset_index(), 2)

        fig_nps = generate_bar_chart_all_agents(weekly_avg_sNPS, 'Agent sNPS', 'NPS', '#FF555F')

        # Weekly AHT
        filtered_df['Voice AHT'] = pd.to_numeric(filtered_df['Voice AHT'], errors='coerce')
        weekly_avg_aht = round(filtered_df.groupby(['Week_Number'])['Voice AHT'].mean().reset_index(), 2)

        fig_aht = generate_bar_chart_all_agents(weekly_avg_aht, 'Voice AHT', 'AHT', '#073161')

        # Weekly Resolution
        filtered_df['Resolution Answer'] = pd.to_numeric(filtered_df['Resolution Answer'], errors='coerce')
        weekly_resolution = round(filtered_df.groupby(['Week_Number'])['Resolution Answer'].mean().reset_index(), 2)

        fig_resolution = generate_bar_chart_all_agents(weekly_resolution, 'Resolution Answer', 'Resolution Score', '#00AECF')

        return fig_aht, fig_nps, fig_resolution

    else:
        # Weekly NPS
        filtered_df['Agent sNPS'] = pd.to_numeric(filtered_df['Agent sNPS'], errors='coerce')
        weekly_avg_sNPS = round(filtered_df.groupby(['Week_Number', 'Agent Name'])['Agent sNPS'].mean().reset_index(), 2)

        fig_nps = generate_stacked_bar_chart_selected_agents(weekly_avg_sNPS, 'Agent sNPS', 'NPS', '#FF555F', selected_agents)

        # Weekly AHT
        filtered_df['Voice AHT'] = pd.to_numeric(filtered_df['Voice AHT'], errors='coerce')
        weekly_avg_aht = round(filtered_df.groupby(['Week_Number', 'Agent Name'])['Voice AHT'].mean().reset_index(), 2)

        fig_aht = generate_stacked_bar_chart_selected_agents(weekly_avg_aht, 'Voice AHT', 'AHT', '#073161', selected_agents)

        # Weekly Resolution
        filtered_df['Resolution Answer'] = pd.to_numeric(filtered_df['Resolution Answer'], errors='coerce')
        weekly_resolution = round(filtered_df.groupby(['Week_Number', 'Agent Name'])['Resolution Answer'].mean().reset_index(), 2)

        fig_resolution = generate_stacked_bar_chart_selected_agents(weekly_resolution, 'Resolution Answer', 'Resolution Score', '#00AECF', selected_agents)
      
        return fig_aht, fig_nps, fig_resolution
        
        
        
# FUNCTION FOR TOPIC WISE TRENDS

def generate_bar_chart_all_topics(weekly_data, y_column, title, color):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weekly_data['Week_Number'],
        y=weekly_data[y_column],
        marker_color=color,  # Set the color of the bars
        text=weekly_data[y_column],  # Add text labels on bars
        textposition='auto',  # Automatically position the text labels
        width=0.75
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_y=1,
        xaxis_title='Week Number',
        yaxis_title=title,
        height=300,
        width=999,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
        xaxis=dict(
        tickmode='array',  # Set the tick mode
        tickvals=weekly_data['Week_Number'].unique(),  # Set the tick values to the unique weeks
        tickformat='d'  # Set the tick format to display integers
        )
    )

    return fig




def generate_stacked_bar_chart_selected_topics(weekly_data, y_column, title, colors, selected_topics):
    fig = go.Figure()
    colors = ['#FF555F', '#073161', '#00AECF', '#FFD700', '#008000']

    for i, topic in enumerate(weekly_data['Major Category'].unique()):
        agent_data = weekly_data[weekly_data['Major Category'] == topic]
        fig.add_trace(go.Bar(
            x=agent_data['Week_Number'],
            y=agent_data[y_column],
#             orientation = 'h',
            name=topic if selected_topics and 'All' not in selected_topics else None,
            showlegend = False if selected_topics and 'All' in selected_topics else True,
            marker_color=colors[i] if selected_topics and 'All' not in selected_topics else colors[i % len(colors)],
#             text=agent_data[y_column],
            textposition='auto',
        ))

    fig.update_layout(
        barmode='stack',  # Set barmode to 'stack' for stacked bar chart
        title=title,
        title_x=0.5,
        title_y=1,
        xaxis_title='Week Number',
        yaxis_title=title,
        height=400,
        width=999,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=20, r=20, t=30, b=30),
        plot_bgcolor="white",
        xaxis=dict(
        tickmode='array',  # Set the tick mode
        tickvals=weekly_data['Week_Number'].unique(),  # Set the tick values to the unique weeks
        tickformat='d'  # Set the tick format to display integers
        )
    )
    
    return fig



def wow_aht_snps_resolution_by_topics(data1, selected_agents, selected_topics, date_range):
    filtered_df = data1.copy()

    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= np.datetime64(start_date)) & (filtered_df['Date'] <= np.datetime64(end_date))]

    if selected_agents and 'All' not in selected_agents:
        filtered_df = filtered_df[filtered_df['Agent Name'].isin(selected_agents)]
        
    if selected_topics and 'All' not in selected_topics:
        filtered_df = filtered_df[filtered_df['Major Category'].isin(selected_topics)]

    if selected_topics and 'All' in selected_topics:
        # Weekly NPS
        filtered_df['Agent sNPS'] = pd.to_numeric(filtered_df['Agent sNPS'], errors='coerce')
        weekly_avg_sNPS = round(filtered_df.groupby(['Week_Number'])['Agent sNPS'].mean().reset_index(), 2)

        fig_nps = generate_bar_chart_all_topics(weekly_avg_sNPS, 'Agent sNPS', 'NPS', '#FF555F')

        # Weekly AHT
        filtered_df['Voice AHT'] = pd.to_numeric(filtered_df['Voice AHT'], errors='coerce')
        weekly_avg_aht = round(filtered_df.groupby(['Week_Number'])['Voice AHT'].mean().reset_index(), 2)

        fig_aht = generate_bar_chart_all_topics(weekly_avg_aht, 'Voice AHT', 'AHT', '#073161')

        # Weekly Resolution
        filtered_df['Resolution Answer'] = pd.to_numeric(filtered_df['Resolution Answer'], errors='coerce')
        weekly_resolution = round(filtered_df.groupby(['Week_Number'])['Resolution Answer'].mean().reset_index(), 2)

        fig_resolution = generate_bar_chart_all_topics(weekly_resolution, 'Resolution Answer', 'Resolution Score', '#00AECF')

        return fig_nps, fig_aht , fig_resolution

    else:
        # Weekly NPS
        filtered_df['Agent sNPS'] = pd.to_numeric(filtered_df['Agent sNPS'], errors='coerce')
        weekly_avg_sNPS = round(filtered_df.groupby(['Week_Number', 'Main_issue'])['Agent sNPS'].mean().reset_index(), 2)

        fig_nps = generate_stacked_bar_chart_selected_topics(weekly_avg_sNPS, 'Agent sNPS', 'NPS', '#FF555F', selected_topics)

        # Weekly AHT
        filtered_df['Voice AHT'] = pd.to_numeric(filtered_df['Voice AHT'], errors='coerce')
        weekly_avg_aht = round(filtered_df.groupby(['Week_Number', 'Major Category'])['Voice AHT'].mean().reset_index(), 2)

        fig_aht = generate_stacked_bar_chart_selected_topics(weekly_avg_aht, 'Voice AHT', 'AHT',  '#073161', selected_topics)

        # Weekly Resolution
        filtered_df['Resolution Answer'] = pd.to_numeric(filtered_df['Resolution Answer'], errors='coerce')
        weekly_resolution = round(filtered_df.groupby(['Week_Number', 'Major Category'])['Resolution Answer'].mean().reset_index(), 2)

        fig_resolution = generate_stacked_bar_chart_selected_topics(weekly_resolution, 'Resolution Answer', 'Resolution Score','#00AECF',selected_topics)
      
        return fig_nps, fig_aht , fig_resolution


# Function to assign colors to topics
def get_topic_color(idx):
    # Define a list of colors
    colors = ['#FF555F', '#073161', '#00AECF']
    # Return color based on index, with cycling for additional topics
    return colors[idx % len(colors)]



def wow_p_n_d(data1, date_range, selected_agents, selected_topics):
    
    filtered_df = data1.copy()
      
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= np.datetime64(start_date)) & (filtered_df['Date'] <= np.datetime64(end_date))]
     
    if selected_agents and 'All' not in selected_agents:
        filtered_df = filtered_df[filtered_df['Agent Name'].isin(selected_agents)]
        
    if selected_topics and 'All' not in selected_topics:
        filtered_df = filtered_df[filtered_df['Major Category'].isin(selected_topics)]
    
    # Calculate the top topics available
    top_topics = filtered_df['Major Category'].value_counts().index[:3]
    
    figs = {}

    for i, topic in enumerate(top_topics, 1):
        topic_df = filtered_df[filtered_df['Major Category'] == topic]
        
        # Weekly Promoter count
        weekly_count_p = topic_df['P'].sum()
        # Weekly Neutral count
        weekly_count_n = topic_df['N'].sum()
        # Weekly Detractor count
        weekly_count_d = topic_df['D'].sum()
        
        # Create stacked bar chart for each topic
        fig = go.Figure()

        weeks = topic_df['Week_Number'].unique()
        fig.add_trace(go.Bar(
            x=weeks,
            y=topic_df.groupby('Week_Number')['P'].sum(),
            name='Promoter',
            marker_color='#FF555F'
        ))
        fig.add_trace(go.Bar(
            x=weeks,
            y=topic_df.groupby('Week_Number')['N'].sum(),
            name='Neutral',
            marker_color='#073161',
            base=topic_df.groupby('Week_Number')['P'].sum().tolist()
        ))
        fig.add_trace(go.Bar(
            x=weeks,
            y=topic_df.groupby('Week_Number')['D'].sum(),
            name='Detractor',
            marker_color='#00AECF',
            base=(topic_df.groupby('Week_Number')['P'].sum() + topic_df.groupby('Week_Number')['N'].sum()).tolist()
        ))

        fig.update_layout(
            barmode='stack',  # Set barmode to 'stack' for stacked bar chart
            title=topic,  # Add the title
            title_y=1,  # Adjust the vertical position of the title
            title_x=0.5,  # Position the title in the middle horizontally
            xaxis_title='Week Number',  # Add x-axis title
            yaxis_title='Count',  # Add y-axis title
            height=400, 
            width=999, 
            margin=dict(l=20, r=20, t=30, b=30),  # Adjust margins for the border
            plot_bgcolor ="white",  # Background color
            legend=dict(
                orientation="v",  # Set legend orientation to vertical
                yanchor="top",  # Anchor legend to the top
                xanchor="right",  # Anchor legend to the right side
                y=1,  # Adjust y position of the legend
                x=1.1,  # Adjust x position of the legend
            ),
            xaxis=dict(
            tickmode='array',  # Set the tick mode
            tickvals=weeks,  # Set the tick values to the unique weeks
            tickformat='d'  # Set the tick format to display integers
        )
)
        
        figs[f'fig_{i}'] = fig
    
    return figs


# #function to split main topic and sub topic:
def split_topics(agent_training_df):
    
    # Filter relevant columns
    agent_training_df = agent_training_df[['Agent Name', 'Major Category']]

    # Group by 'Agent Name_x' and count the occurrences of each 'Main topic'
    agent_topic_counts = agent_training_df.groupby('Agent Name')['Major Category'].value_counts().reset_index(name='topic_count')

    # Sort in descending order based on the count of 'Main topic'
    agent_topic_counts.sort_values(by=['Agent Name', 'topic_count'], ascending=[True, False], inplace=True)

    # Initialize an empty DataFrame to store the results
    top_topics_per_agent = pd.DataFrame(columns=['Agent Name', 'Major Category'])

    # Iterate through unique agents
    for agent in agent_topic_counts['Agent Name'].unique():
        # Filter data for the current agent
        agent_data = agent_topic_counts[agent_topic_counts['Agent Name'] == agent]

        # Calculate the total count of topics for the current agent
        total_count = agent_data['topic_count'].sum()

        # Calculate the percentage for each topic and append it to the result DataFrame
        agent_data['Percentage'] = round((agent_data['topic_count'] / total_count) * 100,2)
        agent_data['Major Category'] = agent_data['Major Category'] + ' : ' + agent_data['Percentage'].astype(str) + '%'

        # Take the top topics for the current agent
        top_topics = agent_data.head(min(3, len(agent_data)))

        # Append the top topics to the result DataFrame
        top_topics_per_agent = pd.concat([top_topics_per_agent, top_topics], ignore_index=True)

    # Pivot the table to create new columns based on the number of top topics for each agent
    top_topics_per_agent_pivot = top_topics_per_agent.pivot_table(index='Agent Name', columns=top_topics_per_agent.groupby('Agent Name').cumcount() + 1, values='Major Category', aggfunc='first')

    # Rename columns
    top_topics_per_agent_pivot.columns = [f"Issue area{i}" for i in range(1, top_topics_per_agent_pivot.shape[1] + 1)]

    # Reset index
    top_topics_per_agent_pivot.reset_index(inplace=True)
    #top_topics_per_agent_pivot.rename(columns={'Agent Name_x': 'Agent Name'}, inplace=True)

    # Reorder columns
    top_topics_per_agent_pivot = top_topics_per_agent_pivot[['Agent Name'] + list(top_topics_per_agent_pivot.columns[1:])]
    
    return top_topics_per_agent_pivot



# Function to fetch Agent Training data based on selected agent
def fetch_agent_training_data(data1,date_range, selected_agents):
    agent_training_df = data1.copy()
    
    if date_range:
        start_date, end_date = date_range
        agent_training_df = agent_training_df[(agent_training_df['Date'] >= np.datetime64(start_date)) & (agent_training_df['Date'] <= np.datetime64(end_date))]
        
    if selected_agents and 'All' not in selected_agents:
        agent_training_df = agent_training_df[agent_training_df['Agent Name'].isin(selected_agents)]
#     elif not selected_agents or 'All' in selected_agents:
#         # No agents selected or 'All' is selected, return all data
#         pass
    
    # Check if filtered dataframe is empty
    agent_training_df = agent_training_df[['Agent Name', 'Major Category']]
    agent_training_df.dropna(inplace=True)
    if not agent_training_df.empty:
        agent_training_df = split_topics(agent_training_df)
    
    return agent_training_df


# Function to filter data
def filter_data(data1, date_range, agents):
    df_filtered = data1.copy()
    
    if date_range:
        start_date, end_date = date_range
        df_filtered = df_filtered[(df_filtered['Date'] >= np.datetime64(start_date)) & (df_filtered['Date'] <= np.datetime64(end_date))]
        
        # Convert columns to numeric data types
        
        #numeric_columns = ['Agent sNPS', 'Voice AHT', 'Resolution Answer', 'P', 'N', 'D']

        required_columns = ['Voice AHT', 'Resolution Answer']
        #old_df = old_df[['Ticket ID',sentences,agent,NPS_score,Date,'Voice AHT','Resolution Answer']]
    
        if set(required_columns).issubset(['Agent sNPS', 'Voice AHT', 'Resolution Answer', 'P', 'N', 'D']):
            numeric_columns = ['Agent sNPS', 'Voice AHT', 'Resolution Answer', 'P', 'N', 'D']
        #data1.columns = ['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer','Major Category','Sub Category','P','N','D', 'Week_Number']
        else:
            numeric_columns = ['Agent sNPS', 'P', 'N', 'D']

  
       
        df_filtered[numeric_columns] = df_filtered[numeric_columns].apply(pd.to_numeric, errors='coerce')


        if set(required_columns).issubset(['Agent sNPS', 'Voice AHT', 'Resolution Answer', 'P', 'N', 'D']):
            numeric_columns = ['Agent sNPS', 'Voice AHT', 'Resolution Answer', 'P', 'N', 'D']
        #data1.columns = ['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer','Major Category','Sub Category','P','N','D', 'Week_Number']
        else:
            numeric_columns = ['Agent sNPS', 'P', 'N', 'D']

        
        
        df_filtered = df_filtered.groupby('Agent Name').agg({'Agent sNPS':'mean','Ticket ID':'count','Voice AHT':'mean','Resolution Answer':'mean','D':'sum','P':'sum','N':'sum'}).reset_index()


        df_filtered.columns = ['Agent Name','Average sNPS','Ticket count','Average AHT','Resolution%','Promoter%','Neutral%','Detractor%']
        df_filtered['Resolution%'] = df_filtered['Resolution%']*100
        # Calculate percentages
        df_filtered['Promoter%'] = df_filtered['Promoter%'] / df_filtered['Ticket count'] * 100
        df_filtered['Neutral%'] = df_filtered['Neutral%'] / df_filtered['Ticket count'] * 100
        df_filtered['Detractor%'] = df_filtered['Detractor%'] / df_filtered['Ticket count'] * 100
        df_filtered = df_filtered.round(2)
        
    if agents and 'All' not in agents:
        df_filtered = df_filtered[df_filtered['Agent Name'].isin(agents)]

    return df_filtered


def select_top3_topics(data1,date_range,selected_agents):
    filtered_df = data1.copy()

    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= np.datetime64(start_date)) & (filtered_df['Date'] <= np.datetime64(end_date))]

    if selected_agents and 'All' not in selected_agents:
        filtered_df = filtered_df[filtered_df['Agent Name'].isin(selected_agents)]

    top3topics = list(filtered_df.groupby('Major Category').agg({'Ticket ID':'count'}).reset_index().sort_values(by='Ticket ID', ascending=False)['Major Category'].head(3))

    return top3topics


def select_all_topics(data1,date_range,selected_agents):
    # Update available topics based on the selected agent
    available_topics = []
    if date_range and selected_agents and 'All' not in selected_agents:
        filtered_df = data1.copy()
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= np.datetime64(start_date)) & (filtered_df['Date'] <= np.datetime64(end_date))]
        available_topics = filtered_df[filtered_df['Agent Name'].isin(selected_agents)]['Major Category'].unique().tolist()
        available_topics = [value for value in available_topics if not pd.isna(value)]
    else:
        filtered_df = data1.copy()
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= np.datetime64(start_date)) & (filtered_df['Date'] <= np.datetime64(end_date))]
        available_topics = data1['Major Category'].unique().tolist()
        available_topics = [value for value in available_topics if not pd.isna(value)]

    available_topics.insert(0, 'All')
    return available_topics


def fetch_actions(row):
    agent_action_df = pd.read_excel('agent_training_actions.xlsx')
    actions = []
    for col in ['Issue area1', 'Issue area2']:
        issue = row[col]
        issues_list = [i.strip() for i in str(issue).split(',')]  # Splitting by comma and stripping spaces
        
        for single_issue in issues_list:
            if single_issue in agent_action_df['Issues'].values:
                actions.append(agent_action_df[agent_action_df['Issues'] == single_issue]['Actions'].iloc[0])

    if not actions:
        actions.append(agent_action_df[agent_action_df['Issues'] == 'common issues']['Actions'].iloc[0])

    return ', '.join(actions)

model_path = r'Phi-3-mini-4k-instruct-q4.gguf'

def preprocess_recomendation(text):
    lines = text.split('\n')
    processed_text = ""
    for line in lines:
        if line.strip():
            match = re.match(r'^(.*?)\s*<|end|><|assistant|>', line)
            if match:
                processed_text += " - " + match.group(1).strip() + "\n"
            else:
                processed_text += " - " + line.strip() + "\n"
    return processed_text

def get_recomendations(old_df,selected_agents,agent_training_df):
    llm = Llama(
    model_path=model_path,
    n_ctx=2000,  # Context length to use
    n_threads=4  # Number of CPU threads to use
#     # n_gpu_layers=0  # Number of model layers to offload to GPU
    )
    generation_kwargs = {
         "max_tokens": 100,
         "stop": ["</s>"],
         "echo": False,  # Echo the prompt in the output
         "top_k": 1  # Greedy decoding
    }


    df_recommendations = pd.read_excel('MArch_Demo_Issue_Recommendation_file_new.xlsx')

    generate_summary = pd.DataFrame()
    # Extract the list of recommendations
    recommendations_list = df_recommendations["Recommendations"].tolist()

    for i, agent in enumerate(selected_agents):
        agent_total = len(selected_agents)
        st.write("Agent ",i+1,"/",agent_total," Started")
        # agent_name = row["Agent Name"]
        customer_feedback = old_df[old_df['Agent Name']==agent]['Sentences'].tolist()
        # context = customer_feedback
        context = ". ".join(customer_feedback)
        #st.write(context)

        # Generate a summary of the feedback
        question = f"Summarize the given input in maximum 2 lines"
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        res = llm(prompt, **generation_kwargs)  # Res is a dictionary
        summary = res['choices'][0]['text'].split('\n')[0]
        #st.write("Summary ",summary)
        #generate_summary.loc[i,'Agent Name']= agent
        #generate_summary.loc[i,'Summary'] = summary

        # Use the summary as context for generating recommendations
        context = summary

        # Generate recommendation using the model
        # question = f"Given the input, what would be the appropriate recommendation to customer service by taking reference from the below list: {', '.join(recommendations_list)}"
        question = f"Given the input, list in maximum two bullet points, what would be the appropriate recommendation to customer service by taking reference from the below list: {', '.join(recommendations_list)}"
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        res = llm(prompt, **generation_kwargs)
        recommendation = res['choices'][0]['text']

        
        reco = preprocess_recomendation(recommendation)

        #st.write(recommendation)
        #st.write(reco)
        agent_training_df.loc[agent_training_df['Agent Name']==agent, 'Recommendation'] = reco
        #st.write(agent_training_df)
    agent_training_df.to_csv('agent_training_with_recomendation.csv')
    return agent_training_df
        #print(f"Agent: {agent}\nSummary: {summary}\nRecommendation: {recommendation}")

def agent_issue_action_mapping(agent_training_df,old_df):
    agent_training = agent_training_df.copy()

    

    
    
    if 'Issue area1' in agent_training.columns:
        agent_training['Issue area1'] = agent_training['Issue area1'].str.split(':').str[0].str.strip()
    
    if 'Issue area2' in agent_training.columns:
        agent_training['Issue area2'] = agent_training['Issue area2'].str.split(':').str[0].str.strip()
    
    if 'Issue area3' in agent_training.columns:
        agent_training['Issue area3'] = agent_training['Issue area3'].str.split(':').str[0].str.strip()
    
    # Create 'Actions' column by iterating over each row
    #agent_training_df['Actions'] = agent_training.apply(fetch_actions, axis=1)

    
    
    return agent_training


def custom_week_number(date):
    # Get the year and month of the date
    year = date.year
    month = date.month
    # Find the first day of the month
    first_day_of_month = pd.Timestamp(year, month, 1)
    # Calculate the week number based on the day of the month
    week_number = (date - first_day_of_month).days // 7 + 1
    return week_number



def section3():

    # Retrieve data from session state
    #if 'final_combined_data.xlsx' in os.listdir():
    #    data = pd.read_excel('final_combined_data.xlsx')
    #else:
    #    if 'data' not in st.session_state:
    #        data = old_df
#             st.session_state.data = load_data() 
#         data = st.session_state.data.copy()
    data = pd.read_csv('Tagged_data_category.csv')
    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

    # Extract the week number from the 'Date' column
#     data['Week_Number'] = data['Created At Date'].dt.isocalendar().week
    data['Week_Number'] = data['Date'].apply(custom_week_number)
    
    
    # Create new columns with initial values set to 0
    data['P'] = 0
    data['N'] = 0
    data['D'] = 0

    # Update values based on conditions using .loc
    data.loc[data['Agent sNPS'] == 100, 'P'] = 1
    data.loc[data['Agent sNPS'] == 0, 'N'] = 1
    data.loc[data['Agent sNPS'] == -100, 'D'] = 1
    

        # # Columns to be checked for existence in the DataFrame
    required_columns = ['Ticket ID', 'Voice AHT', 'Resolution Answer']
    #old_df = old_df[['Ticket ID',sentences,agent,NPS_score,Date,'Voice AHT','Resolution Answer']]
    
    if set(required_columns).issubset(data.columns):
        data1 = data[['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer','Major Category','Sub Category','P','N','D', 'Week_Number']]
        #data1.columns = ['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer','Major Category','Sub Category','P','N','D', 'Week_Number']
    else:
        data1 = data[['Sentences','Agent Name','Agent sNPS','Date','Major Category','Sub Category','P','N','D', 'Week_Number']]
        #data1.columns = ['Sentences','Agent Name','Agent sNPS','Date','Major Category','Sub Category','P','N','D', 'Week_Number']
    # Creating a dataframe
    #data1 = data[['Date','Ticket ID','Agent Name', 'Voice AHT', 'Agent sNPS', 'Major Category','Sub Category','P','N','D', 'Resolution Answer', 'Week_Number']]

    st.sidebar.header('Filter')
    # Filters for date range
    start_date, end_date = st.sidebar.columns(2)
    start_date = start_date.date_input("Start Date", data1['Date'].min())
    end_date = end_date.date_input("End Date", data1['Date'].max())
    date_range = start_date,end_date
        
    # Filters for Agents
    agents = data1['Agent Name'].unique().tolist()
    agents.insert(0, 'All')
    default_agents = ['All'] if 'All' in agents else agents[:1]
    selected_agents = st.sidebar.multiselect('Agent', agents, default=default_agents)
    available_topics = select_all_topics(data1,date_range,selected_agents)

        
    # Streamlit app
    st.title('Snapshot')

    # Filter data
    df_filtered = filter_data(data1, date_range, selected_agents)

    # Display filtered data using st_aggrid
#     AgGrid(df_filtered,height=len(df_filtered)*50, theme=AgGridTheme.STREAMLIT, fit_columns_on_grid_load=True,autoHeight=True, fit_columns_to_grid=True)
    st.dataframe(df_filtered, width=1024,  hide_index=True)

    # Agent Training table
    st.title('Agent Training')
    # Fetch and display Agent Training data based on selected agents
    if selected_agents:
        agent_training_df = fetch_agent_training_data(data1,date_range, selected_agents)
        # Extracting values before colon
        agent_training_df = agent_issue_action_mapping(agent_training_df,data)
        #AgGrid(agent_training_df,height=400, theme=AgGridTheme.STREAMLIT, fit_columns_on_grid_load=True, wrap_text=True)
        st.dataframe(agent_training_df, width=1024, hide_index=True)



        if st.button('Get Recomendations'):
            if os.path.isfile('agent_training_with_recomendation.csv'):
                
                # Read the CSV file into a DataFrame
                prev_reco = pd.read_csv('agent_training_with_recomendation.csv')
                
                # Display the DataFrame
                #st.subheader('Previous Recomendation')
                #st.write(prev_reco)
            recomendation_df = get_recomendations(data,selected_agents,agent_training_df)
            st.subheader('Updated Recomendation')
            st.write(recomendation_df[['Agent Name','Recommendation']])
    else:
        st.write("No agent selected.")

    # Display the title1
    st.write('### Agent-wise WOW Trends')
    fig_aht, fig_nps, fig_resolution = wow_aht_snps_resolution_agent_training(data1,selected_agents,'All',date_range)

    # Create a single column layout
    for fig in [fig_nps, fig_aht, fig_resolution]:
        st.plotly_chart(fig)
    
    # Display the title2
    st.write('### Topic-wise WOW Trends')
    
    #TOPIC FILTER
    top3_topics = select_top3_topics(data1,date_range,selected_agents)
    top3_topics.insert(0, 'All')
    default_topics = ['All'] if 'All' in top3_topics else top3_topics
    selected_topics = st.sidebar.multiselect('Topic Filter', top3_topics, default=default_topics)
    
    topic_fig_nps,topic_fig_aht,topic_fig_resolution = wow_aht_snps_resolution_by_topics(data1,selected_agents,selected_topics,date_range)
    
    # Create a single column layout
    for fig in [topic_fig_nps,topic_fig_aht,topic_fig_resolution]:
        st.plotly_chart(fig)


#     Display the title3
    st.write('### Tag-wise WOW Trends for selected Agents and Topics')

    figs = wow_p_n_d(data1, date_range, selected_agents, selected_topics)
    
    # Create a single column layout
    for fig_name, fig in figs.items():
        st.plotly_chart(fig)
