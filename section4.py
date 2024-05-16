import streamlit as st
import json
from streamlit_lottie import st_lottie
import os
from IPython.display import JSON
from st_aggrid import AgGrid, GridOptionsBuilder, AgGridTheme
import pandas as pd

from bs4 import BeautifulSoup
import re
import unicodedata
from llama_cpp import Llama
from transformers import AutoTokenizer
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from transformers import BertModel, BertTokenizer
import torch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from transformers import BertModel



st.image(r"header1.png", use_column_width=True,)

def load_lottiefile(filepath:str):
            with open(filepath, "r") as f:
                return json.load(f)



model_path = r'Phi-3-mini-4k-instruct-q4.gguf'

import streamlit as st
import time

def custom_progress_bar(iterable):
    total = len(iterable)
    progress_bar = st.progress(0)
    for i, item in enumerate(iterable):
        progress_bar.progress((i + 1) / total)
        yield item

def run_llm_categories(cate_list):
    st.write("Creating categories by LLM")

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context length to use
        n_threads=4  # Number of CPU threads to use
        # n_gpu_layers=0  # Number of model layers to offload to GPU
    )

    generation_kwargs = {
         "max_tokens": 1000,
         "stop": ["</s>"],
         "echo": False,  # Echo the prompt in the output
         "top_k": 1  # Greedy decoding
     }

    context = ""
    question = f"categorize the provided list of issues effectively,group them based on their primary themes, ensuring each item is placed in a single, relevant category: {cate_list}"
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    res = llm(prompt, **generation_kwargs)  # Res is a dictionary
    # st.write(res['choices'][0]['text'])

    # Extract 'choices' from the dictionary
    choices_text = res['choices'][0]['text']

    st.write(res)
    st.write(choices_text)

    choices_text = choices_text.replace('<|end|>','')
    choices_text = choices_text.replace('<|assistant|>','')

    # Use regular expressions to find all categories and subcategories
    categories = re.findall(r'(?:\d+\.\s+)?(.*?):\s+(?:\[)?(.*?)(?:\])?(?=\n|$)', choices_text, re.DOTALL)

    #st.write(categories)

    # Initialize a list to store all categories and subcategories
    all_categories = []

    # Iterate through found categories and subcategories
    for category, subcategories_text in categories:
        # Split subcategories by comma and strip them
        subcategories = [subcategory.strip() for subcategory in subcategories_text.split(',')]
        all_categories.append((category.strip(), subcategories))

    # Print the list of categories and subcategories
    for category, subcategories in all_categories:
        print(category)
        for subcategory in subcategories:
            print('-', subcategory)
        print()


    category_df = pd.DataFrame(all_categories)

    #st.write(category_df)

    # Splitting content of column 1 into separate rows
    df_expanded = category_df.explode(1).reset_index(drop=True)
    df_expanded.columns = ['Major category', 'Sub Category']
    df_expanded['Sub Category'] = df_expanded['Sub Category'].str.strip("'")
    #ddf['Detail'] = ddf['Detail'].str.strip()
    return df_expanded



def run_llm_categories2(cate_list):
    st.write("Creating categories by LLM")

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context length to use
        n_threads=4  # Number of CPU threads to use
        # n_gpu_layers=0  # Number of model layers to offload to GPU
    )

    generation_kwargs = {
         "max_tokens": 100,
         "stop": ["</s>"],
         "echo": False,  # Echo the prompt in the output
         "top_k": 1  # Greedy decoding
     }

    max_records_per_sublist = 200
    # Initialize an empty list to store sublists
    sublists = []

    # Iterate over the original list, creating sublists
    #while cate_list:
    #    sublist = cate_list[:max_records_per_sublist]
    #    sublists.append(sublist)
    #    cate_list = cate_list[max_records_per_sublist:]


    st.write(cate_list[:max_records_per_sublist])
    
    broad_categories = run_llm_categories(cate_list[:max_records_per_sublist])
    st.write(broad_categories)

    # Convert the DataFrame column to a set
    df_categories_set = set(broad_categories['Sub Category'])

    # Convert the list to a set
    list_categories_set = set(cate_list)

    # Find the categories that are not common between the DataFrame and the list
    unique_categories = df_categories_set.symmetric_difference(list_categories_set)

    # Get the count of unique categories
    unique_categories_count = len(unique_categories)

    st.write(f"Unique categories not common between the DataFrame and the list: {unique_categories}")
    st.write(f"Count of unique categories: {unique_categories_count}")
    
    #for i, item in enumerate(sublists):
    #    st.write(i)
    #    st.write(item)
    #    context = ""
    #    question = f"categorize the provided list of issues effectively,group them based on their primary themes, ensuring each item is placed in a single, relevant category.Provide me the list of categories and end your answer with |end|: {item}"
    #    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    #    res = llm(prompt, **generation_kwargs)  # Res is a dictionary
        # st.write(res['choices'][0]['text'])

        # Extract 'choices' from the dictionary
    #    choices_text = res['choices'][0]['text']
    #    st.write(choices_text)
        #choices_text = choices_text.replace('<|end|>','')
        #choices_text = choices_text.replace('<|assistant|>','')

        # Use regular expressions to find all categories and subcategories
        #categories = re.findall(r'(?:\d+\.\s+)?(.*?):\s+(?:\[)?(.*?)(?:\])?(?=\n|$)', choices_text, re.DOTALL)

    #st.write(categories)

    # Initialize a list to store all categories and subcategories
    #all_categories = []

    # Iterate through found categories and subcategories
    #for category, subcategories_text in categories:
        # Split subcategories by comma and strip them
    #    subcategories = [subcategory.strip() for subcategory in subcategories_text.split(',')]
    #    all_categories.append((category.strip(), subcategories))

    # Print the list of categories and subcategories
    #for category, subcategories in all_categories:
    #    print(category)
    #    for subcategory in subcategories:
    #        print('-', subcategory)
    #    print()


    #category_df = pd.DataFrame(all_categories)

    #st.write(category_df)

    # Splitting content of column 1 into separate rows
    #df_expanded = category_df.explode(1).reset_index(drop=True)
    #df_expanded.columns = ['Major category', 'Sub Category']
    #df_expanded['Sub Category'] = df_expanded['Sub Category'].str.strip("'")
    #ddf['Detail'] = ddf['Detail'].str.strip()
    #return df_expanded




def run_clustering_categories(categories_list):
    
    st.write("Creating categories by cluster")

    

    # Load the model locally
    local_model_directory = r'bert-base-uncased'
    model = BertModel.from_pretrained(local_model_directory)


    # Load pre-trained BERT model and tokenizer
    #model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    #model = BertModel.from_pretrained(model_name)

    # Assuming you have a DataFrame 'df' with a column 'text_column' that you want to cluster
    #text_data = df['text_column'].tolist()

    # Function to encode text data to BERT embeddings
    def encode_text(text_list):
        model.eval()
        with torch.no_grad():
            encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
            output = model(**encoded_input)
        return output.last_hidden_state.mean(dim=1).numpy()

    # Generate BERT embeddings
    embeddings = encode_text(categories_list)

    # Perform Agglomerative Clustering without specifying the number of clusters
    clustering_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
    clustering_model.fit(embeddings)

    # Create the linkage matrix
    Z = linkage(embeddings, method='ward')

    # Determine the number of clusters using the elbow method
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    if acceleration_rev.size == 0:
        k = 2  # Default to 2 clusters if there's no data for acceleration
    else:
        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters

    #k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters

    # Perform Agglomerative Clustering with the determined number of clusters
    final_clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    st.write(final_clustering)
    st.write(final_clustering.fit_predict(embeddings))

    df = pd.DataFrame()
    df['Detail'] = categories_list

    df['cluster'] = final_clustering.fit_predict(embeddings)

    #st.write(df)

    # Create a new DataFrame to hold the text data for each cluster
    #clustered_df = pd.DataFrame(columns=['Cluster', 'Text_Data'])

    # Group by 'cluster' and aggregate the text data
    grouped_df = df.groupby('cluster')['Detail'].apply(list).reset_index()

    # Now 'clustered_df' contains each cluster and the corresponding list of text data
    #st.write(grouped_df)

    # Now 'df' has an additional column 'cluster' with cluster labels
                    

    llm = Llama(
        model_path=model_path,
        n_ctx=3000,  # Context length to use
        n_threads=4  # Number of CPU threads to use
        # n_gpu_layers=0  # Number of model layers to offload to GPU
    )

    generation_kwargs = {
         "max_tokens": 10,
         "stop": ["</s>"],
         "echo": False,  # Echo the prompt in the output
         "top_k": 1  # Greedy decoding
    }


    st.write("Fine tuning Category Names")
    total_records = len(grouped_df)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx,row in grouped_df.iterrows():
        current_index = grouped_df.index.get_loc(idx) + 1
        #st.write(current_index)
        #st.write(total_records)
        progress_percent = current_index/total_records
        #st.write(progress_percent)
        progress_bar.progress(progress_percent)
        status_text.text(f"Processing record {current_index}/{total_records}")

        
        context = row['Detail']
        question = "Give a three word name to these cluster of words."
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        res = llm(prompt, **generation_kwargs)  # Res is a dictionary
        #st.write(i)
        #st.write(res['choices'][0]['text'])

        # Extract 'choices' from the dictionary
        choices_text = res['choices'][0]['text'].split('<|assistant|>')[0]
        #choices_text2 = choices_text.split('\n')[0]
        #st.write(choices_text)
        grouped_df.loc[idx,'Category'] = choices_text

    status_text.empty()

    grouped_df.to_csv('Grouped_category.csv')
    
    st.write("Categories Created. Processing Data Now..")
    #st.write(grouped_df)
     # Splitting content of column 1 into separate rows
    #
    #st.write(grouped_df)
    # Assuming 'df' is your DataFrame after the groupby and list operation
    df_exploded = grouped_df.explode('Detail').reset_index(drop=True)
    
    #split_df = grouped_df['Detail'].str.split(',', expand=True)
    #st.write(split_df)
    #stacked_df = split_df.stack().reset_index(level=1, drop=True).rename('Detail')
    #st.write(stacked_df)
    #result_df = pd.merge(stacked_df, grouped_df[['Category']], left_index=True, right_index=True, how='left')
    #st.write(result_df)

    #result_df['Detail'] = result_df['Detail'].strip()
    #result_df['Category'] = result_df['Category'].strip()
    df_exploded = df_exploded[['Detail','Category']]
    st.write(df_exploded)

    df_exploded.columns = ['Sub Category','Major category']
    #st.write(df_exploded)

    
    return df_exploded




def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove punctuation
    #text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    #text = re.sub(r'\d+', '', text)
    # Tokenization
    #words = word_tokenize(text)
    # Remove stopwords
    #words = [word for word in words if word not in stopwords.words('english')]
    # Re-create text from words
    #text = ' '.join(words)
    return text

def run_analysis(temp_data):
    out = data_tuning(temp_data)
    out = analyse_NP(out)
    out = analyse_p(out)
    #out = pd.read_csv('Tagged_df.csv')
    out = create_categories(out)
    return(out)
    

def data_tuning(temp_data):
    df = temp_data

    #['Sentences','Agent Name','Agent sNPS','NPS Type']
    df.loc[df['Agent sNPS']==-100,'NPS Type']='Detractor'
    df.loc[df['Agent sNPS']==100,'NPS Type']='Promoter'
    df.loc[df['Agent sNPS']==0,'NPS Type']='Neutral'
    #df = df[['Sentences','Tag','Agent sNPS']]
#    df.head()

    #st.write(df['NPS Type'].value_counts())

    #df.columns = ['Sentences','Tag','Agent sNPS','Survey Date']
    #['Sentences','Agent Name','Agent sNPS','NPS Type']
    df = df[~df['Sentences'].isna()]
    df = df[df['Sentences']!=0]

    ddf = df

    st.write("Data Loaded")
    #st.write(ddf)
    
    st.write("1/5 PreProcessing Data")
    ddf['Processed_Sentences'] = ddf['Sentences'].apply(preprocess_text)
    return(ddf)

    
def analyse_NP(ddf):

    ## Instantiate model from downloaded file
    llm = Llama(
        model_path=model_path,
        n_ctx=1000,  # Context length to use
        n_threads=4  # Number of CPU threads to use
        # n_gpu_layers=0  # Number of model layers to offload to GPU
    )

    generation_kwargs = {
         "max_tokens": 20,
         "stop": ["</s>"],
         "echo": False,  # Echo the prompt in the output
         "top_k": 1  # Greedy decoding
     }


    st.write("2/5 Analysing Non Promoters")

    question = "What is customer unhappy about in the given input? Your Answere should begin with The customers are unhappy about. Answere in maximum 3 words."

    ddf_temp = ddf[ddf['NPS Type'] != 'Promoter']

    ##total_iterations = len(ddf_temp['Processed_Sentences'])
    total_records = len(ddf_temp)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in ddf_temp.iterrows():
        #st.write(idx)
        current_index = ddf_temp.index.get_loc(idx) + 1
        #st.write(current_index)
        #st.write(total_records)
        progress_percent = current_index/total_records
        #st.write(progress_percent)
        progress_bar.progress(progress_percent)
        status_text.text(f"Processing record {current_index}/{total_records}")
        
        context = row['Processed_Sentences']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        res = llm(prompt, **generation_kwargs)  # Res is a dictionary

            ## Unpack and print the generated text from the LLM response dictionary
            #print(f"Context: {context}")
        print(f"Answer: {res['choices'][0]['text']}")
            #print("-----")
        word = res['choices'][0]['text'].split('\n')[0]
        ddf.loc[idx,'Detail'] = word
            
        #    pbar.update(1)

    status_text.empty()

    return(ddf)


# In[26]:
def analyse_p(ddf):
    st.write("3/5 Analysing Promoters")

        ## Instantiate model from downloaded file
    llm = Llama(
        model_path=model_path,
        n_ctx=1000,  # Context length to use
        n_threads=4  # Number of CPU threads to use
        # n_gpu_layers=0  # Number of model layers to offload to GPU
    )

    generation_kwargs = {
         "max_tokens": 20,
         "stop": ["</s>"],
         "echo": False,  # Echo the prompt in the output
         "top_k": 1  # Greedy decoding
     }

    question = "What is customer happy about in the given input? Your Answere should begin with The customers are happy about. Answere in maximum 3 words."

    ddf_temp = ddf[ddf['NPS Type'] == 'Promoter']

    total_records = len(ddf_temp)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in ddf_temp.iterrows():
        #st.write(idx)
        current_index = ddf_temp.index.get_loc(idx) + 1
        #st.write(current_index)
        #st.write(total_records)
        progress_percent = current_index/total_records
        #st.write(progress_percent)
        progress_bar.progress(progress_percent)
        status_text.text(f"Processing record {current_index}/{total_records}")
            
        context = row['Sentences']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        res = llm(prompt, **generation_kwargs)  # Res is a dictionary

            ## Unpack and print the generated text from the LLM response dictionary
            #print(f"Context: {context}")
        print(f"Answer: {res['choices'][0]['text']}")
        print("-----")
        word = res['choices'][0]['text'].split('\n')[0]
        ddf.loc[idx,'Detail'] = word
            
        #pbar.update(1)

    status_text.empty()




    #st.write(ddf)

    ddf['Detail'] = ddf['Detail'].str.lower()


    # In[521]:


    #ddf['Detail'] = ddf['Detail'].str.replace("the fact that", "")
    #ddf['Detail'] = ddf['Detail'].str.replace("the fact", "")
    #ddf['Detail'] = ddf['Detail'].str.replace("fact", "")
    ddf['Detail'] = ddf['Detail'].str.replace("the customers are happy about", "")
    ddf['Detail'] = ddf['Detail'].str.replace("customers are happy about", "")
    ddf['Detail'] = ddf['Detail'].str.replace("the customers are unhappy about", "")
    ddf['Detail'] = ddf['Detail'].str.replace("customers are unhappy about", "")
    #ddf['Detail'] = ddf['Detail'].str.replace("the", "")



    ddf.to_csv('Tagged_df.csv')

    return(ddf)

    #st.write(ddf)
def create_categories(ddf):
    st.write("4/5 Creating Categories")
    cate_list = ddf['Detail'].unique().tolist()
    #cate_list = [item.lower() for item in cate_list]

        # ## Instantiate model from downloaded file
    


    st.write(cate_list)
    #if(len(ddf['Detail'])<=100):
    df_expanded = run_clustering_categories(cate_list)
    #else:
    #    df_expanded = run_clustering_categories(cate_list)



    st.write(df_expanded.head())
    st.write(ddf.head())

    df_expanded['Sub Category'] = df_expanded['Sub Category'].str.strip()
    ddf['Detail'] = ddf['Detail'].str.strip()

    # Merge df2 with categories_df on Category column
    merged_df = ddf.merge(df_expanded, left_on='Detail', right_on='Sub Category', how='left')

    # Drop the Sub Category column
    #merged_df.drop('Sub Category', axis=1, inplace=True)

    # Rename the Major category column
    merged_df.rename(columns={'Major category': 'Major Category'}, inplace=True)

    #st.write(merged_df)
    
    #output_ddf = ddf[['Survey Date','Sentences', 'Tag', 'Agent sNPS','Major Category','Detail']]

    #st.write(output_ddf)

    #output_ddf.columns = ['Survey Date','Sentences', 'Tag', 'Agent sNPS','Major Category','Detail']
    df_expanded.to_csv('Category.csv')
    merged_df.to_csv('Tagged_data_category.csv')
    # In[590]:
    st.write("5/5 Analysis Completed")

    return merged_df


def display_results(df):
    st.write("Processed DataFrame:")
    st.write(df) 
#==========================================SECTION 4=======================================
# st.image("header1.png", use_column_width=True,)

# Define Section 4
def section4():

    old_df = pd.DataFrame(columns=['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer'])

        # Upload DataFrame file
    st.sidebar.header("Upload DataFrame")
    uploaded_file = st.sidebar.file_uploader("Upload DataFrame (Excel file)", type=["xlsx"], key="data_frame")

    if uploaded_file is None:
        st.write("Please Upload file.")
    else:
        old_df = pd.read_excel(uploaded_file)

            
    if st.sidebar.button('Preview'):
        if uploaded_file is None:
            st.write("Please Upload file before Preview.")
        else:
            st.subheader("Preview of Uploaded DataFrame:")
            st.write(old_df.head())

    #st.title("Issue Analysis App")
    col1, col2 = st.columns(2)
    with col1:
            st.markdown("<h5 class='instruction_heading'> </h5>", unsafe_allow_html=True)
            lottie_creator = load_lottiefile("intro.json")
            st_lottie(lottie_creator, speed=1, reverse=False, loop=True, height=450)
            # Video Section
            # st.subheader("Introduction Video:")
            # video_url = "1.mp4"  # Replace with your actual video URL
            # st.video(video_url)
    with col2:
            st.markdown(
                """
                <div style="">
                        <div style="
                            position: absolute;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 50%;
                        "></div>   
                            <style>
                    ul.custom-list {
                        list-style-type: none;
                        padding-left: 20px;
                        color: #1b3461;
                        font-size:14px;
                    }

                    ul.custom-list li::before {
                        content: '◼'; /* Unicode character for an em dash */
                        color: #ff555f; /* Line color */
                        display: inline-block;
                        width: 2em;
                        font-size:9px;
                        margin-left: -1em;
                    }
                </style>

                <h5 style="color: #ffffff; background: #1ec677; padding: 10px; padding-left:20px;  font-size:16px;">Business purpose</h5>
                <p style="padding-top:10px; font-size:14px; color: #1b3461">The application is designed to empower businesses with
                actionable insights by analyzing customer feedback data.</p>
                <ul class="custom-list">
                    <li style="font-size:14px;">&nbsp;    Utilizing a state-of-the-art language learning model (LLM),
                the app provides a comprehensive analysis of customer sentiments and experiences. </li>
                    <li style="font-size:14px;">&nbsp;   Running fully locally,
                it ensures the utmost confidentiality and security of sensitive customer data.</li>
                     </ul>
                
                <h5 style="color: #ffffff; background: #1ec677; padding: 10px; padding-left:20px; font-size:16px;">Value proposition</h5>
                <ul class="custom-list">
                    <li style="font-size:14px;">&nbsp; Data Security: With local system operation, customer data remains secure and private, eliminating risks associated with cloud storage.</li>
                    <li style="font-size:14px;">&nbsp; Insightful Analytics: The LLM-based model meticulously parses feedback, identifying key issues and trends that affect customer satisfaction.</li>
                    <li style="font-size:14px;">&nbsp; Targeted Training: By pinpointing specific areas of concern, the app guides businesses in developing focused training programs for customer service agents.</li>
                    <li style="font-size:14px;">&nbsp; Performance Enhancement: It offers tailored recommendations for agent performance improvement, fostering a culture of excellence in customer service.</li>
                    <li style="font-size:14px;">&nbsp; Customer Retention: By understanding and addressing customer needs, the app aids in enhancing customer loyalty and retention.</li>
           
                </ul>
                """,
                unsafe_allow_html=True
            )
            
    # Upload issue list file
    #st.sidebar.header("Upload Issue List")
    #issue_list_file = st.sidebar.file_uploader("Upload Issue List (Excel file)", type=["xlsx"], key="issue_list")



    #st.sidebar.header("Upload DataFrame")
    #df_file = st.sidebar.file_uploader("Upload DataFrame (Excel file)", type=["xlsx"], key="data_frame")

    sentences = st.sidebar.selectbox('Select the Feedback Column', old_df.columns)
    agent = st.sidebar.selectbox('Select the Agent Column', old_df.columns)
    NPS_score = st.sidebar.selectbox('Select the NPS Score', old_df.columns)
    # NPS = st.sidebar.selectbox('Select the NPS Column', old_df.columns)
    Date = st.sidebar.selectbox('Select the Date Column', old_df.columns)

    # # Columns to be checked for existence in the DataFrame
    required_columns = ['Ticket ID', 'Voice AHT', 'Resolution Answer']
    #old_df = old_df[['Ticket ID',sentences,agent,NPS_score,Date,'Voice AHT','Resolution Answer']]

    if set(required_columns).issubset(old_df.columns):
        print('')
    else:
        old_df['Resolution Answer'] = 0
        old_df['Voice AHT'] = 0
        old_df['Ticket ID'] = range(1, len(old_df) + 1)

    old_df = old_df[['Ticket ID',sentences,agent,NPS_score,Date,'Voice AHT','Resolution Answer']]
    old_df.columns = ['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer']

    old_df['Date'] = pd.to_datetime(old_df['Date'], format='%d-%m-%Y')
    
    
    #if set(required_columns).issubset(old_df.columns):
    #    old_df = old_df[['Ticket ID',sentences,agent,NPS_score,Date,'Voice AHT','Resolution Answer']]
    #    old_df.columns = ['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer']
    #else:
    #    old_df = old_df[[sentences,agent,NPS_score,Date,]]
    #    old_df.columns = ['Sentences','Agent Name','Agent sNPS','Date']

    
    # old_df = old_df[['Ticket ID',sentences,agent,NPS_score,Date,'Voice AHT','Resolution Answer']]
    # old_df.columns = ['Ticket ID','Sentences','Agent Name','Agent sNPS','Date','Voice AHT','Resolution Answer']




    df = old_df[~old_df['Sentences'].isna()]
    df = df[df['Sentences']!=0]
    df = df[df['Sentences']!='0']

    # Calculate metrics
    total_records = total_unique_agents = total_issue_counts = 0

    #df = pd.read_excel(df_file)

        # Calculate metrics
    total_records = old_df.shape[0]
    total_unique_agents = old_df['Agent Name'].nunique()
    total_issue_counts = len(df['Sentences'])

    # Display metric cards
    col1, col2, col3, = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div style='background: linear-gradient(90deg, #ffffff, #e0e0e0); height: 100px; text-align: center; width: 100%; border-radius: 10px; box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.1); position: relative; overflow: hidden; border-collapse: collapse;'>
                <h4>Total Records</h2>
                <p style='color:red; font-weight:bold; text-align:center'>{total_records}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style='background: linear-gradient(90deg, #ffffff, #e0e0e0); height: 100px; text-align: center; width: 100%; border-radius: 10px; box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.1); position: relative; overflow: hidden; border-collapse: collapse;'>
                <h4>Unique Agents</h2>
                <p style='color:red; font-weight:bold; text-align:center'>{total_unique_agents}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style='background: linear-gradient(90deg, #ffffff, #e0e0e0); height: 100px; text-align: center; width: 100%; border-radius: 10px; box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.1); position: relative; overflow: hidden; border-collapse: collapse;'>
                <h4>Issue Counts</h2>
                <p style='color:red; font-weight:bold; text-align:center'>{total_issue_counts}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    
    num_records = st.sidebar.slider("Select number of records", min_value=1, max_value=df.shape[0], value=3)
    st.sidebar.write("Expected Time to Analyse: ", round((num_records*20)/60), " minutes")
    
    if st.sidebar.button('Analyse'):
        temp_data = df.head(num_records)
        result_df = run_analysis(temp_data)