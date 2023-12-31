# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:12:58 2023

@author: ctsass
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Finish Time Predictions", 
    )

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data('test_data.csv')

races = data['race_name'].sort_values().unique()

st.title("Test Set Finish Time Predictions")

min_count = data.participant_count.min()
max_count = data.participant_count.max()

min_dist = data.dist.min()
max_dist = data.dist.max()

with st.sidebar:
    
    st.title('Race selection')
    
    part_count_choice = st.slider(
        'Filter on participant count range', 
        min_value = min_count,
        max_value = max_count,
        value = [min_count, max_count]
        )
    
    updated_races = data[
        (data.participant_count >= part_count_choice[0])
        & (data.participant_count <= part_count_choice[1])
        ].race_name.sort_values().unique()
    
    dist_choice = st.slider(
        'Filter on distance range (in miles)', 
        min_value = min_dist,
        max_value = max_dist,
        value = [min_dist, max_dist]
        )
    
    updated_races = data[
        data.race_name.isin(updated_races) &
        (data.dist >= dist_choice[0]) &
        (data.dist <= dist_choice[1])
        ].race_name.sort_values().unique()
    
    if len(updated_races) == 0:
        st.warning('There are no races within filtered ranges')
    else:
        st.info(
            f'There are {len(updated_races)} races within filtered ranges.'
            )
    
    race_chosen = st.selectbox('Select a race', updated_races)
    
tab1, tab2 = st.tabs(['Prediction performance', 'Background information'])

df = data[data.race_name == race_chosen]

mean_err = df[
    ['USU_abs_err', 
     'MED_abs_err', 
     'XGB_abs_err']
    ].mean().map("{:,.2f}".format)

med_err = df[
    ['USU_abs_err', 
     'MED_abs_err', 
     'XGB_abs_err']
    ].median().map("{:,.2f}".format)

in_tar = df[['USU_in_target', 'MED_in_target', 'XGB_in_target']].mean()
in_tar *= 100
in_tar = in_tar.map("{:,.1f}%".format)

with tab1:
    
    dash_1 = st.container(border=True)
    dash_2 = st.container(border=True)
    dash_3 = st.container()
    dash_4 = st.container(border=True)
    dash_5 = st.container(border=True)

    with dash_1:
        
        if len(updated_races) == 0:
            st.header('None')
        else:
            disp_1 = ' '.join(race_chosen.split()[:-5])
            disp_2 = race_chosen.split()[-5][1:]
            disp_3 = race_chosen.split()[-1]
            st.header(f'{disp_1}', divider='blue')
            st.subheader(f'{disp_2} miles, N = {disp_3} participants')
    
    with dash_2:
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label = 'USU mean AE',
                value = mean_err['USU_abs_err']
                )
            st.metric(
                label='USU median AE', 
                value = med_err['USU_abs_err']
                )
            st.metric(
                label='USU \u00B11 std dev', 
                value = in_tar['USU_in_target']
                )
            
            with col2:
                st.metric(
                    label='MED mean AE', 
                    value = mean_err['MED_abs_err']
                    )
                st.metric(
                    label='MED median AE', 
                    value = med_err['MED_abs_err']
                    )
                st.metric(
                    label='MED \u00B11 std dev', 
                    value = in_tar['MED_in_target']
                    )
            
            with col3:
                st.metric(
                    label='XGB mean AE', 
                    value = mean_err['XGB_abs_err']
                    )
                st.metric(
                    label='XGB median AE', 
                    value = med_err['XGB_abs_err']
                    )
                st.metric(
                    label='XGB \u00B11 std dev', 
                    value = in_tar['XGB_in_target']
                    )
    
    with dash_3:
        
        with st.expander('Notes'):
            
            notes = '''
            - AE is Absolute Error 
            - All error measurements are in minutes
            '''
            st.markdown(notes)
        
    with dash_4:
        
        st.subheader('Signed error violin plots', divider=False)
            
        fig_violin = go.Figure()
        fig_violin.add_trace(
            go.Violin(
                x=df['XGB_err'], 
                box_visible=True, 
                meanline_visible=True, 
                line_color='black', 
                fillcolor='cadetblue', 
                opacity=0.8, 
                name='XGB'
                )
            )
        fig_violin.add_trace(
            go.Violin(
                x=df['MED_err'], 
                box_visible=True,
                meanline_visible=True, 
                line_color='black', 
                fillcolor='salmon', 
                opacity=0.8, 
                name='MED'
                )
            )
        fig_violin.add_trace(
            go.Violin(
                x=df['USU_err'], 
                box_visible=True,
                meanline_visible=True, 
                line_color='black', 
                fillcolor='tan', 
                opacity=0.8, 
                name='USU'
                )
            )
        fig_violin.update_traces(orientation='h')
        fig_violin.update_layout(
            xaxis_title=dict(text='Prediction minus result (minutes)'),
            margin=dict(t=20)
        )
            
        st.plotly_chart(fig_violin, use_container_width=True, theme=None)
    
    with dash_5:
        
        st.subheader('Prediction and result distributions', divider=False)
        
        col_chosen = st.radio(
            'Choose a prediction scheme', 
            ['USU', 'MED', 'XGB'], 
            horizontal=True
            )
        
        col = col_chosen
        col_chosen += '_pred'
        
        df = data[data.race_name == race_chosen]
        n = len(df)
        
        _, bins = np.histogram(
            np.hstack(
                (
                    df.time, 
                    df.USU_pred, 
                    df.XGB_pred, 
                    df.MED_pred
                    )
                ), 
            bins=max(1, min(n//5, 20))
            )
    
        y_max = 0
    
        for s in ['time', 'USU_pred', 'XGB_pred', 'MED_pred']:
            count, _ = np.histogram(df[s], bins=bins)
            tmp = max(count)
            y_max = max(tmp, y_max)
            
        y_max += 1
    
        time_count, index = np.histogram(df['time'], bins=bins)
        
        color = {'USU_pred':'tan', 'MED_pred':'salmon', 'XGB_pred':'cadetblue'}
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=df[col_chosen],
                xbins=dict(
                    start=bins[0], 
                    end=bins[-1], 
                    size=bins[1]-bins[0]+0.00001
                    ), 
                autobinx=False,
                name=col,
                marker_color=color[col_chosen],
                opacity=0.8,
                bingroup=1
                )
            )
        fig.add_trace(
            go.Histogram(
                x=df['time'],
                name='results',
                marker_color='rgba(245, 157, 39, 0)',
                marker_line_color='black',
                marker_line_width=1.5,
                bingroup=1
                )
            )
        fig.update_layout(
            barmode='overlay', 
            margin=dict(t=20), 
            xaxis_title=dict(text='Minutes')
            )
        fig.update_yaxes(range = [0, y_max])
        fig.update_xaxes(range = [bins[0]-1, bins[-1]+1])
        
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
with tab2:
    
    info = '''
    - The original dataset contained 1,429,253 race results with
    dates ranging from 08-24-2018 to 08-05-2023.
    - After removing probable non-running races, 
    there were 1,191,678 results in the filtered dataset.
    - In order to engineer features to be used for predictions, 
    two pieces of information were needed:
    at least one earlier result for the participant and the results from
    at least one previous running of the race.
    - So a large portion of the filtered dataset could only be used to
    engineer features for examples occurring later (by date) in the filtered
    dataset.
    - The final dataset contained 459,433 examples with features for
    predictions.
    - This dataset was split by date into training and test sets in order to
    train and evaluate an XGBoost algorithm for predicting finish times.
    - The train set contained 419,861 examples, with dates ranging from 
    02-17-2019 to 04-30-2023.
    - The test set contained 39,572 examples, with dates ranging from 
    05-01-2023 to 08-05-2023.
    - This app contains information about the performance of the three 
    prediction schemes on the test set.
    '''
    st.markdown(info)
    
    
    