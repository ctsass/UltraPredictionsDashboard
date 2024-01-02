import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Ultramarathon Predictions", 
    )

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data('test_data.csv')

median_times = load_data('median_times.csv')

min_times = load_data('min_times.csv')

races = data['race_name'].sort_values().unique()

st.title("Ultramarathon Predictions")

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

if len(updated_races) == 0:
    st.warning('There are no races within filtered ranges')

else:
    @st.cache_data
    def params(race_chosen):
    
        df = data[data.race_name == race_chosen]
        
        winning_times = df[['ppt_gender', 'time']].copy()
        winning_times = winning_times.groupby('ppt_gender')['time'].min()
        winning_times = winning_times.map("{:,.1f}".format)
    
        med_times = df[['ppt_gender', 'time']].copy()
        med_times = med_times.groupby('ppt_gender')['time'].median()
        med_times = med_times.map("{:,.1f}".format)
    
        SD = df[['ppt_gender', 'time']].copy()
        SD = SD.groupby('ppt_gender')['time'].std(ddof=0)
        SD = SD.map("{:,.1f}".format)
        
        ppt_count = df[['ppt_gender', 'time']].copy()
        ppt_count = ppt_count.groupby('ppt_gender')['time'].count()
    
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
    
        in_tar = df[['USU_in_target', 
                     'MED_in_target', 
                     'XGB_in_target']].mean()
        in_tar *= 100
        in_tar = in_tar.map("{:,.1f}%".format)
    
        ppt_total = len(df)
    
        _, bins = np.histogram(
            np.hstack(
                (
                    df.time, 
                    df.USU_pred, 
                    df.XGB_pred, 
                    df.MED_pred
                    )
                ), 
            bins=max(1, min(ppt_total//5, 20))
            )
        
        y_max = 0
        
        for s in ['time', 'USU_pred', 'XGB_pred', 'MED_pred']:
            count, _ = np.histogram(df[s], bins=bins)
            tmp = max(count)
            y_max = max(tmp, y_max)
                
        y_max += 1
        
        name_only = df.race_name_only.iloc[0]
        dist_only = df.dist.iloc[0]
        
        meds = median_times[(median_times.race_name == name_only)
                            & (median_times.distance_miles == dist_only)]
        
        mins = min_times[(min_times.race_name == name_only)
                            & (min_times.distance_miles == dist_only)]

        col_color = {'USU_pred':'tan', 
                     'MED_pred':'salmon', 
                     'XGB_pred':'cadetblue'}
    
        return (df, winning_times, med_times, SD, ppt_count, mean_err, 
                med_err, in_tar, bins[0], bins[-1], bins[1]-bins[0], 
                y_max, name_only, dist_only, ppt_total, meds, mins, 
                col_color)
    
    (df, winning_times, med_times, SD, ppt_count, mean_err, 
     med_err, in_tar, x_min, x_max, delta_x, 
     y_max, name_only, dist_only, ppt_total, 
     meds, mins, col_color) = params(race_chosen)
    
    tab1, tab2, tab3, tab4 = st.tabs(['Prediction performance', 
                                      'Race metrics', 
                                      'Race history', 
                                      'Background information',])
    
    with tab1:
        
        dash_11 = st.container(border=True)
        dash_12 = st.container(border=True)
        dash_13 = st.container()
        dash_14 = st.container(border=True)
        dash_15 = st.container(border=True)
    
        with dash_11:
    
            st.header(f'{name_only}', divider='gray')
            st.subheader(f'{dist_only} miles, N = {ppt_total} participants')
                
        with dash_12:
    
            st.subheader('Error metrics', divider='gray')
        
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
        
        with dash_13:
            
            with st.expander('Notes'):
                
                notes = '''
                - All time measurements are in minutes
                - AE is Absolute Error
                '''
                st.markdown(notes)
        
        with dash_14:
            
            st.subheader('Prediction and result distributions', divider=False)
            
            col_chosen = st.radio(
                'Choose a prediction scheme', 
                ['USU', 'MED', 'XGB'], 
                horizontal=True
                )
            
            col = col_chosen
            col_chosen += '_pred'
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=df[col_chosen],
                    xbins=dict(
                        start=x_min, 
                        end=x_max, 
                        size=delta_x + 0.00001
                        ), 
                    autobinx=False,
                    name=col,
                    marker_color=col_color[col_chosen],
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
            fig.update_xaxes(range = [x_min-1, x_max+1])
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
            
        with dash_15:
            
            st.subheader('Signed error violin plots', divider=False)
                
            fig_violin = go.Figure()
            fig_violin.add_trace(
                go.Violin(
                    x=df['XGB_err'], 
                    box_visible=True, 
                    meanline_visible=True, 
                    line_color='black', 
                    fillcolor=col_color['XGB_pred'], 
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
                    fillcolor=col_color['MED_pred'], 
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
                    fillcolor=col_color['USU_pred'], 
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
            
    with tab2:

        dash_21 = st.container(border=True)
        dash_22 = st.container(border=True)
        dash_23 = st.container()
        
        with dash_21:

            st.header(f'{name_only}', divider='gray')
            st.subheader(f'{dist_only} miles, N = {ppt_total} participants')
                
        with dash_22:
    
            st.subheader('Race metrics', divider='gray')
            
            m = len(SD)
            cols = st.columns(m)
            i = 0
            
            for col in cols:
                with col:
                    st.metric(
                        label=f'{ppt_count.index[i]} count',
                        value = ppt_count[i]
                        )
                    st.metric(
                        label=f'{winning_times.index[i]} win time',
                        value = winning_times[i]
                        )
                    st.metric(
                        label=f'{med_times.index[i]} median time',
                        value = med_times[i]
                        )
                    st.metric(
                        label=f'{SD.index[i]} std dev',
                        value = SD[i]
                        )
                    i += 1
        with dash_23:
            
            with st.expander('Note'):
                
                note = '''
                All time measurements are in minutes
                '''
                st.markdown(note)
    
    with tab3:

        dash_31 = st.container(border=True)
        dash_32 = st.container(border=True)
        
        with dash_31:

            st.header(f'{name_only}', divider='gray')
            st.subheader(f'{dist_only} miles)
            
        with dash_32:
            
            st.subheader('Median and Winning Times by Date')
            
            med_color = {'M':'firebrick' , 'F':'royalblue' , 'X':'teal'}
            
            fig = go.Figure()
            
            for gen in meds.ppt_gender.unique():
        
                gen_med = meds[meds.ppt_gender == gen].sort_values(by='date')
                gen_min = mins[mins.ppt_gender == gen].sort_values(by='date')
            
                fig.add_trace(go.Scatter(
                    x=gen_med.date, y=gen_med.med_time, name=gen+' median',
                    line = dict(color=med_color[gen])
                    ))
                fig.add_trace(go.Scatter(
                    x=gen_min.date, y=gen_min.min_time, name=gen+' win',
                    line = dict(color=med_color[gen], dash='dot')
                    ))
                
            fig.update_layout(margin=dict(t=20),
                              xaxis_title='Date',
                              yaxis_title='Time (minutes)')
            fig.update_layout(showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
            
    with tab4:
        
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
        - The final engineered dataset contained 459,433 examples with features for
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
