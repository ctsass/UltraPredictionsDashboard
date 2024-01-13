import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

st.set_page_config(
    page_title="Ultramarathon Predictions", 
    )

@st.cache_data
def load_text(path):
    with open(path, 'r', encoding="utf-8") as file:
        text = file.read()
    pattern = re.compile("\n\n")
    return pattern.split(text)

info_1 = load_text('info_1.txt')

info_2 = load_text('info_2.txt')

info_3 = load_text('info_3.txt')

info_4 = load_text('info_4.txt')

usu_notes = load_text('usu_notes.txt')

data_notes = load_text('data_notes.txt')

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data('test_data.csv')

median_times = load_data('median_times.csv')

min_times = load_data('min_times.csv')

data_med = load_data('test_median.csv')

data_mean = load_data('test_mean.csv')

races = data['race_name'].sort_values().unique()

st.title("Ultramarathon Predictions")

min_count = data.participant_count.min()
max_count = data.participant_count.max()

min_dist = data.dist.min()
max_dist = data.dist.max()

with st.sidebar:
    
    st.title('Race selection')
    
    test_note = '''**Note**: All test set races 
    occurred in 2023.
    '''
    st.markdown(test_note)
    
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
        st.info(f'There are {len(updated_races)} races within filtered ranges.')
    
    race_chosen = st.selectbox('Select a race', 
                               updated_races, 
                               index=742 if len(updated_races) == len(races) else 0)

if len(updated_races) == 0:
    st.warning('There are no races within filtered ranges')

else:
    @st.cache_data
    def params(race_chosen):
    
        df = data[data.race_name == race_chosen]
        
        winning_times = df[['ppt_gender', 'time']].copy()
        winning_times = winning_times.groupby('ppt_gender')['time'].min()
        winning_times = winning_times.map("{:,.2f}".format)
    
        med_times = df[['ppt_gender', 'time']].copy()
        med_times = med_times.groupby('ppt_gender')['time'].median()
        med_times = med_times.map("{:,.2f}".format)
    
        SD = df[['ppt_gender', 'time']].copy()
        SD = SD.groupby('ppt_gender')['time'].std(ddof=0)
        SD = SD.map("{:,.2f}".format)
        
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
        
        preds = ['USU', 'MED', 'XGB']
    
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
        
        medians = median_times[(median_times.race_name == name_only)
                            & (median_times.distance_miles == dist_only)]
        
        minimums = min_times[(min_times.race_name == name_only)
                            & (min_times.distance_miles == dist_only)]
        
        gen_meds = [(gen, medians[medians.ppt_gender == gen].sort_values(by='date'))
                        for gen in medians.ppt_gender.unique()]
        
        gen_mins = [(gen, minimums[minimums.ppt_gender == gen].sort_values(by='date'))
                        for gen in minimums.ppt_gender.unique()]
        
        col_color = {'USU_pred':'tan', 
                     'MED_pred':'salmon', 
                     'XGB_pred':'cadetblue'}
        
        history_color = {'M':'firebrick' , 'F':'royalblue' , 'X':'teal'}
    
        return (df, winning_times, med_times, SD, ppt_count, mean_err, 
                med_err, in_tar, preds, bins[0], bins[-1], bins[1]-bins[0], 
                y_max, name_only, dist_only, ppt_total, 
                col_color, history_color, gen_meds, gen_mins)
    
    (df, winning_times, med_times, SD, ppt_count, mean_err, 
     med_err, in_tar, preds, x_min, x_max, delta_x, 
     y_max, name_only, dist_only, ppt_total, col_color, history_color, 
     gen_meds, gen_mins) = params(race_chosen)
    
    tab1, tab2, tab4 = st.tabs(['Project information', 
                                      'Race-specific prediction performance', 
                                      'Race-specific history'])
    
    with tab1:
        
        with st.container():
            
            for par in info_1:
                st.markdown(par)
        
        with st.container(border=True):
            
            st.subheader('Median percent error', 
                         divider=False)
        
            fig = go.Figure()
            
            preds = ['USU', 'MED', 'XGB']
            col_color = {'USU_pred':'tan', 
                         'MED_pred':'salmon', 
                         'XGB_pred':'cadetblue'}
            
            for pred in preds:
                fig.add_trace(
                    go.Bar(
                    x=data_med.length_range,
                    y=data_med[pred+'_pe'], 
                    marker_color=col_color[pred+'_pred'],
                    name=pred
                    )
                )
            
            fig.update_layout(
                barmode='group', 
                bargroupgap=0.1,
                xaxis_tickangle=-45,
                xaxis_title=dict(text='Length ranges (miles)'),
                yaxis=dict(tickformat=".2%"),
                margin=dict(t=25)
            )
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
        
        with st.container():
            
            for par in info_2:
                st.markdown(par)
        
        with st.container(border=True):
            
            st.subheader('Percent within 1 standard deviation (by gender)', 
                         divider=False)
        
            fig = go.Figure()
            
            for pred in preds:
                fig.add_trace(
                    go.Bar(
                    x=data_mean.length_range,
                    y=data_mean[pred+'_in_target'], 
                    marker_color=col_color[pred+'_pred'],
                    name=pred
                    )
                )
            
            fig.update_layout(
                barmode='group', 
                bargroupgap=0.1,
                xaxis_tickangle=-45,
                xaxis_title=dict(text='Length ranges (miles)'),
                yaxis=dict(tickformat=".2%"),
                margin=dict(t=25)
            )
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
        
        with st.container():
            
            for par in info_3:
                st.markdown(par)
                
        with st.container():
            
            st.subheader('Summary', 
                         divider='gray')
            
            for par in info_4:
                st.markdown(par)
        
        with st.container():
            
            with st.expander('Further details about runner rank and target times'):
                
                for par in usu_notes:
                    st.markdown(par)
        
        with st.container():
            
            with st.expander('Further details about dataset and test set'):
                
                for par in data_notes:
                    st.markdown(par)
    
    with tab2:
        
        with st.container(border=True):
    
            st.header(f'{name_only}', divider='gray')
            st.subheader(f'{dist_only} miles, N = {ppt_total} participants')
                
        with st.container(border=True):
    
            st.subheader('Accuracy and error metrics', 
                         divider='gray')
            
            for col, pred in zip(st.columns(3), preds):
                
                with col:
                    st.metric(
                        label= pred+' \u00B11 std dev', 
                        value = in_tar[pred+'_in_target']
                        )
                    st.metric(
                        label = pred+' mean AE',
                        value = mean_err[pred+'_abs_err']
                        )
                    st.metric(
                        label=pred+' median AE', 
                        value = med_err[pred+'_abs_err']
                        )
        
        with st.container():
            
            with st.expander('Notes'):
                
                notes = '''
                - All time measurements are in hours
                - AE is Absolute Error
                - Standard deviation is calculated by gender
                '''
                st.markdown(notes)
                    
        with st.container():
            
            with st.expander('Race metrics'):
                
                m = len(SD)
                i = 0
                
                for col in st.columns(m):
                    with col:
                        st.metric(
                            label=f'{SD.index[i]} std dev',
                            value = SD[i]
                            )
                        st.metric(
                            label=f'{ppt_count.index[i]} count',
                            value = ppt_count[i]
                            )
                        st.metric(
                            label=f'{med_times.index[i]} median time',
                            value = med_times[i]
                            )
                        st.metric(
                            label=f'{winning_times.index[i]} win time',
                            value = winning_times[i]
                            )
                        i += 1
        
        with st.container(border=True):
            
            st.subheader('Prediction and result distributions', 
                         divider=False)
            
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
                    marker_color='rgba(0, 0, 0, 0)',
                    marker_line_color='black',
                    marker_line_width=1.5,
                    bingroup=1
                    )
                )
            fig.update_layout(
                barmode='overlay', 
                margin=dict(t=20), 
                xaxis_title=dict(text='Time (hours)')
                )
            fig.update_yaxes(range = [0, y_max])
            fig.update_xaxes(range = [x_min-0.02, x_max+0.02])
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
            
        with st.container(border=True):
            
            st.subheader('Signed error violin plots', divider=False)
                
            fig = go.Figure()
            
            for pred in reversed(preds):
                fig.add_trace(
                    go.Violin(
                    x=df[pred+'_err'], 
                    marker_symbol='line-ns-open', 
                    marker_color=col_color[pred+'_pred'],
                    marker_size=18,
                    points='all',
                    jitter=0,
                    box_visible=True,
                    fillcolor=col_color[pred+'_pred'],
                    line_color='black',
                    pointpos=-1.7,
                    name=pred
                    )
                )
                
            fig.update_traces(orientation='h')
            fig.update_layout(
                xaxis_title=dict(text='Prediction minus result (hours)'),
                margin=dict(t=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
            
        with st.container(border=True):
            
            st.subheader('Absolute error box plots', divider=False)
                
            fig = go.Figure()
            
            for pred in reversed(preds):
                fig.add_trace(
                    go.Box(
                    x=df[pred+'_abs_err'], 
                    boxmean=True,
                    marker_symbol='line-ns-open', 
                    marker_color=col_color[pred+'_pred'],
                    marker_size=18,
                    boxpoints='all',
                    jitter=0,
                    fillcolor=col_color[pred+'_pred'],
                    line_color='black',
                    pointpos=-1.7,
                    name=pred
                    )
                )
            
            fig.update_traces(orientation='h')
            fig.update_layout(
                xaxis_title=dict(text='Absolute error (hours)'),
                margin=dict(t=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
            
    # with tab3:
        
    #     with st.container(border=True):

    #         st.header(f'{name_only}', divider='gray')
    #         st.subheader(f'{dist_only} miles, N = {ppt_total} participants')
                
    #     with st.container(border=True):
    
    #         st.subheader('Race metrics', 
    #                      divider='gray')
            
    #         m = len(SD)
    #         i = 0
            
    #         for col in st.columns(m):
    #             with col:
    #                 st.metric(
    #                     label=f'{SD.index[i]} std dev',
    #                     value = SD[i]
    #                     )
    #                 st.metric(
    #                     label=f'{ppt_count.index[i]} count',
    #                     value = ppt_count[i]
    #                     )
    #                 st.metric(
    #                     label=f'{med_times.index[i]} median time',
    #                     value = med_times[i]
    #                     )
    #                 st.metric(
    #                     label=f'{winning_times.index[i]} win time',
    #                     value = winning_times[i]
    #                     )
    #                 i += 1
                    
    #     with st.container():
            
    #         with st.expander('Note'):
                
    #             note = '''
    #             All time measurements are in hours
    #             '''
    #             st.markdown(note)
    
    with tab4:
        
        with st.container(border=True):

            st.header(f'{name_only}', divider='gray')
            st.subheader(f'{dist_only} miles')
            
        with st.container(border=True):
            
            st.subheader('Median and Winning Times by Date')
            
            fig = go.Figure()
            
            for gen, meds in gen_meds:
            
                fig.add_trace(go.Scatter(
                    x=meds.date, 
                    y=meds.med_time, 
                    name=gen+' median',
                    line = dict(color=history_color[gen])))
            
            for gen, mins in gen_mins:
            
                fig.add_trace(go.Scatter(
                    x=mins.date, 
                    y=mins.min_time, 
                    name=gen+' median',
                    line = dict(color=history_color[gen],
                                dash='dot')))
            fig.update_layout(margin=dict(t=20),
                              xaxis_title='Date',
                              yaxis_title='Time (hours)')
            fig.update_layout(showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
