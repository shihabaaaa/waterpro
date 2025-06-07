import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

#load the csv file for water accumulation  shihab-step 1
df = pd.read_csv(r'h:\courstream\datasensors.csv')
#Convert 'times' to datetime timestamp     shihab-step 2
df['time'] = pd.to_datetime(df['time'], dayfirst=True)

# verify the table created and data types and convert   shihab-step 3
# print("Data loaded successfully.")          #verify data loaded
# print(df[['time', 'level']].head())         #verify data types

# Define Thresholds & Hysteresis  shihab-step 4
location_threshold = 42  # Example threshold value for location
def get_thresholds(location_threshold):
    hysteresis = 2  # Hysteresis value
    # Print the thresholds for debugging
    # print(f"Location threshold: {location_threshold}, Hysteresis: {hysteresis}")  # verify thresholds
    return {
        'low_enter': 42 + hysteresis,
        'low_exit': 42 - hysteresis,
        'medium_enter': 55 + hysteresis,
        'medium_exit': 55 - hysteresis,
        'high_enter': 69 + hysteresis,
        'high_exit': 69 - hysteresis
    }
# get_thresholds(location_threshold) # verify thresholds

# Function to Apply Event Classification with Hysteresis  shihab-step 5
def classify_events(df, threshold_base=location_threshold):
    thresholds = get_thresholds(threshold_base)
    current_state = "Normal"
    event_states = []

    for i, row in df.iterrows():
        level = row['level']

        if current_state == "Normal":
            if level >= thresholds['high_enter']:
                current_state = "High accumulation"
            elif level >= thresholds['medium_enter']:
                current_state = "Medium accumulation"
            elif level >= thresholds['low_enter']:
                current_state = "Low accumulation"
        elif current_state == "Low accumulation":
            if level < thresholds['low_exit']:
                current_state = "Normal"
            elif level >= thresholds['medium_enter']:
                current_state = "Medium accumulation"
            elif level >= thresholds['high_enter']:
                current_state = "High accumulation"
        elif current_state == "Medium accumulation":
            if level < thresholds['medium_exit']:
                current_state = "Low accumulation"
            elif level >= thresholds['high_enter']:
                current_state = "High accumulation"
        elif current_state == "High accumulation":
            if level < thresholds['high_exit']:
                current_state = "Medium accumulation"

        event_states.append(current_state)

    df['event_type'] = event_states
    return df
df = classify_events(df, location_threshold)  # Apply the classification function to the DataFrame


#### -------------------- verify event classification --------------------####
# Apply the classification function to the DataFrame  
# df = classify_events(df, location_threshold)
# Print the first few rows of the DataFrame to verify the results  
# print(df[['time', 'level', 'event_type']].head())  # verify event classification      
# Save the results to a new CSV file  
#df.to_csv(r'h:\courstream\wateraccumulation44.csv', index=False)   # export to csv file
#Print a message indicating that the file has been saved
#print("Water accumulation data has been saved to 'wateraccumulation44.csv'.")  # verify file saved
#### -------------------- verify event classification --------------------####

# Merge Nearby Events (1-point gap rule) shihab-step 6
def merge_events(df):
    events = []
    prev_type = None
    start_time = None
    values = []

    for i in range(len(df)):
        row = df.iloc[i]
        cur_type = row['event_type']

        if cur_type != "Normal":
            if prev_type != cur_type:
                # Close previous event
                if prev_type is not None:
                    end_time = df.iloc[i-1]['time']
                    duration = (end_time - start_time).total_seconds() / 60
                    avg_level = sum(values) / len(values)
                    events.append({
                        "Event type": prev_type,
                        "Start date-time": start_time,
                        "Ending date-time": end_time,
                        "Event duration (min)": round(duration, 2),
                        "Average level (cm)": round(avg_level, 2)
                    })
                # Start new event
                start_time = row['time']
                values = [row['level']]
            else:
                values.append(row['level'])

            prev_type = cur_type
        else:
            # If only one normal row between same event type, skip it
            if i+2 < len(df) and df.iloc[i+2]['event_type'] == prev_type:
                continue
            # Close event
            if prev_type is not None:
                end_time = df.iloc[i-1]['time']
                duration = (end_time - start_time).total_seconds() / 60
                avg_level = sum(values) / len(values)
                events.append({
                    "Event type": prev_type,
                    "Start date-time": start_time,
                    "Ending date-time": end_time,
                    "Event duration (min)": round(duration, 2),
                    "Average level (cm)": round(avg_level, 2)
                })
                prev_type = None
                values = []

    return pd.DataFrame(events)
events_df = merge_events(df)
# Print the first few rows of the merged DataFrame to verify the results
print(events_df.head())  # verify merged events



#### -------------------- verify event classification --------------------####

#""" shihab important step to apply this verify event classification you need to run the previous steps first
 #   go to line 68: df = classify_events(df, location_threshold) and apply this code
#"""
# Apply the merge function to the DataFrame
# df_merged = merge_events(df)
# Print the first few rows of the merged DataFrame to verify the results
# print(df_merged.head())  # verify merged events
# Save the merged results to a new CSV file
# df_merged.to_csv(r'h:\courstream\wateraccumulation44_merged.csv', index=False)  # export to csv file
# Print a message indicating that the file has been saved
# print("Water accumulation data has been saved to 'wateraccumulation44_merged.csv'.")  # verify file saved
# End of the script
#### -------------------- verify event classification --------------------####

## 6: Full Pipeline Execution shihab step 7
events_df.to_csv("water_accumulation_events2.csv", index=False)

################### visualizatoin #############################
#crate visualization with pandas and visualization shihab step-7
# Read file origianl and events
original_df = pd.read_csv(r'h:\courstream\datasensors.csv')
events_df = pd.read_csv(r'h:\courstream\water_accumulation_events2.csv')

#display 2 tab
tab1, tab2, tab3 = st.tabs(["Original Data", "Water Accumulation Events","Map Data"])
with tab1:
    st.markdown("<h6 style='text-align: center; color: blue;'>Original Sensors Information Viewer</h6>", unsafe_allow_html=True)
    #range slider
    range_slider = st.slider(label="select level", min_value=original_df['level'].min()
                             ,max_value=original_df['level'].max()
                             ,value=(original_df['level'].min(), original_df['level'].max()))
    #Display Table with filter
    # filtered_df = original_df[(original_df['level'] >= range_slider[0]) & (original_df['level'] <= range_slider[1])]
    # st.dataframe(filtered_df)
    #Display other way used plot
    plot_df = original_df[original_df['level'].between(range_slider[0], range_slider[1])]    
    # create Column
    col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")
    # Calculate Avarage for parameters
    avrage_level = round(plot_df['level'].mean(),2)
    avrage_heiht = round(plot_df['air_height'].mean(),2)
    avrage_temp = round(plot_df['temperature'].mean(),2)
    avrage_rsrp = round(plot_df['rsrp'].mean(),2)
    # Column 1
    with col1:
        st.metric("Avarage Level", avrage_level)
    # Column 2
    with col2:
        st.metric("Avarage Height", avrage_heiht)
    # Column 3
    with col3:
        st.metric("Avarage Temperature", avrage_temp)
    # Column 4
    with col4:
        st.metric("Avarage Rsrp", avrage_rsrp)
    # Optional: line chart
    st.line_chart(plot_df[['time', 'level']].set_index('time'))      
    # Add the Scatterplot
    scatter_title= 'Water Level Over Time by Accumulation Event'
    scatter_plot = px.scatter(data_frame=plot_df,x='temperature', y='level', color='message_type'
                              , title= scatter_title
                              ,hover_data=['sensor_id', 'project_id', 'infrastructure_id', 'latitude', 'longitude'])
    # Display the scatter plot in Streamlit
    # scatter template
    px.scatter(template='plotly_dark')
    st.plotly_chart(scatter_plot, use_container_width=True)
    # Dipaly Table
    st.write(plot_df)
    
with tab2:
    st.markdown("<h6 style='text-align: center; color: blue;'>Water Accumulation Event Viewer</h6>", unsafe_allow_html=True)
    # add dropdown multiselect
    unique_event_types = events_df['Event type'].unique()
    
    selected_event_type = st.multiselect("Select Event Type", unique_event_types)
    st.dataframe(events_df[events_df['Event type'].isin(selected_event_type)])
    
    # add dropdown on select
    # selected_event_type = st.selectbox("Select Event Type", unique_event_types)
    # st.dataframe(events_df[events_df['Event type'] == selected_event_type])
    
    # Add two columns
    # Add two columns
    col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
    with col1:
        # Display the scatter plot in Streamlit
        # scatter template
        scatter_title2 = "Water Accumulation Event Viewer"
        plot_df2 = events_df[events_df['Event type'].isin(selected_event_type)]

        scatter_plot = px.scatter(data_frame=plot_df2,x='Event type', y='Average level (cm)'
                                    , color='Event type',title= scatter_title2)
        st.plotly_chart(scatter_plot, use_container_width=True)

    with col2:
        box_plot = px.violin(plot_df2, x='Event type', y='Average level (cm)',box=True, title=scatter_title2)
        box_plot.update_layout(xaxis_title="Event Type", yaxis_title="Average Level (cm)")
        st.plotly_chart(box_plot, use_container_width=True)
    
with tab3:
    st.markdown("<h6 style='text-align: center; color: blue;'>Map Data</h6>", unsafe_allow_html=True)
    map_title = "Map Data of sensors"
    plot3 = px.scatter_mapbox(data_frame=original_df,lat="latitude",lon="longitude"
                              ,zoom=2,color="message_type",title=map_title,mapbox_style="carto-positron")
    st.plotly_chart(plot3, use_container_width=True)