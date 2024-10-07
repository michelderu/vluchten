import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from vinc import v_direct
import numpy as np

st.title("Hoi Mel!")

def create_normal_distribution(df, column):
    # Calculate mean and standard deviation of Heading
    mean_heading = df[column].mean()
    std_heading = df[column].std()

    # Create a new dataframe with a normal distribution of Heading
    num_samples = len(flight_data)
    normal_heading = np.round(np.random.normal(mean_heading, std_heading, num_samples))
    normal_distribution = pd.DataFrame(normal_heading, columns=[column])

    return normal_distribution


def find_nearest_airport(lat, lon):
    location = (lat, lon)
    # Calculate the distance from the location to each airport using the provided v_direct function
    distances = [(v_direct(location, (lat, lon)), name) for lat, lon, name in zip(airport_data['Latitude'], airport_data['Longitude'], airport_data['Name'])]
    distances = pd.DataFrame(distances, columns=['Distance', 'Name'])
    # Find the row with the lowest value in the 'Distance' column
    nearest_idx = distances['Distance'].idxmin()
    return distances.loc[nearest_idx, 'Name']

def load_airport_data():
    df = pd.read_csv('data/airports-extended.csv')
    # Fill missing values with 'Unknown'
    for column in ['City', 'IATA', 'ICAO']:
        df[column] = df[column].fillna('Unknown')
    # Replace '\N' with 'Unknown' in the IATA, ICAO and Timezone columns
    for column in ['IATA', 'ICAO', 'Timezone']:
        df[column] = df[column].replace('\\N', 'Unknown')
    df['Timezone'] = df['Timezone'].replace('\\N', 'Unknown')
    # Filter out user contributed data / also filter out airports with no IATA code to keep to public airports
    return df[(df["Source"] == 'OurAirports') & (df["IATA"] != 'Unknown')]

def load_schedule_data():
    df = pd.read_csv('data/schedule_airport.csv')
    # Fill missing values with 'Unknown'
    df['Org/Des'] = df['Org/Des'].fillna('Unknown')
    # Enrich the data with airport information / join on Org/Ds-ICAO
    df = df.merge(airport_data[['Name', 'City', 'Country', 'Latitude', 'Longitude', 'ICAO']], left_on='Org/Des', right_on='ICAO', how='left')
    return df

def load_flight_data(filename: str):
    df = pd.read_excel(filename)
    # Rename columns
    df = df.rename(columns={
        'Time (secs)': 'Time',
        '[3d Latitude]': 'Latitude',
        '[3d Longitude]': 'Longitude',
        '[3d Altitude M]': 'Altitude_M',
        '[3d Altitude Ft]': 'Altitude_F',
        '[3d Heading]': 'Heading',
        'TRUE AIRSPEED (derived)': 'Speed'
    })
    # Remove leading * from Speed column
    df['Speed'] = pd.to_numeric(df['Speed'].replace('^\\*', '', regex=True), errors='coerce')
    # Remove rows with None data
    df = df.dropna()
    # Enrich the first and last line with airport names based on a distance search with lat/lon against airport data
    first_row = df.iloc[0]
    last_row = df.iloc[-1]
    df.loc[df.index[0], 'Airport'] = find_nearest_airport(first_row['Latitude'], first_row['Longitude'])
    df.loc[df.index[-1], 'Airport'] = find_nearest_airport(last_row['Latitude'], last_row['Longitude'])
    return df

airport_data = load_airport_data()
schedule_data = load_schedule_data()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Airport Data", "Schedule Data", "Flight Data"])

with tab1:
    # Display statistical information
    st.write(f"There are {len(airport_data)} airports across {airport_data['Country'].nunique()} countries and {airport_data['City'].nunique()} cities.")
    st.write(f"The minimum altitude is {round(airport_data['Altitude'].min())} meters and the maximum is {round(airport_data['Altitude'].max())}. It's mean and median are {round(airport_data['Altitude'].mean())} and {round(airport_data['Altitude'].median())} respectively.")

    # Top 10 countries by number of airports
    st.subheader("Top 10 Countries by Number of Airports")
    country_counts = airport_data['Country'].value_counts().head(10)
    st.bar_chart(country_counts)

    # Show on map
    st.subheader("Map of All Airports")
    st.map(airport_data, latitude="Latitude", longitude="Longitude")

    # Show table
    st.subheader("Table of All Airports")
    st.dataframe(airport_data, use_container_width=True)

with tab2:
    st.header("Schedule Data")
    st.dataframe(schedule_data, use_container_width=True)

with tab3:
    # Create a dropdown to select the flight file
    flight_files = [
        'data/30Flight 1.xlsx',
        'data/30Flight 2.xlsx',
        'data/30Flight 3.xlsx',
        'data/30Flight 4.xlsx',
        'data/30Flight 5.xlsx',
        'data/30Flight 6.xlsx',
        'data/30Flight 7.xlsx'
    ]
    selected_flight = st.selectbox("Select a flight:", flight_files)

    # Load the selected flight data
    flight_data = load_flight_data(selected_flight)

    # Display statistical information
    #st.write(f"The flight is {round(flight_data['Time'].max())} seconds long and covers {round(flight_data['Distance'].sum())} meters.")
    st.subheader("Height and speed over time")
    st.write(f"The median altitude is {round(flight_data['Altitude_F'].median())} feet. Here's a chart of the altitude over time:")
    st.line_chart(flight_data['Altitude_F'], x_label="Time", y_label="Height (ft)")
    st.write(f"The median speed is {round(flight_data['Speed'].median())} knots. Here's a chart of the speed over time:")
    st.line_chart(flight_data['Speed'], x_label="Time", y_label="Speed (knots)")

    # Create a map centered on the mean of the flight path
    st.subheader("Map of Flight Path")
    total_seconds = round(flight_data['Time'].max())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    st.write(f"The total time of the flight is {days} days, {hours} hours and {minutes} minutes.")

    center_lat = flight_data['Latitude'].mean()
    center_lon = flight_data['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add scatter plot for flight path
    for idx, row in flight_data.iloc[1:-1].iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=2,
            color='blue',
            fill=True,
            fillColor='blue'
        ).add_to(m)

    # Add markers for start and end points
    start_point = flight_data.iloc[0][['Latitude', 'Longitude']].values.tolist()
    end_point = flight_data.iloc[-1][['Latitude', 'Longitude']].values.tolist()

    folium.Marker(
        start_point,
        popup=flight_data.loc[flight_data.index[0], 'Airport'],
        tooltip=flight_data.loc[flight_data.index[0], 'Airport'],
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    folium.Marker(
        end_point,
        popup=flight_data.loc[flight_data.index[-1], 'Airport'],
        tooltip=flight_data.loc[flight_data.index[-1], 'Airport'],
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

    # Fit the map to the flight path
    sw = flight_data[['Latitude', 'Longitude']].min().values.tolist()
    ne = flight_data[['Latitude', 'Longitude']].max().values.tolist()
    m.fit_bounds([sw, ne])

    # call to render Folium map in Streamlit
    st_data = st_folium(m, width=725)

    # Show normal distribution of heading
    st.subheader("Normal Distribution of Heading")
    st.dataframe(create_normal_distribution(flight_data, 'Heading'), use_container_width=True)

    st.subheader("Table of Flight Data")
    st.dataframe(flight_data, use_container_width=True)