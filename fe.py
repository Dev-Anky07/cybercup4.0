'''import streamlit as st

# Define the city-district-intersection relationships

def main():
    st.set_page_config(page_title="Traffic Management System", page_icon="🚦",layout="wide")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main", "Stats", "About"])

    if page == "Main Feed":
        main_page()
    elif page == "Stats":
        stats_page()
    elif page == "About":
        about_page()

def stats_page():
    st.title("Traffic Statistics")
    st.write("This page will contain graphical representations of traffic data.")
    
    # Placeholder for graphs
    st.write("Graphs and insights will be displayed here.")

def about_page():
    st.title("About Our Traffic Management System")

    st.markdown("""
    ## Smart Traffic Management System

    Our Smart Traffic Management System is designed to optimize traffic flow and reduce congestion in urban areas. By leveraging advanced technologies and data analytics, we provide real-time insights and predictive capabilities to traffic management authorities.

    ### Key Features:
    1. **Real-time Monitoring**: Our system provides up-to-the-minute information on traffic conditions across the city.
    2. **Predictive Analytics**: Using historical data and machine learning algorithms, we forecast traffic patterns to preemptively address potential congestion.
    3. **Adaptive Signal Control**: Traffic signals are dynamically adjusted based on current traffic conditions to optimize flow.
    4. **Incident Detection**: Quick identification and response to traffic incidents, reducing their impact on overall traffic.
    5. **Data Visualization**: Intuitive dashboards and reports for easy interpretation of complex traffic data.

    ### How It Works:
    1. Data Collection: Traffic sensors and cameras collect real-time data from various intersections.
    2. Data Processing: Our advanced algorithms process the collected data to extract meaningful insights.
    3. Analysis & Prediction: The system analyzes current conditions and predicts future traffic patterns.
    4. Optimization: Based on the analysis, traffic signals are optimized, and recommendations are made to traffic authorities.
    5. Continuous Learning: The system continuously learns from new data, improving its predictions and recommendations over time.

    By implementing our Smart Traffic Management System, cities can expect reduced congestion, lower emissions, and improved overall urban mobility.
    """)

city_data = {
    "NCT of Delhi": {
        "New Delhi": ["Intersection X", "Intersection Y"],
        "Central Delhi": ["Intersection X", "Intersection Y"],
        "South Delhi": ["Intersection X", "Intersection Y"],
        "South East Delhi": ["Intersection X", "Intersection Y"],
        "South West Delhi": ["Intersection X", "Intersection Y"],
        "North East Delhi":["Intersection X", "Intersection Y"],
        "North West Delhi":["Intersection X", "Intersection Y"],
        "North Delhi": ["Intersection X", "Intersection Y"],
        "West Delhi":["Intersection X", "Intersection Y"],
        "East Delhi":["Intersection X", "Intersection Y"],
        "Shahdra": ["Intersection X", "Intersection Y"]
    },
    "Mumbai": {
        "District 1": ["Intersection Y"],
        "District 2": ["Intersection Y"]
    },
    "Bengaluru": {
        "District 3": ["Intersection Z"]
    },
    "Kolkata":{
        "Momta": ["Intersection X", "Intersection Y"],
    }
}

def main_page():
    st.title("Traffic Management System")

    # City dropdown
    city = st.selectbox("Select City", list(city_data.keys()))

    # District dropdown (reactive)
    districts = list(city_data[city].keys())
    district = st.selectbox("Select District", districts)

    # Intersection dropdown (reactive)
    intersections = city_data[city][district]
    intersection = st.selectbox("Select Intersection", intersections)

    # Search button
    if st.button("Search"):
        st.write(f"Feed for {city}, {district}, {intersection} loading :")
        st.spinner("Loading....🤔")

if __name__ == "__main__":
    main()'''

import streamlit as st
st.set_page_config(page_title="Traffic Management System", page_icon="🚦",layout="wide",initial_sidebar_state="expanded")
# Define the city-district-intersection relationships
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main", "Stats", "About"])

    if page == "Main":
        main_page()
    elif page == "Stats":
        stats_page()
    elif page == "About":
        about_page()

def main_page():
    st.title("Traffic Management System")

    # City dropdown
    city = st.selectbox("Select City", list(city_data.keys()))

    # District dropdown (reactive)
    districts = list(city_data[city].keys())
    district = st.selectbox("Select District", districts)

    # Check if the district has tehsils
    if isinstance(city_data[city][district], dict):
        # Tehsil dropdown (reactive)
        tehsils = list(city_data[city][district].keys())
        tehsil = st.selectbox("Select Tehsil", tehsils)

        # Intersection dropdown (reactive)
        intersections = city_data[city][district][tehsil]
        intersection = st.selectbox("Select Intersection", intersections)
    else:
        # If no tehsils, directly show intersections
        intersections = city_data[city][district]
        intersection = st.selectbox("Select Intersection", intersections)

    # Search button
    if st.button("Search"):
        if 'tehsil' in locals():
            st.write(f"Feed for {city}, {district}, {tehsil}, {intersection} loading:")
        else:
            st.write(f"Feed for {city}, {district}, {intersection} loading:")
        st.spinner("Loading....🤔")

def stats_page():
    st.title("Traffic Statistics")
    st.write("This page will contain graphical representations of traffic data.")
    
    # Placeholder for graphs
    st.write("Graphs and insights will be displayed here.")

def about_page():
    st.title("About Our Traffic Management System")

    st.markdown("""
    ## Smart Traffic Management System

    Our Smart Traffic Management System is designed to optimize traffic flow and reduce congestion in urban areas. By leveraging advanced technologies and data analytics, we provide real-time insights and predictive capabilities to traffic management authorities.

    ### Key Features:
    1. **Real-time Monitoring**: Our system provides up-to-the-minute information on traffic conditions across the city.
    2. **Predictive Analytics**: Using historical data and machine learning algorithms, we forecast traffic patterns to preemptively address potential congestion.
    3. **Adaptive Signal Control**: Traffic signals are dynamically adjusted based on current traffic conditions to optimize flow.
    4. **Incident Detection**: Quick identification and response to traffic incidents, reducing their impact on overall traffic.
    5. **Data Visualization**: Intuitive dashboards and reports for easy interpretation of complex traffic data.

    ### How It Works:
    1. Data Collection: Traffic sensors and cameras collect real-time data from various intersections.
    2. Data Processing: Our advanced algorithms process the collected data to extract meaningful insights.
    3. Analysis & Prediction: The system analyzes current conditions and predicts future traffic patterns.
    4. Optimization: Based on the analysis, traffic signals are optimized, and recommendations are made to traffic authorities.
    5. Continuous Learning: The system continuously learns from new data, improving its predictions and recommendations over time.

    By implementing our Smart Traffic Management System, cities can expect reduced congestion, lower emissions, and improved overall urban mobility.
    """)

city_data = {
    "NCT of Delhi": {
        "New Delhi": {
            "Parliament Street":["Intersection X", "Intersection Y"],
            "Connaught Place":["Intersection X", "Intersection Y"],
            "Chanakyapuri": ["Intersection X", "Intersection Y"]},
        "Central Delhi": {
            "Karol Bagh":["Intersection X", "Intersection Y"],
            "Pahar Ganj":["Intersection X", "Intersection Y"],
            "Darya Ganj":["Intersection X", "Intersection Y"]},
        "South Delhi": {
            "Defence Colony":["Intersection X", "Intersection Y"],
            "Hauz Khas":["Intersection X", "Intersection Y"],
            "Kalkaji":["Intersection X", "Intersection Y"]},
        "South East Delhi": ["Intersection X", "Intersection Y"],
        "South West Delhi": ["Intersection X", "Intersection Y"],
        "North East Delhi":["Intersection X", "Intersection Y"],
        "North West Delhi":["Intersection X", "Intersection Y"],
        "North Delhi": ["Intersection X", "Intersection Y"],
        "West Delhi":{
            "Punjabi Bagh":["Intersection X", "Intersection Y"],
            "Patel Nagar":["Intersection X", "Intersection Y"],
            "Rajouri Garden":["Intersection X", "Intersection Y"] },
        "East Delhi":{
            "Gandhi Nagar":["Intersection X", "Intersection Y"],
            "Vivek Vihar": ["Intersection X", "Intersection Y"],
            "Preet Vihar":["Intersection X", "Intersection Y"]},
        "Shahdra": ["Intersection X", "Intersection Y"]
    },
    "Mumbai": {
        "District 1": ["Intersection Y"],
        "District 2": ["Intersection Y"]
    },
    "Bengaluru": {
        "District 3": ["Intersection Z"]
    },
    "Kolkata":{
        "Momta": ["Intersection X", "Intersection Y"],
    }
}

if __name__ == "__main__":
    main()
