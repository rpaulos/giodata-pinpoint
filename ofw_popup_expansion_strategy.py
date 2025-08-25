import pandas as pd
import folium
import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium
import requests
import folium
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

def show_filters(df, column, continent=None):
    if column == 'Country' and continent and continent != 'All':
        filter_list = list(df[df['Continent'] == continent][column].unique())
    else:
        filter_list = list(df[column].unique())

    filter_list.sort()

    if column == 'Continent':
        return st.selectbox(column, ['All'] + filter_list)
    elif column == 'Country':
        return st.selectbox(column, ['All'] + filter_list, index=0)
    elif column == 'Type':
        return st.selectbox(column, ['Combined'] + filter_list, index=0)
    else:
        return st.selectbox(column, filter_list, index=2)
    
def apply_filters(df, continent, country, type_, year):
    filtered_df = df.copy()

    # Continent filter
    if continent != "All":
        filtered_df = filtered_df[filtered_df["Continent"] == continent]

    # Country filter
    if country != "All":
        filtered_df = filtered_df[filtered_df["Country"] == country]

    # Year filter
    if year != "All":
        filtered_df = filtered_df[filtered_df["Year"] == year]

    # Type filter
    if type_ == "Land-based":
        filtered_df = filtered_df[filtered_df["Type"] == "Land-based"]

    elif type_ == "Sea-based":
        filtered_df = filtered_df[filtered_df["Type"] == "Sea-based"]

    elif type_ == "Combined":
        filtered_df = (
            filtered_df.groupby(["Continent", "Country"], as_index=False)["Value"].sum()
        )
        filtered_df["Type"] = "Combined"
        return filtered_df  # already aggregated

    # Final groupby (always apply after filtering)
    filtered_df = (
        filtered_df.groupby(["Continent", "Country", "Type"], as_index=False)["Value"].sum()
    )

    return filtered_df

def calculate_ofw_remittances(df, continent, country, type_, year):
    # Apply filters for the selected year and previous year
    filtered_df = apply_filters(df, continent, country, type_, year)
    prev_df = apply_filters(df, continent, country, type_, year - 1)

    # Calculate current total and delta
    current_total = filtered_df["Value"].sum()
    prev_total = prev_df["Value"].sum()
    delta = f"{current_total - prev_total:,.2f}"

    return filtered_df, current_total, delta

def show_top_countries_by_remittances(filtered_df, top_n=10):
    # Aggregate by country and get top N
    df_selected_year_sorted = (
        filtered_df.groupby("Country", as_index=False)["Value"].sum()
        .sort_values("Value", ascending=False)
        .head(top_n)
    )

    # Display in Streamlit
    st.markdown("##### Top Countries by Remittances")
    st.dataframe(
        df_selected_year_sorted,
        column_order=("Country", "Value"),
        hide_index=True,
        width=None,
        height=399,
        column_config={
            "Country": st.column_config.TextColumn("Country"),
            "Value": st.column_config.ProgressColumn(
                "Total Remittances",
                format="dollar",
                min_value=0,
                max_value=max(df_selected_year_sorted.Value),
            ),
        },
    )

    return df_selected_year_sorted

def show_remittance_pie_chart():

    df = pd.read_excel('./data/Cash Remittances by Mode.xlsx', sheet_name='Mode of Remittance')

    # Pie Chart
    fig = px.pie(
        df,
        values="Total Cash Remittance",
        names="Mode of Remittance",
        # title="Mode of Remittance Distribution",
        hole=0.3  # 0 for pie, >0 for donut
    )

     # Remove legend & show labels inside
    fig.update_traces(textposition="inside", textinfo="label+percent")
    fig.update_layout(showlegend=False)

    # Display
    st.plotly_chart(fig, use_container_width=True)

def show_region_barchart():

    df = pd.read_excel('./data/Regional OFW Employment Statistics.xlsx', sheet_name='Regional OFW Employment')

    # Melt into long format for stacked bars
    df_melted = df.melt(id_vars="Region", value_vars=["Landbased", "Seabased"],
                        var_name="Type", value_name="Count")

    # Horizontal stacked bar chart
    fig = px.bar(
        df_melted,
        x="Count",
        y="Region",
        color="Type",
        orientation="h",
        # title="OFW Deployment by Region (Landbased vs Seabased)",
        text="Count",
        color_discrete_map={"Landbased": "#2e7d32", "Seabased": "#66bb6a"}
    )

    # Formatting
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        barmode="stack"
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_ofw_remittance_map(filtered_df, continent=None):
    """
    Plots OFW remittance data on a world map with choropleth and tooltips.
    
    Parameters:
        filtered_df (pd.DataFrame): Must have columns ['Country', 'Value'].
        continent (str, optional): Continent name for auto-zoom. Default is None.
    """
    # Load GeoJSON
    geojson_url = 'https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json'
    response = requests.get(geojson_url)
    geojson = response.json()

    # Initialize map
    map_ = folium.Map(location=[0, 0], zoom_start=2, scrollWheelZoom=False, tiles='CartoDB positron')
    
    # Add Choropleth
    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=filtered_df,
        columns=['Country', 'Value'],
        key_on='feature.properties.name',
        fill_color='Greens',
        fill_opacity=0.7,
        line_opacity=0.2,
        highlight=True,
        legend_name='OFW Cash Remittances'
    ).add_to(map_)

    # Index DF for lookup
    df_indexed = filtered_df.set_index('Country')

    # Add properties for tooltips
    for feature in geojson['features']:
        country_name = feature['properties']['name']
        if country_name in df_indexed.index:
            feature['properties']['country'] = f"Country: {country_name}"
            feature['properties']['value'] = f"Remittances: {df_indexed.loc[country_name, 'Value']:,.2f}"
        else:
            feature['properties']['country'] = f"Country: {country_name}"
            feature['properties']['value'] = "Value: N/A"

    # Add custom tooltip
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['country', 'value'], labels=False)
    )

    # Auto-zoom to continent
    if continent:
        continents = {
            "Africa": [
                "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon","Cabo Verde","Central African Republic",
                "Chad","Republic of the Congo","Democratic Republic of the Congo","Côte d'Ivoire","Djibouti","Equatorial Guinea",
                "Eritrea","Ethiopia","Eswatini","Gabon","Gambia","Ghana","Guinea","Guinea Bissau","Kenya","Lesotho","Liberia",
                "Madagascar","Malawi","Mali","Mauritania","Mauritius","Mayotte","Morocco","Mozambique","Namibia","Niger",
                "Nigeria","Réunion","Rwanda","Saint Helena","Sao Tome and Principe","Senegal","Seychelles","Sierra Leone",
                "Somalia","South Africa","South Sudan","Sudan","United Republic of Tanzania","Togo","Tunisia","Uganda",
                "Western Sahara","Zambia","Zimbabwe"
            ],

            "Americas": [
                "Colombia", "Canada"
            ],

            "Asia": [
               "Afghanistan", "Armenia", "Japan", "Kazakhstan", "North Korea", "South Korea", "Kyrgyzstan", "Macao, SAR China", "Tajikistan", "Timor-Leste", "Turkmenistan", "Uzbekistan", "Brunei Darussalam", "Thailand", "Vietnam", "Bahrain", "Iran", "Iraq", "Palestinian Territory", "Qatar", "Saudi Arabia", "Syria", "United Arab Emirates", "Yemen"
            ],

            "Asia - ASEAN": [
                "Brunei Darussalam","Cambodia","Indonesia","Laos","Malaysia","Myanmar", "Singapore", "Thailand", "Vietnam"
            ],

            "Asia - Middle East": [
                "Bahrain","Egypt","Iran","Iraq","Israel","Jordan","Kuwait","Lebanon","Libya","Oman","Palestinian Territory","Qatar","Saudi Arabia","Syria","United Arab Emirates","Yemen"
            ],

            "Europe ": [
                "Albania","Andorra","Belarus","Bonaire, Sint Eustatius and Saba","Bosnia and Herzegovina","British Indian Ocean Territory","Faroe Islands","Georgia","Gibraltar","Greenland","Guernsey","Holy See (Vatican City State)","Iceland","Isle of Man","Liechtenstein","Macedonia","Martinique","Moldova","Monaco","Montenegro","Montserrat","Norway","San Marino","Republic of Serbia","South Georgia and the South Sandwich Islands","Svalbard and Jan Mayen Islands","Switzerland","Turkey","Ukraine","United Kingdom"
            ],

            "Europe - European Union": [
                "Aland Islands","Austria","Belgium","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden"
            ], 

            "Oceania": [
                "Australia", "New Zealand", "Papua New Guinea"
            ]
        }

        if continent in continents:
            bounds = []
            for feature in geojson['features']:
                if feature['properties']['name'] in continents[continent]:
                    geometry = feature['geometry']
                    if geometry['type'] == "Polygon":
                        bounds.extend(geometry['coordinates'][0])
                    elif geometry['type'] == "MultiPolygon":
                        for poly in geometry['coordinates']:
                            bounds.extend(poly[0])
            # Set map bounds
            if bounds:
                lats = [coord[1] for coord in bounds]
                lons = [coord[0] for coord in bounds]
                map_.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # Display map in Streamlit
    st_folium(map_, height=585, width=1300)

def sarimax_forecast(df_group, steps=12):
    # Combine Year + Month into a datetime column
    df_group['Date'] = pd.to_datetime(
        df_group['Year'].astype(str) + '-' + df_group['Month'], 
        format='%Y-%b'
    )
    df_group = df_group.set_index('Date').asfreq('MS')

    # Build SARIMAX model
    model = SARIMAX(df_group['Value'], order=(1,1,1), seasonal_order=(1,1,1,12))
    fit = model.fit(disp=False)

    # Forecast
    forecast = fit.get_forecast(steps=steps)
    forecast_df = forecast.summary_frame()
    forecast_df['Date'] = forecast_df.index
    forecast_df.rename(columns={'mean':'Forecast'}, inplace=True)

    # Add metadata
    forecast_df['Country'] = df_group['Country'].iloc[0]
    forecast_df['Type'] = df_group['Type'].iloc[0]

    return forecast_df[['Date','Forecast','mean_ci_lower','mean_ci_upper','Country','Type']]

def forecast_and_plot(df, countries_selected, types_selected, horizon):
    """
    Generate SARIMAX forecasts for selected countries/types and plot historical + forecasted values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ['Country','Type','Year','Month','Value']
    countries_selected : list
        List of countries to filter
    types_selected : list
        List of types to filter
    horizon : int
        Number of forecast steps
    """

    # Filter input df
    filtered_df_forecast = df[
        df['Country'].isin(countries_selected) &
        df['Type'].isin(types_selected)
    ].copy()

    # Generate forecasts for all selected countries/types
    forecast_list = []
    for (country, type_), group in filtered_df_forecast.groupby(['Country', 'Type']):
        forecast_list.append(sarimax_forecast(group, steps=horizon))  # calls directly
    all_forecasts = pd.concat(forecast_list)

    # Prepare historical data
    filtered_df_forecast['Date'] = pd.to_datetime(
        filtered_df_forecast['Year'].astype(str) + '-' + filtered_df_forecast['Month'],
        format='%Y-%b'
    )

    historical = filtered_df_forecast[['Date', 'Value', 'Country', 'Type']].copy()
    historical['Source'] = "Historical"
    all_forecasts['Source'] = "Forecast"

    # Merge historical + forecast
    combined_df = pd.merge(
        historical,
        all_forecasts,
        on=['Date', 'Country', 'Type'],
        how='outer'
    )

    # Plot per group
    for (country, type_), group in combined_df.groupby(['Country','Type']):
        fig, ax = plt.subplots(figsize=(10, 3))

        # Historical
        hist = group.dropna(subset=['Value'])
        ax.plot(hist['Date'], hist['Value'], label="Historical", marker='o')

        # Forecast
        fore = group.dropna(subset=['Forecast'])
        ax.plot(fore['Date'], fore['Forecast'], label="Forecast", linestyle='--', marker='x')

        if not fore.empty:
            ax.fill_between(
                fore['Date'],
                fore['mean_ci_lower'],
                fore['mean_ci_upper'],
                color='gray', alpha=0.3, label="Confidence Interval"
            )

        ax.set_title(f"{country} - {type_}")
        ax.legend()
        st.pyplot(fig)

    return filtered_df_forecast, all_forecasts

def agentic_ai_analysis(
    continent, countries_selected, types_selected, year,
    current_total, top_countries, horizon,
    filtered_df_forecast, all_forecasts,
    openai_api_key
):
    """
    Run AI analysis on historical + forecasted remittance data and display results in Streamlit chat.

    Parameters
    ----------
    continent : str
        Selected continent
    countries_selected : list
        Selected countries
    types_selected : list
        Selected types
    year : int
        Selected year
    current_total : float
        Total remittances for current selection
    top_countries : pd.DataFrame
        DataFrame of top countries by remittances
    horizon : int
        Forecast horizon
    filtered_df_forecast : pd.DataFrame
        Filtered historical dataframe
    all_forecasts : pd.DataFrame
        Forecast results dataframe
    openai_api_key : str
        OpenAI API key
    """

    st.markdown(
        """<hr style="margin-top:2px;margin-bottom:10px;">""",
        unsafe_allow_html=True
    )

    # --- Prepare Historical Summary ---
    historical_summary = filtered_df_forecast[['Country', 'Type', 'Date', 'Value']].copy()
    historical_summary.rename(columns={'Value': 'Remittances'}, inplace=True)
    historical_summary['Source'] = 'Historical'

    # --- Prepare Forecast Summary ---
    forecast_summary = all_forecasts[['Country', 'Type', 'Date', 'Forecast']].copy()
    forecast_summary.rename(columns={'Forecast': 'Remittances'}, inplace=True)
    forecast_summary['Source'] = 'Forecast'

    # --- Combine ---
    summary_df = pd.concat([historical_summary, forecast_summary], ignore_index=True)
    summary_df = summary_df.sort_values(by=['Country', 'Type', 'Date']).reset_index(drop=True)

    # --- Prompt ---
    prompt = f"""
    You are a strategic financial analyst for BPI Bank.

    Context Provided:
    - Selected Continent: {continent}
    - Selected Country(s): {', '.join(countries_selected)}
    - Selected Type(s): {', '.join(types_selected)}
    - Selected Year: {year}
    - Total OFW Remittances: ${current_total:,.2f}
    - Top Countries by Remittances: {top_countries.to_markdown(index=False)}
    - Historical and Forecasted Remittances for next {horizon} months: {summary_df.to_markdown(index=False)}
    - Pop-Up Strategy Definition: A pop-up strategy refers to setting up a temporary, small-scale location, like a shop or branch that appears for a short time in a busy area. It helps businesses test markets, reach more people, and provide services in spots without a permanent presence. In banking, this means creating pop-up micro-branches offering services like ATM access, account sign-ups, or quick customer help, all in places like malls or stores. Key Elements: Temporary setup: Fast to open/close in high-traffic areas; Limited services: ATM, account opening, loan inquiries, quick assistance; Tech-driven tools: Tablets, kiosks, smart ATMs, POS devices; Low-cost, flexible: No long leases, ideal for testing markets

    Your Task:
    Analyze the historical and forecasted cash remittance data to identify patterns, trends, and insights. Then, evaluate whether a Pop-Up Expansion Strategy is feasible for the selected country(s) and type(s).

    Output Requirements:
    - Present insights in clear, concise bullet points.
    - Highlight key patterns and anomalies (seasonality, growth/decline trends, volatility).
    - Identify opportunities and risks per country/type.
    - Recommend whether a pop-up branch is viable, and if so, suggest priority locations, timing, and focus (landbased vs seabased).
    - Provide actionable, data-driven suggestions to guide decision-making.
    """

    # --- LLM Setup ---
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0
    )

    # --- Conversation State ---
    if "popup_messages" not in st.session_state:
        st.session_state.popup_messages = []

    if "analysis_done" not in st.session_state:
        response = llm([
            SystemMessage(content="You are a strategic financial analyst for BPI Bank."),
            HumanMessage(content=prompt)
        ])
        st.session_state.analysis_done = response.content
        st.session_state.popup_messages.append({"role": "assistant", "content": response.content})

    # --- Display Messages ---
    for msg in st.session_state.popup_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- User Chat Input ---
    if user_input := st.chat_input("Ask about the strategy or insights..."):
        st.session_state.popup_messages.append({"role": "user", "content": user_input})

        messages = [SystemMessage(content="You are a strategic financial analyst for BPI Bank.")] + [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        response = llm(messages)
        st.session_state.popup_messages.append({"role": "assistant", "content": response.content})

        with st.chat_message("assistant"):
            st.markdown(response.content)

