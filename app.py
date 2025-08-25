import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from ofw_popup_expansion_strategy import *
from hero_product_mapping import *
from branch import agentic_ai_branch_analyzer

st.set_page_config(layout="wide", page_title="PinPoint")

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #F8F8F8;
        text-align: center;
        padding: 15px 0;
    }
            
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
            
</style>
""", unsafe_allow_html=True)

logo_image = "img/logo.png"
st.sidebar.image(logo_image)

with st.sidebar:
    selected = option_menu(None, ['Hero Product Mapping', 'Competitor Analysis', 'Pop-Up Strategy'],
                           icons=['coin', 'bank', 'globe'], default_index=0)
    st.write(""" #### Copyright © 2025 GioData Solutions. All rights reserved. """)

openai_api_key = st.secrets["api_keys"]["openai_key"]

if selected == 'Hero Product Mapping':
    @st.cache_data
    def load_wealth_data():
        try:
            df = pd.read_excel('data/Wealth Indicator.xlsx', sheet_name='City Municipality')
            return df
        except:
            st.error("Could not load Wealth Indicator.xlsx file")
            st.stop()

    @st.cache_data
    def load_shapefile(scope):
        try:
            if scope == 'Regional':
                return gpd.read_file('data/gadm41_PHL_shp/gadm41_PHL_1.shp')
            elif scope == 'Provincial':
                return gpd.read_file('data/gadm41_PHL_shp/gadm41_PHL_1.shp')
            else:
                return gpd.read_file('data/gadm41_PHL_shp/gadm41_PHL_2.shp')
        except:
            st.error(f"Could not load Philippines shapefile for {scope.lower()} level")
            st.stop()

    df = load_wealth_data()
    wealth_metrics = ['City/Municipality Total GDP', 'GDP Growth (%)', 'Poverty Rate (%)', 
                    'Annual LGU Income', 'Condominium', 'Retail Hubs', 'Developers', 
                    'Car Showrooms', 'International Schools', 'Hospitals', 
                    'Luxury Hotel Presence', 'Casinos']

    try:
        df = load_wealth_data()
        wealth_metrics = ['City/Municipality Total GDP', 'GDP Growth (%)', 'Poverty Rate (%)', 
                        'Annual LGU Income', 'Condominium', 'Retail Hubs', 'Developers', 
                        'Car Showrooms', 'International Schools', 'Hospitals', 
                        'Luxury Hotel Presence', 'Casinos']
    except:
        st.error("Could not load Wealth Indicator.xlsx file")
        st.stop()

    st.title('Hero Product Mapping')
    st.caption('Data Sources: Philippine Statistics Authority (PSA), Department of Trade and Industry (DTI)')


    # Get region-province mapping
    region_province_mapping = get_region_province_mapping(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Region filter
        region_options = ['All'] + list(region_province_mapping.keys())
        selected_region = st.selectbox('Select Region', region_options)

    with col2:
        # Province filter - updates based on selected region
        if selected_region == 'All':
            province_options = ['All'] + [province for provinces in region_province_mapping.values() for province in provinces]
        else:
            province_options = ['All'] + region_province_mapping[selected_region]
        
        selected_province = st.selectbox('Select Province', province_options)

    with col3:
        # City/Municipality filter - updates based on selected region and province
        city_options = get_city_options(df, selected_region, selected_province)
        selected_city = st.selectbox('Select City/Municipality', city_options)

    with col4:
        # Wealth indicator selection
        wealth_indicators = ['All'] + wealth_metrics
        indicator = st.selectbox('Wealth Indicator', wealth_indicators)

    # Determine scope and filter data based on selections
    scope = determine_scope_from_filters(selected_region, selected_province, selected_city)
    filtered_df = filter_data_by_all_selections(df, selected_region, selected_province, selected_city)


    gdf = load_shapefile(scope)
    if scope == 'Regional':
        shapefile_column = 'NAME_1'
        data_column = 'Region'
        zoom_level = 6
    elif scope == 'Provincial':
        shapefile_column = 'NAME_1'
        data_column = 'Province'
        zoom_level = 7
    else:
        shapefile_column = 'NAME_2'
        data_column = 'City'
        zoom_level = 8

    # Special handling when showing all provinces (when All regions selected)
    if selected_region == 'All' and selected_province == 'All' and selected_city == 'N/A' and scope == 'Provincial':
        # Process data at regional level first, then distribute to provinces
        regional_processed_df = process_wealth_data(filtered_df, 'Regional', indicator)
        data_column = 'Region'  # This will trigger the regional-to-provincial mapping in merge function
        processed_df = regional_processed_df
    elif selected_city == 'All' and scope == 'City':
        # When "All" cities are selected, show city-level data
        processed_df = process_wealth_data(filtered_df, 'City', indicator)
    else:
        processed_df = process_wealth_data(filtered_df, scope, indicator)

    col_left, col_right = st.columns([2, 6])

    with col_left:
        # Use the processed_df that was already calculated above
        if not processed_df.empty:
            if indicator == 'All':
                metric_name = "Total Wealth Score"
                display_column = 'Total_Wealth_Score'
            else:
                metric_name = indicator
                display_column = indicator

            df_grouped = processed_df.sort_values(display_column, ascending=False).head(10)

            # st.markdown(f'##### Top {scope}\'s by {metric_name}')
            st.markdown(f'##### Top Cities by {metric_name}')
            if not df_grouped.empty:
                display_df = df_grouped[[data_column, display_column]].copy()
                display_df.columns = [scope, 'Value']
                
                display_df['Formatted Value'] = display_df['Value'].apply(lambda x: format_large_values(x, metric_name))
                
                st.dataframe(
                    display_df[[scope, 'Formatted Value']],
                    hide_index=True,
                    width=None,
                    height=399,
                    column_config={
                        scope: st.column_config.TextColumn(scope),
                        "Formatted Value": st.column_config.TextColumn(
                            "Value"
                        ),
                    },
                )

            # st.markdown(f"##### MSME's Distribution")
            msmes_data = pd.read_excel('data/Wealth Indicator.xlsx', sheet_name='MSMEs')
            msmes_data = msmes_data[msmes_data['Region'] == selected_region]

            # Pie Chart
            fig = px.pie(
                msmes_data,
                values="MSMEs",
                names="City/Province",
                title="MSMEs Distribution",
                hole=0.3  # 0 for pie, >0 for donut
            )

            # Remove legend & show labels inside
            fig.update_traces(textposition="inside", textinfo="label+percent")
            fig.update_layout(showlegend=False)

            # Display
            st.plotly_chart(fig, use_container_width=True)


        elif selected_city == 'N/A' and selected_province != 'All':
            st.info("💡 Select 'All' or a specific city/municipality to view the heatmap for this province.")
        elif selected_city == 'N/A':
            st.info("💡 Please select a region, province, or city/municipality to view the heatmap.")

    with col_right:
        map_center = [12.8797, 121.7740]
        m = folium.Map(location=map_center, zoom_start=zoom_level, scrollWheelZoom=False, tiles='CartoDB positron')
        
        # Only render map data if we have processed data and not in N/A state
        if not processed_df.empty and selected_city != 'N/A':
            if indicator == 'All':
                map_data = processed_df[[data_column, 'Total_Wealth_Score']].copy()
                map_data.columns = [data_column, 'Value']
                metric_name = "Total Wealth Score"
            else:
                map_data = processed_df[[data_column, indicator]].copy()
                map_data.columns = [data_column, 'Value']
                metric_name = indicator
            
            # Merge with shapefile
            merged_gdf = merge_shapefile_data(gdf, map_data, shapefile_column, data_column, scope)
            ai_map_data = map_data
            if selected_province != 'All':
                merged_gdf = merged_gdf[merged_gdf['NAME_1'] == selected_province]
                
            if not merged_gdf.empty and 'Value' in merged_gdf.columns:
                merged_gdf = merged_gdf.dropna(subset=['Value'])
                
                if not merged_gdf.empty:
                    min_val = merged_gdf['Value'].min()
                    max_val = merged_gdf['Value'].max()

                      # === AUTO ZOOM TO REGION/PROVINCE/CITY ===
                    bounds = merged_gdf.total_bounds  # [minx, miny, maxx, maxy]
                    sw = [bounds[1], bounds[0]]  # south-west corner
                    ne = [bounds[3], bounds[2]]  # north-east corner
                    m.fit_bounds([sw, ne])  # <-- this makes it auto zoom!
                    
                    choropleth = folium.Choropleth(
                        geo_data=merged_gdf.__geo_interface__,
                        data=merged_gdf,
                        columns=[shapefile_column, 'Value'],
                        key_on=f'feature.properties.{shapefile_column}',
                        fill_color='YlOrRd',
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        highlight=True,
                        legend_name=f'{metric_name}',
                        nan_fill_color='lightgray',
                        nan_fill_opacity=0.3,
                    ).add_to(m)
                    
                    legend_html = f'''
                    <div style="position: fixed; 
                                bottom: 50px; left: 50px; width: 300px; height: 90px; 
                                background-color: white; border:2px solid grey; z-index:9999; 
                                font-size:12px; padding: 10px; border-radius: 5px;
                                box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                    <h4 style="margin-top:0; margin-bottom:8px; text-align:center; font-size:14px;">{metric_name}</h4>
                    
                    <!-- Color bar -->
                    <div style="display: flex; height: 20px; margin-bottom: 5px; border: 1px solid #ccc;">
                        <div style="background: #FFEDA0; flex: 1;"></div>
                        <div style="background: #FED976; flex: 1;"></div>
                        <div style="background: #FEB24C; flex: 1;"></div>
                        <div style="background: #FD8D3C; flex: 1;"></div>
                        <div style="background: #FC4E2A; flex: 1;"></div>
                        <div style="background: #E31A1C; flex: 1;"></div>
                        <div style="background: #BD0026; flex: 1;"></div>
                        <div style="background: #800026; flex: 1;"></div>
                    </div>
                    
                    <!-- Value labels -->
                    <div style="display: flex; justify-content: space-between; font-size: 10px; color: #666;">
                        <span>{format_large_values(min_val, metric_name)}</span>
                        <span style="text-align: center;">Avg: {format_large_values(merged_gdf['Value'].mean(), metric_name)}</span>
                        <span>{format_large_values(max_val, metric_name)}</span>
                    </div>
                    
                    <div style="text-align: center; font-size: 10px; color: #999; margin-top: 3px;">
                        Low ← → High
                    </div>
                    </div>
                    '''
                    m.get_root().html.add_child(folium.Element(legend_html))

                    tooltip_features = []
                    for idx, row in merged_gdf.iterrows():
                        if pd.notna(row['Value']):
                            tooltip_features.append({
                                'geometry': row['geometry'],
                                'properties': {
                                    'name': row[shapefile_column],
                                    'value': row['Value'],
                                    'metric': metric_name
                                }
                            })
                    
                    for feature in tooltip_features:
                        formatted_value = format_large_values(feature['properties']['value'], feature['properties']['metric'])
                        folium.GeoJson(
                            feature['geometry'],
                            style_function=lambda x: {'fillOpacity': 0, 'weight': 0, 'color': 'transparent'},
                            tooltip=folium.Tooltip(
                                f"{scope}: {feature['properties']['name']}<br>{feature['properties']['metric']}: {formatted_value}",
                                sticky=True
                            )
                        ).add_to(m)
        elif selected_city == 'N/A':
            # Add a text overlay when N/A is selected
            instruction_html = '''
            <div style="position: fixed; 
                        top: 50%; left: 50%; transform: translate(-50%, -50%);
                        background-color: rgba(255, 255, 255, 0.9); 
                        padding: 20px; border-radius: 10px; 
                        border: 2px solid #ccc; z-index: 9999;
                        text-align: center; font-size: 16px;">
                <h3 style="margin-top: 0; color: #333;">Please make a selection</h3>
                <p style="margin-bottom: 0; color: #666;">
                    Select 'All' or a specific city/municipality<br>
                    to view the heatmap data.
                </p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(instruction_html))

        map_data = st_folium(m, height=900, width=1300, returned_objects=["last_object_clicked"], key=f"map_{scope}_{indicator}_{selected_city}")

    st.markdown("##### Agentic AI Recommended Hero Product Analysis")
    hero_product_ai_analysis(selected_region, selected_province, indicator, ai_map_data, openai_api_key)


elif selected == 'Pop-Up Strategy':
    st.title('Kababayan Connect: OFW Pop-Up Expansion Strategy')
    st.caption('Data Sources: Bangko Sentral ng Pilipinas (BSP), Philippine Statistics Authority (PSA), Department of Migrant Workers (DMW)')

    df = pd.read_excel('data/OFW Cash Remittances - All Countries.xlsx')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        continent = show_filters(df, 'Continent')
    with col2:
        country = show_filters(df, 'Country', continent)
    with col3:
        type_ = show_filters(df, 'Type')
    with col4:
        year = show_filters(df, 'Year')

    col5, col6 = st.columns([2,6])

    with col5:
        filtered_df, current_total, delta = calculate_ofw_remittances(df, continent, country, type_, year)

        st.metric(
            label="""Total OFW Remittances""",
            value=f"${current_total:,.2f}",
            delta=delta
        )

        top_countries = show_top_countries_by_remittances(filtered_df)

        st.markdown("##### Mode of Remittance Distribution")
        show_remittance_pie_chart()
    
    with col6:
        plot_ofw_remittance_map(filtered_df, continent)

        st.markdown("##### OFW Deployment by Region (Landbased VS Seabased)")
        show_region_barchart()
    
    st.markdown("##### OFW Cash Remittances Forecasting")
    st.markdown(
        """<hr style="margin-top:2px;margin-bottom:10px;">""", 
        unsafe_allow_html=True
    )

    countries_selected = st.multiselect("Select Country(s)", sorted(df['Country'].unique()), default=top_countries['Country'].tolist()[0])

    col7, col8 = st.columns(2)
    with col7:
        if type_ == 'Combined':
            type_selected = ['Land-based', 'Sea-based']
        else:
            type_selected = type_
        types_selected = st.multiselect("Select Type(s)", df['Type'].unique(), default=type_selected)
    with col8:
        horizon = st.selectbox("Select Forecast Horizon (months)", options=[6,12,24], index=1)
    
    filtered_df_forecast, all_forecasts = forecast_and_plot(df, countries_selected, types_selected, horizon)

    st.markdown("##### Agentic AI Pop-Up Expansion Strategy Analysis")
    
    agentic_ai_analysis(continent, countries_selected, type_selected, year, current_total, top_countries, horizon, filtered_df_forecast, all_forecasts, openai_api_key=openai_api_key)

elif selected == 'Competitor Analysis':
    st.title('Domestic Branches')
    st.caption('Source: Google Maps')

    # Bank selection first
    bank = st.selectbox('Select Bank', ['All', 'BPI', 'BDO', 'UnionBank', 'Metrobank', 'Landbank', 'Other Financial Institutions'])

    # Map each bank to its dataset
    bank_files = {
        'All': "data/Branch Location/ALL_Establishments.csv",
        'BPI': "data/Branch Location/BPI_LUZON.csv",
        'BDO': "data/Branch Location/BDO_LUZON.csv",
        'UnionBank': "data/Branch Location/UNIONBANK_LUZON.csv",
        'Metrobank': "data/Branch Location/METROBANK_LUZON.csv",
        'Landbank': "data/Branch Location/LANDBANK_LUZON.csv",
        'Other Financial Institutions': "data/Branch Location/OTHERS_LUZON.csv"
    }

    # Load dataset for selected bank
    df = pd.read_csv(bank_files[bank], encoding='latin-1')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')

    # Drop rows with invalid coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])

    default_region = "NCR"
    default_province = "NCR"
    default_city = "Manila"

    if 'region' not in locals():
        region = default_region
    if 'province' not in locals():
        province = default_province
    if 'city' not in locals():
        city = default_city

    # Dependent filters (Region → Province → City)
    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox('Select Region', ['All'] + sorted(df['Region'].unique().tolist()), index=2)
    with col2:
        province_options = df[df['Region'] == region]['Province'].unique().tolist() if region != 'All' else df['Province'].unique().tolist()
        province = st.selectbox('Select Province', ['All'] + sorted(province_options), index=2)
    with col3:
        city_options = df[df['Province'] == province]['City'].unique().tolist() if province != 'All' else df['City'].unique().tolist()
        city = st.selectbox('Select City', ['All'] + sorted(city_options))

    # Apply filters
    if region != 'All':
        df = df[df['Region'] == region]
    if province != 'All':
        df = df[df['Province'] == province]
    if city != 'All':
        df = df[df['City'] == city]

    # Default center = Philippines
    map_center = [14.5995, 120.9842]
    zoom_level = 6

    # If the filtered dataframe is not empty, re-center on selection
    if not df.empty:
        avg_lat = df['Latitude'].mean()
        avg_lon = df['Longitude'].mean()
        map_center = [avg_lat, avg_lon]

        # Adjust zoom depending on selection granularity
        if city != "All":
            zoom_level = 15
        elif province != "All":
            zoom_level = 10
        elif region != "All":
            zoom_level = 8

    #st.dataframe(df)

    # Load all datasets into one combined DataFrame for summary counts
    dfs = []
    for b, path in bank_files.items():
        if b != "All":  # Skip "All" placeholder
            temp = pd.read_csv(path, encoding='latin-1')
            temp['Bank'] = b
            dfs.append(temp)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['Latitude'] = pd.to_numeric(combined_df['Latitude'], errors='coerce')
    combined_df['Longitude'] = pd.to_numeric(combined_df['Longitude'], errors='coerce')
    combined_df = combined_df.dropna(subset=['Latitude', 'Longitude'])

    # Apply the same filters to combined_df
    if region != 'All':
        combined_df = combined_df[combined_df['Region'] == region]
    if province != 'All':
        combined_df = combined_df[combined_df['Province'] == province]
    if city != 'All':
        combined_df = combined_df[combined_df['City'] == city]

    bank_counts = combined_df['Bank'].value_counts()

    # Create a container for the cards
    st.subheader("Bank Branch and ATM Summary")

    banks = ['BPI', 'BDO', 'UnionBank', 'Metrobank', 'Landbank']
    card_cols = st.columns(5)

    bank_summary = {}

    for idx, bank_name in enumerate(banks):
        bank_df = combined_df[combined_df['Bank'] == bank_name]
        total = len(bank_df)
        atm_count = len(bank_df[bank_df['Types'].str.contains('ATM', na=False)])
        branch_count = total - atm_count

        bank_summary[bank_name] = {"Branches": branch_count, "ATMs": atm_count, "Total": total}

        with card_cols[idx]:
            st.metric(
                label=bank_name,
                value=f"{total} locations",
                delta=f"{branch_count} Branch / {atm_count} ATM"
            )

    # Folium Map
    # m = folium.Map(location=[14.5995, 120.9842], zoom_start=6, scrollWheelZoom=False, tiles='CartoDB positron')
# m = folium.Map(location=map_center, zoom_start=zoom_level, scrollWheelZoom=False, tiles='CartoDB positron')

map_col, stats_col = st.columns([3, 1])

with map_col:
    # Create the map once
    m = folium.Map(location=map_center, zoom_start=zoom_level, scrollWheelZoom=False, tiles='CartoDB positron')

    # Add all markers and circles
    for _, row in df.iterrows():
        popup_text = (
            f"<b>{row['Branch Name'] if 'Branch Name' in df.columns else row['Place ID']}</b><br>"
            f"⭐ {row['Rating'] if 'Rating' in df and not pd.isna(row['Rating']) else 'N/A'} "
            f"({row['User Ratings Count'] if 'User Ratings Count' in df.columns and not pd.isna(row['User Ratings Count']) else 0} reviews)"
        )

        folium.Circle(
            location=[row['Latitude'], row['Longitude']],
            radius=3000,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.3
        ).add_to(m)

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_text
        ).add_to(m)

    # Display map inside this column
    st_folium(m, height=500, width=800)

with stats_col:
    st.subheader('Branch Summary')
    st.metric(
        label='Number of ATMs & Branches',
        value=len(df)
    )

AI_col = st.container() 

with AI_col:
    ai_recommendation = agentic_ai_branch_analyzer(df, bank_summary)
    st.subheader('Agentic AI Branch Improvement Recommendation')
    st.write(ai_recommendation)
