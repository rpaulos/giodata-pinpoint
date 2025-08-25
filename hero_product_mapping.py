import pandas as pd
import folium
import streamlit as st
import geopandas as gpd
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

def get_region_province_mapping(df):
    """
    Get mapping of regions to their provinces from the dataset
    """
    region_province_mapping = {}
    for region in df['Region'].unique():
        provinces = df[df['Region'] == region]['Province'].unique()
        region_province_mapping[region] = sorted(list(provinces))
    return region_province_mapping

def get_city_options(df, selected_region, selected_province):
    """
    Get city/municipality options based on selected region and province
    """
    filtered_df = df.copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_province != 'All':
        filtered_df = filtered_df[filtered_df['Province'] == selected_province]
    
    cities = ['All'] + sorted(filtered_df['City'].unique().tolist())
    return cities

def filter_data_by_region_province(df, selected_region, selected_province):
    """
    Filter the dataframe based on selected region and province
    """
    filtered_df = df.copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_province != 'All':
        filtered_df = filtered_df[filtered_df['Province'] == selected_province]
    
    return filtered_df

def filter_data_by_all_selections(df, selected_region, selected_province, selected_city):
    """
    Filter the dataframe based on all selections: region, province, and city
    """
    filtered_df = df.copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_province != 'All':
        filtered_df = filtered_df[filtered_df['Province'] == selected_province]
        
    if selected_city not in ['All', 'N/A']:
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    
    return filtered_df

def determine_scope_from_filters(selected_region, selected_province, selected_city=None):
    """
    Determine the appropriate scope based on the selected filters
    """
    if selected_city and selected_city not in ['All', 'N/A']:
        return 'City'  # Show specific city
    elif selected_city == 'All':
        return 'City'  # Show all cities within the selected province
    elif selected_province != 'All':
        return 'Provincial'  # Show provincial level when province is selected but city is N/A
    elif selected_region != 'All':
        return 'Provincial'  # Show provinces within the region
    else:
        # When all regions are selected, show provincial level (all provinces)
        return 'Provincial'

def format_large_values(value, metric_name):
    """
    Format large values (GDP, LGU Income) to show in millions for better readability
    """
    millions_metrics = ['City/Municipality Total GDP', 'Annual LGU Income', 'Total_Wealth_Score']
    
    if metric_name in millions_metrics and abs(value) >= 1000000:
        millions_value = value / 1000000
        return f"{millions_value:,.1f}M"
    elif metric_name in millions_metrics and abs(value) >= 1000:
        thousands_value = value / 1000
        return f"{thousands_value:,.1f}K"
    elif metric_name in ['GDP Growth (%)', 'Poverty Rate (%)']:
        return f"{value:.1f}%"
    else:
        return f"{value:,.0f}"

@st.cache_data
def process_wealth_data(df, scope, indicator):
    """
    Process wealth data based on scope (Regional/Provincial/City) and indicator
    """
    try:
        if df.empty:
            return pd.DataFrame()
            
        numeric_columns = ['City/Municipality Total GDP', 'GDP Growth (%)', 'Poverty Rate (%)', 
                          'Annual LGU Income', 'Condominium', 'Retail Hubs', 'Developers', 
                          'Car Showrooms', 'International Schools', 'Hospitals', 
                          'Luxury Hotel Presence', 'Casinos']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if scope == 'Regional':
            group_column = 'Region'
        elif scope == 'Provincial':
            group_column = 'Province'
        else:
            group_column = 'City'
        
        if indicator == 'All':
            additive_metrics = ['City/Municipality Total GDP', 'Annual LGU Income', 
                              'Condominium', 'Retail Hubs', 'Developers', 'Car Showrooms',
                              'International Schools', 'Hospitals', 'Luxury Hotel Presence', 'Casinos']
            
            percentage_metrics = ['GDP Growth (%)', 'Poverty Rate (%)']
            
            grouped = df.groupby(group_column).agg({
                **{metric: 'sum' for metric in additive_metrics if metric in df.columns},
                **{metric: 'mean' for metric in percentage_metrics if metric in df.columns}
            }).reset_index()
            
            wealth_score = 0
            
            if 'City/Municipality Total GDP' in grouped.columns:
                wealth_score += grouped['City/Municipality Total GDP'] * 0.5
            
            if 'Annual LGU Income' in grouped.columns:
                wealth_score += grouped['Annual LGU Income'] * 0.2
            
            infrastructure_cols = ['Condominium', 'Retail Hubs', 'Developers', 'Car Showrooms',
                                 'International Schools', 'Hospitals', 'Luxury Hotel Presence', 'Casinos']
            infrastructure_score = 0
            for col in infrastructure_cols:
                if col in grouped.columns:
                    infrastructure_score += grouped[col]
            wealth_score += infrastructure_score * 0.2
            
            if 'Poverty Rate (%)' in grouped.columns:
                max_poverty = grouped['Poverty Rate (%)'].max()
                if max_poverty > 0:  # Avoid division by zero
                    poverty_penalty = (grouped['Poverty Rate (%)'] / max_poverty) * wealth_score * 0.1
                    wealth_score -= poverty_penalty
            
            grouped['Total_Wealth_Score'] = wealth_score
            return grouped
            
        else:
            if indicator in df.columns:
                if indicator in ['GDP Growth (%)', 'Poverty Rate (%)']:
                    grouped = df.groupby(group_column)[indicator].mean().reset_index()
                else:
                    grouped = df.groupby(group_column)[indicator].sum().reset_index()
                return grouped
            else:
                st.error(f"Indicator '{indicator}' not found in data")
                return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error processing wealth data: {str(e)}")
        return pd.DataFrame()

@st.cache_data

def merge_shapefile_data(_gdf, data_df, shapefile_column, data_column, scope=None):
    """
    Merge shapefile geodataframe with data dataframe
    """
    try:
        if data_df.empty:
            return _gdf.copy()
        
        gdf_clean = _gdf.copy()
        data_clean = data_df.copy()
        
        # Special handling for regional data when showing provincial scope
        if data_column == 'Region' and shapefile_column == 'NAME_1' and scope == 'Provincial':
            # Map regional data to provinces
            region_to_provinces = {
                'National Capital Region (NCR)': ['Metropolitan Manila'],
                'Cordillera Administrative Region (CAR)': ['Abra', 'Apayao', 'Benguet', 'Ifugao', 'Kalinga', 'Mountain Province'],
                'Ilocos Region (Region I)': ['Ilocos Norte', 'Ilocos Sur', 'La Union', 'Pangasinan'],
                'Cagayan Valley (Region II)': ['Batanes', 'Cagayan', 'Isabela', 'Nueva Vizcaya', 'Quirino'],
                'Central Luzon (Region III)': ['Aurora', 'Bataan', 'Bulacan', 'Nueva Ecija', 'Pampanga', 'Tarlac', 'Zambales'],
                'CALABARZON (Region IV-A)': ['Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal'],
                'MIMAROPA (Region IV-B)': ['Marinduque', 'Occidental Mindoro', 'Oriental Mindoro', 'Palawan', 'Romblon'],
                'Bicol Region (Region V)': ['Albay', 'Camarines Norte', 'Camarines Sur', 'Catanduanes', 'Masbate', 'Sorsogon'],
            }
            
            province_data = []
            for _, row in data_df.iterrows():
                region = row[data_column]
                
                if region in region_to_provinces:
                    provinces = region_to_provinces[region]
                    for province in provinces:
                        province_row = {'Province': province}
                        for col in row.index:
                            if col != data_column:
                                if col == 'Total_Wealth_Score':
                                    province_row[col] = row[col] / len(provinces)
                                elif col in ['GDP Growth (%)', 'Poverty Rate (%)']:
                                    province_row[col] = row[col]
                                else:
                                    province_row[col] = row[col] / len(provinces)
                        province_data.append(province_row)
            
            if province_data:
                province_df = pd.DataFrame(province_data)
                merged = _gdf.merge(province_df, left_on='NAME_1', right_on='Province', how='left')
                return merged
        
        # For city-level data, filter to only Luzon regions to prevent incorrect matching
        if data_column == 'City':
            # Define Luzon provinces to filter shapefile
            luzon_provinces = [
                'Metropolitan Manila', 'Abra', 'Apayao', 'Benguet', 'Ifugao', 'Kalinga', 'Mountain Province',
                'Ilocos Norte', 'Ilocos Sur', 'La Union', 'Pangasinan', 'Batanes', 'Cagayan', 'Isabela', 
                'Nueva Vizcaya', 'Quirino', 'Aurora', 'Bataan', 'Bulacan', 'Nueva Ecija', 'Pampanga', 
                'Tarlac', 'Zambales', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 
                'Occidental Mindoro', 'Oriental Mindoro', 'Palawan', 'Romblon', 'Albay', 'Camarines Norte', 
                'Camarines Sur', 'Catanduanes', 'Masbate', 'Sorsogon'
            ]
            
            # Filter shapefile to Luzon provinces only
            gdf_clean = gdf_clean[gdf_clean['NAME_1'].isin(luzon_provinces)]
        
        elif data_column == 'Province' and shapefile_column == 'NAME_1':
            # For provincial data, filter shapefile to only provinces that exist in the data
            provinces_in_data = data_clean['Province'].unique()
            gdf_clean = gdf_clean[gdf_clean['NAME_1'].isin(provinces_in_data)]
        
        # Normalize city names for better matching
        if data_column == 'City' and shapefile_column == 'NAME_2':
            data_clean['City_normalized'] = data_clean['City'].apply(normalize_place_name)
            gdf_clean['NAME_2_normalized'] = gdf_clean['NAME_2'].apply(normalize_place_name)
            
            merged = gdf_clean.merge(data_clean, left_on='NAME_2_normalized', right_on='City_normalized', how='left')
        else:
            merged = gdf_clean.merge(data_clean, left_on=shapefile_column, right_on=data_column, how='left')
        
        return merged
        
    except Exception as e:
        st.error(f"Error merging data: {str(e)}")
        return _gdf.copy()

def normalize_place_name(name):
    """
    Normalize place names for better matching between datasets
    """
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Common normalization rules
    replacements = {
        'Ñ': 'N', 'ñ': 'n',
        'City of ': '',
        'Municipality of ': '',
        ' City': '',
        ' Municipality': '',
        'St.': 'Saint',
        'Sto.': 'Santo',
        'Sta.': 'Santa',
        'Mt.': 'Mount',
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove extra spaces and convert to title case
    name = ' '.join(name.split()).title()
    
    return name

def create_wealth_choropleth(gdf, data_df, location_column, value_column='Value'):
    """
    Create a choropleth map for wealth indicators
    """
    try:
        map_center = [12.8797, 121.7740]
        m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')
        
        if data_df.empty or value_column not in data_df.columns:
            return m
            
        merged_gdf = merge_shapefile_data(gdf, data_df, location_column, location_column)
        
        if not merged_gdf.empty and value_column in merged_gdf.columns:
            merged_gdf = merged_gdf.dropna(subset=[value_column])
            
            if not merged_gdf.empty:
                folium.Choropleth(
                    geo_data=merged_gdf.__geo_interface__,
                    data=merged_gdf,
                    columns=[location_column, value_column],
                    key_on=f'feature.properties.{location_column}',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Wealth Indicator Value'
                ).add_to(m)
                
                for idx, row in merged_gdf.iterrows():
                    if pd.notna(row[value_column]):
                        folium.GeoJson(
                            row['geometry'],
                            style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
                            tooltip=folium.Tooltip(
                                f"{location_column}: {row[location_column]}<br>Value: {row[value_column]:,.2f}"
                            )
                        ).add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error creating choropleth: {str(e)}")
        map_center = [12.8797, 121.7740]
        return folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')

def load_philippines_shapefile(level=1):
    """
    Load Philippines shapefile based on administrative level
    level 0: Country
    level 1: Regions
    level 2: Provinces
    level 3: Municipalities/Cities
    """
    try:
        shapefile_path = f'GADM PHL shapefile/gadm41_PHL_{level}.shp'
        gdf = gpd.read_file(shapefile_path)
        return gdf
    except Exception as e:
        st.error(f"Error loading shapefile level {level}: {str(e)}")
        return None

def get_summary_statistics(df, value_column='Value'):
    """
    Get summary statistics for the wealth indicator data
    """
    try:
        if df.empty or value_column not in df.columns:
            return {}
            
        stats = {
            'total': df[value_column].sum(),
            'mean': df[value_column].mean(),
            'median': df[value_column].median(),
            'std': df[value_column].std(),
            'min': df[value_column].min(),
            'max': df[value_column].max(),
            'count': len(df)
        }
        
        return stats
        
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return {}

def format_wealth_data(df):
    """
    Format and clean wealth indicator data
    """
    try:
        df_clean = df.copy()
        
        numeric_columns = ['Value', 'Year']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        critical_columns = ['Value']
        for col in critical_columns:
            if col in df_clean.columns:
                df_clean = df_clean.dropna(subset=[col])
        
        text_columns = ['Region', 'Province', 'Indicator', 'Category']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error formatting data: {str(e)}")
        return df

def hero_product_ai_analysis(region, province, indicator, data, openai_api_key):

    bank_products_df = pd.read_excel("data/Bank Products.xlsx", sheet_name='Accounts')

    # --- Prompt ---
    prompt = f"""
    You are a strategic financial analyst for BPI Bank.

    Context Provided:
    - Selected region: {region}
    - Selected province: {province}
    - Selected wealth indicator: {indicator}
    - Data : {data}
    - Bank Products: {bank_products_df.to_markdown(index=False)}

    Your task is to:  
    1. Analyze the city’s economic profile based on the given wealth indicator.  
    2. Interpret what this wealth indicator suggests about the city’s economic capacity, opportunities, and risks.  
    3. Identify the possible financial behaviors or needs of residents and businesses in the city given the context. 
    4. Recommend the most relevant products from the Bank Products list above (do not invent new products).
    5. Provide a concise explanation for each recommended product, linking it directly to the city’s context. 
    6. Keep the analysis concise, professional, and actionable.  

     Format your response as follows:  
    **Analysis:** [analysis]  
    **Recommended Hero Products:**  
    1. [Product] → [Reasoning]  
    2. [Product] → [Reasoning]  
    3. [Product] → [Reasoning]
    ...
    """

     # --- LLM Setup ---
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0
    )

    # --- Conversation State ---
    if "hero_product_messages" not in st.session_state:
        st.session_state.hero_product_messages = []

    if "hero_last_inputs" not in st.session_state:
        st.session_state.last_inputs = {
            "region": None,
            "province": None,
            "indicator": None,
            "data": pd.DataFrame(),
        }

    # Check if inputs changed
    inputs_changed = (
        st.session_state.last_inputs["region"] != region or
        st.session_state.last_inputs["province"] != province or
        st.session_state.last_inputs["indicator"] != indicator or
        not st.session_state.last_inputs["data"].equals(data)
    )

    # Generate analysis if inputs changed
    if inputs_changed is not None:
        response = llm([
            SystemMessage(content="You are a strategic financial analyst for BPI Bank."),
            HumanMessage(content=prompt)
        ])
        st.session_state.hero_product_messages = [{"role": "assistant", "content": response.content}]

        # Save last inputs
        st.session_state.hero_last_inputs = {
            "region": region,
            "province": province,
            "indicator": indicator,
            "data": data,
        }

    # --- Display Messages ---
    for msg in st.session_state.hero_product_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- User Chat Input ---
    if user_input := st.chat_input("Ask about the strategy or insights..."):
        st.session_state.hero_product_messages.append({"role": "user", "content": user_input})

        messages = [SystemMessage(content="You are a strategic financial analyst for BPI Bank.")] + [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.hero_product_messages
        ]

        response = llm(messages)
        st.session_state.hero_product_messages.append({"role": "assistant", "content": response.content})

        with st.chat_message("assistant"):
            st.markdown(response.content)