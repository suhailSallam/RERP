import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set page configuration

st.set_page_config(page_title='Real Estate Rental Prices Data Analysis System نظام تحليل بيانات أسعار تأجير العقارات',page_icon=None,
                   layout='wide',initial_sidebar_state='auto', menu_items=None)
with st.sidebar:
    st.markdown("""
    <style>
    :root {
      --header-height: 50px;
    }
    .css-z5fcl4 {
      padding-top: 2.5rem;
      padding-bottom: 5rem;
      padding-left: 2rem;
      padding-right: 2rem;
      color: blue;
    }
    .css-1544g2n {
      padding: 0rem 0.5rem 1.0rem;
    }
    [data-testid="stHeader"] {
        background-image: url(/app/static/icons8-astrolabe-64.png);
        background-repeat: no-repeat;
        background-size: contain;
        background-origin: content-box;
        color: blue;
    }

    [data-testid="stHeader"] {
        background-color: rgba(28, 131, 225, 0.1);
        padding-top: var(--header-height);
    }
    [data-testid="stSidebar"] {
        background-color: #e3f2fd; /* Soft blue */
        margin-top: var(--header-height);
        color: blue;
        position: fixed; /* Ensure sidebar is fixed */
        width: 250px; /* Fixed width */
        height: 100vh; /* Full height of the viewport */
        z-index: 999; /* Ensure it stays on top */
        overflow-y: auto; /* Enable scrolling for overflow content */
        padding-bottom: 2rem; /* Extra padding at the bottom */
    }
    [data-testid="stToolbar"]::before {
        content: "Real Estate Rental Prices Data Analysis System نظام تحليل بيانات أسعار تأجير العقارات";
    }
    [data-testid="collapsedControl"] {
        margin-top: var(--header-height);
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
    /* Responsive Design */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100%; /* Sidebar takes full width on small screens */
            height: auto; /* Adjust height for small screens */
            position: relative; /* Sidebar is not fixed on small screens */
            z-index: 1000; /* Ensure it stays on top */
        }

        .css-z5fcl4 {
            padding-left: 1rem; /* Adjust padding for smaller screens */
            padding-right: 1rem;
        }
        [data-testid="stHeader"] {
            padding-top: 1rem; /* Adjust header padding */
        }
        [data-testid="stToolbar"] {
            font-size: 1.2rem; /* Adjust font size for the toolbar */
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
# Language Selector
language = st.sidebar.radio("Select Language | اختر اللغة", ["English", "العربية"])

# Define a dictionary for labels and texts
texts = {
    "English": {
        "page_title": "Real Estate Rental Prices Data Analysis System",
        "general_overview": "General Overview",
        "total_properties": "Total Properties",
        "average_price": "Average Nightly Price",
        "average_evaluation": "Average Evaluation",
        "most_common_category": "Most Common Category",
        "property_distribution": "Distribution of Property Categories",
        "city_insights": "City Insights",
        "select_city": "Select a City",
        "area_vs_price": "Area vs Price in",
        "neighborhood_insights": "Neighborhood Insights",
        "select_neighborhood": "Select a Neighborhood",
        "price_distribution": "Price Distribution in",
        "price_distribution_all": "Price Distribution in All Regions",
        "property_category_analysis": "Property Category Analysis",
        "select_category": "Select a Property Category",
        "predictive_modeling": "Predictive Modeling",
        "price_prediction": "Price Prediction",
        "predicted_price": "Predicted Nightly Price",
        "closest_price": "Closest Real Nightly Price",
        "enter_area": "Enter Property Area (sq. meters)",
        "evaluation_score": "Enter Evaluation Score",
        "scatter_plot_title": "Real Prices and Predicted Price",
        "city":"City",
        "rsCategory":"Real State Category",
        "title_location":"0.0",
        "area":"Area (sq. meters)",
        "one_night_price":"One Night Price",
        "NeighborhoodsVsEvaluation":"Top 10 Neighborhoods by Average Evaluation in",
        "NeighborhoodsVsPropertyCount":"Top 10 Neighborhoods by Property Count",
        "neighborhood":"Neighborhood",
        "average_evaluation_city":"Average Evaluation by City",
        "top_cities_highst_average_price":"Top Cities with Highest Average Rental Prices",
        "count":"Count",
        "areaVsEvaluation":"Relationship Between Area and Evaluation",
        "evaluation":"Evaluation",
        "evaluation_distribution":"Evaluation Distribution in",
        "area_dist_Properties":"Area Distribution of Properties",
    },
    "العربية": {
        "page_title": "نظام تحليل بيانات أسعار تأجير العقارات",
        "general_overview": "نظرة عامة",
        "total_properties": "إجمالي العقارات",
        "average_price": "متوسط السعر الليلي",
        "average_evaluation": "متوسط التقييم",
        "most_common_category": "الفئة الأكثر شيوعاً",
        "property_distribution": "توزيع فئات العقارات",
        "city_insights": "رؤى المدن",
        "select_city": "اختر مدينة",
        "area_vs_price": "المساحة مقابل السعر في",
        "neighborhood_insights": "رؤى الأحياء",
        "select_neighborhood": "اختر حيًا",
        "price_distribution": "توزيع الأسعار في",
        "price_distribution_all": "توزيع الأسعار في كافة المناطق",
        "property_category_analysis": "تحليل فئات العقارات",
        "select_category": "اختر فئة عقارية",
        "predictive_modeling": "النمذجة التنبؤية",
        "price_prediction": "توقع السعر",
        "predicted_price": "السعر الليلي المتوقع",
        "closest_price": "أقرب سعر ليلي حقيقي",
        "enter_area": "أدخل مساحة العقار (بالمتر المربع)",
        "evaluation_score": "أدخل درجة التقييم",
        "scatter_plot_title": "الأسعار الحقيقية والسعر المتوقع",
        "city":"المدينة",
        "rsCategory":"فئة العقار",
        "title_location":"1.0",
        "area":"المساحة (بالمتر المربع)",
        "one_night_price":"سعر الليلة الواحدة",
        "NeighborhoodsVsEvaluation":"أعلى 10 أحياء تقييماً في",
        "NeighborhoodsVsPropertyCount":"أفضل 10 أحياء حسب عدد العقارات",
        "neighborhood":"الحيّ",
        "average_evaluation_city":"متوسط التقييم حسب المدينة",
        "top_cities_highst_average_price":"أعلى المدن حسب متوسط أسعار التأجير",
        "count":"العدد",
        "areaVsEvaluation":"العلاقة بين المساحة والتقييم",
        "evaluation":"التقييم",
        "evaluation_distribution":"توزيع فئات التقييم",
        "area_dist_Properties":"توزيع العقارات حسب المساحة",
    }
}

# Get selected language texts
t = texts[language]

# Apply RTL or LTR based on language
if language == "العربية":
    st.markdown('<style>body { direction: RTL; text-align: right; }</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body { direction: LTR; text-align: left; }</style>', unsafe_allow_html=True)

# Load the dataset
df = pd.read_excel('real_estate_rental_prices_extended.xlsx')

# Sidebar Navigation
options = st.sidebar.radio(t["page_title"], 
                            [t["general_overview"], 
                             t["city_insights"], 
                             t["neighborhood_insights"],
                             t["property_category_analysis"], 
                             t["predictive_modeling"]])

# General Overview Dashboard
if options == t["general_overview"]:
    st.header(t["general_overview"])
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #FF6347; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['total_properties']}</b><br>{len(df)}</div>", unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #4682B4; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_price']}</b><br>${df['OneNightPrice'].mean():.2f}</div>", unsafe_allow_html=True)
    col3.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #32CD32; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_evaluation']}</b><br>{df['Evaluation'].mean():.2f}</div>", unsafe_allow_html=True)
    col4.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #FFD700; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['most_common_category']}</b><br>{df['RsCategory'].mode()[0]}</div>", unsafe_allow_html=True)

    # Visualizations
    col21, col22 = st.columns(2)
    with col21:
        st.subheader(t["property_distribution"])
        fig1 = px.pie(df, names='RsCategory', title=t["property_distribution"])
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    with col22:
        st.subheader(t["city_insights"])
        fig2 = px.bar(df['City'].value_counts().reset_index(),
                      x='index', y='City', title=t["city_insights"],
                      labels={'index': t["select_city"], 'City': t["total_properties"]})
        st.plotly_chart(fig2, theme=None, use_container_width=True)

# City Insights Dashboard
elif options == t["city_insights"]:
    st.header(t["city_insights"])
    # Select City
    city_selected = st.sidebar.selectbox(t["select_city"], df['City'].unique())
    city_data = df[df['City'] == city_selected]

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #FF6347; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['total_properties']}</b><br>{len(city_data)}</div>", unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #4682B4; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_price']}</b><br>${city_data['OneNightPrice'].mean():.2f}</div>", unsafe_allow_html=True)
    col3.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #32CD32; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_evaluation']}</b><br>{city_data['Evaluation'].mean():.2f}</div>", unsafe_allow_html=True)
    col4.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #FFD700; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['most_common_category']}</b><br>{city_data['RsCategory'].mode()[0]}</div>", unsafe_allow_html=True)

    # Visualizations
    col21, col22 = st.columns(2)
    with col21:
        st.subheader(f"{t['area_vs_price']} {city_selected}")
        fig = px.scatter(city_data, x='Area', y='OneNightPrice', title=f"{t['area_vs_price']} {city_selected}",
                         labels={'Area': t["enter_area"], 'OneNightPrice': t["average_price"]})
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with col22:
        st.subheader(f"{t['price_distribution_all']}")
        # Average Rental Price by City
        # What is the average nightly rental price for properties in each city?
        # Group by city and calculate the average price
        avg_price_by_city = df.groupby('City')['OneNightPrice'].mean().sort_values().reset_index()
        # Plot the bar chart
        fig1 = px.bar(avg_price_by_city, x='City', y='OneNightPrice', title=t["price_distribution_all"], color='City', labels={'color': 'Legend'})
        fig1.update_layout(
        title_x=float(t["title_location"]),
        xaxis_title=t['city'],           # Custom x-axis label
        yaxis_title=t['average_price']   # Custom y-axis label
        )
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    col31, col32 = st.columns(2)
    with col31:
        st.subheader(f"{t['price_distribution']}  {city_selected}")
        # Price Distribution Across Categories
        # How does the rental price vary across different property categories?
        # Plot the box plot
        fig1 = px.box(city_data,x='RsCategory', y='OneNightPrice', title=f"{t['price_distribution']} {city_selected}", color='RsCategory', labels={'color': 'Legend'})
        fig1.update_layout(
        title_x=float(t["title_location"]),
        xaxis_title=t['rsCategory'],     # Custom x-axis label
        yaxis_title=t['one_night_price']   # Custom y-axis label
            )
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    with col32:
        st.subheader(f"{t['top_cities_highst_average_price']}")
        # Top Cities with Highest Average Rental Prices
        # Which cities have the most expensive properties on average?
        avg_price_by_city = df.groupby('City')['OneNightPrice'].mean().sort_values(ascending=False).reset_index()
        # Plot the horizontal bar plot
        fig1 = px.bar(avg_price_by_city, x='OneNightPrice', y='City', title=t["top_cities_highst_average_price"], color='City', labels={'color': 'Legend'}, orientation='h')
        fig1.update_layout(
        title_x=float(t["title_location"]),
        xaxis_title=t['city'],     # Custom x-axis label
        yaxis_title=t['average_price']   # Custom y-axis label
            )
        st.plotly_chart(fig1, theme=None, use_container_width=True)

# Neighborhood Insights Dashboard
elif options == t["neighborhood_insights"]:
    st.header(t["neighborhood_insights"])
    # City and Neighborhood Filters
    city_selected = st.sidebar.selectbox(t["select_city"], df['City'].unique())
    neighborhoods_in_city = df[df['City'] == city_selected]['Neighbourhood'].unique()
    neighborhoods_with_evaluation = df[df['City'] == city_selected][['Neighbourhood', 'Evaluation']].drop_duplicates()
    df_neighborhoods_with_evaluation = pd.DataFrame(neighborhoods_with_evaluation)
    neighborhood_selected = st.sidebar.selectbox(t["select_neighborhood"], neighborhoods_in_city)

    # Filtered Data
    df_filtered = df[(df['City'] == city_selected) & (df['Neighbourhood'] == neighborhood_selected)]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #FF6347; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['total_properties']}</b><br>{len(df_filtered)}</div>", unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #4682B4; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_price']}</b><br>${df_filtered['OneNightPrice'].mean():.2f}</div>", unsafe_allow_html=True)
    col3.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #32CD32; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_evaluation']}</b><br>{df_filtered['Evaluation'].mean():.2f}</div>", unsafe_allow_html=True)

    # Visualizations
    col11, col12 = st.columns(2)
    with col11:
        st.subheader(f"{t['price_distribution']} {neighborhood_selected}")
        fig1 = px.box(df_filtered, y='OneNightPrice', title=f"{t['price_distribution']} {neighborhood_selected}")
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    with col12:
        st.subheader(f"{t['NeighborhoodsVsEvaluation']} {city_selected} ")
        # Popular Neighborhoods Based on Evaluations
        # Which neighborhoods have the highest average evaluations?
        # Group by neighborhood and calculate the average evaluation
        avg_evaluation_by_neighborhood = (
            df_neighborhoods_with_evaluation.groupby('Neighbourhood')['Evaluation'].mean().sort_values(ascending=False).head(10).reset_index())
        # Plot the bar plot
        fig1 = px.bar(avg_evaluation_by_neighborhood, x='Neighbourhood', y='Evaluation', title=f"{t['NeighborhoodsVsEvaluation']} {city_selected}", color='Neighbourhood', labels={'color': 'Legend'})
        fig1.update_layout(
        title_x=float(t["title_location"]),
        xaxis_title=t['neighborhood'],     # Custom x-axis label
        yaxis_title=t['average_evaluation']   # Custom y-axis label
            )
        st.plotly_chart(fig1, theme=None, use_container_width=True)
# Property Category Analysis Dashboard
elif options == t["property_category_analysis"]:
    st.header(t["property_category_analysis"])
    # Select Property Category
    category_selected = st.sidebar.selectbox(t["select_category"], df['RsCategory'].unique())
    df_filtered = df[df['RsCategory'] == category_selected]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #FF6347; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['total_properties']}</b><br>{len(df_filtered)}</div>", unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #4682B4; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_price']}</b><br>${df_filtered['OneNightPrice'].mean():.2f}</div>", unsafe_allow_html=True)
    col3.markdown(
        f"<div style='font-size: 20px; color: white; background-color: #32CD32; padding: 10px; border-radius: 5px;'>"
        f"<b>{t['average_evaluation']}</b><br>{df_filtered['Evaluation'].mean():.2f}</div>", unsafe_allow_html=True)

    # Visualizations
    col11, col12 = st.columns(2)
    with col11:
        st.subheader(f"{t['property_distribution']} ")
        # Distribution of Property Categories
        # How are the different types of properties distributed across the dataset?
        # Count property categories
        category_distribution = df['RsCategory'].value_counts().reset_index()
        # Plot the bar chart
        fig1 = px.pie(df, names='RsCategory',title=t["property_distribution"], color='RsCategory', labels={'color': 'Legend'})
        fig1.update_layout(
        title_x=float(t['title_location']),
            )
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    with col12:
        st.subheader(f"{t['price_distribution']}")
        # Price Distribution Across Categories
        # How does the rental price vary across different property categories?
        # Plot the box plot
        fig1 = px.box(df,x='RsCategory', y='OneNightPrice', title=t['price_distribution'], color='RsCategory', labels={'color': 'Legend'})
        fig1.update_layout(
        title_x=float(t["title_location"]),
        xaxis_title=t['rsCategory'],     # Custom x-axis label
        yaxis_title=t['one_night_price']   # Custom y-axis label
            )
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    col21, col22 = st.columns(2)
    with col21:
        st.subheader(f"{t['most_common_category']}")
        # Most Popular Property Categories
        #Which property types are most common in the dataset?
        property_category_count = df['RsCategory'].value_counts()
        # Plot the bar plot
        fig1 = px.bar(property_category_count, x=property_category_count.index, y=property_category_count.values, title=t['most_common_category'], color='RsCategory', labels={'color': 'Legend'})
        fig1.update_layout(
        title_x=float(t["title_location"]),
        xaxis_title=t['rsCategory'],     # Custom x-axis label
        yaxis_title=t['count']   # Custom y-axis label
            )
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    with col22:
        st.subheader(f"{t['price_distribution']} {category_selected}")
        fig1 = px.box(df_filtered, y='OneNightPrice', title=f"{t['price_distribution']} {category_selected}")
        st.plotly_chart(fig1, theme=None, use_container_width=True)
        
# Predictive Modeling Dashboard
elif options == t["predictive_modeling"]:
    st.header(t["predictive_modeling"])
    # Inputs
    col11, col12 = st.columns(2)
    col21, col22 = st.columns(2)
    col31, col32 = st.columns(2)
    with col11:
        area_input = st.number_input(t["enter_area"], min_value=10, max_value=1000, step=10)
    with col12:
        evaluation_input = st.slider(t["evaluation_score"], min_value=0, max_value=10, step=1)
    with col21:
        city_input = st.selectbox(t["select_city"], df['City'].unique())

    # Filter neighborhoods based on selected city
    neighborhoods_in_city = df[df['City'] == city_input]['Neighbourhood'].unique()
    with col22:
        neighborhood_input = st.selectbox(t["select_neighborhood"], neighborhoods_in_city)
    with col31:
        category_input = st.selectbox(t["select_category"], df['RsCategory'].unique())

    # Prepare the model
    features = ['Area', 'Evaluation', 'City_code', 'RsCategory_code', 'Neighbourhood_code']
    target = 'OneNightPrice'
    X = df[features]
    y = df[target]

    # Encode categorical inputs
    city_code = df[df['City'] == city_input]['City_code'].iloc[0]
    category_code = df[df['RsCategory'] == category_input]['RsCategory_code'].iloc[0]
    neighborhood_code = df[df['Neighbourhood'] == neighborhood_input]['Neighbourhood_code'].iloc[0]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict button
    if st.button(t["price_prediction"]):
        input_data = [[area_input, evaluation_input, city_code, category_code, neighborhood_code]]
        predicted_price = model.predict(input_data)[0]

        # Find the closest real price
        df_filtered = df[(df['City'] == city_input) & 
                         (df['RsCategory'] == category_input) & 
                         (df['Neighbourhood'] == neighborhood_input)]
        
        if df_filtered.empty:
            closest_real_price = 0
            st.warning("No matching data found for the selected parameters. Defaulting closest price to $0.")
        else:
            df_filtered['distance'] = abs(df_filtered['Area'] - area_input) + abs(df_filtered['Evaluation'] - evaluation_input)
            closest_real_price = df_filtered.loc[df_filtered['distance'].idxmin(), 'OneNightPrice']

        # Display Predicted and Closest Real Price
        col41, col42 = st.columns(2)
        col41.success(f"{t['predicted_price']}: ${predicted_price:.2f}")
        col42.info(f"{t['closest_price']}: ${closest_real_price:.2f}")

        # Scatter plot: Predicted vs Real Values
        fig3 = px.scatter(df, x='Area', y='OneNightPrice', title=t["scatter_plot_title"],
                          color=px.Constant("Real Value"), labels={'color': 'Legend'})
        fig3.add_scatter(x=[area_input], y=[predicted_price], mode='markers', name='Predicted Value',
                         marker=dict(color='red', size=10))
        st.plotly_chart(fig3, theme=None, use_container_width=True)
