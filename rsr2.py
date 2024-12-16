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
        background-image: url(i.png);
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
        "price_distribution_all":"Price Distribution in All Reagions",
        "property_category_analysis": "Property Category Analysis",
        "select_category": "Select a Property Category",
        "area_insights":"Area Insights",
        "evaluation_insights":"Evaluation Insights",
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
        "NeighborhoodsVsEvaluation":"Top 10 Neighborhoods by Average Evaluation",
        "NeighborhoodsVsPropertyCount":"Top 10 Neighborhoods by Property Count",
        "neighborhood":"Neighborhood",
        "average_evaluation_city":"Average Evaluation by City",
        "top_cities_highst_average_price":"Top Cities with Highest Average Rental Prices",
        "count":"Count",
        "areaVsEvaluation":"Relationship Between Area and Evaluation",
        "evaluation":"Evaluation",
        "evaluation_distribution":"Evaluation Distribution",
        "area_dist_Properties":"Area Distribution of Properties",
        "most_common_Neighborhood":"Most Common Neighborhood",
        "total_properties_area": "Total Properties",
        "average_area": "Average Area",
        "largest_property": "Largest Property",
        "total_evaluated_properties": "Total Evaluated Properties",
        "highest_evaluation": "Highest Evaluation",
        "area_input_W":"Area should be between 10-100,000 Sq. Meter",
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
        "price_distribution_all":"توزيع الأسعار في كل المناطق",
        "property_category_analysis": "تحليل فئات العقارات",
        "select_category": "اختر فئة عقارية",
        "area_insights":"رؤى المساحة",
        "evaluation_insights":"رؤى التقييم",
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
        "NeighborhoodsVsEvaluation":"أعلى 10 أحياء تقييماً",
        "NeighborhoodsVsPropertyCount":"أفضل 10 أحياء حسب عدد العقارات",
        "neighborhood":"الحيّ",
        "average_evaluation_city":"متوسط التقييم حسب المدينة",
        "top_cities_highst_average_price":"أعلى المدن حسب متوسط أسعار التأجير",
        "count":"العدد",
        "areaVsEvaluation":"العلاقة بين المساحة والتقييم",
        "evaluation":"التقييم",
        "evaluation_distribution":"توزيع فئات التقييم",
        "area_dist_Properties":"توزيع العقارات حسب المساحة",
        "most_common_Neighborhood":"الحي الأكثر شيوعاً",
        "total_properties_area": "إجمالي العقارات",
        "average_area": "متوسط المساحة",
        "largest_property": "أكبر عقار",
        "total_evaluated_properties": "إجمالي العقارات المقيمة",
        "highest_evaluation": "أعلى تقييم",
        "area_input_W":"المساحة يجب أن تكون بين 10 -100،000 متر مربع",
    }
}

# Get selected language texts
t = texts[language]

def render_metrics(columns, metrics):
    """
    Render metrics dynamically in columns.

    :param columns: List of st.columns
    :param metrics: List of tuples containing (label, value, color)
    """
    for col, (label, value, color) in zip(columns, metrics):
        col.markdown(
            f"<div style='font-size: 20px; color: white; background-color: {color}; padding: 10px; border-radius: 5px;'>"
            f"<b>{label}</b><br>{value}</div>", unsafe_allow_html=True)
def pie_render(data,names,title):
    fig = px.pie(data, names=names, title=t[title])
    fig.update_layout(title_x=float(t["title_location"]))
    st.plotly_chart(fig, theme=None, use_container_width=True)

def bar_render(data,x,y,title,index_label, value_label):
    fig = px.bar(df[y].value_counts().reset_index(),
        x=x, y=y, title=t[title],color=x,
        labels={'index': t[index_label], y: t[value_label], 'color':'Legend'})
    fig.update_layout(title_x=float(t["title_location"]))
    st.plotly_chart(fig, theme=None, use_container_width=True)

def bar_b_render(data, x, y, title, xaxis_title, yaxis_title,sp_title, color,orientation):
    fig1 = px.bar(data, x=x, y=y, title=f"{t[title]} {sp_title}", color=color, labels={'color': 'Legend'}, orientation=orientation)
    fig1.update_layout(
    title_x=float(t["title_location"]),
    xaxis_title=t[xaxis_title],  # Custom x-axis label
    yaxis_title=t[yaxis_title]   # Custom y-axis label
    )
    st.plotly_chart(fig1, theme=None, use_container_width=True)

def scatter_render(data,x,y,title,x_label,y_label,hover1,hover2,size,color):
    fig = px.scatter(data, x=x, y=y, title=title,
        labels={x: t[x_label], y: t[y_label]},
        hover_data=[hover1, hover2],size=size, color=color
        )
    if hover1 != None and hover2 != None:
        fig.update_traces(marker=dict(size=15, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(title_x=float(t["title_location"]))
    st.plotly_chart(fig, theme=None, use_container_width=True)

def box_render(data,x,y,title,sp_title,xaxis_title,yaxis_title,color):
    fig = px.box(data,x=x, y=y, title=f"{t[title]} {sp_title}", color=color, labels={'color': 'Legend'})
    fig.update_layout(
    title_x=float(t["title_location"]),
    xaxis_title=t[xaxis_title],  # Custom x-axis label
    yaxis_title=t[yaxis_title]   # Custom y-axis label
        )
    st.plotly_chart(fig, theme=None, use_container_width=True)

def histogram_render(data,x,title,color,xaxis_title,yaxis_title):
    fig = px.histogram(data, x=x, title=t[title], color=color,labels={'color': 'Legend'})
    fig.update_layout(
    title_x=float(t["title_location"]),
    xaxis_title=t[xaxis_title],  # Custom x-axis label
    yaxis_title=t[yaxis_title]   # Custom y-axis label
        )
    st.plotly_chart(fig, theme=None, use_container_width=True)
                
# Apply RTL or LTR based on language
if language == "العربية":
    st.markdown('<style>body { direction: RTL; text-align: right; }</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body { direction: LTR; text-align: left; }</style>', unsafe_allow_html=True)

# Load the dataset
@ st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

df = load_data('real_estate_rental_prices_extended.xlsx')

if df.empty:
    st.error("The dataset is empty or invalid.")
else:
    # Sidebar Navigation
    options = st.sidebar.radio(t["page_title"], 
                                [t["general_overview"], 
                                 t["city_insights"], 
                                 t["neighborhood_insights"],
                                 t["property_category_analysis"],
                                 t["area_insights"],
                                 t["evaluation_insights"],                             
                                 t["predictive_modeling"]])
    ########################################################################################################################################################################
    # General Overview Dashboard
    if options == t["general_overview"]:
        st.header(t["general_overview"])
        # Key Metrics
        columns = st.columns(4)
        metrics = [
            (t['total_properties'], len(df), "#FF6347"),
            (t['average_price'], f"${df['OneNightPrice'].mean():.2f}", "#4682B4"),
            (t['average_evaluation'], f"{df['Evaluation'].mean():.2f}", "#32CD32"),
            (t['most_common_category'], f"{df['RsCategory'].mode()[0]}", "#FFD700")

        ]
        render_metrics(columns, metrics)
        # Visualizations
        col21, col22 = st.columns(2)
        with col21:
            st.subheader(t["property_distribution"])
            pie_render(df,names='RsCategory',title ="property_distribution") 
        with col22:
            st.subheader(t["city_insights"])
            bar_render(df,x='index',y='City',title='city_insights',index_label='city',value_label='total_properties')
    ########################################################################################################################################################################
    # City Insights Dashboard
    elif options == t["city_insights"]:
        st.header(t["city_insights"])
        # Select City
        city_selected = st.sidebar.selectbox(t["select_city"], df['City'].unique())
        city_data = df[df['City'] == city_selected]

        # Metrics
        columns = st.columns(4)
        metrics = [
            (t['total_properties'], len(city_data), "#FF6347"),
            (t['average_price'], f"${city_data['OneNightPrice'].mean():.2f}", "#4682B4"),
            (t['average_evaluation'], f"{city_data['Evaluation'].mean():.2f}", "#32CD32"),
            (t['most_common_category'], f"{city_data['RsCategory'].mode()[0]}", "#FFD700")

        ]
        render_metrics(columns, metrics)

        # Visualizations
        col21, col22 = st.columns(2)
        with col21:
            st.subheader(f"{t['area_vs_price']} {city_selected}")
            scatter_render(data=city_data,x='Area',y='OneNightPrice',title=f"{t['area_vs_price']} {city_selected}", x_label='area', y_label='average_price', hover1='City',hover2='RsCategory',size='OneNightPrice',color='Area')
        with col22:
            # Average Rental Price by City
            # What is the average nightly rental price for properties in each city?
            # Group by city and calculate the average price
            st.subheader(f"{t['price_distribution_all']}")
            avg_price_by_city = df.groupby('City')['OneNightPrice'].mean().sort_values().reset_index()
            bar_b_render(data=avg_price_by_city,x='City', y='OneNightPrice', title='price_distribution_all', xaxis_title='city', yaxis_title='average_price',sp_title='', color='City',orientation='v')
            
        col31, col32 = st.columns(2)
        with col31:
            # Price Distribution Across Categories
            # How does the rental price vary across different property categories?
            # Plot the box plot
            st.subheader(f"{t['price_distribution']}  {city_selected}")
            box_render(data=city_data,x='RsCategory', y='OneNightPrice', title='price_distribution',sp_title=city_selected,xaxis_title='rsCategory',yaxis_title='one_night_price',color='RsCategory')
        with col32:
            # Top Cities with Highest Average Rental Prices
            # Which cities have the most expensive properties on average?
            st.subheader(f"{t['top_cities_highst_average_price']}")
            avg_price_by_city = df.groupby('City')['OneNightPrice'].mean().sort_values(ascending=False).reset_index()
            bar_b_render(data=avg_price_by_city, x='OneNightPrice', y='City',title='top_cities_highst_average_price', xaxis_title='city', yaxis_title='average_price',sp_title='', color='City',orientation='h')
    ########################################################################################################################################################################
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
        columns = st.columns(3)
        metrics = [
            (t['total_properties'], len(df_filtered), "#FF6347"),
            (t['average_price'], f"${df_filtered['OneNightPrice'].mean():.2f}", "#4682B4"),
            (t['average_evaluation'], f"{df_filtered['Evaluation'].mean():.2f}", "#32CD32")

        ]
        render_metrics(columns, metrics)
        # Visualizations
        col11, col12 = st.columns(2)
        with col11:
            st.subheader(f"{t['price_distribution']} {neighborhood_selected}")
            box_render(data=df_filtered,x='Neighbourhood', y='OneNightPrice', title='price_distribution',sp_title=neighborhood_selected,xaxis_title='neighborhood',yaxis_title='one_night_price',color='Neighbourhood')
        with col12:
            # Popular Neighborhoods Based on Evaluations
            # Which neighborhoods have the highest average evaluations?
            # Group by neighborhood and calculate the average evaluation
            st.subheader(f"{t['NeighborhoodsVsEvaluation']} {city_selected} ")
            avg_evaluation_by_neighborhood = (
                df_neighborhoods_with_evaluation.groupby('Neighbourhood')['Evaluation'].mean().sort_values(ascending=False).head(10).reset_index())
            bar_b_render(data=avg_evaluation_by_neighborhood, x='Neighbourhood', y='Evaluation',title='NeighborhoodsVsEvaluation', xaxis_title='neighborhood', yaxis_title='average_evaluation',sp_title=city_selected, color='Neighbourhood',orientation='v')
        col21, col22 = st.columns(2)
        with col21:
            # Property Distribution by Neighborhood
            #  Which neighborhoods have the highest number of properties?
            st.subheader(f"{t['most_common_Neighborhood']} {city_selected} ")
            neighborhood_count = df['Neighbourhood'].value_counts().head(10)
            bar_b_render(data=neighborhood_count, x=neighborhood_count.index, y=neighborhood_count.values,title='most_common_Neighborhood', xaxis_title='neighborhood', yaxis_title='count',sp_title=city_selected, color='Neighbourhood',orientation='v')
    ########################################################################################################################################################################
    # Property Category Analysis Dashboard
    elif options == t["property_category_analysis"]:
        st.header(t["property_category_analysis"])
        # Select Property Category
        category_selected = st.sidebar.selectbox(t["select_category"], df['RsCategory'].unique())
        df_filtered = df[df['RsCategory'] == category_selected]

        # Metrics
        columns = st.columns(3)
        metrics = [
            (t['total_properties'], len(df_filtered), "#FF6347"),
            (t['average_price'], f"${df_filtered['OneNightPrice'].mean():.2f}", "#4682B4"),
            (t['average_evaluation'], f"{df_filtered['Evaluation'].mean():.2f}", "#32CD32")

        ]
        render_metrics(columns, metrics)
        # Visualizations
        col11, col12 = st.columns(2)
        with col11:
            # Distribution of Property Categories
            # How are the different types of properties distributed across the dataset?
            # Count property categories
            st.subheader(f"{t['property_distribution']} ")
            category_distribution = df['RsCategory'].value_counts().reset_index()
            pie_render(df,names='RsCategory',title ="property_distribution") 
        with col12:
            # Price Distribution Across Categories
            # How does the rental price vary across different property categories?
            st.subheader(f"{t['price_distribution']}")
            box_render(data=df,x='RsCategory', y='OneNightPrice', title='price_distribution',sp_title='',xaxis_title='rsCategory',yaxis_title='one_night_price',color='RsCategory')
        col21, col22 = st.columns(2)
        with col21:
            # Most Popular Property Categories
            #Which property types are most common in the dataset?
            st.subheader(f"{t['most_common_category']}")
            property_category_count = df['RsCategory'].value_counts()
            bar_b_render(data=property_category_count, x=property_category_count.index, y=property_category_count.values,title='most_common_category', xaxis_title='rsCategory', yaxis_title='count',sp_title='', color='RsCategory',orientation='v')
        with col22:
            st.subheader(f"{t['price_distribution']} {category_selected}")
            box_render(data=df_filtered,x='RsCategory', y='OneNightPrice', title='price_distribution',sp_title=category_selected,xaxis_title='rsCategory',yaxis_title='one_night_price',color='RsCategory')
    ########################################################################################################################################################################
    # Area Insights Dashboard
    elif options == t["area_insights"]:
        st.header(t["area_insights"])
        # Metrics
        columns = st.columns(3)
        metrics = [
            (t['total_properties'], len(df), "#FF6347"),
            (t['average_area'], f"${df['Area'].mean():.2f} sq.m", "#4682B4"),
            (t['largest_property'], f"{df['Area'].max()} sq.m", "#32CD32")

        ]
        render_metrics(columns, metrics)
        # Visualizations
        col11, col12 = st.columns(2)
        with col11:
            # Relationship Between Area and Price
            # How does the area of a property affect its nightly rental price?
            st.subheader(f"{t['area_vs_price']}")
            scatter_render(data=df,x='Area', y= 'OneNightPrice', title=t["area_vs_price"],x_label='area',y_label='one_night_price',hover1='City',hover2='Neighbourhood',size='OneNightPrice', color='Area')
        with col12:
            # Relationship Between Area and Evaluation
            # Is there a correlation between the area of a property and its evaluation?
            st.subheader(f"{t['areaVsEvaluation']}")
            scatter_render(data=df,x='Area', y= 'Evaluation', title=t["areaVsEvaluation"],x_label='area',y_label='evaluation',size='Evaluation', color='Area',hover1=None,hover2=None)
        col21, col22 = st.columns(2)
        with col21:
            # Area Distribution
            # What is the distribution of property areas?
            st.subheader(f"{t['area_dist_Properties']}")
            box_render(data=df,x='RsCategory', y='Area', title='area_dist_Properties',sp_title='',xaxis_title='rsCategory',yaxis_title='area',color='RsCategory')
    ########################################################################################################################################################################
    # Evaluation Insights Dashboard
    elif options == t["evaluation_insights"]:
        st.header(t["evaluation_insights"])
        # Metrics
        columns = st.columns(3)
        metrics = [
            (t['total_evaluated_properties'], df['Evaluation'].count(), "#FF6347"),
            (t['average_evaluation'], f"{df['Evaluation'].mean():.2f}", "#4682B4"),
            (t['highest_evaluation'], f"{df['Evaluation'].max()}", "#32CD32")

        ]
        render_metrics(columns, metrics)

        # Visualizations
        col11, col12 = st.columns(2)
        with col11:
            # Average Evaluation by City
            # Which cities have the highest average property evaluations?
            st.subheader(f"{t['average_evaluation_city']}")
            avg_evaluation_by_city = df.groupby('City')['Evaluation'].mean().sort_values().reset_index()
            bar_b_render(data=avg_evaluation_by_city, x='Evaluation', y='City',title='average_evaluation_city', xaxis_title='city', yaxis_title='average_evaluation',sp_title='', color='City',orientation='h')
        with col12:
            # Relationship Between Area and Evaluation
            # Is there a correlation between the area of a property and its evaluation?
            st.subheader(f"{t['areaVsEvaluation']}")
            scatter_render(data=df,x='Area', y= 'Evaluation', title=t["areaVsEvaluation"],x_label='area',y_label='evaluation',size='Evaluation', color='Area',hover1=None,hover2=None)
        col21, col22 = st.columns(2)
        with col21:
            # Evaluation Distribution
            # How are property evaluations distributed across the dataset?
            st.subheader(f"{t['evaluation_distribution']}")
            histogram_render(data=df,x='Evaluation',title='evaluation_distribution',color='Evaluation',xaxis_title='evaluation',yaxis_title='count')
    ########################################################################################################################################################################
    # Predictive Modeling Dashboard
    elif options == t["predictive_modeling"]:
        st.header(t["predictive_modeling"])
        # Inputs
        col11, col12 = st.columns(2)
        col21, col22 = st.columns(2)
        col31, col32 = st.columns(2)
        with col11:
            area_input = st.number_input(t["enter_area"], min_value=10, max_value=100000, step=10)
            if area_input < 10 or area_input > 100000:
                st.warning(t["area_input_W"])
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
    ########################################################################################################################################################################
