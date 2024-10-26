import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import skew
import numpy as np
import folium
import streamlit.components.v1 as components
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import cross_validate,KFold
from sklearn.preprocessing import MinMaxScaler
import pickle
from PIL import Image
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import streamlit as st
from streamlit_option_menu import option_menu



with open("model.pkl","rb") as file:
    class_model = pickle.load(file)

df= pd.read_csv(r"C:\Users\ASUS\Documents\GUVI ZEN CLASSES\MAINT BOOT\Crime Analyser\Sample Crime Dataset - Sheet1.csv")
df = df.dropna()

df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%y %H:%M')

df['Updated On'] = pd.to_datetime(df['Updated On'],format='%m/%d/%y %H:%M')

df['month_name'] = df['Date'].dt.month_name()

df['day'] = df['Date'].dt.day_name()

df['hour'] = df['Date'].dt.hour

df['year'] = df['Date'].dt.year

df.drop(columns=['ID','Case Number','Block','IUCR','FBI Code','X Coordinate','Y Coordinate'],inplace=True)

summer_months = ['June','July','August']
spring = ['March','April','May']
autumn = ['September','October','November']
winter = ['December','January','February']


df['summer'] = df['month_name'].apply(lambda x: 'summer' if x in summer_months else 'not summer' )
df['autumn'] = df['month_name'].apply(lambda x: 'autumn' if x in autumn else 'not autumn')
df['spring'] = df['month_name'].apply(lambda x: 'spring' if x in spring else 'not spring')
df['winter'] = df['month_name'].apply(lambda x: 'winter' if x in winter else 'not winter')

desc_list = df['Description'].unique().tolist()

desc_list_encoded =  {key : i for  i,key in enumerate(desc_list)}

prime_type_list = df['Primary Type'].unique().tolist()

prime_type_list_encoded = {key : i for i,key in enumerate(prime_type_list)}

reversed_prime_type = {i : key for i,key in enumerate(prime_type_list)}

domestic_list = df['Domestic'].unique().tolist()

domestic_list_encoded = {key : i for i,key in enumerate(domestic_list)}

location_list = df['Location'].unique().tolist()

location_list_encoded = {key : i for i,key in enumerate(location_list)}

district_list = df['District'].unique().tolist()

district_list_encoded = {key : i for i,key in enumerate(district_list)}

def map_data(df):
    df['Primary Type'] = df['Primary Type'].map(prime_type_list_encoded)
    df['Description'] = df['Description'].map(desc_list_encoded)
    df['Domestic'] = df['Domestic'].map(domestic_list_encoded)
    df['Location'] = df['Location'].map(location_list_encoded)
    df['District'] = df['District'].map(district_list_encoded)
    return df

def data_type_convert(df):     
    df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%y %H:%M')
    df['Updated On'] = pd.to_datetime(df['Updated On'],format='%m/%d/%y %H:%M')
    df['month_name'] = df['Date'].dt.month_name()
    df['day'] = df['Date'].dt.day_name()
    df['hour'] = df['Date'].dt.hour
    df['year'] = df['Date'].dt.year
    return df

crime_grp = df.groupby('Primary Type')

no_of_crime_year = crime_grp['Year'].value_counts().sort_values(ascending=False)
crime_yr_df = pd.DataFrame(no_of_crime_year).reset_index()

def year_freq():
    no_of_crime_year = crime_grp['Year'].value_counts().sort_values(ascending=False)
    crime_yr_df = pd.DataFrame(no_of_crime_year).reset_index()
    return crime_yr_df

def year_plot():
    fig,ax = plt.subplots(figsize=(6,4))
    ax.bar(crime_yr_df['Year'],crime_yr_df['count'],align='edge')
    ax.set_xlabel("years")
    ax.set_ylabel("No of crimes")
    ax.set_title('Yearwise crime Occurence')
    ax.legend()
    return fig

def month_wise_crime():
    no_of_crime_month = crime_grp['month_name'].value_counts().sort_values(ascending=False)
    crime_month_df_month = pd.DataFrame(no_of_crime_month).reset_index()
    return crime_month_df_month

def monthly_plot():
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(month_wise_crime()['month_name'],month_wise_crime()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('months')
    ax.set_ylabel('No of crime')
    ax.set_title('Monthwise crime occurence')
    return fig

def day_wise_crime():
    no_of_crime_day = crime_grp[['Year','month_name','day']].value_counts().sort_values(ascending=False)
    crime_day_df = pd.DataFrame(no_of_crime_day).reset_index()
    return crime_day_df

def day_plot():
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(day_wise_crime()['day'],day_wise_crime()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Days')
    ax.set_ylabel('No of crimes')
    ax.set_title('Daywise crime Occurence')
    return fig

def hour_crime():
    no_of_crime_hour =  crime_grp[['Date','hour']].value_counts().sort_values(ascending=False)
    crime_hour_df = pd.DataFrame(no_of_crime_hour).reset_index()
    return crime_hour_df

def hourly_crime_plot():
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(hour_crime()['hour'],hour_crime()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Hours')
    ax.set_ylabel('No of crimes')
    ax.set_title('Hourwise_Crime')
    return fig

def crime_rich_area():
    data = df[['Primary Type','Location Description']].value_counts()
    crime_area_df = pd.DataFrame(data).reset_index()
    return crime_area_df

def crime_rich_area_plot():
    fig,ax=plt.subplots(figsize=(12,8))
    ax.bar(crime_rich_area()['Location Description'],crime_rich_area()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Locations')
    ax.set_ylabel('No of crimes')
    ax.set_title('Crime rich Areas')
    return fig

def community_area():
    data_9 = crime_grp['Community Area'].value_counts().sort_values(ascending=False)
    com_area_df = pd.DataFrame(data_9).reset_index()
    return com_area_df

def community_area_plot():
    fig,ax = plt.subplots(figsize=(12,10))
    ax.bar(community_area()['Primary Type'],community_area()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('crime_count')
    ax.set_title('Community_Area_Crime_Count')
    return fig

def beat_area_crime():
    data_10 = crime_grp['Beat'].value_counts().sort_values(ascending=False)
    beat_df = pd.DataFrame(data_10).reset_index()
    return beat_df

def beat_area_crime_plot():
    fig, ax = plt.subplots(figsize=(10,8))
    ax.bar(beat_area_crime()['Primary Type'],beat_area_crime()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('crime_count')
    ax.set_title('Beat_Area_Crime_Count')
    return fig

def com_and_beat():
    com_and_beat = pd.concat([community_area(),beat_area_crime()],axis=1,join='inner')
    return com_and_beat

def com_and_beat_df():
    data_14 = {'COMMUNITY AREA CRIME COUNT':479,'BEAT AREA CRIME COUNT':749}
    com_vs_beat_df = pd.DataFrame(data_14,index=['A'])
    return com_vs_beat_df

def com_vs_beat():
    fig,ax = plt.subplots(figsize=(12,7))
    values = [community_area().value_counts().sum(),beat_area_crime().value_counts().sum()]
    category = [community_area().columns[1],beat_area_crime().columns[1]]
    color = ['red','orange']
    #explod = [0.1,0.2]
    ax.pie(values,labels=category,colors = color,autopct="%1.1f%%")
    ax.set_title("Community_area vs Beat crime")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig

    

df = df.dropna(subset=['Latitude', 'Longitude'])

# ## Heat Map to show the density of the crime incidents -View -I##

def crime_density():
    map_center = [41.8337734,-88.0616153]
    crime_map = folium.Map(location=map_center, zoom_start=8)

    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]

    HeatMap(heat_data).add_to(crime_map)

    for index, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=(f"crime_type:{row['Primary Type']} location :{row['Location Description']}")
        ).add_to(crime_map)

    return crime_map

map_ = crime_density()

map_.save("crime_map.html")

## Map to show the density of the crime incidents - View -II ##

def crime_location_2():
    crime_map_1= folium.Map(location=[41.8337734,-88.0616153],
                       zoom_start=12)

    marker_cluster = MarkerCluster().add_to(crime_map_1)

    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,  
            color='red', 
            fill=True,
            fill_color='blue',  
            fill_opacity=0.6,  
            popup=(f"crime_type:{row['Primary Type']} location :{row['Location Description']}")  
        ).add_to(crime_map_1)    

    return crime_map_1

map_1= crime_location_2()

map_1.save("crime_map_2.html")

## Repeat Crime Location ##

def repeat_crime_location():

    lat_str, lon_str = df['Location'].mode()[0].strip("()").split(",")

    lat_str_1, lon_str_1 = df['Location'].mode()[1].strip("()").split(",")

    mapObj = folium.Map(location=[lat_str,lon_str],zoom_start=12,zoom_control = False)

    marker_cluster = MarkerCluster().add_to(mapObj)



    folium.CircleMarker(location=[float(lat_str),float(lon_str)],radius=5,color='red',fill=True,
                        fill_color='blue',fill_opacity=1.5,
                        popup=f"latitude ;{lat_str},longitude:{lon_str}").add_to(mapObj)

    folium.CircleMarker(location=[float(lat_str_1),float(lon_str_1)],radius=5,color='red',fill=True,
                        fill_color='blue',fill_opacity=1.5,
                        popup=f"latitude ;{lat_str_1},longitude:{lon_str_1}").add_to(mapObj)



    return mapObj

map_2 = repeat_crime_location()

map_2.save("crime_map_2.html")

## District and Ward Analysis ##

def district_crime_data():
    data_7 = crime_grp['District'].value_counts().sort_values(ascending=False)
    dist_df = pd.DataFrame(data_7).reset_index()
    return dist_df

def district_crime_plot():
    fig,ax = plt.subplots(figsize=(12,7))
    ax.bar(district_crime_data()['Primary Type'],district_crime_data()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('Districts')
    ax.set_title('Districtwise_Crime_Count')
    return fig

def ward_crime_data():
    data_8 = crime_grp['Ward'].value_counts().sort_values(ascending=False)
    ward_df = pd.DataFrame(data_8).reset_index()
    return ward_df

def ward_crime_plot():
    fig,ax =plt.subplots(figsize=(12,7))
    ax.bar(ward_crime_data()['Primary Type'],ward_crime_data()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('Wardwise_count')
    ax.set_title('Wardwise_Crime_Count')
    return fig

## Which one has more crime consistency - District or Wards? ##

def crime_consistency():
    data_7 = crime_grp['District'].value_counts().sort_values(ascending=False)
    data_8 = crime_grp['Ward'].value_counts().sort_values(ascending=False)
    std_dev_data = {'dist_std_dev':data_7.std(),'ward_std_dev':data_8.std()}
    plt.rcParams['figure.figsize'] = [4, 4] 
    fig,ax = plt.subplots(figsize=(6,4))
    ax.bar(std_dev_data.keys(),std_dev_data.values(),color=['blue','green'])
    bars = ax.bar(std_dev_data.keys(),std_dev_data.values(),color=['blue','green'])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    ax.set_title('Standard deviation of District and Ward')
    ax.set_ylabel('Standard_deviation')
    return fig

def std_dev_df():
    data_7 = crime_grp['District'].value_counts().sort_values(ascending=False)
    data_8 = crime_grp['Ward'].value_counts().sort_values(ascending=False)
    std_dev_data = {'DISTRCT-STD-DEIVATION':data_7.std(),'WARD-STD-DEVIATION':data_8.std()}
    std_dev_df = pd.DataFrame(std_dev_data,index=([0]))
    return std_dev_df


def std_deviation():

    data_7 = crime_grp['District'].value_counts().sort_values(ascending=False)
    dist_df = pd.DataFrame(data_7).reset_index()
    dist_std_dev = dist_df.std()
    data_8 = crime_grp['Ward'].value_counts().sort_values(ascending=False)
    ward_df = pd.DataFrame(data_8).reset_index()
    ward_std_dev = ward_df.std()
    std_dev_data = {'dist_std_dev':data_7.std(),'ward_std_dev':data_8.std()}
    plt.figure(figsize=(6,4))
    plt.bar(std_dev_data.keys(),std_dev_data.values(),color=['orange','red'])
    plt.title('Standard deviation of District and Ward')
    plt.ylabel('Standard_deviation')
    plt.show()

## Most frequent crime type in districts ans wards: ##

def frq_crime_type_district():

    data_7 = crime_grp['District'].value_counts().sort_values(ascending=False)
    dist_df = pd.DataFrame(data_7).reset_index()
    fig,ax=plt.subplots(figsize=(11,7))
    ax.bar(dist_df['Primary Type'].mode().tolist(),dist_df['Primary Type'].mode().value_counts())
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime Type')
    ax.set_ylabel('Crime Count')
    ax.set_title('Most frequent crime type in districts')
    return fig

def frq_crime_type_ward():

    data_8 = crime_grp['Ward'].value_counts().sort_values(ascending=False)
    ward_df = pd.DataFrame(data_8).reset_index()
    fig,ax = plt.subplots(figsize=(11,9))
    ax.bar(ward_df['Primary Type'].mode().tolist(),ward_df['Primary Type'].mode().value_counts())
    ax.set_title('Most frequent crime type in wards')
    ax.set_xlabel('Crime Type')
    ax.set_ylabel('Crime Frequency')
    return fig

## Arrest and Domestic incident analysis ##

def arrest_rate():
    data_3 = crime_grp[['Date','Location Description','Arrest']].value_counts(normalize=True).sort_values(ascending=False)
    arrest_df = pd.DataFrame(data_3).reset_index()
    arrest_df.rename(columns={'proportion':'Arrest_rate'},inplace=True)
    arrest_df.drop(['Arrest'],axis=1,inplace=True)
    return arrest_df
    
def arrest_rate_plot():
    fig,ax= plt.subplots(figsize=(12,7))
    ax.bar(arrest_rate()['Primary Type'],arrest_rate()['Arrest_rate'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('Arrest_Rates')
    ax.set_title('Arrest_Rates')
    return fig


def domestic_crime():
    domestic_crimes = df['Domestic'] == True
    data_5 = df.loc[domestic_crimes]['Primary Type']
    domestic_crime_df = pd.DataFrame(data_5)
    domestic_crime_df.rename(columns={'Primary Type':'Domestic_crime'},inplace=True)
    return domestic_crime_df

def non_domestic_crime():
    non_dome_crime = df['Domestic'] == False
    data_6 = df[non_dome_crime]['Primary Type']
    non_dom_crime_df = pd.DataFrame(data_6)
    non_dom_crime_df.rename(columns={'Primary Type':'Non_domestic_crime'},inplace=True)
    return non_dom_crime_df

def crime_type():
    crime_type_df = pd.concat([domestic_crime(),non_domestic_crime()],axis=1)
    crime_type_df.rename(columns={'Primary Type':'Domestic_crimes'})
    return crime_type_df

def domeestic_crime_new():
    dome_data = crime_type()['Domestic_crime'].value_counts().sort_values(ascending=False)
    dome_data_df = pd.DataFrame(dome_data).reset_index()
    return dome_data_df

def non_dome_crime_new():
    non_dome_data = crime_type()['Non_domestic_crime'].value_counts().sort_values(ascending=False)
    non_dome_data_df = pd.DataFrame(non_dome_data).reset_index()
    return non_dome_data_df


def domestic_crime_count_plot():
    domestic_crime_counts = crime_type()['Domestic_crime'].value_counts()
    fig,ax = plt.subplots(figsize=(12,8))
    ax.bar(domestic_crime_counts.index.astype(str),crime_type()['Domestic_crime'].value_counts())
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('No of crimes')
    ax.set_title('Domestic_crime')
    return fig

def non_domestic_crime_plot():
    non_domestic_crime_counts = crime_type()['Non_domestic_crime'].value_counts()
    fig,ax =plt.subplots(figsize=(12,7))
    ax.bar(non_domestic_crime_counts.index.astype(str),crime_type()['Non_domestic_crime'].value_counts())
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('No of crimes')
    ax.set_title('Non_domestic_crime')
    return fig

## Domestic vs. Non-Domestic Crimes : Which is high? ##

def domestic_vs_non_domestic():
    fig,ax = plt.subplots(figsize=(10,6))
    crime_data = crime_type()
    values = [crime_data['Domestic_crime'].value_counts().sum(),crime_data['Non_domestic_crime'].value_counts().sum()]
    category = [crime_data.columns[0],crime_data.columns[1]]
    color = ['orange','red']
    explod = [0.1,0.2]
    ax.pie(values,labels=category,colors = color,explode=explod,autopct="%1.1f%%")
    ax.set_title('DOMESTIC VS NON-DOMESTIC CRIMES -WHICH IS HIGH?')
    ax.legend()
    return fig

## Crime Type Analysis ##

def crime_distribution():
    crm_type = crime_grp['Description'].value_counts().sort_values(ascending=False)
    crm_type_df = pd.DataFrame(crm_type).reset_index()
    return crm_type_df

def crime_distribution_plot():
    fig,ax = plt.subplots(figsize=(12,7))
    ax.bar(crime_distribution()['Primary Type'],crime_distribution()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Type')
    ax.set_ylabel('Crime_frequency')
    ax.set_title('Distribution of Crime_types')
    return fig

## Severity Analysis ##

def crime_severity():
    severe_crimes = ['HOMICIDE', 'ASSAULT', 'BATTERY', 'ROBBERY', 'CRIM SEXUAL ASSAULT']
    less_severe_crimes = ['THEFT', 'FRAUD', 'DECEPTIVE PRACTICE', 'CRIMINAL DAMAGE', 'PUBLIC PEACE VIOLATION']
    crime_grp = df.groupby('Primary Type')
    df['severity'] = df['Primary Type'].apply(lambda x: 'severe' if x in severe_crimes else 'less_severe' if x in less_severe_crimes else 'other')
    data_11 = crime_grp['severity'].value_counts().sort_values(ascending=False)
    severe_df = pd.DataFrame(data_11).reset_index()
    return severe_df

def crime_severity_plot():
    fig,ax = plt.subplots(figsize=(8,6))
    ax.bar(crime_severity()['severity'],crime_severity()['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Crime_Severity')
    ax.set_ylabel('Crime_frequency')
    ax.set_title('Distribution of Severe vs. Less Severe Crime')
    return fig

## Seasonal and Weather Impact ##

def seasons_crime():
    heads = pd.Series(df[['summer','autumn','winter','spring']].value_counts().index.names)
    values = pd.Series(df[['summer','autumn','winter','spring']].value_counts().values)
    data_13 = {'seasons':heads,'crime_numbers':values}
    season_df = pd.DataFrame(data_13)
    return season_df


def seasons_plot():
    fig,ax=plt.subplots(figsize=(8,6))
    ax.bar(df[['summer','autumn','winter','spring']].value_counts().index.names,df[['summer','autumn','winter','spring']].value_counts().values)
    plt.xticks(rotation=90)
    ax.set_xlabel('Seasons')
    ax.set_ylabel('Crime_frequency')
    ax.set_title('Seasonal Trends vs. Crime Frequency')
    return fig

def get_prediction(description,domestic,district,location):
    arr = np.array([description,domestic,district,location])
    input_data = arr.reshape(1,4)
    prediction =class_model.predict(input_data)
    pred = prediction.tolist()
    pred_1 = int(pred[0])
    pred_new = reversed_prime_type[pred_1]
    return pred_new
    

st.set_page_config(layout='wide')

title_text = '''<h1 style = 'font-size : 55px;text-align:center;color:purple;background-color:lightgrey;'>Chicago Crime Analyzer</h1>'''
st.markdown(title_text,unsafe_allow_html=True)

with st.sidebar:

    image = Image.open('chicago_police_logo_1.jpeg')

    st.image(image)

    select = option_menu('MAIN_MENU',['HOME','ABOUT','ANALYTICS','PREDICTION'])

if select == 'HOME':
        
    st.write(" ")
    st.write(" ")
    st.header(":violet[Dataset Description:]")
    with st.container(border=True):
        st.markdown('''<h5 style='color:#00ffff;font-size:21px'> The dataset contains records of reported crimes in Chicago. 
                    Each record includes details such as the type of crime, location, arrest status, and other relevant information..''',
                    unsafe_allow_html=True)
        
        st.write(" ")
    st.subheader(":blue[Objective]")
    with st.container(border=True):
        st.markdown('''<h5 style='color:#00ffff;font-size:21px'>The primary objective of this project is to leverage historical and recent crime data to identify patterns, trends, 
                    and hotspots within Chicago. By conducting a thorough analysis of this data, the outcome will support strategic 
                    decision-making, improve resource allocation, and contribute to reducing crime rates and enhancing public safety.
                    The task is to provide actionable insights that can shape our crime prevention strategies, 
                    ensuring a safer and more secure community. This project will be instrumental in aiding law enforcement
                    operations and enhancing the overall effectiveness of our efforts in combating crime in Chicago.''',unsafe_allow_html=True)

elif select == 'ABOUT':

    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.markdown('''<h6 style ='color:#ff1a66;font-size:31px'>Project Title:Chicago Crime Analyser''',unsafe_allow_html=True)

    st.markdown('''<h5 style ='color:#007acc;font-size:31px'>Domain:<br>Crimonology and Crime Prevention''',unsafe_allow_html=True)

    st.markdown('''<h5 style ='color:#00ffff;font-size:31px'>Take away skills:<br>Python Scipting<br>Data manipulation<br>EDA<br>
                Model Building<br>Model Deployment in Streamlit''',unsafe_allow_html=True)
               


elif select == 'ANALYTICS':

    st.markdown("""
<style>

	.stTabs [role="tab"] {font-size: 52px;
		gap: 2px;
    }

	.stTabs [role="tab"] {
		height: 45px;
        white-space: pre-wrap;
		background-color: #C0C0C0;
		border-radius: 4px 4px 0px 0px;
		gap: 10px;
    padding-top: 10px;
		padding-bottom: 10px;
    padding: 30px 40px;
    width: 800px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #C0C0C0;
      color:red;font-weight:bold;color:blue;
      font-size: 31px;
	}

</style>""", unsafe_allow_html=True)
  
    st.markdown("""
    <style>
    .stTabs [role="tab"] {
        font-size: 84px;font-weight: bold;
        padding: 10px 20px;color:blue;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;  
        color: black;  
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    
    tab1,tab2,tab3,tab4,tab5 = st.tabs(['TIME SERIES ANALYSIS','LOCATION - SPECIFIC ANALYSIS','CRIME PATTERN ANALYSIS','GEOSPATIAL ANALYSIS','DISTRICT AND WARD ANALYSIS'])

    tab6,tab7,tab8,tab9,tab10 = st.tabs(['ARREST RATES','DOMESTIC AND NON-DOMESTIC CRIME','CRIME DISTRIBUTION','SEVERITY ANALYSIS','SEASONAL AND WEATHER IMPACT'])

    video_file = open("C:/Users/ASUS/Downloads/crime_analytic_image_horizontally (1).mp4", "rb")
    
    video_bytes = video_file.read()

    st.video(video_bytes)

    with tab1:

        option_1 =  st.radio("Select the Option",['NONE','YEAR','MONTH','DAY','HOUR'],index=None)

        if option_1 == 'NONE':

            st.write('PLEASE CHECK OTHERS')        

        if option_1 == 'YEAR':

            col1,col2 = st.columns(2)

            with col1:
                yr_crime = year_freq()
                st.dataframe(yr_crime,width=600,height=430)
                
            with col2:
                fig = year_plot()            
                st.pyplot(fig)

        if option_1 == 'MONTH':

            col1,col2 =st.columns(2)

            with col1:
                
                month_crime = month_wise_crime()
                st.dataframe(month_crime,width=600,height=510)

            with col2:

                fig_1 = monthly_plot()
                st.pyplot(fig_1)

        if option_1 == 'DAY':

            col1,col2 = st.columns(2)

            with col1:

                day_crime = day_wise_crime()
                st.dataframe(day_crime,width=600,height=510)

            with col2:

                daywise_plot = day_plot()
                st.pyplot(daywise_plot)

        if option_1 == 'HOUR':

            col1,col2 = st.columns(2)

            with col1:

                hourly_crime = hour_crime()
                st.dataframe(hourly_crime,width=600,height=440)

            with col2:

                hour_plot = hourly_crime_plot()
                st.pyplot(hour_plot)

        if st.button('Recommendations',use_container_width=True):
            with st.container(border=True):
                st.subheader('YEARWISE CRIME OCCURENCE')
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Investigate the cause:<br> The spike in 2024 suggests an anomaly or 
                            significant events leading to increased crime. A thorough investigation into local or national events that might 
                            have triggered this surge is necessary.''',unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Allocate resources effectively:<br>Since crime has drastically increased
                                in 2024, local authorities should prioritize and allocate more resources (e.g., police presence, surveillance) 
                                for crime prevention.''',unsafe_allow_html=True)

            with st.container(border=True):
                st.subheader('MONTHWISE CRIME OCCURENCE')
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Observation:<br>September shows a significant spike in crime, followed by August, while other months have a much lower crime rate.
                            Seasonal patterns:<br> September might be a high-risk month due to specific events (festivals, public gatherings, 
                            vacations). Local law enforcement should be prepared for increased crime during this period and take preventive 
                            actions.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Increase security in high-risk months:<br> As September and August show 
                            heightened activity, security measures should be intensified during these months. This could involve 
                            increased patrols, surveillance, and community engagement.''',unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Predictive policing:<br> Use this trend to predict and prepare for higher 
                            crime rates in upcoming years during these months. Authorities can implement preventive measures, such as crime awareness 
                            drives and quick-response teams.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.subheader('DAYWISE CRIME OCCURENCE')
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Reinforce Law Enforcement on High-Crime Days<br>Deploy additional patrol 
                            units on Wednesdays:<br> Since Wednesday has the highest crime rate, local law enforcement should consider deploying more 
                            officers, especially in high-crime areas, to deter criminal activity.
                            Adjust resource allocation throughout the week:<br> Focus more resources and manpower on Wednesdays, 
                            Thursdays, and Fridays, as these are the peak crime days. Lesser resources could be allocated to low-crime days 
                            like Saturday.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.subheader('HOURWISE CRIME OCCURENCE')
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Increase Police Presence<br>During Peak HoursDeploy more patrol officers:<br>Reinforce law enforcement visibility from 18:00 to 20:00 in high-crime areas. 
                            Uniformed officers or even undercover teams can act as a deterrent to criminal activity.<br>
                            Focus on hotspots:<br> Use crime data to identify specific locations where incidents are most frequent during these hours, such as busy intersections, shopping districts, parks, or public transport hubs.<br>
                            Create temporary checkpoints:<br>Set up security checkpoints or increase vehicle inspections during these hours, especially in areas prone to vehicle-related crimes (theft, illegal street racing, etc.).''',unsafe_allow_html=True)
            


            
    with tab2:

        option_2 = st.radio("Select the Option",['NONE','AREAS WHERE MORE CRIMES ARE HAPPENING?'],index=None)

        if option_2 == "NONE":

            st.write('PLEASE CHECK OTHERS')

        if option_2 == 'AREAS WHERE MORE CRIMES ARE HAPPENING?':

            col1,col2 = st.columns(2)

            with col1:

                crime_rich = crime_rich_area()
                st.dataframe(crime_rich,width=800,height=650)
            with col2:           
                crime_rich_bar = crime_rich_area_plot()
                st.pyplot(crime_rich_bar)

        

        if st.button('RECOMMENDATION',use_container_width=True):
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Focus on High-Crime Areas:<h5>The locations on the far left,
                            such as Street, Apartment, Small Retail Store, and Parking Lot/Garage, have the highest number of reported crimes. 
                            Authorities should prioritize these locations for enhanced surveillance, patrolling, and safety measures.''',
                            unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Install Security Measures:<h5>Locations like Parking Lots and Garages 
                            can benefit from better lighting, surveillance cameras, and security personnel to deter crime.Residential areas,
                             particularly Apartment Buildings and Residences, should consider additional security systems like alarms, CCTV, 
                            and neighborhood watch programs.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Increase Police Patrols:<h5>Areas with high crime incidence, especially Streets,
                             Apartments, and Retail Stores, should be regularly patrolled by law enforcement officers.Police presence could deter 
                            criminals in public places like Parking Lots and Garages.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Policy Adjustments:<h5>Crime data like this could lead to policy adjustments,
                            such as reallocating resources to higher-crime areas or creating stricter laws for offenders in these locations.''',
                            unsafe_allow_html=True)                


    with tab3:

        option_3 = st.radio("Select the Option",['NONE','COMMUNITY AREA CRIME COUNT','BEAT AREA CRIME COUNT','COMMUNITY AREA VS BEAT AREA'],index=None)

        if option_3 == 'NONE':

            st.write('PLEASE CHECK OTHERS')
        
        
        if option_3 == 'COMMUNITY AREA CRIME COUNT':

            col1,col2 = st.columns(2)

            with col1:

                comm_area = community_area()
                st.dataframe(comm_area,width=800,height=675)

            with col2:

                comm_plot = community_area_plot()
                st.pyplot(comm_plot)

        if option_3 == 'BEAT AREA CRIME COUNT':

            col1,col2 =st.columns(2)

            with col1:

                beat_crime = beat_area_crime()

                st.dataframe(beat_crime,width=800,height=675)

            with col2:

                beat_crime_plot = beat_area_crime_plot()

                st.pyplot(beat_crime_plot)

        if option_3 == 'COMMUNITY AREA VS BEAT AREA':

            col1,col2 =st.columns(2)

            with col1:
                compare_df = com_and_beat_df()
                st.dataframe(compare_df,width=600,height=50)
            with col2:
                grp_plot = com_vs_beat()
                st.pyplot(grp_plot)

        if st.button('RECOMMENDATIONS',use_container_width=True):
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Focus on Crime Prevention for Specific Offenses:<h5>Theft and criminal damage 
                            are leading in both areas, indicating a need for targeted interventions such as increased surveillance and public 
                            awareness campaigns.Motor vehicle theft and battery are also significant and should be addressed through law 
                            enforcement patrols in vulnerable areas.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Resource Allocation:<h5>Since Beat Areas account for 61% of total crimes,
                            it would be strategic to allocate more resources (police presence, crime prevention programs) to these areas to address
                             the higher volume of incidents.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Enhanced Monitoring in Community Areas:<h5>Even though the crime rate is lower 
                            in Community Areas, there are still notable instances of theft, battery, and assault. Improved community engagement,
                            neighborhood watches, and public safety programs could help further reduce these numbers.''',unsafe_allow_html=True)

    with tab4:

        option_4 = st.radio('Select the option',['NONE','CRIME DENSITY- VIEW -I- HEAT MAP','CRIME DENSITY - VIEW - II','REPEAT CRIME LOCATIONS'])

        if option_4 == 'CRIME DENSITY- VIEW -I- HEAT MAP':
                crime_view_one = crime_density()
                st.components.v1.html(map_._repr_html_(), width=1200, height=400,scrolling=True)

        if option_4 == 'CRIME DENSITY - VIEW - II':

            crime_view_two = crime_location_2()
            st.components.v1.html(map_1._repr_html_(),width=1200,height=400,scrolling=True)

        if option_4 == 'REPEAT CRIME LOCATIONS':
            repeat_crime_loc = repeat_crime_location()
            st.components.v1.html(map_2._repr_html_(),width=1200,height=400,scrolling=True)

    with tab5:

        option_5 = st.radio('Select the option',['NONE','DISTRICTWISE CRIME COUNT','WARDWISE CRIME COUNT','WHICH ONE HAS MORE CRIME CONSISTENCY - DISTRICT 0R WARD' ,'RECURRING CRIME TYPE'
                                                     ])
        if option_5 == 'DISTRICTWISE CRIME COUNT':
            
            col1,col2 = st.columns(2)
            with col1:
                dist_crim_data = district_crime_data()
                st.dataframe(dist_crim_data,width=600,height=530)
            with col2:
                dist_crimes_plot = district_crime_plot()
                st.pyplot(dist_crimes_plot)

        if option_5 == 'WARDWISE CRIME COUNT':

            col1,col2 = st.columns(2)
            with col1:
                ward_crime_count = ward_crime_data()
                st.dataframe(ward_crime_count,width=600,height=530)
            with col2:
                ward_crime_count_plot = ward_crime_plot()
                st.pyplot(ward_crime_count_plot)

        if option_5 =='WHICH ONE HAS MORE CRIME CONSISTENCY - DISTRICT 0R WARD':
            col1,col2 =st.columns(2)
            with col1:
                std_dev_data = std_dev_df()
                st.dataframe(std_dev_data,width=600,height=10)
            with col2:
                crime_reg = crime_consistency()
                st.pyplot(crime_reg)
        
        if option_5 == 'RECURRING CRIME TYPE':
            col1,col2= st.columns(2)
            with col1:
                dist_crime = frq_crime_type_district()
                st.pyplot(dist_crime)
            with col2:
                ward_crime = frq_crime_type_ward()
                st.pyplot(ward_crime)

        if st.button('STEPS TO BE TAKEN',use_container_width=True):
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Focused Crime Prevention in Districts:<h5> Since the district-level 
                            standard deviation is higher, this indicates significant variability in crime levels across districts.
                            Resources should be allocated based on high-crime districts, particularly for addressing theft, criminal damage,
                            and battery. A more dynamic, district-specific crime prevention strategy should be adopted to 
                            address the fluctuations.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Ward-Level Patrol Consistency:<h5> Since wards have a lower standard deviation,
                            suggesting more consistency in crime occurrence, a steady patrol and enforcement presence in wards may be more
                            effective in curbing recurring offenses. Focused ward-level patrolling for the most frequent crimes like criminal damage
                             and battery should be maintained.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Permanent Police Beats for Consistent Crimes:<h5> Since crime types like
                            theft, criminal damage, and battery occur consistently, especially at the ward level, consider establishing permanent 
                            police beats in high-crime areas. This would allow for a constant, visible police presence and quick response times 
                            to deter these recurring offenses.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Resource Allocation Based on Crime Consistency:<h5> Given the more variable 
                            crime rates in districts compared to wards, law enforcement resources should be adaptively allocated to districts with 
                            spiking crime levels, while maintaining routine patrols in wards where crime patterns are more stable.''',unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Conclusion:<h5>The data suggests that a dual approach is necessary:<h5>
                             More flexible, district-specific crime responses for fluctuating crimes, and more consistent, ward-focused measures 
                            for persistent offenses. Emphasis should be placed on high-frequency crimes across both districts and wards, 
                            with targeted resource allocation for maximum impact.''',unsafe_allow_html=True)

    with tab6:

        option_6 = st.radio('Select the option',['NONE','ARREST RATES'],index=None)

        if option_6 == 'ARREST RATES':

            col1,col2 = st.columns(2)
            
            with col1:
                arrest_rate_df = arrest_rate()
                st.dataframe(arrest_rate_df,width=600,height=530)

            with col2:
                arrest_rate_graph = arrest_rate_plot()
                st.pyplot(arrest_rate_graph)
        
    with tab7:

        option_7 = st.radio('Select the option',['NONE','DOMESTIC CRIME','NON-DOMESTIC CRIME','DOMESTIC VS  NON-DOMESTICE CRIMES - WHICH ONE IS HIGH'],index=None)

        if option_7 == 'DOMESTIC CRIME':

            col1,col2 = st.columns(2)

            with col1:                
                domestic_crime_df = domeestic_crime_new()
                st.dataframe(domestic_crime_df,width=600,height=530)

            with col2:

                domestic_crime_graph = domestic_crime_count_plot()
                st.pyplot(domestic_crime_graph)

        if option_7 == 'NON-DOMESTIC CRIME':

            col1,col2 = st.columns(2)
            with col1:
                non_dome_crime_df = non_dome_crime_new()
                st.dataframe(non_dome_crime_df,width=600,height=530)
            with col2:
                non_dome_crime_graph = non_domestic_crime_plot()
                st.pyplot(non_dome_crime_graph)

        if option_7 == 'DOMESTIC VS  NON-DOMESTICE CRIMES - WHICH ONE IS HIGH':
            col1,col2 = st.columns(2)
            with col1:
                comapre_graph = domestic_vs_non_domestic()
                st.pyplot(comapre_graph)

        if st.button('CORRECTIVE MEASURES',use_container_width=True):
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>DOMESTIC CRIMES:<h5>Enhance Reporting Mechanisms<h5>Improving Reporting Access:<h5>
                            Since domestic crimes are left unreported by victims,local authorities should ensure that victims have multiple,
                            accessible avenues to report crimes (e.g., apps, online platforms, community centers) and ensure that people feel
                             safe doing so.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Anonymous Reporting Systems:<h5> Implementing anonymous reporting systems for 
                            domestic crimes can encourage more victims to come forward without fear of reprisal.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Victim Protection Plans:<h5> Strengthening protective measures for victims,
                           such as restraining orders, safe shelters, and financial support, can help reduce the long-term impact on those 
                           affected by domestic violence.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Community Engagement:<h5> Law enforcement agencies could work more closely with
                             communities to create neighborhood watch programs or other community-led initiatives aimed at reducing domestic violence.''',unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>NON-DOMESTIC CRIMES:<h5>Enhance Surveillance Systems:<h5>Install more CCTV cameras in public places,
                            commercial areas, parking lots, and neighborhoods to deter thieves and assist in post-incident investigations.
                            Implement community-driven security systems, such as neighborhood watch programs or mobile apps that allow residents 
                            to report suspicious activity quickly.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'><h5>Encourage Smart Device Use:<h5>Promote the use of anti-theft technology, such as GPS tracking 
                            for valuable items, smart locks, and alarms, especially in high-risk areas like shopping districts and transport hubs.''',unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'><h5>Secure Retail and Commercial Spaces:<h5>Work with local businesses 
                            to improve security protocols, including better alarm systems, controlled access, and trained 
                            security personnel.''',unsafe_allow_html=True)
    with tab8:

        option_8 = st.radio('Select the option',['NONE','CRIME TYPE DISTRIBUTION'],index=None)

        if option_8 == 'CRIME TYPE DISTRIBUTION':

            col1,col2 = st.columns(2)

            with col1:

                crime_dist_df = crime_distribution()
                st.dataframe(crime_dist_df,width=600,height=530)

            with col2:

                crime_dist_plot = crime_distribution_plot()
                st.pyplot(crime_dist_plot)           


    with tab9:

        option_9 = st.radio('Select the option',['NONE','DISTRIBUTION OF SEVERE VS LESS-SEVERE CRIMES'],index=None)

        if option_9 == 'DISTRIBUTION OF SEVERE VS LESS-SEVERE CRIMES':

            col1,col2 = st.columns(2)

            with col1:

                crime_severe_df = crime_severity()
                st.dataframe(crime_severe_df,width=600,height=530)

            with col2:

                crime_severity_graph = crime_severity_plot()
                st.pyplot(crime_severity_graph)

        if st.button('SUGGESTED ACTIONS',use_container_width=True):
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Increased Patrols in High-Incidence Areas:<h5>
                                Focus law enforcement resources on neighborhoods with high rates of non-severe crimes. 
                                Increased visibility can deter potential offenders and reassure residents.''',unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Diversion Programs for Youth:<h5>
                            Implement diversion programs targeting young offenders involved in non-severe crimes. 
                            These programs can provide alternatives to criminal behavior through mentorship, 
                            job training, and educational support.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Utilize Data-Driven Policing:<h5>Use crime data analytics 
                                to identify patterns and trends in non-severe crime occurrences. This can inform targeted interventions
                                 and resource allocation.''',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Diversion Programs for Youth:<h5>Implement diversion programs targeting 
                                young offenders involved in non-severe crimes. These programs can provide alternatives to criminal behavior
                             through mentorship, job training, and educational support.''',unsafe_allow_html=True)



    with tab10:

        option_10 = st.radio('Select the option',['NONE','SEASONAL TRENDS ON CRIME FREQUENCY'],index=None)

        if option_10 == 'SEASONAL TRENDS ON CRIME FREQUENCY':

            col1,col2 = st.columns(2)

            with col1:
                seasons_crime_df = seasons_crime()
                st.dataframe(seasons_crime_df,width=600,height=150)

            with col2:
                seasons_impact_plot = seasons_plot()
                st.pyplot(seasons_impact_plot)


        if st.button('RECOMMENDED ACTIONS',use_container_width=True):
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Enhanced Law Enforcement Presence During Summer:<h5>Increase police patrols 
                            and visibility in high-crime areas during the summer months when crime rates peak. This can act as a deterrent and 
                            provide a sense of safety for residents.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Resource Allocation for Seasonal Trends:<h5>Allocate resources and funding for 
                            crime prevention programs that specifically target the summer months. This could include overtime pay for officers or 
                            funding for community initiatives.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Data-Driven Policing Strategies:<h5>Utilize historical crime data to identify 
                            specific days or events that correlate with higher crime rates during the summer. This information can help law enforcement 
                            strategize their deployment of resources effectively.''',unsafe_allow_html=True)
                
            with st.container(border=True):
                st.markdown('''<h5 style='color:#00ffff;font-size:21px'>Promote Safe Summer Activities:<h5>Encourage local businesses to create safe 
                            summer activities and events, such as outdoor movie nights, concerts, and fairs, to bring the community together and 
                            distract from criminal behavior.''',unsafe_allow_html=True)


                             

elif select == 'PREDICTION':
    
    title = '''<h1 style='font-size: 30px;text-align:center;color:#00ff80;'>To predict the crime type,Please provide the 
                following information</h1>'''
    st.markdown(title,unsafe_allow_html=True)

    st.write()
    st.write()
    st.write()

    col1,col2 = st.columns(2)

    with col1:
         
         description = st.selectbox('Crime_Description',desc_list,index=None)

         crime_type = st.selectbox('Domestic',domestic_list,index=None)

             
    with col2:
         
         district = st.selectbox('District',district_list,index=None)
         
         location = st.selectbox('Location',location_list,index=None)

         
    if st.button(':violet[Predict]',use_container_width=True):

        if None in [description,crime_type,district,location]:
            st.error("Required input is missing",icon="🚨")
        else:

            description = desc_list_encoded[description]

            location = location_list_encoded[location]

            crime_type = domestic_list_encoded[crime_type]

            district = district_list_encoded[district]

            prediction = get_prediction(description,crime_type,district,location)   

            st.subheader((f":green[Crime_Type :] {prediction}"))
