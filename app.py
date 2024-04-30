import streamlit as st
import pandas as pd
import numpy as np
# import altair as alt
import plotly.express as px
import re
import pickle


st.set_page_config(layout="wide")
st.title('Medical Cost Prediction')
st.sidebar.header('Parameters')

df = pd.read_csv('./Medical_insurance.csv')
# st.write(df.head())

######### SIDEBAR DETAILS #########
# count filter

age = st.sidebar.text_input('Age', '22')
age = float(age)

Available_sex = tuple(df['sex'].unique())
sex_type = st.sidebar.radio(
        "Sex",
        Available_sex
    )


bmi = st.sidebar.text_input('BMI', '27.5')
bmi = float(bmi)


children = st.sidebar.text_input('Children', '3')
children = float(children)


Available_smoker = tuple(df['smoker'].unique())
smoker_type = st.sidebar.radio(
        "Smoker",
        Available_smoker
    )

Available_region = tuple(df['region'].unique())
region_type = st.sidebar.radio(
        "Region",
        Available_region
    )

#################


# feature engineering
# computing weight status(under-weight, over-weight..) and age category based on bmi and age respectively

def bmi_catg(x):
    if x < 18.5:
        return 'Under Weight'
    elif x>=18.5 and x<=24.9:
        return 'Normal Weight'
    elif x>=25 and x<=29.9:
        return 'Overweight'
    else:
        return 'Obese'

def age_catg(x):
    if x <= 35:
        return 'Young Adult'
    elif x>=36 and x<=55:
        return 'Senior Adult'
    else:
        return 'Elder'
    

def parenting_stage(age, num_children):
    early_parent_threshold = 30
    if age < early_parent_threshold:
        if num_children == 0:
            return "Early Parent (No Children)"
        else:
            return "Early Parent (With Children)"
    else:
        if num_children == 0:
            return "Experienced Parent (No Children)"
        else:
            return "Experienced Parent (With Children)"

final_cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest',
       'region_southeast', 'region_southwest', 'Weight_status_Obese',
       'Weight_status_Overweight', 'Weight_status_Under Weight',
       'Age_status_Senior Adult', 'Age_status_Young Adult',
       'Parenting_status_Early Parent (With Children)',
       'Parenting_status_Experienced Parent (No Children)',
       'Parenting_status_Experienced Parent (With Children)']


user_input_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
user_input_df = pd.DataFrame([[age, sex_type, bmi, children, smoker_type, region_type]], columns=user_input_cols)
user_input_df['Weight_status'] = user_input_df['bmi'].apply(bmi_catg)
user_input_df['Age_status'] = user_input_df['age'].apply(age_catg)
user_input_df['Parenting_status'] = user_input_df[['age', 'children']].apply(lambda row: parenting_stage(row['age'], row['children']), axis=1)
st.write(user_input_df)


final_user_input_df = pd.DataFrame(np.zeros(len(final_cols)).reshape(1,-1), columns=final_cols)
final_user_input_df.loc[:, 'age'] = user_input_df['age']
final_user_input_df.loc[:, 'bmi'] = user_input_df['bmi']
final_user_input_df.loc[:, 'children'] = user_input_df['children']

for col in final_cols[3:]:
    value = col.split('_')[-1]
    col_ui = '_'.join(col.split('_')[:-1])
    if user_input_df[col_ui][0] == value:
        final_user_input_df.loc[0, col] = 1
         
# st.write(final_user_input_df)

filename = 'RandomForest_Regressor.sav'
randomforest_regressor = pickle.load(open(filename, 'rb'))

charges = np.e**(randomforest_regressor.predict(final_user_input_df)[0])
st.subheader('Predicted Charges:')
st.write(charges)