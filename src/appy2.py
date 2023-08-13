import streamlit as st
import pandas as pd
import joblib, os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.write("""# TIME-SERIES REGRESSION PREDICTION APP
         My fisrt interface using streamlit to predict a time-series regression""")


# Write a Subheader
st.subheader('Kindly Enter Your Information')


# load ml components

cwd = os.getcwd()
relative_path = "src\\ml_Assets\\ml.pkl"


absolute_path = os.path.join(cwd, relative_path)
print(absolute_path)

#ML components Reading

with open(absolute_path, "rb") as f:
    ml_components = pickle.load(f)
    
labels = ml_components['label']
pip = ml_components["pipeline"]
scaler = ml_components["scaler"]
models = ml_components["model"]

# Execution
#ml_components_dict = load_ml_components(fp=ml_core_fp) 


#idx_to_labels = {i: l for (i, l) in enumerate(labels)}

#end2end_pipeline = ml_components_dict['pipeline']

# INPUTS DECLARATIONS

date = st.date_input('Kindly Select your date')
oil_prices = st.number_input('Enter oil price', min_value = 10.00, max_value = 1000.00)
onpromotion = st.selectbox('Enter the promotion status on the selected date, 1 for onpromotion and 0 for no promotion]', ('1', '0'))
transactions = st.number_input('Enter transactions Amount on the date chosen')

# num_cols = ['store_nbr', 'onpromotion', 'transactions','transferred', 'oil_prices', 'cluster', 'Year', 'Month', 'DayOfMonth', 'DaysInMonth', 'DayOfYear', 'Week']
       

# cat_cols = ['family', 'holidays_type', 'locale', 'locale_name', 'description','city', 'state', 'store_type']

backgroundColor = "green"
with st.form(key = "my_form", clear_on_submit = True):
    
    # Output Presentation
    submitted = st.form_submit_button("predict")
    if submitted:
        
        try:
            inputs = [date, oil_prices, onpromotion, transactions]
            
            df = pd.dataFrame(inputs[1:], columns = inputs[0])
            
            df = df.set_index('date')
            
            # selectingexogenous variables
            #df = df[['oil_prices', 'onpromotion']]
            #print(df)
            
            predicted = pip.predict(df)
            
            labels = predicted
            
            #f['Sales'] = predicted
            
            st.balloons()
            st.success('Successfully Predicted', icon = '✅')
            print(df)
            
            st.write(labels)
            st.write(df)
            
            
        except:
            st.error('something wrong has happened....', icon = '🚨')





# Prediction as Executed

#if st.button('predict'):
    # DataFrame creation
    #df = pd.DataFrame({
        #"date":[date],"oil_prices":[oil_prices], "transactions":[transactions], "onpromotion":[onpromotion]
    #})
    #print(f"[Info]Input data as dataframe:\n{df.to_markdown()}")
    
    
    
    #st.text(f"The Total Sales within the period is : '{''}' .")
