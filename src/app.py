import streamlit as st
import pandas as pd
import joblib, os
import pickle
import warnings

# Useful Functions
def load_ml_components(fp):
    'load the ml components to re-use in app'
    with open(fp, 'rb') as f:
        object = pickle.load(f)
    return object

# Variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'ml_Assets','ml.pkl')

# Execution
ml_components_dict = load_ml_components(fp=ml_core_fp) 

labels = ml_components_dict['label']
idx_to_labels = {i: l for (i, l) in enumerate(labels)}

end2end_pipeline = ml_components_dict['pipeline']

print(f"\n[Info] Predictable labels: {labels}")
print(f"\n[Info] Predictable Idexes to labels: {idx_to_labels}")

print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")

st.write("""# TIME-SERIES REGRESSION PREDICTION APP
         My fisrt interface using streamlit to predict a time-series regression""")

# INPUTS
date = st.date_input('Kindly Select your date')
oil_prices = st.number_input('Enter oil price', min_value = 10.00, max_value = 1000.00)
onpromotion = st.selectbox('Enter the promotion status on the selected date, 1 for onpromotion and 0 for no promotion]', ('1', '0'))
transactions = st.number_input('Enter transactions Amount on the date chosen')


#num_cols = ['store_nbr', 'onpromotion', 'transactions','transferred', 'oil_prices', 'cluster', 'Year', 'Month', 'DayOfMonth', 'DaysInMonth', 'DayOfYear', 'Week']
       

#cat_cols = ['family', 'holidays_type', 'locale', 'locale_name', 'description','city', 'state', 'store_type']

# Prediction as Executed

if st.button('Predict'):

    #DataFrame creation
    
    #df = pd.DataFrame({
        #"date":[date],"oil_prices":[oil_prices], "transactions":[transactions], "onpromotion":[onpromotion], 
    #})
    
    
    #df = pd.DataFrame(columns=['store_nbr', 'onpromotion', 'transactions','transferred', 'oil_prices', 'cluster', 'Year', 'Month', 
                               #'DayOfMonth', 'DaysInMonth', 'DayOfYear', 'Week', 'family', 'holidays_type', 'locale', 'locale_name', 
                               #'description','city', 'state', 'store_type','Unnamed: 0_x', 'Unnamed: 0_y', 'Unnamed: 0'])
    
    
    df = pd.DataFrame({'date':[date], 'store_nbr':[store_nbr], 'onpromotion':[onpromotion], 'transactions':[transactions],
                               'transferred':[transfered], 'oil_prices':[oil_prices], 'cluster':[cluster], 'Year':[Year], 'Month':[Month], 
                               'DayOfMonth':[DayOfMonth], 'DaysInMonth':[DaysInMonth], 'DayOfYear':[DayOfMonth], 'Week':[Week], 'family':[family], 
                               'holidays_type':[holidays_type], 'locale':[locale], 'locale_name':[locale_name], 
                               'description':[description],'city':[city], 'state':[state], 'store_type':[store_type]
                               })
    
    
    print(f"[Info]Input data as dataframe:\n{df.to_markdown()}")
    
# ML PART
    output = end2end_pipeline.predict(df)

## store confidence score/probability for the predicted class
    df['confidence score'] = output.max(axis=-1)

# store index then replace by the matching label

    df['predicted label'] = predicted_idx
    predicted_label = df['predicted label'].replace(idx_to_labels)
    df['predicted label'] = predicted_label

    print(f'[Info] Input dataframe with prediction :\n{df.to_markdown()}')
    
#st.text(f"The Total Sales for the chosen date is : '{}' .")
