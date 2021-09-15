import streamlit as st 
import pandas as pd
import lightgbm as lgb
import numpy as np 
from sklearn.preprocessing import StandardScaler

st.header("Churn Prediction")
st.write("""Customer Churn prediction is very important for subscription companies, it can allow companies to customize their marketing or services for customers that are predicted to Churn(Unsubscribe).
            This web app is displaying predictions of a pretrained model on the Telco Customer Churn Dataset. This is of course unique for each company. 
            **To use it** simply fill in all of the values of a customer and the model will predict whether that 
            customer will Churn or not.""")
st.image('customer-churn-edit.jpeg')

bst = lgb.Booster(model_file=r'new_churn.txt')
mean_tenure = 32.37114865824223
std_tenure = 24.55948102309423
mean_charges =64.76169246059922
std_charges = 30.09004709767854




col1, col2,col3 = st.columns(3)
display = ("Yes", "No")
options = list(range(len(display)))

with st.form("myform"): 
    with col1:
        mult_line = st.selectbox( "Multiple Lines",options, format_func=lambda x: display[x])
        InternetService_DSL = st.selectbox('Internet Service DSL',options, format_func=lambda x: display[x])
        fiber_optics = st.selectbox('Fiber optics',options, format_func=lambda x: display[x])
        online_security = st.selectbox("Online Security", options, format_func=lambda x: display[x])
        online_backup = st.selectbox('Online Backup', options, format_func=lambda x: display[x])
        device_protection = st.selectbox('Device Protection',options, format_func=lambda x: display[x])
    with col2:
        tech_support = st.selectbox('Tech Support',options, format_func=lambda x: display[x])
        streaming_tv = st.selectbox('Streaming TV',options, format_func=lambda x: display[x])
        streaming_movies = st.selectbox('Streaming Movies',options, format_func=lambda x: display[x])
        contract_length = st.selectbox('Contract Lenght',['Month to month','One year','Two year'])
        payment_method = st.selectbox('Payment Method',['Bank transfer','Credit Card','Electronic Check','Mailed Check'])
        partner = st.selectbox('Customer has a Partner',options, format_func=lambda x: display[x])
    with col3:     
        Dependant = st.selectbox('Customer has dependants',options, format_func=lambda x: display[x])
        paperless_billing = st.selectbox("Paperless Billing",options, format_func=lambda x: display[x])
        tenure = st.number_input("Tenure in Months")
        Monthly_charges = st.number_input('Average Monthly Bill')
        Senior_citizen = st.selectbox("Senior Citizen",options, format_func=lambda x: display[x])
    submitted = st.form_submit_button('Press when done')

#st.write(online_security)
#st.write(type(online_security)) debugging

def contract_type(contract_length):
    month_to_month = 0.0
    one_year = 0.0
    two_year = 0.0
    if contract_length == 'Month to month':
        month_to_month = 1.0        
    elif contract_length == 'One year':
        one_year = 1.0        
    elif contract_length == 'Two year':
        two_year = 1.0
    return month_to_month, one_year, two_year

def payment_type(payment_method):
    bank_transfer = 0.0
    credit_card = 0.0 
    e_check =0.0
    mail_check = 0.0
    if payment_method == 'Bank transfer':
        bank_transfer =1.0
    elif payment_method == 'Credit Card':
        credit_card =1.0
    elif payment_method == 'Electronic Check':
        e_check =1.0
    elif payment_method == 'Mailed Check':
        mail_check =1.0
    return bank_transfer,credit_card,e_check,mail_check
    
if submitted:        
    month_to_month, one_year, two_year = contract_type(contract_length)
    bank_transfer,credit_card,e_check,mail_check = payment_type(payment_method)
    tenure = (tenure -mean_tenure)/std_tenure
    Monthly_charges = (Monthly_charges - mean_charges)/std_charges

    
   
    #predict
    prediction = bst.predict(np.array([mult_line,InternetService_DSL,fiber_optics,
            online_security,online_backup,device_protection,tech_support,
            streaming_tv,streaming_movies,month_to_month,one_year,two_year,bank_transfer,
            credit_card,e_check,mail_check,partner,Dependant,paperless_billing,tenure,Monthly_charges,Senior_citizen]).reshape(1,-1))
    st.write("The probability of Churning is", prediction[0])
    
    if prediction > 0.5: 
            st.warning('Customer is likely to Churn')
    else: 
        st.write("Customer is not likely to churn")
        st.balloons()


features_list = ['MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
       'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
       'Partner_Yes', 'Dependents_Yes', 'PaperlessBilling_Yes', 'tenure',
       'MonthlyCharges', 'SeniorCitizen']
