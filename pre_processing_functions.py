import pandas as pd
from datetime import datetime


def feature_transformation(customer_info):
    #Gender mapping
    customer_info['customer_gender'] = customer_info['customer_gender'].map({'female': 1, 'male': 0})

    #Customer birthdate transformation to customer age
    customer_info['customer_birthdate'] = pd.to_datetime(
    customer_info['customer_birthdate'], format='%m/%d/%Y %I:%M %p')

    current_year = datetime.now().year
    customer_info['customer_age'] = current_year - customer_info['customer_birthdate'].dt.year

    #Loyalty card flag
    customer_info['loyalty_card_number'] = customer_info['loyalty_card_number'].notna().astype(int)

    #Years active
    customer_info['years_active'] = 2025 - customer_info['year_first_transaction']

    #percentage
    customer_info['percentage_of_products_bought_promotion'] = customer_info['percentage_of_products_bought_promotion']*100

    #Education splitting 
    education_titles = ['Phd.', 'Msc.', 'Bsc.', 'MBA.']

    def split_name(name):
        parts = str(name).strip().split()
        if parts and parts[0] in education_titles:
            educ = parts[0]
            clean_name = ' '.join(parts[1:])
        else:
            educ = None
            clean_name = name.strip()
        return pd.Series([clean_name, educ])

    customer_info[['customer_name_clean', 'customer_educlevel']] = customer_info['customer_name'].apply(split_name)
    customer_info['customer_name'] = customer_info['customer_name_clean']
    customer_info.drop(columns='customer_name_clean', inplace=True)

    return customer_info