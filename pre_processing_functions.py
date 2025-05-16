from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler  
from sklearn.impute import KNNImputer, SimpleImputer 
from sklearn.impute import SimpleImputer 
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def load_basket(filepath):
    basket = pd.read_csv(filepath)
    return basket

def load_info(filepath):
    info = pd.read_csv(filepath)
    return info

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

def missing_values(df, n_neighbors=5):
    handled_missing = df.copy()
    
    # create new df for numeric and categorical columns
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

    # Use simple imputer to impute numeric columns by median
    if len(num_cols) > 0:
        num_imputer = KNNImputer(n_neighbors=n_neighbors)
        handled_missing[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Use simple imputer to impute categorical columns by most frequent
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy= 'most_frequent')
        handled_missing[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return handled_missing

def encoding(df):
    return df

def scalling(df, scaler = 'robust'):
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()
    else:     
        scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled


def preprocess(path):
    df = load_info(path)
    df = feature_transformation(df)
    df = missing_values(df)
    #df = encoding(df)
    #df = scalling(df)
    return df

def feature_selection(path, method, threshold=0.01, n_components=3, correlation_threshold=0.9):
    df = preprocess(path)
    original_features = df.columns.tolist()

    result_dict = {feature: False for feature in original_features} 

    if method == "variance":
        selected_features = df.columns[VarianceThreshold(threshold=threshold).fit(df).get_support()]
        result_dict.update({f: True for f in selected_features})
        return pd.Series(result_dict, name="Keep (Variance)")
    
    elif method == 'correlation':
        corr_matrix = df.corr().abs()
        to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    to_drop.add(col2)  
        df_selected = df.drop(columns=list(to_drop))
        for f in df.columns:
            result_dict[f] = f not in to_drop
        return pd.Series(result_dict, name="Keep (Correlation)")

    elif method == "pca":
        pca = PCA(n_components=n_components)
        pca.fit(df)
        matrix = pd.DataFrame(pca.components_.T, index=df.columns, columns=[f"PC{i+1}" for i in range(n_components)])
        
        # A feature is important if it contributes highly to a PC
        importance_threshold = 0.3 
        important_features = set()

        for col in matrix.columns:
            important_features.update(matrix[matrix[col].abs() >= importance_threshold].index.tolist())

        for f in df.columns:
            result_dict[f] = f in important_features
        return pd.Series(result_dict, name="Important in PCA")

    else:
        raise ValueError("Choose method from 'variance', 'correlation', or 'pca'")
