from datetime import datetime
from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


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
    
    #joining kids and teens columns, turning children into binary
    customer_info["children"] = customer_info["kids_home"] + customer_info["teens_home"]
    customer_info["has_children"] = customer_info["children"].apply(lambda x: 1 if x > 0 else 0)

    #Loyalty card flag
    customer_info['loyalty_card_number'] = customer_info['loyalty_card_number'].notna().astype(int)

    #Years active
    customer_info['years_active'] = 2025 - customer_info['year_first_transaction']

    #typical time period into binary 
    customer_info['typical_time_period'] = customer_info['typical_hour'].apply(lambda x: 1 if x < 12 else 1 )
    
    #percentage
    customer_info['percentage_of_products_bought_promotion'] = customer_info['percentage_of_products_bought_promotion']*100

    #Education splitting and turning into binary
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
    customer_info['customer_educlevel'] = customer_info['customer_educlevel'].isin(education_titles).astype(int)

    #drop column Unnamed: 0, customer_birthdate, and customer_name
    customer_info = customer_info.drop(columns=['Unnamed: 0', 'customer_birthdate', 'customer_name'])
    
    return customer_info

def missing_values(df, n_neighbors=5):
    handled_missing = df.copy()
    
    # create new df for numeric and categorical columns
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Use simple imputer to impute numeric columns by KNN
    if len(num_cols) > 0:
        num_imputer = KNNImputer(n_neighbors=n_neighbors)
        handled_missing[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Use simple imputer to impute categorical columns by most frequent
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy= 'most_frequent')
        handled_missing[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return handled_missing

def manual_outliers(df):
    manual_thresholds = {
        'lifetime_spend_groceries': 140000,
        'lifetime_spend_electronics': 30000,
        'lifetime_spend_vegetables': 4000,
        'lifetime_spend_nonalcohol_drinks': 1600,
        'lifetime_spend_alcohol_drinks': 4000,
        'lifetime_spend_meat': 2800,
        'lifetime_spend_fish': 3600,
        'lifetime_spend_hygiene': 3000,
        'lifetime_spend_videogames': 2000,
        'lifetime_spend_petfood': 900,
        'lifetime_total_distinct_products': 800,
    }

    for col, threshold in manual_thresholds.items():
        df[col] = df[col].clip(upper=threshold)
    
    df['percentage_of_products_bought_promotion'] = df['percentage_of_products_bought_promotion'].clip(lower=0)
    
    return(df)

def multidimensional_outliers(df):
    
    #features we selected for DBSCAN
    features = [
    'lifetime_spend_groceries',
    'lifetime_spend_electronics',
    'lifetime_spend_meat',
    'lifetime_total_distinct_products',
    'percentage_of_products_bought_promotion',
    'lifetime_spend_alcohol_drinks',
    'number_complaints',
    'distinct_stores_visited',
    'typical_hour'
    ]

    # Standardize the selected features
    X_scaled = StandardScaler().fit_transform(df[features])

    # Apply DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # Add results to DataFrame and remove said outliers
    df['dbscan_label'] = labels
    df['is_outlier'] = (labels == -1)
    df_clean = df[df['is_outlier'] == False].copy().drop(columns=['dbscan_label', 'is_outlier'])

    return df_clean


def scaling(df):
    df_scaled = df.copy()
    # select numeric columns excluding binary columns
    num_cols = df_scaled.select_dtypes(include=['number']).columns
    non_binary_num_cols = [col for col in num_cols if df_scaled[col].nunique() > 2]
    # Scale only numeric columns
    scaler_ = StandardScaler()
    df_scaled[non_binary_num_cols] = scaler_.fit_transform(df_scaled[non_binary_num_cols])
    return df_scaled


def preprocess(path):
    df = load_info(path)
    df = feature_transformation(df)
    df = missing_values(df)
    df = manual_outliers(df)
    df = multidimensional_outliers(df)
    df = scaling(df)
    return df

def preprocess_semscalling(path):
    df = load_info(path)
    df = feature_transformation(df)
    df = missing_values(df)
    df = manual_outliers(df)
    df = multidimensional_outliers(df)
    return df

def feature_selection(path, method, threshold=0.01,correlation_threshold=0.9):
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

   
    else:
        raise ValueError("Choose method from 'variance', 'correlation'.")
