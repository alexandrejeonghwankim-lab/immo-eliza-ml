
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


METRO_AREAS = {
    "Brussels": (50.8503, 4.3517),
    "Antwerp": (51.2194, 4.4025),
    "Ghent": (51.0543, 3.7174),
    "Liege": (50.6326, 5.5797),
    "Charleroi": (50.4108, 4.4446)
}

CITIES_50K = {
    "Bruges": (51.2093, 3.2247),
    "Namur": (50.4674, 4.8718),
    "Leuven": (50.8798, 4.7005),
    "Mons": (50.4542, 3.9523),
    "Aalst": (50.9360, 4.0355),
    "Hasselt": (50.9307, 5.3325),
    "Mechelen": (51.0259, 4.4775),
    "SintNiklaas": (51.1651, 4.1437),
    "LaLouviere": (50.4866, 4.1878),
    "Kortrijk": (50.8266, 3.2645),
    "Ostend": (51.2300, 2.9200),
    "Tournai": (50.6056, 3.3886),
    "Genk": (50.9650, 5.5000),
    "Roeselare": (50.9465, 3.1227),
    "Mouscron": (50.7449, 3.2064),
    "Verviers": (50.5890, 5.8624),
    "Lokeren": (51.1036, 3.9934)
}

ALL_CITIES = {**METRO_AREAS, **CITIES_50K}



def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c



def remove_outliers_by_group(df, group_col = "postal_code", k = 1.5, min_group_size = 20):
    """
    Remove outliers using IQR per group (postal code / region / city)
    
    
    Parameters:
    - df: pandas DataFrame
    - group_col: column to group by (e.g., 'postal_code')
    - k: IQR multiplier (1.5 standard, 2 or 3 = more tolerant)
    - min_group_size: minimum rows per group to apply filtering

    """

    def filter_group(group):

        if len(group) < min_group_size:
            return group
        
        Q1 = group["price"].quantile(0.25)
        Q3 = group["price"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        return group[
            (group["price"] >= lower_bound) &
            (group["price"] <= upper_bound)
        ]
    df_filtered = df.groupby(group_col, group_keys = False).apply(filter_group)

    return df_filtered


def clean_data(df):

    if "longitude" in df.columns and "latitude" in df.columns:
        df = df.dropna(subset=["longitude", "latitude"])

    if "category" in df.columns:
        allowed = ["house", "apartment"]
        df = df[df["category"].isin(allowed)]

    
    #print("Initial rows:", len(df))

    df = df[(df["price"] > 50000) & (df["price"] < 5000000)]

    #print("After global price filter", len(df))

    df = remove_outliers_by_group(df, group_col = "postal_code",k = 2, min_group_size= 20)

    #print("After IQR filtering:", len(df))

    q_99 = df["price"].quantile(0.99)
    df = df[df["price"] <=q_99]

    return df


def features_engineering(df):
    epc_map = {
        "FlandersDoubleA": 4,
        "FlandersSingleA": 4,
        "BrusselsA": 4,
        "WalloniaTripleA": 4,
        "WalloniaDoubleA": 4,
        "WalloniaSingleA": 4,
        "FlandersB": 3,
        "BrusselsB": 3,
        "BrusselsC": 3,
        "WalloniaB": 3,
        "FlandersC": 2,
        "FlandersD": 2,
        "BrusselsD": 2,
        "BrusselsE": 2,
        "WalloniaC": 2,
        "WalloniaD": 2,
        "WalloniaE": 2,
        "FlandersE": 1,
        "FlandersF": 1,
        "BrusselsF": 1,
        "BrusselsG": 1,
        "WalloniaF": 1,
        "WalloniaG": 1
      }
    
    df["epc_ordinal"] = df["epc"].map(epc_map)
    
    kitchen_map ={
    "Super equipped" :4,
    "Fully equipped" :3,
    "Partially equipped" :2,
    "Not equipped" :1, 
    }
    df["kitchen_level"] = df["kitchen_equipment"].map(kitchen_map)
    
    """
    glazing_map = {
    "Triple glass": 3,
    "Double glass": 2,
    "Simple glass": 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    }
    df["glazing_level"] = df["glazing_type"].map(glazing_map)
    """



    for city, (city_lat, city_lon) in ALL_CITIES.items():
        df[f"dist_{city.lower()}"] = df.apply(
            lambda row: haversine(
                row["latitude"],
                row["longitude"],
                city_lat,
                city_lon
            ),
            axis=1
        )
    
    




    return df

    


# ✅ ONLY for Linear Regression
def fill_missing_values(df):

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical(df):

    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols)

    return df

 