from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer

def column_transformer():
    minmax_scaler = MinMaxScaler(feature_range=(0,3))
    num_col = ['class_index,height,width']
    cat_col = ['class']
    transformer = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), cat_col),
        (minmax_scaler,num_col ),
        remainder='passthrough'
    )
    return transformer