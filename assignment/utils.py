import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def get_data(min_size=None, min_size_test=None, fill_nan=None):
    X_train = pd.read_csv("/home/levcsi/IPCV/DMaML/assignment/data/X_train.csv")
    Y_train = pd.read_csv("/home/levcsi/IPCV/DMaML/assignment/data/Y_train.csv")
    X_test = pd.read_csv("/home/levcsi/IPCV/DMaML/assignment/data/X_test.csv")
    
    targets = Y_train.TARGET.unique()
    target_map = {targets[i]: i for i in range(len(targets))}
    inverse_target_map = {v: k for k, v in target_map.items()}
    
    Admission_types = X_train.Admission_type.unique()
    Admission_type_map = {Admission_types[i]: i for i in range(len(Admission_types))}
    
    Marital_statuses = X_train.Marital_status.unique()
    Marital_status_map = {Marital_statuses[i]: i for i in range(len(Marital_statuses))}
    
    Ethnicities = X_train.Ethnicity.unique()
    Ethnicity_map = {Ethnicities[i]: i for i in range(len(Ethnicities))}
    
    Genders = X_train.Gender.unique()
    Gender_map = {Genders[i]: i for i in range(len(Genders))}

    XY_train = X_train.join(Y_train["TARGET"])
    XY_train['Admission_type_NUM'] = XY_train['Admission_type'].map(Admission_type_map)
    XY_train['Marital_status_NUM'] = XY_train['Marital_status'].map(Marital_status_map)
    XY_train['Ethnicity_NUM'] = XY_train['Ethnicity'].map(Ethnicity_map)
    XY_train['Gender_NUM'] = XY_train['Gender'].map(Gender_map)
    XY_train['TARGET_NUM'] = XY_train['TARGET'].map(target_map)

    X_test['Admission_type_NUM'] = X_test['Admission_type'].map(Admission_type_map)
    X_test['Marital_status_NUM'] = X_test['Marital_status'].map(Marital_status_map)
    X_test['Ethnicity_NUM'] = X_test['Ethnicity'].map(Ethnicity_map)
    X_test['Gender_NUM'] = X_test['Gender'].map(Gender_map)

    columns_for_drop = list()
    for c in XY_train.columns:
        if pd.api.types.is_string_dtype(XY_train[c]):
          columns_for_drop.append(c)
    XY_train.drop(columns_for_drop, inplace=True, axis=1)

    columns_for_drop.remove("TARGET")
    X_test.drop(columns_for_drop, inplace=True, axis=1)

    if min_size is not None:
        for c in XY_train.columns:
            if c != "TARGET_NUM":
                XY_train[c][XY_train.groupby(c)[c].transform('size') <= min_size] = -1

    if min_size_test is not None:
        for c in X_test.columns:
            X_test[c][X_test.groupby(c)[c].transform('size') <= min_size_test] = -1

    if fill_nan is not None:
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=fill_nan)
        XY_train_full = imp.fit_transform(XY_train)
        XY_train = pd.DataFrame(XY_train_full, columns=XY_train.columns, index=XY_train.index)
        X_test_full = imp.fit_transform(X_test)
        X_test = pd.DataFrame(X_test_full, columns=X_test.columns, index=X_test.index)

    return XY_train, X_test, inverse_target_map
