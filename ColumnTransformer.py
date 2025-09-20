import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from titanic.data_clean import num_cols, cat_cols, int_cols

# 构造流水线
numeric_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessing = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
        ('int', 'passthrough', int_cols)
    ]
)

rf_clf = Pipeline(steps=[
    ('prep', preprocessing),                  
    ('model', RandomForestClassifier(random_state=42))
])

lgb_clf = Pipeline(steps=[
    ('prep', preprocessing),
    ('model', lgb.LGBMClassifier(objective='binary', random_state=42))
])
