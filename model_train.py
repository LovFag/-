import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from titanic.ColumnTransformer import preprocessing
from titanic.data_clean import titanic_labels, titanic_train_data, PassengerId

# 决策树模型
tree_reg = make_pipeline(preprocessing, DecisionTreeClassifier(random_state=42))
tree_reg.fit(titanic_train_data, titanic_labels)
tree_scores = cross_val_score(tree_reg, titanic_train_data, titanic_labels,
                             scoring="accuracy", cv=10)

# 随机森林模型（参数是分类器）
forest_reg = make_pipeline(preprocessing, RandomForestClassifier(random_state=42))
forest_reg.fit(titanic_train_data, titanic_labels)
# 交叉验证指标，回归用 neg_mean_squared_error，分类用 accuracy
cv_score = cross_val_score(forest_reg, titanic_train_data, titanic_labels,
                           scoring="accuracy", cv=10)
titanic_predictions = forest_reg.predict(titanic_train_data)

best_model = joblib.load("best_model.pkl")
lgb_predictions = best_model.predict(titanic_train_data)
scoress = cross_val_score(best_model, titanic_train_data, titanic_labels,
                         scoring="accuracy", cv=10)

if __name__ == '__main__':
    # print(pd.Series(forest_rmses).describe())
    # print(forest_reg.score(titanic_train_data, titanic_labels))
    print("True_labels:")
    print(titanic_labels.iloc[:10])
    print("Predict_labels:")
    print(pd.DataFrame({
        'PassengerId': PassengerId[:10],
        'Survived': titanic_predictions[:10]
    }))
    print('Tree accuracy:', tree_scores.mean())
    print('Forest accuracy:', cv_score.mean())
    print('Best model accuracy:', scoress.mean())

