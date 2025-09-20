import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from titanic.ColumnTransformer import lgb_clf
from titanic.data_clean import titanic_labels, titanic_train_data, test_df

param_dist = {
    'model__n_estimators': [400, 600, 800, 1000, 1200],
    'model__learning_rate': [0.03, 0.05, 0.08, 0.1],
    'model__num_leaves': [24, 31, 48, 64],
    'model__subsample': [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.7, 0.8, 0.9],
    'model__min_child_samples': [10, 20, 30],
    'model__verbose': [-1]
}

rand_search = RandomizedSearchCV(
    lgb_clf,
    param_distributions=param_dist,
    n_iter=60,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
    random_state=42
)

rand_search.fit(titanic_train_data, titanic_labels)
print('Best CV accuracy:', rand_search.best_score_)
print('Best params:', rand_search.best_params_)

# ---------- 7. 在完整训练集上重训 ----------
best_model = rand_search.best_estimator_
best_model.fit(titanic_train_data, titanic_labels)        # 重训全部训练数据
joblib.dump(best_model, 'best_model.pkl')

# ---------- 8. 预测 & 生成提交 ----------
pred = best_model.predict(test_df)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': pred.astype(int)
})
submission.to_csv('submission.csv', index=False)
print('submission.csv ready!')