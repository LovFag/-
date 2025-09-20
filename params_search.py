import joblib
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from titanic.ColumnTransformer import rf_clf
from titanic.data_clean import titanic_labels, titanic_train_data

param_distributions = {
    'model__n_estimators': range(200, 1000),        # 树的数量
    'model__max_depth': range(3, 20),               # 最大深度
    'model__min_samples_split': range(2, 20),       # 内部节点分裂最小样本
    'model__min_samples_leaf': range(1, 10),        # 叶节点最小样本
    'model__max_features': ['sqrt', 'log2', None],    # 每棵树考虑的最大特征
    'model__class_weight': [None, 'balanced']         # 类别权重
}

rand_search = RandomizedSearchCV(
    estimator=rf_clf,
    param_distributions=param_distributions,
    n_iter=100,          # 随机尝试 100 组参数
    cv=5,                # 5 折交叉验证
    scoring='accuracy',  # 评价指标
    n_jobs=-1,
    verbose=0,
    random_state=42
)

#verbose
# 取值	含义
# 0	    完全不输出（静默模式）
# 1	    只在 每轮参数组合 的开始和结束时各打印一行
# 2	    每折 (fold) 结束时都打印一行，即你看到的 [CV] END ... total time=...
# ≥3	额外再打印训练得分等更详细的信息

rand_search.fit(titanic_train_data, titanic_labels)

if __name__ == '__main__':
    print('Best CV accuracy: {:.4f}'.format(rand_search.best_score_))
    print('Best parameters:')
    # for k, v in rand_search.best_params_.items():
    #     print(f'  {k}: {v}')

    # 用最佳模型预测
    best_model = rand_search.best_estimator_
    val_pred = best_model.predict(titanic_train_data)

    joblib.dump(best_model, 'rns_model.pkl')

    # 评估优化参数后的模型
    cv_score = cross_val_score(best_model, titanic_train_data, titanic_labels,
                               scoring="accuracy", cv=10)
    print('CV accuracy: {:.4f} +/- {:.4f}'.format(cv_score.mean(), cv_score.std()))