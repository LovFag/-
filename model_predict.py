import pandas as pd

from titanic.data_describe import test_data
from titanic.model_train import forest_reg
from titanic.params_search import rand_search

PassengerId = test_data['PassengerId']
test_data = test_data.drop(['Name', 'Ticket'], axis=1)

# 2. 与训练时做同样的特征工程
best_model = rand_search.best_estimator_

prediction = best_model.predict(test_data)

# 拼合预测结果DataFrame
submission = pd.DataFrame({
    'PassengerId': PassengerId,
    'Survived': prediction
})

print(submission.info)

submission.to_csv('submission.csv', index=False)