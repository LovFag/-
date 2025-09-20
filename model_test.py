import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score

from titanic.data_clean import strat_test_set
from titanic.model_train import tree_reg, forest_reg
from titanic.params_search import rand_search

test_lables = strat_test_set["Survived"].copy()
PassengerId = strat_test_set["PassengerId"].copy()

tree_predictions = tree_reg.predict(strat_test_set)
tree_scores = cross_val_score(tree_reg, strat_test_set, test_lables,
                              scoring="accuracy", cv=10)
print(tree_scores.mean())

forest_predictions = forest_reg.predict(strat_test_set)
forest_scores = cross_val_score(forest_reg, strat_test_set, test_lables,
                                scoring="accuracy", cv=10)
print(forest_scores.mean())

rns = joblib.load("rns_model.pkl")
predictions = rns.predict(strat_test_set)
scores = cross_val_score(rand_search.best_estimator_, strat_test_set, test_lables,
                         scoring="accuracy", cv=10)
print(scores.mean())

best_model = joblib.load("best_model.pkl")
lgb_predictions = best_model.predict(strat_test_set)
scoress = cross_val_score(best_model, strat_test_set, test_lables,
                         scoring="accuracy", cv=10)
print(scoress.mean())

print(test_lables[:10])
print(pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": tree_predictions
}))
print(pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": forest_predictions
}))
print(pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": predictions
}))