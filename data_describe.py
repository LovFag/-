import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

train_data = pd.read_csv("datasets/train.csv")
test_data = pd.read_csv("datasets/test.csv")

# 可视化分析
if __name__ == "__main__":
    train_data.info()

    with pd.option_context('display.max_columns', None, 'display.width', 180):
        print(train_data.head())

    print(len(train_data[train_data["Sex"] == "male"])) # 577
    print(len(train_data[train_data["Sex"] == "female"])) # 314

    # 先把 object/category 类型列做数值编码
    encoded = train_data.copy()
    for col in encoded.select_dtypes(include=['object', 'category']).columns:
        encoded[col] = encoded[col].astype('category').cat.codes

    encoded.hist(bins=50, figsize=(20, 15))

    corr_matrix = encoded.corr()
    print(corr_matrix["Survived"].sort_values(ascending=False))

    attributes = ["Survived", "Sex", "Age", "Pclass"]
    scatter_matrix(encoded[attributes], figsize=(12, 8))
    plt.show()

