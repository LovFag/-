import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from titanic.data_describe import train_data


# 创造新组合pclass的男女
train_data['pclass_sex'] = train_data['Pclass'].astype(str) + '_' + train_data['Sex']

# 根据组合特征进行数据集划分(训练集，验证集)
strat_train_set, strat_test_set = train_test_split(
    train_data,
    stratify=train_data['pclass_sex'],
    test_size=0.2,
    random_state=42
)

if __name__ == '__main__':
    print("划分前：\n",train_data.head())
    print("划分后：\n", strat_train_set.head())

    for gender in ["male", "female"]:
        print("男性存活情况：")
        print(train_data[train_data["Sex"] == "male"]["Survived"].value_counts())
        print("女性存活情况：")
        print(train_data[train_data["Sex"] == "female"]["Survived"].value_counts())

    for pclass in [1, 2, 3]:
        print("Pclass = %d 存活情况：" % pclass)
        print(train_data[train_data["Pclass"] == pclass]["Survived"].value_counts())

    # 每个pclass的男女性存活情况
    for pclass in [1, 2, 3]:
        print("Pclass = %d 的男性存活情况：" % pclass)
        print(train_data[(train_data["Pclass"] == pclass) & (train_data["Sex"] == "male")]["Survived"].value_counts())
        print("Pclass = %d 的女性存活情况：" % pclass)
        print(train_data[(train_data["Pclass"] == pclass) & (train_data["Sex"] == "female")]["Survived"].value_counts())


    # 可视化新属性存活率分布
    # 方法一
    plt.figure(1)
    sns.countplot(data=train_data,
                  x='pclass_sex',
                  hue='Survived')   # 0=遇难，1=生还
    plt.xticks(rotation=45)
    plt.title('Count of Survived vs Not-Survived by Pclass & Sex')

    # 方法二
    plt.figure(2)
    ct = pd.crosstab(train_data['pclass_sex'], train_data['Survived'], normalize='index')
    # normalize='index' 把行转为比例 → 存活率
    sns.heatmap(ct,
                annot=True,
                fmt='.2%',
                cmap='coolwarm')
    plt.title('Survival Rate within each Pclass-Sex group')
    plt.ylabel('Pclass_Sex')
    plt.xlabel('Survived')
    plt.show()

