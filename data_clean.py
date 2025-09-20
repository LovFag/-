from sklearn.model_selection import train_test_split

from titanic.Analyse_data_split import strat_train_set
from titanic.data_describe import test_data, train_data


# ---------- 通用特征工程函数 ----------
def feature_engineering(df):
    # Cabin → Deck
    df['Deck'] = df['Cabin'].str[0].fillna('Missing')
    # Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr',
                                       'Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')
    # Ticket prefix
    df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z\/]+)', expand=False).fillna('None')
    return df

train_df = feature_engineering(train_data)
test_df  = feature_engineering(test_data)

train_df['pclass_sex'] = train_df['Pclass'].astype(str) + '_' + train_df['Sex']

strat_train_set, strat_test_set = train_test_split(
    train_df,
    stratify=train_data['pclass_sex'],
    test_size=0.2,
    random_state=42
)

titanic_train_data = train_df.drop(columns=["Survived"])
titanic_labels = train_df["Survived"].copy()

# # 丢掉无用列
PassengerId = strat_train_set['PassengerId']
# titanic_train_data = strat_train_set.drop(columns=["Survived", "Name", "Ticket", "PassengerId"])
# titanic_labels = strat_train_set["Survived"].copy()

# 划分列
num_cols = ["Age", "Fare", 'FamilySize', 'IsAlone']
int_cols = ["Pclass", "SibSp", "Parch"]
cat_cols = ["Sex", "Embarked", 'Deck', 'Title', 'TicketPrefix']

if __name__ == '__main__':
    print(titanic_train_data.head())