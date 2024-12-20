# %%
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cufflinks as cf
cf.go_offline()


# %%
train= pd.read_csv('train.csv')
train.head()

# %%
train.isnull()

# %%
sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# %%
train.columns

# %%
sb.countplot(x='Survived',hue='Pclass',data=train)

# %%
sb.displot(train['Age'].dropna(),bins=30)

# %%
train['Age'].iplot(kind='hist',bins=30)

# %%
plt.figure(figsize=(12, 7))
sb.boxplot(x='Pclass',y='Age',data=train,palette='winter')

# %%
#fill the null values in Age with the average ages per class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
# %%
#apply the function to the Age column using Age and Pclass columns
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

# %%
#heatmap to confirm all Ages are filled
sb.heatmap(train.isnull(),yticklabels=False,cbar=False)

# %%
#drop the Cabin column since it has too many null values to be filled
train.drop('Cabin',axis=1,inplace=True)

# %%
#drop any other possible null value
train.dropna(inplace=True)

# %%
#Verify that there are no blanks left
sb.heatmap(train.isnull(),yticklabels=False,cbar=False)

# %%
sex = pd.get_dummies(train['Sex'],drop_first=True) 
#we are dropping 'female' column so the algorith doesnt automatically correlate 0 in 'male' to be '1' in female and vice-verse
embark = pd.get_dummies(train['Embarked'],drop_first=True)
#same logic here for the first value in Embarked

# %%
train = pd.concat([train,sex,embark],axis=1)

# %%
train.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
#these are string columns which wont help with the logical analysis

# %%
train.head()

# %%
from sklearn.model_selection import train_test_split

#Can also define separately as train.drop('Survived',axis=1),and Y as train['Survived']

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.4, 
                                                    random_state=101)
# %%
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()

# %%
logmodel.fit(X_train,y_train)

# %%
predictions = logmodel.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print('\n')
confusion_matrix(y_test,predictions)

# %%
