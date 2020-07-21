import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc, save_figure_path):
    """ Plot model loss and accuracy through epochs. """

    green = '#72C29B'
    orange = '#FFA577'

    with plt.xkcd():
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
        ax1.plot(range(1, len(train_loss) + 1), train_loss, green, linewidth=5,
                 label='training')
        ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, orange,
                 linewidth=5, label='validation')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('loss')
        ax1.tick_params('y')
        ax1.legend(loc='upper right', shadow=False)
        ax1.set_title('Model loss through #epochs', fontweight='bold')

        ax2.plot(range(1, len(train_acc) + 1), train_acc, green, linewidth=5,
                 label='training')
        ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, orange,
                 linewidth=5, label='validation')
        ax2.set_xlabel('# epoch')
        ax2.set_ylabel('accuracy')
        ax2.tick_params('y')
        ax2.legend(loc='lower right', shadow=False)
        ax2.set_title('Model accuracy through #epochs', fontweight='bold')

    plt.tight_layout()
    plt.show()
    fig.savefig(save_figure_path)
    plt.close(fig)

# Read in the data files

month = 4
year = 2019


relativePath = os.getcwd()
data = pd.read_csv(relativePath + '/data/OLV_2015-2019.csv')
train = data[(data['Current_Year'] < year) | ((data['Current_Year'] == year) & (data['Current_Month'] < month))]
test = data[(data['Current_Year'] == year) & (data['Current_Month'] == month)]
test0 = test

# Set the column names
columns = ['Month_Between_First_and_Recent','Month_Between_Recent_Donations',
           'Total_Frequency', 'Total_Amount','Avg_Donation', 'Span', 'Persistability',
           'Month_Since_First_Donation', 'Donated_In_Current_Month', 'Current_Month', 'Current_Year', 'Donated_In_Next_Month']

# Separate categorical columns
cat_columns = ['Current_Month', 'Current_Year']
num_columns = [col for col in columns if col not in cat_columns]

# Change categorical columns to dummies
total = pd.concat([train, test], ignore_index=True)
print(train.shape)
print(test.shape)
total1 = total[num_columns]
for var in cat_columns:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(total[var], prefix=var)
    total1=total1.join(cat_list)
print(total1.columns)

trainRowNo = train.shape[0]
testRowNo = test.shape[0]
print(trainRowNo)
train = total1.head(trainRowNo)
test = total1.tail(testRowNo)
train = train.sample(frac=0.1, random_state=1)


# Split the train validation sets
Feature_columns = [col for col in train.columns if col != 'Donated_In_Next_Month']
X = train[Feature_columns]
y = train['Donated_In_Next_Month']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)



# Normalize the data

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_train.columns = Feature_columns


X_val = pd.DataFrame(scaler.transform(X_val))
X_val.columns = Feature_columns

X_test = test[Feature_columns]
y_test = test['Donated_In_Next_Month']
X_test = pd.DataFrame(scaler.transform(X_test))
X_test.columns = Feature_columns



# Running the NN
input_dim = X.shape[1]
model=Sequential()
model.add(Dense(300, input_dim=input_dim, activation="relu", kernel_initializer='random_uniform',
                bias_initializer=initializers.Constant(0.1)))
model.add(Dense(200, activation="relu", kernel_initializer='random_uniform',
                bias_initializer="zeros"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
x = model.fit(X_train,y_train, epochs=10, batch_size=1024, validation_data=(X_val, y_val), shuffle=True, verbose=1)
plot_model_performance(
    train_loss=x.history.get('loss', []),
    train_acc=x.history.get('acc', []),
    train_val_loss=x.history.get('val_loss', []),
    train_val_acc=x.history.get('val_acc', []),
    save_figure_path = relativePath +'/model_performance.png'
)
preds = model.predict(X_test)
preds_prob = np.array(preds, copy=True)
preds[preds>=0.5] = int(1)
preds[preds<0.5] = int(0)

result = test0[['ID', 'Donated_In_Next_Month']]
# prediction = pd.DataFrame(data=np.column_stack((preds, preds_prob)),columns=['prediction_' + str(month) + "_" + str(year), 'prediction_prob_' + str(month) + "_" + str(year)])
result['prediction_' + str(month) + "_" + str(year)] = preds
result['prediction_prob_' + str(month) + "_" + str(year)] = preds_prob
result.to_csv(relativePath + '/Results/test.csv', index=False)


matrix = confusion_matrix(y_test, preds)
print(matrix)
print("baseline point-wise acc: ", 1-y_test.sum()/len(y_test))
print("f1: ", f1_score(y_test, preds))
print("accuracy: ", accuracy_score(y_test, preds))
print("precision: ", precision_score(y_test, preds))
print("recall: ", recall_score(y_test, preds))
