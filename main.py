# Zdrojový kód bol inšpirovaný príkladmi z predmetu SUNS na inžinierskom stupnic štúdia na FEI STU.
# Niektoré (spomínané v dokumentácii) boli taktiež skonštruované za pomoci ChatGPT.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import seaborn as sb

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.min_rows', 20)
pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('zadanie1_dataset.csv')
df = df.drop(columns=['name', 'url', 'genres', 'filtered_genres'])
df['explicit'] = df['explicit'].astype(int)
df = df.drop_duplicates()
df_original = df.copy()

cols_to_limit = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
for col in cols_to_limit:
    df = df[(df[col] >= 0) & (df[col] <= 1)]

df = df[(df['popularity'] >= 0) & (df['popularity'] <= 100)]
df = df[(df['tempo'] > 0)]
df = df[(df['duration_ms'] > 40000)]

cols_to_check = ['loudness', 'tempo', 'duration_ms']
for col in cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    factor = 1.5
    if col == 'duration_ms':
        factor = 2.5
    df = df[~((df[col] < (Q1 - factor * IQR)) | (df[col] > (Q3 + factor * IQR)))]

for col in ['number_of_artists', 'explicit', 'top_genre', 'emotion']:
    missing_values = df[col].isnull().sum()

    if missing_values > 0:
        df = df.dropna(subset=[col])

df = df[df['explicit'].isin([0, 1])]
df = df[df['number_of_artists'] > 0]

df_preEncoding = df.copy()
'''
#Simple neural network using sklearn
#Encoding string features - LabelEncoding
le_emotion = LabelEncoder()
df['emotion'] = le_emotion.fit_transform(df['emotion'])

le_genre = LabelEncoder()
df['top_genre'] = le_genre.fit_transform(df['top_genre'])

X = df.drop('emotion', axis=1)
y = df['emotion']

#Data split and scaling
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

#Model training
mlp = MLPClassifier(hidden_layer_sizes=(100),
                    max_iter=500,
                    random_state=42)
mlp.fit(X_train_scaled, y_train)

class_names = list(le_emotion.classes_)

# Predict on train set using scaled data
y_pred_train = mlp.predict(X_train_scaled)
print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred_train))
cm_train = confusion_matrix(y_train, y_pred_train)

# Predict on test set using scaled data
y_pred_test = mlp.predict(X_test_scaled)
print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred_test))
cm_test = confusion_matrix(y_test, y_pred_test)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
_, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
_, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()
'''

# Neural network with Keras
# Create one-hot encodings for 'emotion' and 'top_genre' columns
X = pd.get_dummies(df.drop('emotion', axis=1), columns=['top_genre'], prefix='', prefix_sep='')
y = pd.get_dummies(df['emotion'], prefix='', prefix_sep='')

X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Define the Keras model for multi-class classificati
# on
# Adjust the model to make it more complex
model = Sequential()
model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

from keras.callbacks import EarlyStopping

# Define the early stopping callback
early_stopper = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=15,
    verbose=1,
    restore_best_weights=True
)

# Compile the model without any regularization
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

# Train the model for a large number of epochs
history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32,  callbacks=[early_stopper])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the train set
train_scores = model.evaluate(X_train, y_train, verbose=0)
print(f"Training accuracy: {train_scores[1]:.4f}")
# Evaluate the model on the test set
test_scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_scores[1]:.4f}")

# Predict the test set results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

'''
def save_box_plots(data, features, dataset_name):
    for feature in features:
        plt.figure(figsize=(12, 8))
        sb.boxplot(x=data[feature], y=[feature] * len(data), orient='h', patch_artist=True, boxprops=dict(alpha=0.7))
        plt.title(f'Box Plot of {feature} ({dataset_name})')
        plt.savefig(f"Box_plots/{feature}_box_{dataset_name}.png", bbox_inches='tight')
        plt.clf()
        plt.close('all')


def save_histograms(data, features, dataset_name):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sb.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Histogram of {feature} ({dataset_name})')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(f"Histograms/{feature}_hist_{dataset_name}.png", bbox_inches='tight')
        plt.clf()
        plt.close('all')


def save_bar_charts(data, features, dataset_name):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sb.countplot(data=data, x=feature, order=data[feature].value_counts().index)
        plt.title(f'Bar Chart of {feature} ({dataset_name})')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"Bar_charts/{feature}_bar_{dataset_name}.png", bbox_inches='tight')
        plt.clf()
        plt.close('all')


continous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                      'liveness',
                      'valence', 'tempo', 'duration_ms', 'popularity']

save_box_plots(df_original, continous_features, 'Original')
save_histograms(df_preEncoding, continous_features, 'Cleaned')

categorical_features = ['explicit', 'number_of_artists', 'top_genre', 'emotion']
save_bar_charts(df_original, categorical_features, 'Original')
save_bar_charts(df_preEncoding, categorical_features, 'Cleaned')
'''

'''
for i in range(len(continous_features)):
    for j in range(i + 1, len(continous_features)):
        plt.figure(figsize=(10, 6))
        sb.scatterplot(x=continous_features[i], y=continous_features[j], data=df)
        plt.title(f'Relationship between {continous_features[i]} and {continous_features[j]}')
        plt.xlabel(continous_features[i])
        plt.ylabel(continous_features[j])
        plt.savefig(f'Scatter_plots2/ScatterPlot_{continous_features[i]}_{continous_features[j]}.png',
                    bbox_inches='tight')
        plt.close()

plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.savefig('CorrelationMatrixHeatmap.png')
plt.show()

sb.violinplot(x='emotion', y='instrumentalness', data=df_preEncoding, inner='quartile')
plt.title('Comparison of intrumentalness of the four emotion categories')
plt.xlabel('Emotion')
plt.ylabel('Instrumentalness')
plt.show()

sb.violinplot(x='emotion', y='valence', data=df_preEncoding, inner='quartile')
plt.title('Comparison of valence of the four emotion categories')
plt.xlabel('Emotion')
plt.ylabel('Valence')
plt.show()
'''