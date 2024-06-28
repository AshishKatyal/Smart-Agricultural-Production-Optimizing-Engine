import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

st.title('Smart Agricultural Production Optimizing Engine')


def load_data():
    df = pd.read_csv('C:/Users/91999/Python/Python311/Crop_recommendation.csv')
    return df

df = load_data()

if st.checkbox('Show raw data'):
    st.write(df.head())

st.write("Shape of the dataframe: ", df.shape)

if st.checkbox('Show dataframe info'):
    buffer = st.empty()
    df.info(buf=buffer)
    st.text(buffer)
if st.checkbox('Show descriptive statistics'):
    st.write(df.describe())

st.write('Data Types:')
st.write(df.dtypes)


# Histograms
st.write('### Histograms')
columns = ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']
for col in columns:
    st.write(f'### {col.capitalize()}')
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=20, kde=True, edgecolor="black", ax=ax)
    plt.title(col.capitalize(), size=20)
    st.pyplot(fig)

# Scatter Plot
st.write('### Scatter Plot')
fig, ax = plt.subplots()
sns.scatterplot(x='rainfall', y='temperature', hue='label', data=df, ax=ax)
plt.title('Rainfall vs Temperature')
st.pyplot(fig)

# Pair Plot
st.write('### Pair Plot')
fig = sns.pairplot(df, hue='label')
st.pyplot(fig)

# Box Plots
st.write('### Box Plots')
for col in ['temperature', 'humidity', 'N', 'P', 'K', 'ph']:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='label', y=col, data=df, ax=ax)
    plt.title(f'{col.capitalize()} by Crop Label', size=20)
    st.pyplot(fig)

# Unique Crops
crops = df['label'].unique()
st.write(f'Number of unique crops: {len(crops)}')
st.write(crops)
st.write(pd.value_counts(df['label']))


# Detect Outliers
def detect_outlier(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3 - q1
    lower_limit = q1 - (1.5 * IQR)
    upper_limit = q3 + (1.5 * IQR)
    outliers = x[(x < lower_limit) | (x > upper_limit)]
    return outliers

st.write('### Outliers in Potassium for Grapes')
st.write(detect_outlier(df['K'][df['label'] == 'grapes']))


# Preprocessing and Model Training
x = df.drop(['label'], axis=1)
Y = df['label']
encode = preprocessing.LabelEncoder()
y = encode.fit_transform(Y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    'decision_tree': {
        'model': DecisionTreeClassifier(criterion='gini'),
        'params': {'decisiontreeclassifier__splitter': ['best', 'random']}
    },
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['rbf', 'linear']}
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {'randomforestclassifier__n_estimators': [1, 5, 10]}
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {'kneighborsclassifier__n_neighbors': [5, 10, 20, 25], 'kneighborsclassifier__weights': ['uniform', 'distance']}
    }
}

score = []
details = []
best_param = {}

for name, params in models.items():
    pipe = make_pipeline(preprocessing.StandardScaler(), params['model'])
    res = model_selection.GridSearchCV(pipe, params['params'], cv=5)
    res.fit(x_train, y_train)
    score.append({
        'Model name': name,
        'Best score': res.best_score_,
        'Best param': res.best_params_
    })
    details.append(pd.DataFrame(res.cv_results_))
    best_param[name] = res.best_estimator_

scores_df = pd.DataFrame(score)
st.write('### Model Scores')
st.write(scores_df)


st.write('### Best Model Performance')
for model_name, model in best_param.items():
    st.write(f'{model_name}: {model.score(x_test, y_test)}')


# Confusion Matrix
st.write('### Confusion Matrix')
predicted = best_param['random_forest'].predict(x_test)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, predicted), annot=True, fmt='d', ax=ax)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)

# Bagging Classifier
pipe1 = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(n_estimators=10))
bag_model = BaggingClassifier(estimator=pipe1, n_estimators=100, oob_score=True, random_state=0, max_samples=0.8)
bag_model.fit(x_train, y_train)

st.write(f'Bagging Classifier Test Score: {bag_model.score(x_test, y_test)}')
bag_predicted = bag_model.predict(x_test)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, bag_predicted), annot=True, fmt='d', ax=ax)
st.pyplot(fig)

# Classification Report
st.write('### Classification Report')
st.text(classification_report(y_test, bag_predicted))


