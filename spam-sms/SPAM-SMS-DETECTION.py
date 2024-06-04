import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Ensure nltk resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize the stemmer
ps = PorterStemmer()

# Load the dataset
df = pd.read_csv("E:\\codsoft machine learing projects\\spam.csv")
print(df.sample(5))
print(df.shape)



# 1. DATA CLEANING ##
print(df.info())

# Drop last 3 columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
print(df.sample(5))

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
print(df.sample(5))

# Encode target labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check for duplicate values
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates(keep='first')
print(df.duplicated().sum())
print(df.shape)



# 2. EDA ##
print(df.head())
print(df['target'].value_counts())

plt.ion()

# Plot pie chart
plt.figure()
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%.2f%%")
plt.show()

# Data imbalance check
df['num_characters'] = df['text'].apply(len)
print(df.head())

# Number of words
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
print(df.head())

# Number of sentences
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(df.head())

print(df[['num_characters', 'num_words', 'num_sentences']].describe())

# Stats for ham messages
print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe())

# Stats for spam messages
print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe())

# Plot histograms
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_words'], kde=False, color='blue', label='ham')
sns.histplot(df[df['target'] == 1]['num_words'], kde=False, color='red', label='spam')
plt.legend()
plt.show()

# Pairplot
sns.pairplot(df, hue='target')
plt.show()

print(df.info())

# Select numerical columns for correlation
df_numeric = df.select_dtypes(include=[np.number])
print(df_numeric.corr())

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, fmt='.2f')
plt.show()

# Fill NaNs in correlation matrix with 0
corr_matrix = df_numeric.corr().fillna(0)
print(corr_matrix)



# 3. Data Preprocessing ##
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)
print(df.head())

# WordCloud for spam messages
spam_corpus = df[df['target'] == 1]['transformed_text'].str.cat(sep=" ")
wc_spam = WordCloud(width=500, height=500, min_font_size=10, background_color='black').generate(spam_corpus)
plt.figure(figsize=(10, 6))
plt.imshow(wc_spam)
plt.axis('off')
plt.title('Word Cloud for Spam Messages')
plt.show()

# WordCloud for ham messages
ham_corpus = df[df['target'] == 0]['transformed_text'].str.cat(sep=" ")
wc_ham = WordCloud(width=500, height=500, min_font_size=10, background_color='black').generate(ham_corpus)
plt.figure(figsize=(10, 6))
plt.imshow(wc_ham)
plt.axis('off')
plt.title('Word Cloud for Ham Messages')
plt.show()

# Create a list of words from the spam messages
spam_words = spam_corpus.split()

# Create a Counter object from the spam_corpus list
s_word_counter = Counter(spam_words)

# Get the most common 30 words and their frequencies
s_common_words = s_word_counter.most_common(30)

# Create a DataFrame from the most common words and frequencies
df_s_common_words = pd.DataFrame(s_common_words, columns=['Word', 'Frequency'])

# Plot the barplot using DataFrame df_common_words
plt.figure(figsize=(10, 6))
sns.barplot(data=df_s_common_words, x='Word', y='Frequency')
plt.xticks(rotation='vertical')
plt.title('Top 30 Most Common Words in Spam Messages')
plt.show()

# Create a list of words from the ham messages
ham_words = ham_corpus.split()

# Create a Counter object from the ham_corpus list
h_word_counter = Counter(ham_words)

# Get the most common 30 words and their frequencies
h_common_words = h_word_counter.most_common(30)

# Create a DataFrame from the most common words and frequencies
df_h_common_words = pd.DataFrame(h_common_words, columns=['Word', 'Frequency'])

# Plot the barplot using DataFrame df_common_words
plt.figure(figsize=(10, 6))
sns.barplot(data=df_h_common_words, x='Word', y='Frequency')
plt.xticks(rotation='vertical')
plt.title('Top 30 Most Common Words in Ham Messages')
plt.show()

# Print the DataFrame to check its contents
print(df_s_common_words)
print(df_h_common_words)



# 4. Model Building ##
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()

#scaler=MinMaxScaler()
#X = scaler.fit_transform(X)

print(X.shape)

y = df['target'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize models
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Train and evaluate GaussianNB
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print("GaussianNB:")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1))

# Train and evaluate MultinomialNB
mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print("MultinomialNB:")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))

# Train and evaluate BernoulliNB
bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print("BernoulliNB:")
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred3))
print("Precision:", precision_score(y_test, y_pred3))

# Define classifiers
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2, algorithm='SAMME')
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

# Function to train and evaluate classifiers
def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

# Evaluate each classifier
accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print(f"For {name}:")
    print(f"Accuracy = {current_accuracy}")
    print(f"Precision = {current_precision}")
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

# Create and display performance DataFrame
performance_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}).sort_values('Precision', ascending=False)
print(performance_df)

# Create and display melted performance DataFrame
performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
print(performance_df1)

# Create and display barplot based on melted performance
sns.catplot(x = 'Algorithm', y = 'value', hue = 'variable', data=performance_df1, kind='bar', height = 5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()

# model improve
# 1. Change the max_features of TfIdf
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)
new_df = performance_df.merge(temp_df,on='Algorithm')
new_df_scaled = new_df.merge(temp_df,on='Algorithm')
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
print(new_df_scaled.merge(temp_df,on='Algorithm'))

# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
voting.fit(X_train,y_train)

VotingClassifier(estimators=[('svm',SVC(gamma=1.0, kernel='sigmoid',probability=True)),('nb', MultinomialNB()),('et',ExtraTreesClassifier(n_estimators=50,random_state=2))],voting='soft')
y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator = RandomForestClassifier()

from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
        
