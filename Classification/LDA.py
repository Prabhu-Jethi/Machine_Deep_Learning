import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris_df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\iris\iris.data", header=None)
print(iris_df)

iris_df.columns = ['SL', 'SW', 'PL', 'PW', 'Flowers']

print(iris_df.isnull().sum())

flower_species = {'Iris-setosa':0, 'Iris-virginica':1, 'Iris-versicolor':2}
iris_df['Flowers'] = iris_df['Flowers'].map(flower_species)

X = iris_df.drop('Flowers', axis=1)
y = iris_df['Flowers']
print(X)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#### LDA
lda = LinearDiscriminantAnalysis(n_components=2)
model = lda.fit_transform(X, y)
print("Original Shape:", X.shape)
print("Reduced Shape:", model.shape)


df = pd.DataFrame(model, columns=["LD1", "LD2"])
df["Species"] = y
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="LD1",
    y="LD2",
    hue="Species",
    palette="deep",
    s=100
)
plt.title("LDA on Iris Dataset")
plt.show()


lda.fit(X_train, y_train)
print("Linear-Discriminant-model:\n", lda)
train_ac = lda.score(X_train, y_train)
test_ac = lda.score(X_test, y_test)
print("Train:ac", train_ac)
print("Test:ac", test_ac)


cv_scores = cross_val_score(lda, X, y, scoring='accuracy', n_jobs=-1, cv=5)
print("\nCross_Validation: %.4f +- %.4f" % (cv_scores.mean(), cv_scores.std()))


y_pred = lda.predict(X_test)


print("\nAccuracy:\n", accuracy_score(y_test, y_pred))
print("\nClass-report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion_matrix:\n", cm)
sns.heatmap(cm, cmap='hot', fmt='d', annot=True,
            xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginia'],
            yticklabels=['Iris-virginia', 'Iris-versicolor', 'Iris-setosa'])
plt.title("Confusion_Matrix")
plt.show()