# Uczenie Maszynowe w Analizie Danych – Projekt 2025Z

## Lab 3: Implementacja klasyfikatorów. Dostrajanie hiperparametrów.

### 1. Wprowadzenie do klasyfikacji

**Klasyfikacja** to jeden z głównych problemów uczenia maszynowego — polega na przypisaniu obserwacji do jednej z kilku z góry określonych klas (np. „spam” / „nie-spam”, „zdrowy” / „chory”).

Dane wejściowe mają postać wektora cech (X), a etykiety (y) oznaczają klasy.

Przykład:

| Wzrost | Waga | Klasa     |
| ------ | ---- | --------- |
| 175    | 70   | Mężczyzna |
| 160    | 55   | Kobieta   |

Celem modelu jest nauczenie się wzorca, który pozwoli klasyfikować nowe, nieznane dane.

---

### 2. Podstawowe klasyfikatory w scikit-learn

#### a) Regresja logistyczna

Mimo nazwy, to może być klasyfikator. Modeluje prawdopodobieństwo przynależności do danej klasy.

```python
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dane
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predykcja i ocena
y_pred = clf.predict(X_test)
print("Dokładność:", accuracy_score(y_test, y_pred))
```

**Zalety:** interpretowalność, szybkość

**Wady:** słabo radzi sobie z nieliniowością

---

#### b) Drzewo decyzyjne

Model dzieli dane według atrybutów, tworząc hierarchię decyzji.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("Dokładność drzewa:", accuracy_score(y_test, y_pred))
```

**Zalety:** łatwa interpretacja, obsługuje dane kategoryczne i liczbowe

**Wady:** skłonność do przeuczenia (overfitting)

Wizualizacja:

```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
tree.plot_tree(tree, filled=True)
plt.show()
```

---

#### c) K-Nearest Neighbors (KNN)

Klasyfikator opiera się na podobieństwie do sąsiadów w przestrzeni cech.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Dokładność KNN:", accuracy_score(y_test, knn.predict(X_test)))
```

**Zalety:** prosty, nie wymaga uczenia parametrów

**Wady:** wolny dla dużych zbiorów danych

---

#### d) Support Vector Machine (SVM)

Tworzy hiperplan maksymalnie oddzielający klasy.

```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
print("Dokładność SVM:", accuracy_score(y_test, svm.predict(X_test)))
```

**Zalety:** dobra skuteczność przy małej liczbie próbek

**Wady:** trudność w doborze hiperparametrów, wolne uczenie przy dużych zbiorach

---

#### e) Random Forest

Zbiór wielu drzew decyzyjnych — klasyfikacja odbywa się przez głosowanie większościowe.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Dokładność RF:", accuracy_score(y_test, rf.predict(X_test)))
```

**Zalety:** wysoka skuteczność, odporny na overfitting

**Wady:** mniejsza interpretowalność

---

### 3. Ocena jakości klasyfikatora

Oceniamy skuteczność modelu za pomocą metryk:

* **accuracy_score** – dokładność klasyfikacji,
* **precision, recall, f1-score** – dla problemów z niezrównoważonymi danymi,
* **confusion matrix** – macierz pomyłek.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Wizualizacja macierzy:

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Przewidziane")
plt.ylabel("Rzeczywiste")
plt.show()
```

---

### 4. Hiperparametry i ich znaczenie

Hiperparametry to szczegółowe ustawienia modelu określane **przed uczeniem**, które wpływają na jego działanie.

Przykłady:

* `max_depth` – maksymalna głębokość drzewa decyzyjnego,
* `n_estimators` – liczba drzew w RandomForest,
* `C` i `kernel` w SVM,
* `n_neighbors` w KNN.

---

### 5. Dostrajanie hiperparametrów (Hyperparameter Tuning)

Dwa najczęściej stosowane podejścia to:

#### a) Grid Search

Przeszukuje wszystkie kombinacje zadanych parametrów.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Najlepsze parametry:", grid.best_params_)
print("Dokładność:", grid.best_score_)
```

**Zalety:** dokładne przeszukanie przestrzeni parametrów

**Wady:** czasochłonne przy dużych zakresach

---

#### b) Random Search

Losowo wybiera kombinacje parametrów – szybszy przy dużych przestrzeniach.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10)
}

rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=10, cv=5)
rand_search.fit(X_train, y_train)

print("Najlepsze parametry:", rand_search.best_params_)
print("Najlepszy wynik:", rand_search.best_score_)
```

---

#### c) Walidacja krzyżowa (Cross-validation)

Podczas dostrajania parametrów warto stosować **podział danych na kilka części (foldów)**, by uniknąć przypadkowych wyników.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)
print("Średnia dokładność:", scores.mean())
```

---

### 6. Wizualizacja wyników dostrajania

Warto analizować wpływ parametrów na wynik, np. dla liczby sąsiadów w KNN:

```python
neighbors = range(1, 21)
scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.plot(neighbors, scores, marker='o')
plt.title("Dokładność dla różnych wartości k")
plt.xlabel("Liczba sąsiadów (k)")
plt.ylabel("Dokładność")
plt.show()
```

---

### 7. Podsumowanie

| Etap                        | Opis                                          |
| --------------------------- | --------------------------------------------- |
| Wybór klasyfikatora         | Dobierz algorytm odpowiedni do problemu       |
| Uczenie modelu              | `fit(X_train, y_train)`                       |
| Predykcja                   | `predict(X_test)`                             |
| Ocena jakości               | `accuracy`, `precision`, `recall`, `f1-score` |
| Dostrajanie hiperparametrów | `GridSearchCV`, `RandomizedSearchCV`          |
| Walidacja                   | `cross_val_score`                             |

---

### 8. Zalecane biblioteki biblioteki

* `pandas` – operacje na danych
* `numpy` – obliczenia numeryczne
* `matplotlib`, `seaborn` – wizualizacje
* `scikit-learn` – klasyfikatory, metryki, optymalizacja

### Zadania
1. Poczytaj o modelach AI do zadania klasyfikacji. Bazując na wybranym przez Ciebie zbiorze przefiltruj modele do klasyfikacji adekwatnej do twojego zbioru (binarna albo wieloklasowa).
Lista najczęściej używanych (stosunkowo prostych) modeli w klasyfikacji:
    - Logistic Regression (Regresja logistyczna)
    - K-Nearest Neighbors (K-NN)
    - Decision Tree (Drzewo decyzyjne)
    - Random Forest (Las losowy)
    - Gradient Boosting (XGBoost / LightGBM / CatBoost)
    - Support Vector Machine (SVM)
    - Naive Bayes
    - Perceptron
    - Multi-Layer Perceptron (MLP)
    - Convolutional Neural Network (CNN)
    - Recurrent Neural Network (RNN)
    - Long Short-Term Memory (LSTM)
    - AdaBoost
    - Ridge Classifier
    - Bagging Classifier
    - Bernoulli Naive Bayes
2. Wybierz 3-5 modeli, które według teorii możesz wykorzystać do zadania klasyfikacji swojego zbioru.
3. Pomyśl, jakich operacji pre-processingu musisz użyć na kopiach dancyh swoje zbioru, aby zastosować wybrane przez Ciebie zbiory (uzupełnianie/ usuwanie wartości brakujacych, kodowanie atrybutów kategorialnych, standaryzacja, normalizacja, itd.)
