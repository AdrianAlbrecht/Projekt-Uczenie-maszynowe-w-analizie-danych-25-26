# Uczenie Maszynowe w Analizie Danych – Projekt 2025Z

## Lab 4: Porównanie modeli i baseline’u. Przygotowanie wyników i wniosków w formie raportu.

### Przygotowanie wyników i wniosków w formie raportu

W procesie uczenia maszynowego nie wystarczy jedynie zbudować model — konieczne jest **krytyczne porównanie wyników** uzyskanych przez różne algorytmy i zestawienie ich z tzw. **baseline’em**, czyli prostą, referencyjną metodą, która stanowi punkt odniesienia. Celem tej części zajęć jest zrozumienie, jak interpretować wyniki, jak je porównywać oraz jak tworzyć raport, który prezentuje cały proces w sposób zrozumiały i transparentny.

---

### 1. Czym jest baseline?

Baseline to **prosty model referencyjny**, który ustala minimalny poziom jakości, jaki powinien osiągnąć bardziej zaawansowany model.
Dzięki niemu wiemy, czy nasze modele są:

* faktycznie użyteczne,
* czy może tylko *wyglądają na skomplikowane, ale nie przewyższają przypadkowego zgadywania*.

#### Typowe baseline'y:

##### 🔹 **Dla klasyfikacji binarnej:**

* przewidywanie zawsze większościowej klasy,
* losowy klasyfikator (random guess),
* prosty model logistyczny z domyślnymi parametrami,
* model „zero rule” (ZeroR): przewiduje zawsze najczęstszy wynik.

##### 🔹 **Dla klasyfikacji wieloklasowej:**

* przewidywanie najczęstszej klasy,
* losowy wybór klasy z wagami proporcjonalnymi do częstości.

#### 🔹 Po co baseline?

* pozwala ocenić, czy model poprawia wyniki względem „naiwnego” podejścia,
* umożliwia porównanie z modelami zaawansowanymi,
* daje kontekst dla metryk — accuracy 85% może być świetne, ale jeśli baseline ma 84%, to już niekoniecznie.

---

### 2. Porównywanie modeli — jakie aspekty bierzemy pod uwagę?

Porównanie modeli to nie tylko spojrzenie na accuracy. Należy uwzględnić:

* **różne metryki jakości**,
* **stabilność modelu**,
* **czas trenowania**,
* **złożoność**,
* **przeciążenie (overfitting)** i **niedouczenie (underfitting)**,
* **czy model jest interpretowalny**.

Dobre porównanie to takie, które pokazuje **kompromisy** między modelami.

---

### 3. Metryki używane przy porównaniu modeli klasyfikacyjnych

Najczęściej stosowane:

#### 🔹 **Accuracy**

Procent poprawnych klasyfikacji — dobre przy zbalansowanych klasach.

#### 🔹 **Precision i Recall**

Szczególnie ważne w problemach nierównowagi klas.

* Precision — ile z przewidzianych pozytywnych przykładów było poprawnych.
* Recall — ile z faktycznych pozytywów udało się odnaleźć.

#### 🔹 **F1-score**

Harmoniczna średnia precision i recall.

#### 🔹 **Confusion matrix**

Pokazuje dokładnie, gdzie pojawiają się pomyłki.

#### 🔹 **ROC AUC**

Wskazuje, jak dobrze model rozróżnia klasy niezależnie od progu.

#### 🔹 **Log Loss / Cross Entropy**

Pokazuje jakość probabilistycznych predykcji.

#### 🔹 **Balanced Accuracy**

Lepsza miara przy niezbalansowanych klasach.

---

### 4. Jak poprawnie wykonać porównanie?

#### 1. **Ustaw identyczne warunki eksperymentu**

* ten sam zbiór danych,
* ten sam podział trening/test albo ta sama walidacja krzyżowa,
* identyczny preprocessing danych.

#### 2. **Zbuduj baseline**

Nawet jeśli jest słaby — musi istnieć w eksperymencie.

#### 3. **Naucz kilka różnych modeli**

Np.:

* Logistic Regression
* Decision Tree
* Random Forest
* SVM
* kNN
* Naive Bayes
* Perceptron / MLP

#### 4. **Zbieraj metryki**

Najlepiej w formie tabeli, np.:

| Model               | Accuracy | Precision | Recall | F1   | AUC  |
| ------------------- | -------- | --------- | ------ | ---- | ---- |
| Baseline (majority) | 72%      | —         | —      | —    | —    |
| Logistic Regression | 85%      | 0.82      | 0.78   | 0.80 | 0.87 |
| Random Forest       | 90%      | 0.88      | 0.86   | 0.87 | 0.93 |
| SVM RBF             | 88%      | 0.85      | 0.84   | 0.84 | 0.91 |

#### 5. **Zadbaj o powtarzalność**

* ustaw seed (`random_state`),
* opisuj parametry modeli,
* kontroluj losowość np. w k-fold cross-validation.

---

### 5. Jak przygotować czytelny raport?

Dobrze przygotowany raport jest kluczowy.
Powinien zawierać **logiczny, spójny opis całego procesu**, a nie tylko same liczby.

#### Raport powinien mieć:

#### 1. **Opis zbioru danych**

* wymiar,
* liczba klas,
* opis cech,
* występowanie braków danych.

#### 2. **Cel analizy**

* Co model ma klasyfikować?
* Jakie metryki są kluczowe?

#### 3. **Opis baseline’u**

* co wybrano jako baseline i dlaczego,
* jakie osiągnął metryki.

#### 4. **Wyniki każdego modelu**

* w tabeli i/lub wykresach,
* opisowo: co działa dobrze a co źle.

#### 5. **Wpływ tuningu hiperparametrów**

* czy tuning poprawił wyniki?
* o ile?

#### 6. **Wizualizacje**

* confusion matrix dla najlepszego modelu,
* ROC curves, jeśli klasyfikacja binarna,
* bar chart z wynikami modeli.

#### 7. **Wnioski końcowe**

Przykładowe punkty:

* który model jest najlepszy i dlaczego,
* gdzie model się myli,
* czy wyniki są wystarczające dla zastosowań praktycznych,
* co można poprawić (dalszy tuning, inne cechy, większy zbiór).

---

### 7. Jak formułować wartościowe wnioski?

Zamiast pisać:

> Random Forest jest najlepszy.

Napisz:

> Random Forest uzyskał najwyższy F1-score (0.87), co sugeruje, że dobrze radzi sobie z nierównowagą klas.
> Model ma niski błąd na zbiorze testowym, a krzywa ROC wskazuje dużą zdolność separacji klas.
> Ograniczeniem jest natomiast słaba interpretowalność.

---

### Podsumowanie

W tej części zajęć studenci powinni:

* stworzyć baseline baseline jako klasyfikator losowy:


>Najłatwiejszy sposób to użycie gotowego narzędzia z **scikit-learn**:
`DummyClassifier`.
>
>Pozwala on zdefiniować baseline jako:
>
>* klasyfikator przewidujący **losowo**,
>* klasyfikator przewidujący **zawsze większościową klasę**,
>* klasyfikator **stratified** (losowo z zachowaniem proporcji klas).
>
>
>#### 1. Losowy klasyfikator (zupełnie losowy)
>
>```python
>from sklearn.dummy import DummyClassifier
>from sklearn.model_selection import train_test_split
>from sklearn.metrics import accuracy_score
>
># Załóżmy że X, y to Twój dataset
>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>
># Baseline: losowe przewidywanie
>baseline_random = DummyClassifier(strategy="uniform", random_state=42)
>baseline_random.fit(X_train, y_train)
>
>y_pred = baseline_random.predict(X_test)
>
>print("Accuracy (losowy baseline):", accuracy_score(y_test, y_pred))
>```
>
>##### Co robi `strategy="uniform"`?
>
>* Model wybiera **każdą klasę z jednakowym prawdopodobieństwem**.
>* Jeśli masz 3 klasy → każda ma 33% szans.
>
>To najprostszy możliwy baseline.
>
>---
>
>####  2. Losowy baseline z zachowaniem proporcji klas (bardziej sensowny)
>
>```python
>baseline_strat = DummyClassifier(strategy="stratified", random_state=42)
>baseline_strat.fit(X_train, y_train)
>
>y_pred = baseline_strat.predict(X_test)
>
>print("Accuracy (baseline stratified):", accuracy_score(y_test, y_pred))
>```
>
>##### Co robi `strategy="stratified"`?
>
>* Losuje klasy **zgodnie z rozkładem danych treningowych**.
>* Jeśli 80% danych to klasa 0, a 20% to klasa 1 → model będzie losował 80/20.
>
>To najlepszy baseline dla modeli uczących się na niezbalansowanych danych.
>
>---
>
>#### 3. Baseline przewidujący zawsze większościową klasę
>
>Warto dodać — to najczęściej stosowany baseline:
>
>```python
>baseline_majority = DummyClassifier(strategy="most_frequent")
>baseline_majority.fit(X_train, y_train)
>
>y_pred = baseline_majority.predict(X_test)
>
>print("Accuracy (most frequent):", accuracy_score(y_test, y_pred))
>```
>
>---
>
>#### 4. Wypisanie pełnych metryk dla baseline'u
>
>```python
>from sklearn.metrics import classification_report
>
>print(classification_report(y_test, y_pred))
>```

* porównać swoje modele z baseline'm,
* przeanalizować metryki,
* wykonywać tuning hiperparametrów,
* przygotować czytelny, naukowy raport,
* wybrać "najlepszy" model.