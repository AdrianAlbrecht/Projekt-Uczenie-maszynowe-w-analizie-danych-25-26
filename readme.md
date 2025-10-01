# Uczenie Maszynowe w Analizie Danych – Projekt 2025Z

## Opis Zajęć

Celem przedmiotu „Uczenie Maszynowe w Analizie Danych – Projekt” jest praktyczne zastosowanie poznanych metod uczenia maszynowego w analizie danych poprzez przygotowanie indywidualnego projektu w środowisku Jupyter Notebook.

Projekt polega na:

* samodzielnym wyborze zbioru danych do zadania klasyfikacji,
* przeprowadzeniu analizy eksploracyjnej (EDA),
* przygotowaniu danych do uczenia,
* zbudowaniu i dostrojeniu kilku różnych klasyfikatorów (min.3),
* porównaniu ich wyników z klasyfikatorem losowym,
* omówieniu wyników i wyciągnięciu wniosków.

---

## Cele Zajęć

* Samodzielne przeprowadzenie pełnego procesu analizy danych pod kątem klasyfikacji.
* Praktyczne zastosowanie różnych metod klasyfikacji w Pythonie.
* Zdobycie umiejętności przygotowania danych i ich transformacji.
* Umiejętność porównania działania modeli i doboru metryk oceny.
* Rozwinięcie umiejętności dokumentowania i komentowania kodu.

---

## Zasady Zajęć

### 1. Uczestnictwo

* Projekt jest realizowany indywidualnie.
* Obecność na zajęciach jest obowiązkowa – dopuszczalne są 2 nieobecności nieusprawiedliwione.
* Konsultacje projektowe są częścią zajęć – należy przygotowywać postępy na bieżąco.

### 2. Praca indywidualna

* Projekt ma być przygotowany w formacie **Jupyter Notebook (.ipynb)** z pełnymi komentarzami i opisem kroków.
* Zabronione jest korzystanie z narzędzi generujących gotowy kod (ChatGPT, Copilot, itp.).
* Dozwolone jest korzystanie z dokumentacji, tutoriali, książek i artykułów.
* Projekt należy oddać najpóźniej do ostatnich zajęć.

### 3. Ocena

Projekt oceniany będzie na maksymalnie **40 punktów**.
Aby zaliczyć, należy zdobyć minimum **21 pkt (51%)**.

---

## Kryteria Oceny Projektu

1. **Opis i charakterystyka zbioru danych (5 pkt)**

   * źródło danych,
   * liczba próbek, atrybuty, klasy,
   * wstępne obserwacje.

2. **Przygotowanie danych (5 pkt)**

   * oczyszczanie braków i anomalii,
   * kodowanie danych kategorycznych,
   * skalowanie i normalizacja,
   * podział na zbiory treningowy/testowy.

3. **Budowa i porównanie klasyfikatorów (15 pkt)**

   * min. **3 różnych klasyfikatorów** (np. Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, MLP),
   * zastosowanie **klasyfikatora losowego jako baseline**,
   * opis konfiguracji modeli.

4. **Dostrajanie i ewaluacja modeli (10 pkt)**

   * użycie GridSearchCV / RandomizedSearchCV lub innego podejścia,
   * porównanie na podstawie metryk (accuracy, precision, recall, f1-score, AUC),
   * prezentacja wyników w tabeli i na wykresach (np. confusion matrix, ROC).

5. **Wnioski i podsumowanie (5 pkt)**

   * interpretacja wyników,
   * wskazanie najlepszego modelu i uzasadnienie,
   * refleksja nad jakością danych i ewentualnymi ograniczeniami.

---

## Proponowany „Rozkład Jazdy” Projektu

1. Omówienie źródeł danych i wyboru datasetów.
2. Analiza eksploracyjna danych (EDA). Oczyszczanie i przygotowanie danych.
3. Implementacja klasyfikatorów. Dostrajanie hiperparametrów.
4. Porównanie modeli i baseline’u. Przygotowanie wyników i wniosków w formie raportu.
5. -6.Konsultacje i oddanie projektów.

---

## Przykłady Projektów

1. **Klasyfikacja win (Wine Quality Dataset)** – przewidywanie jakości wina na podstawie parametrów chemicznych.
2. **Rozpoznawanie cyfr (MNIST lub Digits)** – klasyfikacja obrazów cyfr 0–9.
3. **Titanic Survival Prediction** – przewidywanie przeżycia pasażerów Titanica.
4. **Breast Cancer Dataset** – diagnoza nowotworu (łagodny/złośliwy).
5. **Spam Classification (SMS Spam Collection)** – wykrywanie spamu w wiadomościach SMS.

---

## Materiały Dodatkowe

* Dokumentacja: [scikit-learn.org](https://scikit-learn.org/)
* Tutoriale Jupyter Notebook
* Książka: Aurélien Géron – *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
* Kaggle Datasets: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)

---

## Kontakt

mgr inż. Adrian Albrecht
[aalbrecht@swps.edu.pl](mailto:aalbrecht@swps.edu.pl)