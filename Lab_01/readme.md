# Uczenie Maszynowe w Analizie Danych – Projekt 2025Z

## Lab 1: Źródła danych i wybór datasetów

---

# 1. Warto zacząć od...

1.1 Przypomnij sobie podstawowe pojęcia związane z danymi w uczeniu maszynowym – krótki materiał:
👉 [https://scikit-learn.org/stable/datasets/toy_dataset.html](https://scikit-learn.org/stable/datasets/toy_dataset.html)

1.2 Warto też zajrzeć do repozytoriów otwartych zbiorów danych:

* **UCI Machine Learning Repository**: [https://archive.ics.uci.edu/ml/index.php](https://archive.ics.uci.edu/ml/index.php)
* **Kaggle Datasets**: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
* **OpenML**: [https://www.openml.org/](https://www.openml.org/)

1.3 Jeżeli chcesz, przejrzyj inne dataset dostępne w pakiecie SciKit Learn.
👉 [https://scikit-learn.org/stable/datasets/real_world.html](https://scikit-learn.org/stable/datasets/real_world.html)

---

# 2. Teoria – Wybór i Źródła Danych

Uczenie maszynowe zaczyna się od **danych**. To one decydują o tym, jak trudne będzie zadanie, jakie algorytmy się sprawdzą i czy projekt ma sens.

### Kluczowe kryteria wyboru datasetu:

* **Cel projektu** – chcemy klasyfikacji, więc dane muszą mieć **etykiety (klasy)**.
* **Rozmiar danych** – minimum kilkaset obserwacji; optymalnie kilka tysięcy.
* **Balans klas** – czy klasy są w miarę równoliczne.
* **Rodzaj cech** – numeryczne, kategoryczne, mieszane (wpływa na preprocessing).
* **Format pliku** – CSV, JSON, Excel, bazy SQL (CSV najprostszy).
* **Jakość danych** – brak duplikatów, poprawność wartości.

---

### Popularne źródła datasetów:

#### a) Wbudowane zbiory w **scikit-learn**

* **Iris** – klasyfikacja gatunków kwiatów (3 klasy).
* **Wine** – ocena win na podstawie cech chemicznych (klasyfikacja wieloklasowa).
* **Digits** – rozpoznawanie cyfr 0–9.
* **Breast Cancer** – diagnoza nowotworu (binarna).

➡️ Zastosowanie: szybki start, nauka podstaw, testowanie modeli.

---

#### b) **Kaggle Datasets** ([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets))

* **Titanic** – przewidywanie przeżycia pasażerów (binary classification).
* **SMS Spam Collection** – klasyfikacja wiadomości jako spam/ham.
* **Heart Disease Dataset** – przewidywanie obecności choroby serca.

➡️ Zastosowanie: dane bardziej realistyczne, możliwość porównań z innymi projektami.

---

#### c) **UCI Machine Learning Repository**

* **Adult (Census Income)** – przewidywanie czy osoba zarabia >50k$/rok.
* **Car Evaluation** – klasyfikacja jakości samochodów (4 klasy).
* **Bank Marketing** – przewidywanie reakcji klienta na kampanię bankową.

➡️ Zastosowanie: projekty średniozaawansowane, często wymagają preprocessing’u.

---

#### d) Inne źródła

* **OpenML** – dane wprost do scikit-learn.
* **Google Dataset Search** – wyszukiwarka datasetów.
* **Dane publiczne (gov.pl, Eurostat, WHO, World Bank)** – świetne do projektów społecznych/ekonomicznych.

---

# 3. Przykłady dopasowania datasetów do zadań

1. **Klasyfikacja binarna**

   * Breast Cancer (złośliwy vs łagodny).
   * Titanic (przeżył vs nie przeżył).
   * Spam Detection (spam vs normal).

2. **Klasyfikacja wieloklasowa**

   * Wine (3 klasy jakości wina).
   * Digits (10 klas cyfr).
   * Car Evaluation (4 klasy jakości auta).

3. **Dane z niezbalansowanymi klasami**

   * Credit Card Fraud (oszustwa vs poprawne transakcje).
   * Rzadkie choroby w medycynie.

---

# 4. Przygotowanie do kolejnych zajęć

Do następnych zajęć każda osoba studencka:

1. **Wybiera dataset** (z podanych źródeł lub inny, uzgodniony z prowadzącym).
2. Przygotowuje **krótką notatkę** (w pliku .ipynb lub .md):

   * źródło danych (link),
   * liczba próbek, cech i klas,
   * charakterystyka cech,
   * wstępne wnioski (np. czy klasy są zrównoważone, jaki preprocessing będzie potrzebny).