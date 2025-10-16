# Uczenie Maszynowe w Analizie Danych – Projekt 2025Z

## Lab 2: Analiza eksploracyjna danych (EDA). Oczyszczanie i przygotowanie danych.

### Czym jest EDA?

**Exploratory Data Analysis (EDA)** to pierwszy i najważniejszy etap w pracy z danymi.  
Polega na **poznaniu struktury danych**, **zrozumieniu relacji między zmiennymi** i **wykryciu nieprawidłowości**.

Można powiedzieć, że EDA to swoisty „wywiad z danymi”, który pozwala lepiej dobrać algorytmy ML i uniknąć błędów w dalszych etapach.

### Etapy EDA

1. **Import i wczytanie danych**  
2. **Podstawowe informacje o zbiorze**  
3. **Statystyki opisowe**  
4. **Analiza braków danych**  
5. **Wykrywanie wartości odstających**  
6. **Analiza korelacji między cechami**  
7. **Wizualizacja danych**  

### Biblioteki

Warto skorzystać z popularnych bibliotek:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
````

---

### 1. Import i podgląd danych

Zacznijmy od wczytania danych (np. z pliku CSV):

```python
df = pd.read_csv("data.csv")
df.head()
```

Sprawdźmy podstawowe informacje o zbiorze:

```python
df.info()          # typy danych i liczba niepustych wartości
df.shape           # liczba wierszy i kolumn
df.columns         # lista nazw kolumn
df.dtypes          # typ danych w każdej kolumnie
```

---

### 2. Statystyki opisowe

Pomagają zrozumieć rozkład danych i zidentyfikować potencjalne błędy:

```python
df.describe(include='all')
```

Dzięki temu możemy od razu zauważyć np. kolumny z wartościami odstającymi lub zdominowane przez jedną kategorię.

---

### 3. Wstępna wizualizacja danych

Wizualizacja to kluczowy etap EDA — pozwala zauważyć zależności, wzorce i anomalie.

#### Histogramy
```python
df['wiek'].hist(bins=20)
plt.title("Rozkład wieku")
plt.xlabel("Wiek")
plt.ylabel("Liczba osób")
plt.show()
```

#### Wykres pudełkowy (boxplot)
```python
sns.boxplot(x=df['dochód'])
plt.title("Dochód – wykres pudełkowy")
plt.show()
```

#### Korelacja między zmiennymi 
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Macierz korelacji")
plt.show()
```

### 4. Czyszczenie danych

#### Braki danych

Braki danych są bardzo częste – mogą wynikać z błędów pomiarowych, problemów technicznych lub braków w ankietach.
Najpierw sprawdźmy, gdzie występują:

```python
df.isnull().sum()
sns.heatmap(df.isnull(), cbar=False)
```

### Sposoby uzupełniania braków:

* usuwanie wierszy:

  ```python
  df = df.dropna()
  ```
* wypełnianie średnią, medianą lub dominującą wartością:

  ```python
  df['wiek'].fillna(df['wiek'].median(), inplace=True)
  ```
* interpolacja:

  ```python
  df['temperatura'] = df['temperatura'].interpolate()
  ```

#### Usuwanie duplikatów

Zdarza się niestety, że w naszych danych będą duplikaty, czyli powtarzające się dane - o ile nie mają one sensu w ciągłości danych (np. historyczny ciąg powtarzających się danych pogodowych, który ma znaczenie w prognozie) to możemy je usunąć.

```python
df.drop_duplicates(inplace=True)
```

#### Poprawa typów danych

Zdarza się również, że dane mają niepoprawny typ chociażby z racji wartości pustych czy po prostu nietypowego formatowania (jak w przypadku daty). Warto wtedy poprawić ten typ po dokonaniu czyszczenia.

```python
df['data'] = pd.to_datetime(df['data'])
df['wiek'] = df['wiek'].astype(int)
```

---

### 5. Wartości odstające (outliers)

#### Wykrywanie

Wartości odstające mogą zakłócać uczenie modeli.
Najczęściej wykrywa się je przy pomocy **boxplotów** lub **z-score**.

```python
sns.boxplot(x=df["dochód"])
```

Wykrywanie za pomocą odchylenia standardowego:

```python
mean = df['dochód'].mean()
std = df['dochód'].std()
df = df[(df['dochód'] > mean - 3*std) & (df['dochód'] < mean + 3*std)]
```

Wykrywanie statystycznie – reguła IQR:

```python
Q1 = df['dochód'].quantile(0.25)
Q3 = df['dochód'].quantile(0.75)
IQR = Q3 - Q1

dolna_granica = Q1 - 1.5 * IQR
górna_granica = Q3 + 1.5 * IQR

outliers = df[(df['dochód'] < dolna_granica) | (df['dochód'] > górna_granica)]
print(outliers)
```

Wykrywanie za pomocą Z-score

```python
from scipy import stats
z_scores = np.abs(stats.zscore(df['dochód']))
df_out = df[z_scores > 3]
```

#### Jak sobie z nimi radzić?

Można poradzić sobie w różny sposób w zależności jak dużo jest tych danych, jak bardzo są kluczowe oraz czy używamy modelu, który będzie na nie czuły.

1. Usunięcie outlierów

```python
df = df[(df['dochód'] >= dolna_granica) & (df['dochód'] <= górna_granica)]
```

2. Zastąpienie wartościami granicznymi

```python
df['dochód'] = np.where(df['dochód'] > górna_granica, górna_granica, df['dochód'])
```

3. Transformacja danych (np. logarytmicznie)

```python
df['dochód_log'] = np.log1p(df['dochód'])
```

4. Modele odporne na outliery

Niektóre modele (np. drzewa decyzyjne, RandomForest) są mniej wrażliwe na wartości odstające – czasem wystarczy je pozostawić.

---

### 6. Kodowanie zmiennych kategorycznych

Modele ML przyjmują dane **numeryczne**, więc napisy należy przekształcić.

**Label Encoding** (dla danych porządkowych):

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['płeć'] = le.fit_transform(df['płeć'])
```

**One-Hot Encoding** (dla danych nieuporządkowanych):

```python
df = pd.get_dummies(df, columns=['miasto'])
```

---

### 7. Skalowanie i standaryzacja

Skalowanie zapewnia, że cechy mają podobny wpływ na model.
Bez tego np. „dochód w złotych” może zdominować inne cechy.

**StandardScaler** – standaryzuje dane (średnia = 0, odchylenie = 1)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['wiek', 'dochód']] = scaler.fit_transform(df[['wiek', 'dochód']])
```

**MinMaxScaler** – skaluje wartości do przedziału [0,1]

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['wiek', 'dochód']] = scaler.fit_transform(df[['wiek', 'dochód']])
```

---

### 8 Korelacja cech

Korelacja pozwala zidentyfikować, które cechy są silnie powiązane (lub zduplikowane).

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

Jeśli dwie zmienne mają korelację powyżej **0.9**, warto rozważyć usunięcie jednej z nich.

---

### 9. Wizualizacja danych

EDA bez wizualizacji to jak książka bez ilustracji — technicznie poprawna, ale trudna do zrozumienia.

Przykłady:

```python
sns.histplot(df['wiek'], bins=20)
sns.countplot(x='płeć', data=df)
sns.scatterplot(x='wiek', y='dochód', hue='płeć', data=df)
```

---

### Wskazówki

* **Nie usuwaj danych bez powodu** – czasem braki mogą być informacją samą w sobie.
* **Uważaj na jednostki** – np. mieszanie wartości w złotych i tysiącach złotych to częsty błąd.
* **EDA to proces iteracyjny** – wykonuj go kilka razy w trakcie projektu.
* **Komentuj każdy etap** *(opcjonalnie, bo kod powinien bronić się sam, ale dobrze jest opisać dlaczego i po co się to robi)*– kod powinien być czytelny i logicznie opisany.


### Wymagania do zaliczenia modułu

* Notebook Jupyter z pełnym procesem EDA:

  * opisy kroków,
  * wizualizacje,
  * komentarze wyjaśniające decyzje,
  * czysty, dobrze sformatowany kod.
