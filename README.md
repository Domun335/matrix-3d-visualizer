# Interaktywny Wizualizator Macierzy 3D

Aplikacja w Pythonie (**Streamlit** + **Plotly** + **NumPy**) dedykowana wyłącznie **geometrycznej wizualizacji macierzy $3 \times 3$** oraz działaniom na nich w przestrzeni trójwymiarowej.

---

## Możliwości Programu

### 1. Wizualizacja Macierzy i Deformacji Przestrzeni
- **Geometryczna istota macierzy**:
  - Kolumny macierzy reprezentują nowe położenia wektorów bazy $\hat{i}, \hat{j}, \hat{k}$:
    - Kolumna 1: $T(\hat{i}) = [a_{11}, a_{21}, a_{31}]^T$ (czerwona strzałka 3D)
    - Kolumna 2: $T(\hat{j}) = [a_{12}, a_{22}, a_{32}]^T$ (zielona strzałka 3D)
    - Kolumna 3: $T(\hat{k}) = [a_{13}, a_{23}, a_{33}]^T$ (niebieska strzałka 3D)
- **Płynna interpolacja transformacji ($t \in [0, 1]$)**:
  - Przekształcenie chwilowe $M(t) = (1-t)I + tA$.
  - Suwak animacji pozwala zobaczyć na żywo płynną metamorfozę przestrzeni z pozycji wyjściowej $I$ do docelowej $A$ (w stylu 3Blue1Brown).
- **Deformowane obiekty 3D**:
  - Sześcian jednostkowy $[0,1]^3 \to$ równoległościan o objętości równej $|\det(M(t))|$.
  - Trójwymiarowa siatka współrzędnych przestrzeni (pokazuje zginanie i rozciąganie siatki kartezjańskiej).
  - Sfera jednostkowa deformująca się w elipsoidę (główne osie rozciągania).
  - Baza wyjściowa $(\hat{i}, \hat{j}, \hat{k})$ jako linie odniesienia.
- **Bogata baza przekształceń**:
  - Obroty wokół osi X, Y, Z oraz **wokół dowolnej osi jednostkowej $\vec{n}$** (wzór Rodriguesa).
  - Skalowanie niejednorodne i odbicia lustrzane.
  - Ścinanie (Shear) w różnych płaszczyznach.
  - Rzuty ortogonalne na płaszczyzny (spłaszczenie do 2D, $\det = 0$) i proste (1D).
  - Macierze symetryczne i jednostkowe.
  - Dowolna macierz wprowadzana w edytorze $3 \times 3$.
- **Algebraiczne właściwości macierzy**:
  - Wyznacznik $\det(A)$ (ze skalą objętości i interpretacją orientacji/odbicia).
  - Ślad $\mathrm{Tr}(A)$, rząd $\mathrm{rank}(A)$.
  - Automatyczna klasyfikacja typu macierzy (ortogonalna, symetryczna, diagonalna, osobliwa).
  - Macierz odwrotna $A^{-1}$ (lub informacja o osobliwości).

### 2. Mnożenie Macierzy jako Złożenie Przekształceń 3D ($B \cdot A$ vs $A \cdot B$)
- Złożenie dwóch niezależnych macierzy $A$ i $B$.
- **Demonstracja nieprzemienności**: naoczny dowód, dlaczego $B \cdot A \neq A \cdot B$.
- Suwak krokowy / dwufazowy ($0 \to 1 \to 2$):
  - Faza 1: nałożenie macierzy $A$.
  - Faza 2: nałożenie macierzy $B$ na wynik przekształcenia $A$.
- Jednoczesne porównanie bryły po $B \cdot A$ (cyjan) i po $A \cdot B$ (róż).
- Własność wyznacznika: $\det(B \cdot A) = \det(B) \cdot \det(A)$.

### 3. Spektrum i Osie Własne Macierzy (Eigen-decomposition)
- Obliczanie wartości własnych $\lambda_i$ oraz wektorów własnych $\vec{v}_i$ ($A\vec{v} = \lambda\vec{v}$).
- Rysowanie **prostych niezmienniczych** w 3D — linii, wzdłuż których macierz jedynie rozciąga lub skraca przestrzeń bez obracania.
- Obsługa wartości rzeczywistych i zespolonych (rotacja podprzestrzeni).

---

## Uruchomienie Programu

```powershell
# Uruchomienie aplikacji w przeglądarce
.venv\Scripts\streamlit.exe run app.py
```
Aplikacja otworzy się pod adresem: `http://localhost:8501`.

---

## Uruchomienie Testów

```powershell
.venv\Scripts\python.exe -m unittest discover tests
```