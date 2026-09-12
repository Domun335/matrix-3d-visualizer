import streamlit as st

from visualizer.compositions import render_composition_tab
from visualizer.eigen import render_eigen_tab
from visualizer.matrix_transforms import render_matrix_tab

st.set_page_config(
    layout="wide",
    page_title="Wizualizator Macierzy 3D",
    page_icon="🔲",
    initial_sidebar_state="expanded",
)

st.title("🔲 Interaktywny Wizualizator Macierzy 3D")
st.markdown(
    "Poznaj geometryczny sens **macierzy $3 \\times 3$** oraz działań na nich. "
    "Zobacz na żywo, jak macierz deformuje przestrzeń trójwymiarową, sześcian jednostkowy i siatkę współrzędnych, "
    "badaj wyznacznik jako zmianę objętości, składaj macierze przez mnożenie oraz odkrywaj ich kierunki własne."
)

with st.sidebar:
    st.header("🎮 Sterowanie Kamerą 3D")
    st.markdown(
        """
        - **Obrót sceny**: Lewy przycisk myszy + przeciągnij.
        - **Przesuwanie (Pan)**: Prawy przycisk myszy + przeciągnij (lub `Shift` + lewy).
        - **Zoom**: Rolka myszy.
        - **Reset kamery**: Podwójne kliknięcie na wykresie.
        - **Włączanie warstw**: Kliknij element w legendzie.
        """
    )
    st.markdown("---")
    st.info(
        "💡 **Klucz do macierzy 3D**:\n\n"
        "Macierz to **przepis na transformację przestrzeni**:\n"
        "- Kolumna 1: dokąd trafia oś X\n"
        "- Kolumna 2: dokąd trafia oś Y\n"
        "- Kolumna 3: dokąd trafia oś Z\n"
        "- $\\det(A)$: jak zmienia się objętość brył."
    )

tab1, tab2, tab3 = st.tabs(
    [
        "🔲 1. Wizualizacja Macierzy i Deformacja Przestrzeni",
        "🔗 2. Mnożenie Macierzy (Złożenie B · A)",
        "🧭 3. Spektrum i Osie Własne Macierzy",
    ]
)

with tab1:
    render_matrix_tab()

with tab2:
    render_composition_tab()

with tab3:
    render_eigen_tab()
