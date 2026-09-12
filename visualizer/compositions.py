import numpy as np
import streamlit as st

from visualizer.plotting_utils import (
    add_cube_mesh,
    add_grid_3d,
    add_vector_3d,
    create_base_figure,
    matrix_to_latex,
)


def render_composition_tab():
    st.header("🔗 Mnożenie Macierzy jako Złożenie Przekształceń 3D")
    st.markdown(
        "Mnożenie dwóch macierzy $B \\cdot A$ odpowiada **złożeniu dwóch przekształceń geometrycznych**: "
        "najpierw przestrzeń przekształca macierz $A$, a następnie na otrzymany wynik nakładana jest macierz $B$. "
        "Mnożenie macierzy z reguły **nie jest przemienne** ($B \\cdot A \\neq A \\cdot B$)!"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Macierz A (Pierwsze Przekształcenie)")
        op_a = st.selectbox(
            "Wybierz operację A:",
            [
                "Obrót wokół osi Z (90°)",
                "Obrót wokół osi X (45°)",
                "Skalowanie osi X (2x) i Y (0.5x)",
                "Ścinanie (Shear w osi X)",
                "Odbicie lustrzane płaszczyzny XY",
            ],
            key="comp_op_a",
        )
        if "Z (90°)" in op_a:
            rad = np.radians(90)
            A = np.array(
                [
                    [np.cos(rad), -np.sin(rad), 0],
                    [np.sin(rad), np.cos(rad), 0],
                    [0, 0, 1],
                ]
            )
        elif "X (45°)" in op_a:
            rad = np.radians(45)
            A = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(rad), -np.sin(rad)],
                    [0, np.sin(rad), np.cos(rad)],
                ]
            )
        elif "Skalowanie" in op_a:
            A = np.diag([2.0, 0.5, 1.0])
        elif "Ścinanie" in op_a:
            A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        else:
            A = np.diag([1.0, 1.0, -1.0])

        st.latex(r"A = " + matrix_to_latex(A))
        st.caption(f"det(A) = {np.linalg.det(A):.2f}")

    with col_b:
        st.subheader("Macierz B (Drugie Przekształcenie)")
        op_b = st.selectbox(
            "Wybierz operację B:",
            [
                "Obrót wokół osi X (90°)",
                "Obrót wokół osi Y (60°)",
                "Skalowanie osi Z (2x)",
                "Ścinanie (Shear w osi Y)",
                "Rzut prostokątny na płaszczyznę XY",
            ],
            key="comp_op_b",
        )
        if "X (90°)" in op_b:
            rad = np.radians(90)
            B = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(rad), -np.sin(rad)],
                    [0, np.sin(rad), np.cos(rad)],
                ]
            )
        elif "Y (60°)" in op_b:
            rad = np.radians(60)
            B = np.array(
                [
                    [np.cos(rad), 0, np.sin(rad)],
                    [0, 1, 0],
                    [-np.sin(rad), 0, np.cos(rad)],
                ]
            )
        elif "Skalowanie" in op_b:
            B = np.diag([1.0, 1.0, 2.0])
        elif "Ścinanie" in op_b:
            B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        else:
            B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        st.latex(r"B = " + matrix_to_latex(B))
        st.caption(f"det(B) = {np.linalg.det(B):.2f}")

    # Obliczenie iloczynów macierzy
    BA = B @ A
    AB = A @ B

    st.markdown("---")
    st.subheader("Wizualizacja Kolejności Mnożenia Macierzy")

    tryb = st.radio(
        "Wybierz widok do analizy:",
        [
            "1. Kolejność B · A (najpierw A, potem B)",
            "2. Kolejność A · B (najpierw B, potem A)",
            "3. Bezpośrednie porównanie brył (B · A vs A · B)",
        ],
        horizontal=True,
    )

    c_opts1, c_opts2 = st.columns(2)
    pokaz_siatke = c_opts1.checkbox(
        "Pokaż siatkę 3D przestrzeni", value=True, key="comp_grid"
    )
    pokaz_kolumny = c_opts2.checkbox(
        "Pokaż wektory kolumn macierzy", value=True, key="comp_cols"
    )

    if tryb.startswith("1."):
        progress = st.slider(
            "Krok animacji: 0 = Stan wyjściowy (I) ➔ 1 = Po macierzy A ➔ 2 = Po macierzy B·A",
            min_value=0.0,
            max_value=2.0,
            value=2.0,
            step=0.05,
        )

        I_mat = np.eye(3)
        if progress <= 1.0:
            M_curr = (1.0 - progress) * I_mat + progress * A
            krok_opis = f"Krok 1: Aplikowanie macierzy A ({int(progress * 100)}%)"
        else:
            s = progress - 1.0
            M_curr = (1.0 - s) * A + s * BA
            krok_opis = f"Krok 2: Aplikowanie macierzy B na wynik A ({int(s * 100)}%)"

        st.caption(f"**Aktualny stan transformacji:** {krok_opis}")

        fig = create_base_figure(
            title="Złożenie: Najpierw A, potem B  ➔  M = B · A", limit=4.5
        )

        if pokaz_siatke:
            add_grid_3d(fig, transform_matrix=M_curr, limit=2.0)

        add_cube_mesh(
            fig,
            transform_matrix=M_curr,
            color="#00e5ff",
            opacity=0.25,
            name="Sześcian M(t)",
        )

        if pokaz_kolumny:
            add_vector_3d(
                fig, M_curr[:, 0], color="#ff1744", name="Kolumna 1: M(t)[:,0]", width=6
            )
            add_vector_3d(
                fig, M_curr[:, 1], color="#00e676", name="Kolumna 2: M(t)[:,1]", width=6
            )
            add_vector_3d(
                fig, M_curr[:, 2], color="#2979ff", name="Kolumna 3: M(t)[:,2]", width=6
            )

        st.plotly_chart(fig, use_container_width=True)

    elif tryb.startswith("2."):
        progress = st.slider(
            "Krok animacji: 0 = Stan wyjściowy (I) ➔ 1 = Po macierzy B ➔ 2 = Po macierzy A·B",
            min_value=0.0,
            max_value=2.0,
            value=2.0,
            step=0.05,
        )

        I_mat = np.eye(3)
        if progress <= 1.0:
            M_curr = (1.0 - progress) * I_mat + progress * B
            krok_opis = f"Krok 1: Aplikowanie macierzy B ({int(progress * 100)}%)"
        else:
            s = progress - 1.0
            M_curr = (1.0 - s) * B + s * AB
            krok_opis = f"Krok 2: Aplikowanie macierzy A na wynik B ({int(s * 100)}%)"

        st.caption(f"**Aktualny stan transformacji:** {krok_opis}")

        fig = create_base_figure(
            title="Złożenie: Najpierw B, potem A  ➔  M = A · B", limit=4.5
        )

        if pokaz_siatke:
            add_grid_3d(fig, transform_matrix=M_curr, limit=2.0)

        add_cube_mesh(
            fig,
            transform_matrix=M_curr,
            color="#ff4081",
            opacity=0.25,
            name="Sześcian M(t)",
        )

        if pokaz_kolumny:
            add_vector_3d(
                fig, M_curr[:, 0], color="#ff1744", name="Kolumna 1: M(t)[:,0]", width=6
            )
            add_vector_3d(
                fig, M_curr[:, 1], color="#00e676", name="Kolumna 2: M(t)[:,1]", width=6
            )
            add_vector_3d(
                fig, M_curr[:, 2], color="#2979ff", name="Kolumna 3: M(t)[:,2]", width=6
            )

        st.plotly_chart(fig, use_container_width=True)

    else:
        # Bezpośrednie porównanie
        fig = create_base_figure(
            title="Porównanie końcowe: Sześcian B · A (cyjan) vs A · B (róż)", limit=4.5
        )
        add_cube_mesh(
            fig,
            transform_matrix=BA,
            color="#00e5ff",
            opacity=0.22,
            name="Sześcian (B · A)",
        )
        add_cube_mesh(
            fig,
            transform_matrix=AB,
            color="#ff4081",
            opacity=0.22,
            name="Sześcian (A · B)",
        )

        if pokaz_kolumny:
            add_vector_3d(fig, BA[:, 0], color="#00e5ff", name="B·A Kolumna 1", width=5)
            add_vector_3d(fig, AB[:, 0], color="#ff4081", name="A·B Kolumna 1", width=5)

        st.plotly_chart(fig, use_container_width=True)

    # Matematyczna analiza iloczynów
    st.markdown("### Analiza Algebraiczna Iloczynów Macierzy")
    c1, c2 = st.columns(2)
    det_BA = float(np.linalg.det(BA))
    det_AB = float(np.linalg.det(AB))

    with c1:
        st.markdown("**Iloczyn $B \\cdot A$ (najpierw A, potem B):**")
        st.latex(r"B \cdot A = " + matrix_to_latex(BA))
        st.write(
            f"- $\\det(B \\cdot A) = {det_BA:.3f}$ (równy $\\det(B) \\cdot \\det(A) = {np.linalg.det(B)*np.linalg.det(A):.3f}$)"
        )
        st.write(f"- Ślad $\\operatorname{{Tr}}(B \\cdot A) = {np.trace(BA):.2f}$")

    with c2:
        st.markdown("**Iloczyn $A \\cdot B$ (najpierw B, potem A):**")
        st.latex(r"A \cdot B = " + matrix_to_latex(AB))
        st.write(
            f"- $\\det(A \\cdot B) = {det_AB:.3f}$ (równy $\\det(A) \\cdot \\det(B) = {np.linalg.det(A)*np.linalg.det(B):.3f}$)"
        )
        st.write(f"- Ślad $\\operatorname{{Tr}}(A \\cdot B) = {np.trace(AB):.2f}$")

    are_commute = np.allclose(BA, AB)
    if are_commute:
        st.success(
            "**W tym przypadku macierze są przemienne**: $B \\cdot A = A \\cdot B$."
        )
    else:
        st.warning(
            "**Mnożenie macierzy NIE jest przemienne!** $B \\cdot A \\neq A \\cdot B$. "
            "Kolejność wykonywania operacji w przestrzeni 3D ma kluczowe znaczenie: "
            "np. obrót a potem ścinanie daje zupełnie inny wynik niż ścinanie a potem obrót!"
        )
