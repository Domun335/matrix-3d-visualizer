import numpy as np
import streamlit as st

from visualizer.plotting_utils import (
    add_line_segment,
    add_sphere_mesh,
    add_vector_3d,
    create_base_figure,
    vector_to_latex,
)


def render_eigen_tab():
    st.header("🧭 Wartości i Wektory Własne (Eigenvalues & Eigenvectors)")
    st.markdown(
        "Wektor własny $\\vec{v}$ to wektor, który pod wpływem macierzy $A$ **nie zmienia swojego kierunku** – "
        "ulega jedynie przeskalowaniu o liczbę $\\lambda$ (wartość własną):"
    )
    st.latex(r"A \cdot \vec{v} = \lambda \cdot \vec{v}")

    col_mat, col_test = st.columns([1, 1])

    with col_mat:
        preset = st.selectbox(
            "Wybierz macierz do analizy:",
            [
                "Macierz symetryczna (3 rzeczywiste osie prostopadłe)",
                "Obrót 3D (1 rzeczywisty wektor własny = oś obrotu)",
                "Skalowanie anizotropowe (osie X, Y, Z)",
                "Rzut prostokątny (wartości λ = 1 oraz λ = 0)",
                "Ścinanie (pojedynczy kierunek niezmienniczy)",
                "Własna macierz 3x3",
            ],
        )

        if preset == "Macierz symetryczna (3 rzeczywiste osie prostopadłe)":
            A = np.array(
                [
                    [2.0, 0.5, 0.0],
                    [0.5, 1.5, 0.5],
                    [0.0, 0.5, 2.5],
                ]
            )
        elif preset == "Obrót 3D (1 rzeczywisty wektor własny = oś obrotu)":
            kat = np.radians(50)
            # Obrót wokół osi [1, 1, 1]
            axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
            nx, ny, nz = axis
            K = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
            A = np.eye(3) + np.sin(kat) * K + (1 - np.cos(kat)) * (K @ K)
        elif preset == "Skalowanie anizotropowe (osie X, Y, Z)":
            A = np.diag([2.0, 0.7, -1.2])
        elif preset == "Rzut prostokątny (wartości λ = 1 oraz λ = 0)":
            A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        elif preset == "Ścinanie (pojedynczy kierunek niezmienniczy)":
            A = np.array([[1.0, 1.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        else:
            r1, r2, r3 = st.columns(3)
            a11 = r1.number_input("a11", value=2.0, step=0.1, key="e11")
            a12 = r2.number_input("a12", value=0.0, step=0.1, key="e12")
            a13 = r3.number_input("a13", value=0.0, step=0.1, key="e13")

            a21 = r1.number_input("a21", value=0.0, step=0.1, key="e21")
            a22 = r2.number_input("a22", value=1.0, step=0.1, key="e22")
            a23 = r3.number_input("a23", value=1.0, step=0.1, key="e23")

            a31 = r1.number_input("a31", value=0.0, step=0.1, key="e31")
            a32 = r2.number_input("a32", value=1.0, step=0.1, key="e32")
            a33 = r3.number_input("a33", value=1.0, step=0.1, key="e33")
            A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    with col_test:
        st.subheader("Wektor testowy (niebędący wektorem własnym)")
        st.caption("Porównaj zachowanie dowolnego wektora z wektorami własnymi:")
        ux = st.slider("u_x", -2.0, 2.0, 1.0, 0.2, key="eig_ux")
        uy = st.slider("u_y", -2.0, 2.0, 1.0, 0.2, key="eig_uy")
        uz = st.slider("u_z", -2.0, 2.0, 1.0, 0.2, key="eig_uz")
        u_test = np.array([ux, uy, uz], dtype=float)
        pokaz_elips = st.checkbox(
            "Pokaż deformację sfery (osie elipsoidy)", value=False
        )

    # Obliczenie wartości i wektorów własnych
    eigenvalues, eigenvectors = np.linalg.eig(A)

    fig = create_base_figure(
        title="Wektory własne i proste niezmiennicze w 3D", limit=4.0
    )

    # Elipsoida
    if pokaz_elips:
        add_sphere_mesh(
            fig,
            transform_matrix=A,
            color="#80d8ff",
            opacity=0.18,
            name="Elipsoida transformacji",
        )

    # Kolory dla wektorów własnych
    colors_eig = ["#00e676", "#ff9100", "#d500f9"]
    real_count = 0

    st.markdown("---")
    st.subheader("Obliczone Wartości i Wektory Własne")

    info_cols = st.columns(3)

    for i in range(3):
        lam = eigenvalues[i]
        vec = eigenvectors[:, i]
        is_real = np.isreal(lam) and abs(lam.imag) < 1e-5

        with info_cols[i]:
            if is_real:
                lam_val = float(lam.real)
                vec_val = vec.real
                real_count += 1
                st.markdown(f"**Wartość $\\lambda_{i+1}$ = {lam_val:.3f}**")
                st.latex(f"\\vec{{v}}_{i+1} = " + vector_to_latex(vec_val))

                # Wizualizacja wektora własnego
                color = colors_eig[i % len(colors_eig)]
                # Prosta niezmiennicza
                line_len = 3.5
                add_line_segment(
                    fig,
                    -line_len * vec_val,
                    line_len * vec_val,
                    color=color,
                    width=2,
                    dash="dash",
                    name=f"Prosta niezmiennicza {i+1}",
                    showlegend=True,
                )

                # Wektor własny przed transformacją
                v_scaled = vec_val * 1.5
                add_vector_3d(
                    fig,
                    v_scaled,
                    color=color,
                    name=f"Wektor własny v_{i+1}",
                    width=5,
                    opacity=0.5,
                    dash="dot",
                )

                # Wektor po transformacji A @ v = lambda * v
                v_transformed = A @ v_scaled
                add_vector_3d(
                    fig,
                    v_transformed,
                    color=color,
                    name=f"A · v_{i+1} = {lam_val:.2f} · v_{i+1}",
                    width=7,
                )
            else:
                st.markdown(f"**Wartość $\\lambda_{i+1}$ (zespolona):**")
                st.write(f"`{lam.real:.3f} + {lam.imag:.3f}j`")
                st.caption(
                    "Wartość zespolona oznacza rotację w danej podprzestrzeni (brak rzeczywistej prostej niezmienniczej)."
                )

    # Rysowanie dowolnego wektora u_test dla kontrastu
    if np.linalg.norm(u_test) > 1e-4:
        add_vector_3d(
            fig,
            u_test,
            color="rgba(255, 255, 255, 0.4)",
            name="Dowolny wektor u",
            width=3,
            dash="dash",
        )
        u_trans = A @ u_test
        add_vector_3d(
            fig,
            u_trans,
            color="#ff1744",
            name="A · u (zmienia kierunek!)",
            width=6,
        )

    st.plotly_chart(fig, use_container_width=True)

    # Podsumowanie i interpretacja
    st.markdown("### Interpretacja Geometryczna")
    st.markdown(f"""
        - **Liczba rzeczywistych wektorów własnych:** {real_count} / 3.
        - Zwróć uwagę na kolorowe przerywane linie: to **proste niezmiennicze**.
          Wektor własny $\\vec{{v}}$ po przemnożeniu przez macierz $A$ **pozostaje na tej samej prostej**!
        - Zwykły wektor (czerwony $\\vec{{u}}$) **zbacza ze swojej prostej** i obraca się w przestrzeni.
        - Jeżeli $\\lambda > 1$: wektor ulega wydłużeniu.
        - Jeżeli $0 < \\lambda < 1$: wektor ulega skróceniu.
        - Jeżeli $\\lambda < 0$: wektor zmienia zwrot na przeciwny (odbija się w tył).
        - Jeżeli $\\lambda = 0$: wektor znika do zera (należy do jądra przekształcenia $\\ker(A)$).
        """)
