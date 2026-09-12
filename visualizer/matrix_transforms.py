import numpy as np
import streamlit as st

from visualizer.plotting_utils import (
    add_cube_mesh,
    add_grid_3d,
    add_line_segment,
    add_sphere_mesh,
    add_vector_3d,
    create_base_figure,
    matrix_to_latex,
)


def rodrigues_rotation_matrix(axis, theta_rad):
    """Zwraca macierz obrotu 3x3 wokół dowolnej osi jednostkowej o kąt theta."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.eye(3)
    n = axis / norm
    nx, ny, nz = n

    K = np.array(
        [
            [0.0, -nz, ny],
            [nz, 0.0, -nx],
            [-ny, nx, 0.0],
        ]
    )
    I = np.eye(3)
    return I + np.sin(theta_rad) * K + (1.0 - np.cos(theta_rad)) * (K @ K)


def identify_matrix_type(A):
    """Rozpoznaje specjalne typy macierzy i ich właściwości geometryczne."""
    det = np.linalg.det(A)
    is_orthogonal = np.allclose(A @ A.T, np.eye(3), atol=1e-4)
    is_symmetric = np.allclose(A, A.T, atol=1e-4)
    is_diagonal = np.allclose(A, np.diag(np.diagonal(A)), atol=1e-4)
    is_singular = abs(det) < 1e-5

    types = []
    if is_orthogonal:
        if np.isclose(det, 1.0, atol=1e-3):
            types.append("Czysty obrót 3D (Izometria właściwa, det=1)")
        elif np.isclose(det, -1.0, atol=1e-3):
            types.append("Odbicie lustrzane / Obrotowe (det=-1)")
        else:
            types.append("Ortogonalna (zachowuje kąty i długości)")
    if is_symmetric:
        types.append("Symetryczna (posiada wzajemnie prostopadłe osie własne)")
    if is_diagonal:
        types.append("Diagonalna (skalowanie wzdłuż osi układu)")
    if is_singular:
        rank = np.linalg.matrix_rank(A)
        types.append(f"Osobliwa / Rzut (det=0, spłaszczenie do wymiaru {rank}D)")

    return types if types else ["Ogólna macierz przekształcenia liniowego"]


def render_matrix_tab():
    st.header("🔲 Wizualizator Macierzy 3D (Przekształcenia Przestrzeni)")
    st.markdown(
        "Każda macierz $3 \\times 3$ to **funkcja geometryczna**, która przekształca całą przestrzeń trójwymiarową. "
        "Jej kolumny to nowe położenia wersorów osi $\\hat{i}, \\hat{j}, \\hat{k}$, a wyznacznik to współczynnik zmiany objętości."
    )

    # Dwie kolumny: 1. Wybór macierzy, 2. Warstwy wizualne
    col_matrix_def, col_vis_options = st.columns([1, 1])

    with col_matrix_def:
        st.subheader("1. Definicja Macierzy A")
        preset = st.selectbox(
            "Wybierz gotową macierz lub stwórz własną:",
            [
                "Obrót wokół osi Z",
                "Obrót wokół osi X",
                "Obrót wokół osi Y",
                "Obrót wokół dowolnej osi n (Rodrigues)",
                "Skalowanie niejednorodne",
                "Ścinanie (Shear)",
                "Rzut prostokątny na płaszczyznę (spłaszczenie do 2D, det=0)",
                "Rzut na prostą (spłaszczenie do 1D, rank=1)",
                "Odbicie lustrzane (symetria, det=-1)",
                "Macierz symetryczna (prostopadłe osie)",
                "Macierz jednostkowa I (brak deformacji)",
                "Własna macierz 3x3",
            ],
        )

        if preset == "Obrót wokół osi Z":
            kat_deg = st.slider("Kąt obrotu θ (°)", -180, 180, 45, 5, key="mat_rot_z")
            kat = np.radians(kat_deg)
            A = np.array(
                [
                    [np.cos(kat), -np.sin(kat), 0.0],
                    [np.sin(kat), np.cos(kat), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        elif preset == "Obrót wokół osi X":
            kat_deg = st.slider("Kąt obrotu θ (°)", -180, 180, 45, 5, key="mat_rot_x")
            kat = np.radians(kat_deg)
            A = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(kat), -np.sin(kat)],
                    [0.0, np.sin(kat), np.cos(kat)],
                ]
            )
        elif preset == "Obrót wokół osi Y":
            kat_deg = st.slider("Kąt obrotu θ (°)", -180, 180, 45, 5, key="mat_rot_y")
            kat = np.radians(kat_deg)
            A = np.array(
                [
                    [np.cos(kat), 0.0, np.sin(kat)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(kat), 0.0, np.cos(kat)],
                ]
            )
        elif preset == "Obrót wokół dowolnej osi n (Rodrigues)":
            kat_deg = st.slider("Kąt obrotu θ (°)", -180, 180, 60, 5, key="mat_rot_n")
            kat = np.radians(kat_deg)
            st.caption("Kierunek osi obrotu:")
            cx, cy, cz = st.columns(3)
            nx = cx.number_input("n_x", value=1.0, step=0.2)
            ny = cy.number_input("n_y", value=1.0, step=0.2)
            nz = cz.number_input("n_z", value=1.0, step=0.2)
            A = rodrigues_rotation_matrix([nx, ny, nz], kat)
        elif preset == "Skalowanie niejednorodne":
            cx, cy, cz = st.columns(3)
            sx = cx.slider("Skala X", -3.0, 3.0, 1.5, 0.1)
            sy = cy.slider("Skala Y", -3.0, 3.0, 0.7, 0.1)
            sz = cz.slider("Skala Z", -3.0, 3.0, 1.2, 0.1)
            A = np.diag([sx, sy, sz])
        elif preset == "Ścinanie (Shear)":
            plaszczyzna = st.selectbox(
                "Płaszczyzna ścinania",
                [
                    "Oś X zależna od Y",
                    "Oś X zależna od Z",
                    "Oś Y zależna od Z",
                    "Oś Z zależna od X",
                ],
            )
            sh = st.slider("Współczynnik ścinania", -2.0, 2.0, 1.0, 0.1)
            A = np.eye(3)
            if plaszczyzna == "Oś X zależna od Y":
                A[0, 1] = sh
            elif plaszczyzna == "Oś X zależna od Z":
                A[0, 2] = sh
            elif plaszczyzna == "Oś Y zależna od Z":
                A[1, 2] = sh
            else:
                A[2, 0] = sh
        elif preset == "Rzut prostokątny na płaszczyznę (spłaszczenie do 2D, det=0)":
            rzut = st.radio(
                "Płaszczyzna rzutu:",
                [
                    "Płaszczyzna XY (zeruje Z)",
                    "Płaszczyzna XZ (zeruje Y)",
                    "Płaszczyzna YZ (zeruje X)",
                ],
            )
            A = np.eye(3)
            if "XY" in rzut:
                A[2, 2] = 0.0
            elif "XZ" in rzut:
                A[1, 1] = 0.0
            else:
                A[0, 0] = 0.0
        elif preset == "Rzut na prostą (spłaszczenie do 1D, rank=1)":
            prosta = st.radio(
                "Kierunek prostej rzutu:",
                ["Oś X (zeruje Y i Z)", "Oś Y (zeruje X i Z)", "Oś Z (zeruje X i Y)"],
            )
            A = np.zeros((3, 3))
            if "X" in prosta:
                A[0, 0] = 1.0
            elif "Y" in prosta:
                A[1, 1] = 1.0
            else:
                A[2, 2] = 1.0
        elif preset == "Odbicie lustrzane (symetria, det=-1)":
            odbicie = st.radio(
                "Odbicie względem:",
                [
                    "Płaszczyzny XY (odwraca oś Z)",
                    "Płaszczyzny XZ (odwraca oś Y)",
                    "Płaszczyzny YZ (odwraca oś X)",
                ],
            )
            A = np.eye(3)
            if "XY" in odbicie:
                A[2, 2] = -1.0
            elif "XZ" in odbicie:
                A[1, 1] = -1.0
            else:
                A[0, 0] = -1.0
        elif preset == "Macierz symetryczna (prostopadłe osie)":
            A = np.array(
                [
                    [2.0, 0.5, 0.0],
                    [0.5, 1.5, 0.5],
                    [0.0, 0.5, 2.5],
                ]
            )
        elif preset == "Macierz jednostkowa I (brak deformacji)":
            A = np.eye(3)
        else:
            st.caption("Wprowadź wartości macierzy 3x3:")
            r1, r2, r3 = st.columns(3)
            a11 = r1.number_input("a11", value=1.0, step=0.1, key="cust_a11")
            a12 = r2.number_input("a12", value=0.0, step=0.1, key="cust_a12")
            a13 = r3.number_input("a13", value=0.0, step=0.1, key="cust_a13")

            a21 = r1.number_input("a21", value=0.0, step=0.1, key="cust_a21")
            a22 = r2.number_input("a22", value=1.0, step=0.1, key="cust_a22")
            a23 = r3.number_input("a23", value=0.0, step=0.1, key="cust_a23")

            a31 = r1.number_input("a31", value=0.0, step=0.1, key="cust_a31")
            a32 = r2.number_input("a32", value=0.0, step=0.1, key="cust_a32")
            a33 = r3.number_input("a33", value=1.0, step=0.1, key="cust_a33")
            A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    with col_vis_options:
        st.subheader("2. Warstwy Wizualizacji 3D")
        c1, c2 = st.columns(2)
        pokaz_baze = c1.checkbox(
            "Kolumny A (Nowe wektory bazy T(i), T(j), T(k))", value=True
        )
        pokaz_stara_baze = c2.checkbox("Baza wyjściowa (i, j, k)", value=True)
        pokaz_szescian = c1.checkbox(
            "Sześcian jednostkowy → Równoległościan", value=True
        )
        pokaz_siatke = c2.checkbox("Deformacja siatki 3D (przestrzeń)", value=True)
        pokaz_elipsoide = c1.checkbox("Sfera → Elipsoida rozciągania", value=False)
        pokaz_eigen = c2.checkbox("Kierunki własne (proste niezmiennicze)", value=False)

    # 3. PŁYNNA ANIMACJA TRANSFOMACJI (M(t) = (1-t)I + tA)
    st.markdown("---")
    c_slider, c_btn1, c_btn2 = st.columns([3, 1, 1])
    with c_slider:
        t = st.slider(
            "Płynna interpolacja przekształcenia: M(t) = (1-t)·I + t·A",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
        )
    with c_btn1:
        st.write("")
        st.write("")
        st.caption(f"Stan: **{int(t*100)}%**")
    with c_btn2:
        st.write("")
        st.write("")
        st.caption("Przesuwaj suwak, by zobaczyć ruch przestrzeni!")

    # Macierz chwilowa interpolacji
    I_mat = np.eye(3)
    M_t = (1.0 - t) * I_mat + t * A

    # Właściwości algebraiczne
    det_A = float(np.linalg.det(A))
    det_Mt = float(np.linalg.det(M_t))
    trace_A = float(np.trace(A))
    rank_A = int(np.linalg.matrix_rank(A))
    matrix_types = identify_matrix_type(A)

    # Wyświetlanie metryk
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wyznacznik det(A)", f"{det_A:.3f}", f"det(M(t)) = {det_Mt:.3f}")
    m2.metric(
        "Skala objętości |det(A)|",
        f"{abs(det_A):.3f}x",
        (
            "Orientacja odwrócona"
            if det_A < 0
            else "Orientacja zachowana" if det_A > 0 else "Utrata wymiaru!"
        ),
    )
    m3.metric("Ślad Tr(A)", f"{trace_A:.2f}")
    m4.metric(
        "Rząd rank(A)",
        f"{rank_A} / 3",
        "Pełny wymiar" if rank_A == 3 else f"Spłaszczenie do {rank_A}D",
    )

    # Tworzenie sceny Plotly 3D
    limit_axis = max(3.5, float(np.max(np.abs(M_t))) * 1.5 + 0.5)
    fig = create_base_figure(
        title=f"Wizualizacja Macierzy: {preset}", limit=min(limit_axis, 7.0)
    )

    # Siatka przestrzenna 3D
    if pokaz_siatke:
        add_grid_3d(fig, transform_matrix=M_t, limit=2.0, step=1.0)

    # Sześcian jednostkowy -> Równoległościan
    if pokaz_szescian:
        add_cube_mesh(
            fig,
            transform_matrix=M_t,
            color="#00e5ff",
            opacity=0.22,
            edge_color="#00bcd4",
            name="Sześcian M(t)",
        )

    # Sfera -> Elipsoida
    if pokaz_elipsoide:
        add_sphere_mesh(
            fig,
            transform_matrix=M_t,
            color="#e040fb",
            opacity=0.2,
            name="Elipsoida deformacji",
        )

    # Oryginalna baza jednostkowa
    if pokaz_stara_baze:
        add_vector_3d(
            fig,
            [1, 0, 0],
            color="rgba(255, 100, 100, 0.4)",
            name="i (baza początkowa)",
            width=3,
            dash="dot",
        )
        add_vector_3d(
            fig,
            [0, 1, 0],
            color="rgba(100, 255, 100, 0.4)",
            name="j (baza początkowa)",
            width=3,
            dash="dot",
        )
        add_vector_3d(
            fig,
            [0, 0, 1],
            color="rgba(100, 150, 255, 0.4)",
            name="k (baza początkowa)",
            width=3,
            dash="dot",
        )

    # Kolumny macierzy M(t) jako wektory bazy
    if pokaz_baze:
        col1 = M_t[:, 0]
        col2 = M_t[:, 1]
        col3 = M_t[:, 2]
        add_vector_3d(fig, col1, color="#ff1744", name="Kolumna 1: T(i)", width=7)
        add_vector_3d(fig, col2, color="#00e676", name="Kolumna 2: T(j)", width=7)
        add_vector_3d(fig, col3, color="#2979ff", name="Kolumna 3: T(k)", width=7)

    # Wektory i proste własne
    if pokaz_eigen:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        colors_eigen = ["#ffd600", "#ff6d00", "#d500f9"]
        for i in range(3):
            lam = eigenvalues[i]
            if np.isreal(lam) and abs(lam.imag) < 1e-5:
                v_eig = eigenvectors[:, i].real
                lam_real = float(lam.real)
                col_e = colors_eigen[i % len(colors_eigen)]
                # Linia prostej własnej
                line_len = 3.5
                add_line_segment(
                    fig,
                    -line_len * v_eig,
                    line_len * v_eig,
                    color=col_e,
                    width=2,
                    dash="dash",
                    name=f"Oś własna λ={lam_real:.2f}",
                    showlegend=True,
                )
                # Wektor własny przekształcony M(t)
                v_t = M_t @ (v_eig * 1.5)
                add_vector_3d(
                    fig,
                    v_t,
                    color=col_e,
                    name=f"Wektor własny M(t)·v_{i+1}",
                    width=5,
                    dash="solid",
                )

    st.plotly_chart(fig, use_container_width=True)

    # Panel matematyczny macierzy
    st.markdown("### Właściwości i Reprezentacja Macierzy")
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("**Postać macierzy $A$:**")
        st.latex(r"A = " + matrix_to_latex(A))
        st.caption(f"**Typ macierzy:** {', '.join(matrix_types)}")

        if abs(det_A) > 1e-5:
            A_inv = np.linalg.inv(A)
            st.markdown("**Macierz odwrotna $A^{-1}$ (cofa to przekształcenie):**")
            st.latex(r"A^{-1} = " + matrix_to_latex(A_inv))
        else:
            st.warning(
                "⚠️ **Macierz osobliwa (det(A) = 0)**: Przestrzeń uległa spłaszczeniu, brak macierzy odwrotnej $A^{-1}$!"
            )

    with col_l2:
        st.markdown("**Wartości i Wektory Własne ($A\\vec{v} = \\lambda\\vec{v}$):**")
        eigenvalues, eigenvectors = np.linalg.eig(A)
        for i in range(3):
            lam = eigenvalues[i]
            if np.isreal(lam) and abs(lam.imag) < 1e-5:
                v_vec = eigenvectors[:, i].real
                st.markdown(
                    f"- $\\lambda_{i+1} = {lam.real:.3f}$ dla $\\vec{{v}}_{i+1} = [{v_vec[0]:.2f}, {v_vec[1]:.2f}, {v_vec[2]:.2f}]^T$"
                )
            else:
                st.markdown(
                    f"- $\\lambda_{i+1} = {lam.real:.2f} \\pm {abs(lam.imag):.2f}i$ *(część zespolona odpowiada rotacji w płaszczyźnie)*"
                )

        st.markdown("---")
        st.markdown(f"""
            **Geometryczna interpretacja kolumn:**
            - Kolumna 1: $\\vec{{a}}_1 = [{A[0,0]:.2f}, {A[1,0]:.2f}, {A[2,0]:.2f}]^T$ to nowy wektor $\\hat{{i}}$ (czerwony)
            - Kolumna 2: $\\vec{{a}}_2 = [{A[0,1]:.2f}, {A[1,1]:.2f}, {A[2,1]:.2f}]^T$ to nowy wektor $\\hat{{j}}$ (zielony)
            - Kolumna 3: $\\vec{{a}}_3 = [{A[0,2]:.2f}, {A[1,2]:.2f}, {A[2,2]:.2f}]^T$ to nowy wektor $\\hat{{k}}$ (niebieski)
            """)
