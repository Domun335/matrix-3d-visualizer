import numpy as np
import plotly.graph_objects as go


def create_base_figure(title="", limit=3.5, dark_mode=True):
    """Tworzy bazowy wykres 3D Plotly ze stałą proporcją osi i estetycznym motywem."""
    fig = go.Figure()

    bg_color = "#0e1117" if dark_mode else "#ffffff"
    grid_color = "rgba(255, 255, 255, 0.12)" if dark_mode else "rgba(0, 0, 0, 0.12)"
    axis_font_color = "#e0e0e0" if dark_mode else "#222222"

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=axis_font_color),
            x=0.05,
            y=0.95,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        scene=dict(
            xaxis=dict(
                range=[-limit, limit],
                title="X",
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                zerolinecolor="rgba(255, 100, 100, 0.7)",
                showbackground=True,
                color=axis_font_color,
                zerolinewidth=2,
            ),
            yaxis=dict(
                range=[-limit, limit],
                title="Y",
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                zerolinecolor="rgba(100, 255, 100, 0.7)",
                showbackground=True,
                color=axis_font_color,
                zerolinewidth=2,
            ),
            zaxis=dict(
                range=[-limit, limit],
                title="Z",
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                zerolinecolor="rgba(100, 150, 255, 0.7)",
                showbackground=True,
                color=axis_font_color,
                zerolinewidth=2,
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.3),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            font=dict(color=axis_font_color, size=12),
            bgcolor=(
                "rgba(20, 25, 35, 0.8)" if dark_mode else "rgba(240, 240, 240, 0.8)"
            ),
            bordercolor="rgba(150, 150, 150, 0.3)",
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
        height=680,
        margin=dict(r=10, l=10, b=10, t=40),
    )
    return fig


def add_vector_3d(
    fig,
    vec,
    start=(0.0, 0.0, 0.0),
    color="royalblue",
    name="v",
    width=6,
    dash=None,
    show_cone=True,
    cone_size=0.3,
    opacity=1.0,
    legendgroup=None,
    showlegend=True,
):
    """Rysuje wektor w przestrzeni 3D z opcjonalnym grotem (stożkiem)."""
    vec = np.asarray(vec, dtype=float)
    start = np.asarray(start, dtype=float)
    end = start + vec
    length = float(np.linalg.norm(vec))

    grp = legendgroup if legendgroup is not None else name

    hover_text = (
        f"<b>{name}</b><br>"
        f"Koniec: ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})<br>"
        f"Długość: {length:.2f}"
    )

    # Linia wektora
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            name=name,
            legendgroup=grp,
            showlegend=showlegend,
            hoverinfo="text",
            hovertext=hover_text,
        )
    )

    # Grot (stożek) na końcu wektora
    if show_cone and length > 1e-4:
        cone_len = min(cone_size, length * 0.35)
        fig.add_trace(
            go.Cone(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                u=[vec[0]],
                v=[vec[1]],
                w=[vec[2]],
                anchor="tip",
                sizemode="absolute",
                sizeref=cone_len,
                showscale=False,
                colorscale=[[0, color], [1, color]],
                opacity=opacity,
                legendgroup=grp,
                showlegend=False,
                hoverinfo="skip",
            )
        )


def add_line_segment(
    fig,
    p1,
    p2,
    color="gray",
    width=2,
    dash="dash",
    name=None,
    showlegend=False,
    legendgroup=None,
):
    """Rysuje odcinek pomocniczy między punktami p1 i p2."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name if name else "pomocnicza",
            showlegend=showlegend,
            legendgroup=legendgroup,
            hoverinfo="none",
        )
    )


def add_point_3d(fig, point, color="gold", name=None, size=5, showlegend=False):
    """Rysuje pojedynczy punkt na scenie 3D."""
    p = np.asarray(point, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers",
            marker=dict(size=size, color=color),
            name=name if name else "punkt",
            showlegend=showlegend,
            hoverinfo="text",
            hovertext=f"{name or 'Punkt'}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
        )
    )


def add_cube_mesh(
    fig,
    transform_matrix=None,
    color="#00e5ff",
    opacity=0.2,
    show_edges=True,
    edge_color="#00bcd4",
    name="Sześcian",
    showlegend=True,
):
    """Rysuje jednostkowy sześcian [0,1]^3 (lub przekształcony przez macierz 3x3)."""
    # 8 wierzchołków sześcianu jednostkowego
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    if transform_matrix is not None:
        M = np.asarray(transform_matrix, dtype=float)
        transformed = (M @ corners.T).T
    else:
        transformed = corners

    # Indeksy trójkątów dla ścian
    i_idx = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j_idx = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k_idx = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    fig.add_trace(
        go.Mesh3d(
            x=transformed[:, 0],
            y=transformed[:, 1],
            z=transformed[:, 2],
            i=i_idx,
            j=j_idx,
            k=k_idx,
            opacity=opacity,
            color=color,
            name=name,
            showlegend=showlegend,
            legendgroup=name,
            hoverinfo="skip",
        )
    )

    # Rysowanie krawędzi sześcianu dla lepszej czytelności bryły
    if show_edges:
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # dół
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # góra
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # piony
        ]
        edge_x = []
        edge_y = []
        edge_z = []
        for u, v in edges:
            edge_x.extend([transformed[u, 0], transformed[v, 0], None])
            edge_y.extend([transformed[u, 1], transformed[v, 1], None])
            edge_z.extend([transformed[u, 2], transformed[v, 2], None])

        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=edge_color, width=3),
                name=f"{name} (krawędzie)",
                showlegend=False,
                legendgroup=name,
                hoverinfo="skip",
            )
        )


def add_sphere_mesh(
    fig,
    transform_matrix=None,
    color="#e040fb",
    opacity=0.15,
    name="Sfera / Elipsoida",
    showlegend=True,
    resolution=24,
):
    """Rysuje jednostkową sferę (lub elipsoidę po przekształceniu macierzą)."""
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution // 2)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    pts = np.vstack([x.flatten(), y.flatten(), z.flatten()])

    if transform_matrix is not None:
        M = np.asarray(transform_matrix, dtype=float)
        pts_trans = M @ pts
    else:
        pts_trans = pts

    x_trans = pts_trans[0, :].reshape(theta_grid.shape)
    y_trans = pts_trans[1, :].reshape(theta_grid.shape)
    z_trans = pts_trans[2, :].reshape(theta_grid.shape)

    fig.add_trace(
        go.Surface(
            x=x_trans,
            y=y_trans,
            z=z_trans,
            opacity=opacity,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=name,
            showlegend=showlegend,
            legendgroup=name,
            hoverinfo="skip",
        )
    )


def add_grid_3d(
    fig,
    transform_matrix=None,
    limit=2.0,
    step=1.0,
    color="rgba(150, 160, 180, 0.25)",
    name="Siatka 3D",
    showlegend=True,
):
    """Rysuje siatkę przestrzenną linii (jak w animacjach 3Blue1Brown)."""
    coords = np.arange(-limit, limit + 0.1, step)
    line_x = []
    line_y = []
    line_z = []

    # Linie wzdłuż osi X dla różnych (y, z)
    for y in coords:
        for z in coords:
            line_x.extend([-limit, limit, None])
            line_y.extend([y, y, None])
            line_z.extend([z, z, None])

    # Linie wzdłuż osi Y dla różnych (x, z)
    for x in coords:
        for z in coords:
            line_x.extend([x, x, None])
            line_y.extend([-limit, limit, None])
            line_z.extend([z, z, None])

    # Linie wzdłuż osi Z dla różnych (x, y)
    for x in coords:
        for y in coords:
            line_x.extend([x, x, None])
            line_y.extend([y, y, None])
            line_z.extend([-limit, limit, None])

    pts = np.array(
        [
            [x if x is not None else np.nan for x in line_x],
            [y if y is not None else np.nan for y in line_y],
            [z if z is not None else np.nan for z in line_z],
        ]
    )

    if transform_matrix is not None:
        M = np.asarray(transform_matrix, dtype=float)
        mask = ~np.isnan(pts[0])
        pts_valid = pts[:, mask]
        pts_trans = M @ pts_valid

        pts_out = np.empty_like(pts)
        pts_out[:, mask] = pts_trans
        pts_out[:, ~mask] = np.nan
    else:
        pts_out = pts

    fig.add_trace(
        go.Scatter3d(
            x=pts_out[0],
            y=pts_out[1],
            z=pts_out[2],
            mode="lines",
            line=dict(color=color, width=1.5),
            name=name,
            showlegend=showlegend,
            legendgroup=name,
            hoverinfo="skip",
        )
    )


def add_parallelogram(
    fig,
    u,
    v,
    start=(0.0, 0.0, 0.0),
    color="rgba(255, 215, 0, 0.35)",
    edge_color="gold",
    name="Równoległobok",
    showlegend=True,
):
    """Rysuje wypełniony równoległobok rozpięty przez wektory u i v."""
    p0 = np.asarray(start, dtype=float)
    p1 = p0 + np.asarray(u, dtype=float)
    p2 = p0 + np.asarray(u, dtype=float) + np.asarray(v, dtype=float)
    p3 = p0 + np.asarray(v, dtype=float)

    verts = np.array([p0, p1, p2, p3])

    # Dwa trójkąty tworzące czworokąt: (0, 1, 2) oraz (0, 2, 3)
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            opacity=0.4,
            color=edge_color,
            name=name,
            showlegend=showlegend,
            legendgroup=name,
            hoverinfo="skip",
        )
    )

    # Obramowanie
    perimeter = np.array([p0, p1, p2, p3, p0])
    fig.add_trace(
        go.Scatter3d(
            x=perimeter[:, 0],
            y=perimeter[:, 1],
            z=perimeter[:, 2],
            mode="lines",
            line=dict(color=edge_color, width=3),
            name=f"{name} (kontur)",
            showlegend=False,
            legendgroup=name,
            hoverinfo="skip",
        )
    )


def matrix_to_latex(M, precision=2):
    """Formatuje macierz NumPy do kodu LaTeX bmatrix."""
    M = np.asarray(M, dtype=float)
    rows = []
    for row in M:
        rows.append(" & ".join([f"{val:.{precision}f}" for val in row]))
    content = r" \\ ".join(rows)
    return r"\begin{bmatrix} " + content + r" \end{bmatrix}"


def vector_to_latex(v, precision=2):
    """Formatuje wektor NumPy do kodu LaTeX bmatrix."""
    v = np.asarray(v, dtype=float)
    content = r" \\ ".join([f"{val:.{precision}f}" for val in v])
    return r"\begin{bmatrix} " + content + r" \end{bmatrix}"
