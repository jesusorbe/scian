from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Set
from functools import lru_cache
import tempfile
import random

import pandas as pd
import networkx as nx
from pyvis.network import Network

from dash import Dash, dcc, html, Input, Output, State, ctx


def resolve_intermediate_dir() -> Path:
    """
    Devuelve la ruta a 'intermediate_data'. Busca en:
    - Carpeta hermana a este archivo (.. / intermediate_data)
    - Carpeta actual del proceso (./intermediate_data)
    - Carpeta actual del archivo (./intermediate_data)
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "intermediate_data",              # 1) local a la app (autocontenida)
        here.parent / "intermediate_data",        # 2) raíz del repo
        Path.cwd() / "intermediate_data",         # 3) cwd
    ]
    for c in candidates:
        if c.exists():
            return c
    # Si no existe, devolvemos la del repo por defecto
    return here.parent / "intermediate_data"


DATA_DIR = resolve_intermediate_dir()


@lru_cache(maxsize=None)
def load_level_data(level: int):
    nodes_path = DATA_DIR / f"l{level}_nodes.parquet"
    edges_path = DATA_DIR / f"l{level}_edges.parquet"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"No se encontraron archivos parquet para el nivel {level}. Esperados: {nodes_path}, {edges_path}"
        )
    df_nodes = pd.read_parquet(nodes_path)
    df_edges = pd.read_parquet(edges_path)
    for col in ("SCIAN_ID", "DESCRIPTOR"):
        if col not in df_nodes.columns:
            raise ValueError(f"Falta columna '{col}' en nodos nivel {level}")
    for col in ("FROM", "TO", "WEIGHT"):
        if col not in df_edges.columns:
            raise ValueError(f"Falta columna '{col}' en aristas nivel {level}")
    # Normaliza a str ids para evitar discrepancias
    df_nodes["SCIAN_ID"] = df_nodes["SCIAN_ID"].astype(str)
    df_edges["FROM"] = df_edges["FROM"].astype(str)
    df_edges["TO"] = df_edges["TO"].astype(str)
    return df_nodes, df_edges


def label_for(node_id: str, id2desc: Dict[str, str]) -> str:
    return f"{node_id} - {id2desc.get(node_id, '')}".strip()


def build_graph(center_ids: Iterable[str], df_nodes: pd.DataFrame, df_edges: pd.DataFrame,
                mode: str, topn: int, min_weight: Optional[float]) -> Network:
    id2desc = {str(row.SCIAN_ID): str(row.DESCRIPTOR) for _, row in df_nodes.iterrows()}
    G = nx.DiGraph()

    centers = [str(c) for c in center_ids]
    for center in centers:
        center_label = id2desc.get(center, center)
        G.add_node(center, label=center_label, color='orange', size=25, title=center_label)

        # Proveedores: edges hacia el centro (p -> c)
        if mode in ("Ambos", "Solo proveedores"):
            prov_df = df_edges[df_edges["TO"] == center].copy()
            if min_weight is not None:
                prov_df = prov_df[prov_df["WEIGHT"] >= float(min_weight)]
            prov_df = prov_df.sort_values("WEIGHT", ascending=False).head(topn)
            for _, r in prov_df.iterrows():
                pid = str(r["FROM"])  # proveedor
                plabel = label_for(pid, id2desc)
                G.add_node(pid, label=plabel, color='blue', size=15, title=plabel)
                G.add_edge(pid, center, value=float(r["WEIGHT"]), title=f"Coeficiente: {float(r['WEIGHT']):.4f}")

        # Clientes: edges desde el centro (c -> cl)
        if mode in ("Ambos", "Solo clientes"):
            cli_df = df_edges[df_edges["FROM"] == center].copy()
            if min_weight is not None:
                cli_df = cli_df[cli_df["WEIGHT"] >= float(min_weight)]
            cli_df = cli_df.sort_values("WEIGHT", ascending=False).head(topn)
            for _, r in cli_df.iterrows():
                cid = str(r["TO"])  # cliente
                clabel = label_for(cid, id2desc)
                G.add_node(cid, label=clabel, color='red', size=15, title=clabel)
                G.add_edge(center, cid, value=float(r["WEIGHT"]), title=f"Coeficiente: {float(r['WEIGHT']):.4f}")

    net = Network(height="800px", width="100%", directed=True)
    net.from_nx(G)
    return net


def build_graph_with_depth(center_ids: Iterable[str], df_nodes: pd.DataFrame, df_edges: pd.DataFrame,
                           direction: str, depth: int, topn: int, min_weight: Optional[float]) -> Network:
    """
    Expande el grafo a partir de center_ids hasta 'depth' capas.
    direction: 'suppliers' (entrantes) o 'clients' (salientes).
    """
    id2desc = {str(row.SCIAN_ID): str(row.DESCRIPTOR) for _, row in df_nodes.iterrows()}
    G = nx.DiGraph()

    prov_colors = ['#1f77b4', '#5dade2', '#aed6f1', '#d6eaf8', '#ebf5fb']
    cli_colors = ['#d62728', '#f1948a', '#f5b7b1', '#fadbd8', '#fdecea']

    centers = [str(c) for c in center_ids]
    for c in centers:
        label = id2desc.get(c, c)
        G.add_node(c, label=label, color='orange', size=25, title=label)

    current: Set[str] = set(centers)
    visited: Set[str] = set(current)
    depth = max(1, min(int(depth), 5))

    for d in range(1, depth + 1):
        next_frontier: Set[str] = set()
        color = prov_colors[min(d-1, len(prov_colors)-1)] if direction == 'suppliers' else cli_colors[min(d-1, len(cli_colors)-1)]

        for node in list(current):
            if direction == 'suppliers':
                df = df_edges[df_edges['TO'] == node].copy()
            else:
                df = df_edges[df_edges['FROM'] == node].copy()

            if min_weight is not None:
                df = df[df['WEIGHT'] >= float(min_weight)]
            df = df.sort_values('WEIGHT', ascending=False).head(topn)

            for _, r in df.iterrows():
                if direction == 'suppliers':
                    src, dst = str(r['FROM']), node
                    nb = src
                else:
                    src, dst = node, str(r['TO'])
                    nb = dst

                if nb not in G:
                    nb_label = id2desc.get(nb, nb)
                    G.add_node(nb, label=nb_label, color=color, size=max(10, 16 - d), title=nb_label)
                G.add_edge(src, dst, value=float(r['WEIGHT']), title=f"Coeficiente: {float(r['WEIGHT']):.4f}")

                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)

        if not next_frontier:
            break
        current = next_frontier

    net = Network(height="800px", width="100%", directed=True)
    net.from_nx(G)
    return net


def net_to_html(net: Network) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        return Path(tmp.name).read_text(encoding="utf-8")


def make_app() -> Dash:
    app = Dash(__name__)
    app.title = "Explorador SCIAN - MIP (Dash)"

    # Layout
    app.layout = html.Div([
        # Encabezado
        html.Div([
            html.Div([
                html.Div("Explorador SCIAN — Matriz Insumo‑Producto", className='brand-title'),
                html.Div("Visualización interactiva de proveedores y clientes", className='brand-subtitle')
            ], className='brand')
        ], className='app-header'),

        # Contenido principal (sidebar + grafo)
        html.Div([
            html.Div([
                html.H4("Controles", className='panel-title'),
                html.P("Selecciona nivel y actividades para visualizar principales proveedores/clientes.", className='muted'),

            html.Div("Nivel SCIAN", className='form-label'),
            dcc.Dropdown(id='level', options=[{'label': str(i), 'value': i} for i in [1, 2, 3, 4]], value=4, clearable=False,
                         style={'marginBottom': '10px'}),

            html.Div("Actividades (SCIAN)", className='form-label'),
            dcc.Dropdown(id='activities', options=[], multi=True, style={'marginBottom': '12px'}),

            html.Div("Mostrar", className='form-label'),
            dcc.RadioItems(id='mode', options=[
                {'label': 'Ambos', 'value': 'Ambos'},
                {'label': 'Solo proveedores', 'value': 'Solo proveedores'},
                {'label': 'Solo clientes', 'value': 'Solo clientes'},
            ], value='Ambos', labelStyle={'display': 'block'}, style={'marginBottom': '12px'}),

            html.Div("Top N por actividad", className='form-label'),
            dcc.Slider(id='topn', min=3, max=50, step=1, value=10,
                       marks={i: str(i) for i in range(5, 51, 5)},
                       tooltip={'placement': 'bottom', 'always_visible': False},
                       ),
            html.Div(style={'height': '10px'}),

            html.Div("Umbral mínimo de peso (opcional)", className='form-label'),
            dcc.Input(id='minw', type='number', value=0.0, step=0.001, style={'width': '100%', 'marginBottom': '10px'}),
            html.Button("Dibujar grafo", id='draw-btn', n_clicks=0, className='btn btn-primary', style={'width': '100%', 'marginBottom': '16px'}),

            html.Hr(),
            html.H4("Expansión por profundidad", className='panel-title'),
            html.Div("Dirección", className='form-label'),
            dcc.RadioItems(id='expand-dir', options=[
                {'label': 'Proveedores de proveedores', 'value': 'suppliers'},
                {'label': 'Clientes de clientes', 'value': 'clients'},
            ], value='suppliers', labelStyle={'display': 'block'}, style={'marginBottom': '12px'}),

            html.Div("Profundidad (2-5)", className='form-label'),
            dcc.Slider(id='expand-depth', min=2, max=5, step=1, value=2,
                       marks={i: str(i) for i in range(2, 6)},
                       tooltip={'placement': 'bottom', 'always_visible': False},
                       ),
            html.Div(style={'height': '10px'}),
            html.Button("Expandir a profundidad", id='expand-btn', n_clicks=0, className='btn btn-secondary', style={'width': '100%'}),
        ], id='sidebar', className='panel', style={
            'flex': '0 0 340px',
            'maxWidth': '360px',
            'padding': '12px',
            'borderRight': '1px solid #eee',
            'position': 'sticky',
            'top': '0'
        }),

            html.Div([
                html.Div(id='message', className='alert'),
                html.Div(id='graph-container', className='graph-card'),
                html.Div([
                    html.A("Descargar HTML del grafo", id='download-link', href='', download='grafo_scian.html', target='_blank', className='btn btn-outline')
                ], className='download-row')
            ], id='main', style={'flex': '1 1 auto', 'padding': '12px'})
        ], id='root', style={
        'display': 'flex',
        'gap': '16px',
        'fontFamily': 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial'
        })
    ], className='app-container')

    # Poblar opciones de actividades al cambiar nivel
    @app.callback(
        Output('activities', 'options'),
        Output('activities', 'value'),
        Input('level', 'value')
    )
    def update_activities(level):
        try:
            df_nodes, _ = load_level_data(int(level))
        except Exception as e:
            # Si falla, limpiar opciones
            return [], []
        options = [
            {
                'label': f"{row.SCIAN_ID} - {row.DESCRIPTOR}",
                'value': str(row.SCIAN_ID)
            }
            for _, row in df_nodes.sort_values('DESCRIPTOR').iterrows()
        ]
        # Selección por defecto aleatoria (1 actividad) para render inicial
        ids = [str(x) for x in df_nodes['SCIAN_ID'].astype(str).tolist()]
        default_sel = random.sample(ids, k=1) if ids else []
        return options, default_sel

    # Render del grafo y enlace de descarga
    @app.callback(
        Output('graph-container', 'children'),
        Output('download-link', 'href'),
        Output('download-link', 'download'),
        Output('message', 'children'),
        Input('draw-btn', 'n_clicks'),
        Input('expand-btn', 'n_clicks'),
        State('level', 'value'),
        State('activities', 'value'),
        State('mode', 'value'),
        State('topn', 'value'),
        State('minw', 'value'),
        State('expand-dir', 'value'),
        State('expand-depth', 'value'),
        prevent_initial_call=False
    )
    def render_graph(draw_clicks, expand_clicks, level, activities, mode, topn, minw, expand_dir, expand_depth):
        triggered = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None

        try:
            df_nodes, df_edges = load_level_data(int(level))
        except Exception as e:
            return html.Div(), '', 'grafo_scian.html', f"Error cargando datos: {e}"

        # Si no hay selección, elige una aleatoria y renderiza
        if not activities:
            ids = [str(x) for x in df_nodes['SCIAN_ID'].astype(str).tolist()]
            activities = random.sample(ids, k=1) if ids else []
            if not activities:
                return html.Div(), '', 'grafo_scian.html', 'No hay datos disponibles.'

        # Construcción del grafo
        if triggered == 'expand-btn':
            net = build_graph_with_depth(
                center_ids=activities,
                df_nodes=df_nodes,
                df_edges=df_edges,
                direction=expand_dir,
                depth=int(expand_depth),
                topn=int(topn or 10),
                min_weight=float(minw) if minw not in (None, '') else None,
            )
            filename = f"grafo_scian_l{level}_expansion.html"
        else:
            net = build_graph(
                center_ids=activities,
                df_nodes=df_nodes,
                df_edges=df_edges,
                mode=mode,
                topn=int(topn or 10),
                min_weight=float(minw) if minw not in (None, '') else None,
            )
            filename = f"grafo_scian_l{level}.html"

        html_content = net_to_html(net)

        # IFrame en vivo
        iframe = html.Iframe(srcDoc=html_content, style={'width': '100%', 'height': '75vh', 'minHeight': '520px', 'border': '0'})

        # Enlace de descarga como data URL
        # Nota: para contenidos grandes puede superar límites del navegador.
        from urllib.parse import quote
        data_url = f"data:text/html;charset=utf-8,{quote(html_content)}"

        return iframe, data_url, filename, ''

    return app


if __name__ == "__main__":
    app = make_app()
    # Ejecutar en modo desarrollo. Cambia host/port según necesidad.
    app.run(debug=False, port=8053)
