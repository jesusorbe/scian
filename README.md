SCIAN App (Dash)

Aplicación Dash autocontenida para explorar el grafo de la Matriz Insumo‑Producto (MIP) por actividades SCIAN. No requiere base de datos en producción; lee archivos Parquet locales bajo `scianapp/intermediate_data/`.

Resumen de capacidades
- Selección de nivel SCIAN (1–4) y de una o varias actividades.
- Visualización de principales proveedores y/o clientes (Top N) del primer nivel.
- Expansión por profundidad (proveedores de proveedores o clientes de clientes) hasta 5 niveles.
- Interfaz responsive y grafo interactivo con descarga en HTML.

Arquitectura
- UI (Dash):
  - `scianapp/dash_app.py` define el layout (header, sidebar, área principal) y callbacks.
  - `assets/style.css` aporta un tema profesional, responsive y estilos para paneles, botones y gráficas.
- Datos (Parquet):
  - Directorio `scianapp/intermediate_data/` con pares por nivel: `l{nivel}_nodes.parquet` y `l{nivel}_edges.parquet`.
  - Búsqueda de datos: `dash_app.py` prioriza `./intermediate_data`, luego `../intermediate_data`, y por último `cwd/intermediate_data`.
- Lógica de grafo:
  - Construcción con NetworkX y render con PyVis.
  - Funciones clave:
    - `load_level_data(level)`: lee y valida Parquet; cacheada con `lru_cache`.
    - `build_graph(...)`: proveedores/clientes de primer nivel según Top N y umbral de peso.
    - `build_graph_with_depth(...)`: expansión BFS acotada por profundidad, dirección y Top N.
    - `net_to_html(net)`: exporta el grafo PyVis a HTML embebible.
- Render del grafo:
  - El HTML generado por PyVis se inserta en un `Iframe` (`srcDoc`).
  - El enlace de descarga usa un data URL. Para grafos muy grandes, es sencillo migrar a un endpoint de descarga.

Estructura de datos esperada
- `l{nivel}_nodes.parquet`:
  - Columnas: `SCIAN_ID` (str), `DESCRIPTOR` (str). Pueden existir columnas adicionales ignoradas.
- `l{nivel}_edges.parquet`:
  - Columnas: `FROM` (str), `TO` (str), `WEIGHT` (float).
  - Semántica: aristas `FROM -> TO`. Proveedores de `X` son filas con `TO == X`; clientes de `X` son filas con `FROM == X`.

Instalación
- Requisitos: Python 3.9+ (recomendado 3.10+).
- Crear entorno y dependencias:
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r scianapp/requirements.txt
```

Ejecución
```
python scianapp/dash_app.py
```
- Por defecto expone en `http://127.0.0.1:8050`.
- Ajusta host/port editando `app.run(debug=True)` si lo necesitas.

Uso de la interfaz
- Sidebar (izquierda):
  - Nivel SCIAN: cambia el conjunto de actividades disponibles.
  - Actividades: multiselección de nodos centrales.
  - Mostrar: “Ambos”, “Solo proveedores” o “Solo clientes”.
  - Top N por actividad: filtra a las relaciones más relevantes.
  - Umbral mínimo de peso: descarta aristas débiles.
  - Expansión por profundidad: dirección y profundidad (2–5) con botón dedicado.
- Panel principal (derecha):
  - Grafo PyVis interactivo (zoom, arrastre, tooltip con coeficiente y etiquetas).
  - Botón de descarga del HTML del grafo.
- Render inicial: al abrir la app se selecciona aleatoriamente una actividad para visualizar sin clicks previos.

Detalles de lógica
- Top N y umbral se aplican en cada nivel de expansión, reduciendo densidad y mejorando legibilidad.
- Colorimetría por tipo de relación y profundidad:
  - Centro: naranja.
  - Proveedores: azules degradados por profundidad.
  - Clientes: rojos/salmón degradados por profundidad.
- IDs y descriptores se normalizan a `str` para consistencia.

Extensión y personalización
- Descarga por endpoint: reemplazar data URL por un callback con `dcc.send_file` (vía `flask.Response`) si el HTML supera límites de URL.
- Más filtros: añadir rangos de `WEIGHT`, excluir/forzar actividades, o límites por grado.
- Estado compartido: memorizar selección en URL (`dcc.Location`) para deep-linking.
- Despliegue: usar un servidor WSGI/ASGI (Gunicorn/Uvicorn) detrás de Nginx; habilitar compresión gzip y cache para activos.

Solución de problemas
- “No se encontraron archivos parquet”: verifica `scianapp/intermediate_data/` que existan `l{nivel}_nodes.parquet` y `l{nivel}_edges.parquet`.
- Grafo vacío o muy denso: ajusta `Top N` y el `Umbral mínimo`.
- Data URL demasiado grande: migra a endpoint de descarga.
- Puerto ocupado: cambia el puerto en `app.run(port=8051)`.

Estructura del proyecto
- `scianapp/dash_app.py` — App principal de Dash y lógica de grafo.
- `scianapp/assets/style.css` — Estilos (cargados automáticamente por Dash).
- `scianapp/intermediate_data/` — Parquet por nivel `l{1..4}_*.parquet`.
- `scianapp/requirements.txt` — Dependencias mínimas (Dash, pandas, pyarrow, networkx, pyvis).
Ejecutar la app (Dash)
```
python scianapp/dash_app.py
```
La app arrancará en `http://127.0.0.1:8050` (ajusta host/port en el `run_server` si lo necesitas).
