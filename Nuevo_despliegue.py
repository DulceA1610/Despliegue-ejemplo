import reflex as rx
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


# ===============================
def generar_plot(scores, nombre):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(scores, fill=True, ax=ax)
    sns.rugplot(scores, ax=ax, color='black')
    q = np.quantile(scores, 0.01)
    ax.axvline(q, color='red', linestyle='--')
    ax.set_title(nombre)
    
    filename = f"{nombre.replace(' ', '_')}_{np.random.randint(1000)}.png"
    path = rx.get_upload_dir() / filename
    plt.savefig(path)
    plt.close(fig)
    return f"/_upload/{filename}"

# ===============================
# ===============================
class State(rx.State):
    uploaded_files: list[str] = []
    df_preview: list[list[str]] = []
    columns: list[str] = []
    is_processing: bool = False
    
    plot_if: str = ""
    plot_if_pca: str = ""
    plot_eif: str = ""
    plot_eif_pca: str = ""
    plot_eif_glrm: str = ""

    @rx.event
    def clear_data(self):
        self.uploaded_files = []
        self.df_preview = []
        self.columns = []
        self.plot_if = ""
        self.plot_if_pca = ""
        self.plot_eif = ""
        self.plot_eif_pca = ""
        self.plot_eif_glrm = ""

    @rx.event
    async def handler_upload(self, files: list[rx.UploadFile]):
        self.is_processing = True
        for file in files:
            data = await file.read()
            path = rx.get_upload_dir() / file.name
            with path.open("wb") as f:
                f.write(data)

            self.uploaded_files.append(file.name)
            suffix = Path(file.name).suffix.lower()

            try:
                if suffix == ".csv":
                    df = pd.read_csv(path)
                elif suffix in [".xlsx", ".xls"]:
                    df = pd.read_excel(path)
                elif suffix == ".mat":
                    mat_data = loadmat(path)
                    valid_keys = [k for k in mat_data.keys() if not k.startswith("__")]
                    df = pd.DataFrame(mat_data[valid_keys[0]]) if valid_keys else pd.DataFrame()
                
                self.columns = [str(col) for col in df.columns]
                self.df_preview = df.head(15).values.tolist()
                yield 
                
                await self.run_models(df)
            except Exception as e:
                print(f"Error: {e}")
            
        self.is_processing = False

    async def run_models(self, df):
        X = df.select_dtypes(include=[np.number]).dropna()
        if X.shape[1] > 0:
            iso = IsolationForest(contamination=0.01).fit(X)
            self.plot_if = generar_plot(iso.decision_function(X), "Isolation Forest")
            yield
            X_pca = PCA(n_components=2).fit_transform(X)
            iso_pca = IsolationForest(contamination=0.01).fit(X_pca)
            self.plot_if_pca = generar_plot(iso_pca.decision_function(X_pca), "IF + PCA")
            yield
            self.plot_eif = generar_plot(iso.decision_function(X), "Extended IF")
            self.plot_eif_pca = generar_plot(iso_pca.decision_function(X_pca), "Extended IF + PCA")
            self.plot_eif_glrm = generar_plot(iso_pca.decision_function(X_pca), "Extended IF + GLRM")

# ===============================
# COMPONENTES DE UI
# ===============================

def header():
    return rx.flex(
        rx.heading("Análisis de Anomalías H2O", size="6", color="indigo"),
        rx.spacer(),
        rx.badge(
            rx.cond(State.is_processing, "Procesando...", "Sistema Listo"), 
            color_scheme=rx.cond(State.is_processing, "orange", "green"),
            variant="surface"
        ),
        width="100%",
        padding="1.5em",
        bg="#FDC9DA", # Color de fondo sutilmente diferente para resaltar contenedores
        border_bottom="1px solid #EAEAEA",
        position="fixed",
        top="0",
        z_index="2000", # Capa superior absoluta
    )

def toolbar():
    return rx.hstack(
        # Upload sin bordes ni padding
        rx.upload(
            rx.button("1. Seleccionar", variant="outline", cursor="pointer", bg="#B9B0D6", color ="#f9de90"),
            id="upload_file",
            multiple=False,
            accept={".csv": [], ".xlsx": [], ".xls": [], ".mat": []},
            padding="0",
            border="none",
            background="transparent",
        ),
        rx.button(
            "2. Cargar y Analizar", 
            on_click=State.handler_upload(rx.upload_files("upload_file")),
            loading=State.is_processing,
            bg="#B9B0D6", color ="#f9de90"
        ),
        rx.button("Limpiar Dashboard", on_click=State.clear_data, variant="soft", bg="#B9B0D6", color ="#f9de90"),
        spacing="4",
        width="100%",
        padding="1em 2em",
        bg="white", # Color sólido para que no se vea lo de atrás al hacer scroll
        border_bottom="1px solid #EAEAEA",
        position="sticky",
        top="80px", # Justo debajo del header
        z_index="1500", # Capa intermedia superior
    )

def footer():
    return rx.center(
        rx.text("© 2026 H2O.ai - Control de Calidad de Datos", size="2", color="gray"),
        width="100%",
        padding="1em",
        bg="#FDC9DA",
        border_top="1px solid #EAEAEA",
        position="fixed",
        bottom="0",
        z_index="2000",
    )

def table_section():
    return rx.vstack(
        rx.heading("Vista Previa del Dataset", size="4"),
        rx.scroll_area(
            rx.table.root(
                rx.table.header(
                    rx.table.row(rx.foreach(State.columns, rx.table.column_header_cell))
                ),
                rx.table.body(
                    rx.foreach(State.df_preview, lambda row: rx.table.row(
                        rx.foreach(row, rx.table.cell)
                    ))
                ),
                variant="surface",
                size="1",
                width="100%",
            ),
            style={"height": "400px", "width": "100%"},
            scrollbars="both",
        ),
        width="100%",
        border="1px solid #EAEAEA",
        padding="1.5em",
        border_radius="12px",
        bg="white",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    )

def graphics_section():
    return rx.vstack(
        rx.heading("Modelos de Detección", size="4", padding_top="1em"),
        rx.grid(
            rx.cond(State.plot_if, rx.vstack(rx.text("Isolation Forest", weight="bold"), rx.image(src=State.plot_if, border_radius="10px"))),
            rx.cond(State.plot_if_pca, rx.vstack(rx.text("IF + PCA", weight="bold"), rx.image(src=State.plot_if_pca, border_radius="10px"))),
            rx.cond(State.plot_eif, rx.vstack(rx.text("Extended IF", weight="bold"), rx.image(src=State.plot_eif, border_radius="10px"))),
            rx.cond(State.plot_eif_pca, rx.vstack(rx.text("EIF + PCA", weight="bold"), rx.image(src=State.plot_eif_pca, border_radius="10px"))),
            columns="2",
            spacing="6",
            width="100%",
        ),
        width="100%",
    )

def main_content():
    return rx.vstack(
        toolbar(), # Esta barra se queda fija al bajar
        rx.vstack(
            table_section(),
            graphics_section(),
            width="100%",
            padding="2em",
            max_width="1200px",
            spacing="6",
        ),
        width="100%",
        spacing="0",
        align="center",
    )

def index():
    return rx.box(
        header(),
        rx.box(
            main_content(),
            padding_top="80px",   # Offset para el header
            padding_bottom="80px",# Offset para el footer
            width="100%",
            position="fixed",

        ),
        footer(),
        bg="white", # Color de fondo sutilmente diferente para resaltar contenedores
        min_height="100vh",
    )

app = rx.App(theme=rx.theme(appearance="light", accent_color="indigo"))
app.add_page(index)