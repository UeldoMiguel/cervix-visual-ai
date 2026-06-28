# -*- coding: utf-8 -*-
"""Diagramas UML (PNG) para a seção de Programação Orientada a Objetos."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
import os

OUT = os.path.dirname(os.path.abspath(__file__))
AZUL = "#1f4e79"
HDR  = "#1f4e79"
HDR_T= "#ffffff"
BODY = "#f4f8fc"
EXT  = "#ececec"
CINZA= "#444444"
plt.rcParams["font.family"] = "DejaVu Sans"


def uml_class(ax, x, y, w, name, attrs, methods, stereotype=None,
              header=HDR, body=BODY, name_color=HDR_T, fs=8.0):
    """Desenha uma classe UML (3 compartimentos). Retorna (altura_total)."""
    line_h = 0.34
    nh = 0.62 + (0.22 if stereotype else 0.0)
    ah = max(line_h * max(len(attrs), 1), line_h)
    mh = max(line_h * max(len(methods), 1), line_h)
    total = nh + ah + mh

    top = y
    # corpo
    ax.add_patch(Rectangle((x, top - total), w, total, facecolor=body,
                           edgecolor=AZUL, linewidth=1.4, zorder=2))
    # cabeçalho
    ax.add_patch(Rectangle((x, top - nh), w, nh, facecolor=header,
                           edgecolor=AZUL, linewidth=1.4, zorder=3))
    cy = top - nh / 2
    if stereotype:
        ax.text(x + w / 2, top - 0.22, stereotype, ha="center", va="center",
                fontsize=fs - 0.5, style="italic", color=name_color, zorder=4)
        ax.text(x + w / 2, top - 0.46, name, ha="center", va="center",
                fontsize=fs + 1.0, fontweight="bold", color=name_color, zorder=4)
    else:
        ax.text(x + w / 2, cy, name, ha="center", va="center",
                fontsize=fs + 1.0, fontweight="bold", color=name_color, zorder=4)

    # separadores
    ax.plot([x, x + w], [top - nh, top - nh], color=AZUL, lw=1.0, zorder=4)
    ax.plot([x, x + w], [top - nh - ah, top - nh - ah], color=AZUL, lw=1.0, zorder=4)

    # atributos
    ay = top - nh - 0.20
    for a in attrs:
        ax.text(x + 0.12, ay, a, ha="left", va="center", fontsize=fs, color="#111", zorder=4)
        ay -= line_h
    if not attrs:
        ax.text(x + 0.12, ay, " ", ha="left", va="center", fontsize=fs, zorder=4)

    # métodos
    my = top - nh - ah - 0.20
    for m in methods:
        ax.text(x + 0.12, my, m, ha="left", va="center", fontsize=fs, color="#111", zorder=4)
        my -= line_h
    return total


def generalization(ax, x1, y1, x2, y2):
    """Linha sólida com triângulo vazado (herança) apontando para o pai (x2,y2)."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                 linewidth=1.5, color=CINZA, zorder=1))
    import numpy as np
    dx, dy = x2 - x1, y2 - y1
    L = (dx**2 + dy**2) ** 0.5
    ux, uy = dx / L, dy / L
    size = 0.22
    bx, by = x2 - ux * size, y2 - uy * size
    px, py = -uy, ux
    tri = [(x2, y2), (bx + px * size * 0.7, by + py * size * 0.7),
           (bx - px * size * 0.7, by - py * size * 0.7)]
    ax.add_patch(Polygon(tri, closed=True, facecolor="white",
                         edgecolor=CINZA, linewidth=1.5, zorder=5))


def association(ax, x1, y1, x2, y2, label="", mult=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.4, color="#2e6a9e", zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center",
                fontsize=7.0, color="#2e6a9e", style="italic")
    if mult:
        ax.text(x2 - 0.05, y2 + 0.18, mult, ha="right", fontsize=7.0, color="#2e6a9e")


def dependency(ax, x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, linewidth=1.2, color=CINZA, linestyle=(0, (5, 3)),
                 zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center",
                fontsize=7.0, color=CINZA, style="italic")


# =====================================================================
# FIG 6 — Diagrama de classes (herança e estrutura)
# =====================================================================
def fig_class():
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 12); ax.axis("off")
    ax.text(6.5, 11.6, "Diagrama de classes UML do Cervix Visual AI",
            ha="center", fontsize=12.5, fontweight="bold", color=AZUL)

    # Classes-base PyTorch (externas)
    uml_class(ax, 1.2, 11.0, 3.4, "Module", [], ["+forward(...)"],
              stereotype="«PyTorch»", header=EXT, body="#f7f7f7", name_color="#222")
    uml_class(ax, 8.4, 11.0, 3.4, "Dataset", [], ["+__len__()", "+__getitem__(idx)"],
              stereotype="«PyTorch»", header=EXT, body="#f7f7f7", name_color="#222")

    # Subclasses de Module (perdas)
    uml_class(ax, 0.4, 8.2, 3.1, "LabelSmoothingBCE",
              ["smoothing: float", "pos_weight: Tensor"],
              ["+forward(logits, targets): Tensor"])
    uml_class(ax, 3.9, 8.2, 3.1, "FocalLoss",
              ["gamma: float", "pos_weight: Tensor"],
              ["+forward(logits, targets): Tensor"])

    # Subclasse de Dataset
    uml_class(ax, 8.2, 8.0, 3.8, "CervixImageDataset",
              ["manifest: DataFrame", "image_root: Path", "transform: Callable"],
              ["+__len__(): int", "+__getitem__(idx): tuple"])

    # GradCAM
    uml_class(ax, 4.2, 4.2, 4.6, "GradCAM",
              ["model: Module", "target_layer: Module",
               "gradients: Tensor", "activations: Tensor", "-_handles: list"],
              ["+generate(tensor): ndarray", "+remove_hooks()", "-_register_hooks()"])

    # Generalizações (filhos → pais)
    generalization(ax, 2.0, 8.2, 2.4, 10.4)   # LabelSmoothingBCE → Module
    generalization(ax, 5.4, 8.2, 2.9, 10.4)   # FocalLoss → Module
    generalization(ax, 10.1, 8.0, 10.1, 10.4) # CervixImageDataset → Dataset

    # Associação: GradCAM → Module (modelo alvo), roteada em cotovelo pela margem esquerda
    col = "#2e6a9e"
    ax.plot([4.2, 0.25], [3.0, 3.0], color=col, lw=1.4, zorder=1)
    ax.plot([0.25, 0.25], [3.0, 10.2], color=col, lw=1.4, zorder=1)
    ax.add_patch(FancyArrowPatch((0.25, 10.2), (1.2, 10.2), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.4, color=col, zorder=1))
    ax.text(0.45, 4.8, "analisa", rotation=90, fontsize=7.5, color=col,
            style="italic", va="center")
    ax.text(1.35, 10.4, "1", fontsize=7.0, color=col)

    # Legenda UML (margem inferior direita)
    lx, ly = 8.4, 2.6
    ax.text(lx, ly + 0.5, "Legenda UML:", fontsize=8.5, fontweight="bold", color=AZUL)
    generalization(ax, lx, ly, lx + 1.1, ly)
    ax.text(lx + 1.3, ly, "generalização (herança)", fontsize=8, va="center")
    association(ax, lx, ly - 0.5, lx + 1.1, ly - 0.5)
    ax.text(lx + 1.3, ly - 0.5, "associação", fontsize=8, va="center")
    dependency(ax, lx, ly - 1.0, lx + 1.1, ly - 1.0)
    ax.text(lx + 1.3, ly - 1.0, "dependência (uso)", fontsize=8, va="center")

    fig.savefig(os.path.join(OUT, "fig6_uml_classes.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 7 — Diagrama UML de colaboração / dependências no treinamento
# =====================================================================
def fig_collab():
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(6.5, 9.6, "Diagrama UML de dependências: orquestração do treinamento",
            ha="center", fontsize=12.5, fontweight="bold", color=AZUL)

    # Orquestrador
    uml_class(ax, 4.6, 9.0, 3.8, "train_model",
              [], ["+executa o laço de treino", "+seleciona o melhor modelo"],
              stereotype="«função orquestradora»", header="#5b3a87", body="#ece4f5")

    # Fábrica de modelo
    uml_class(ax, 0.3, 5.6, 3.4, "build_model",
              [], ["+cria backbone + cabeça"],
              stereotype="«fábrica»", header="#c0561a", body="#fbe6d6")
    uml_class(ax, 0.3, 2.4, 3.4, "Module",
              ["backbone", "classifier (cabeça)"], ["+forward(x)"],
              stereotype="«modelo»", header=EXT, body="#f7f7f7", name_color="#222")

    # Loss
    uml_class(ax, 4.8, 5.6, 3.6, "FocalLoss /\nLabelSmoothingBCE",
              ["pos_weight"], ["+forward(logits, y)"],
              stereotype="«critério»", header=HDR, body=BODY)

    # Dados
    uml_class(ax, 9.2, 5.6, 3.5, "create_dataloaders",
              [], ["+monta DataLoaders"],
              stereotype="«fábrica»", header="#c0561a", body="#fbe6d6")
    uml_class(ax, 9.2, 2.4, 3.5, "CervixImageDataset",
              ["manifest", "transform"], ["+__getitem__(idx)"],
              stereotype="«dados»", header=HDR, body=BODY)

    # GradCAM (interpretabilidade)
    uml_class(ax, 4.8, 2.2, 3.6, "GradCAM",
              ["model", "target_layer"], ["+generate(tensor)"],
              stereotype="«explicabilidade»", header=HDR, body=BODY)

    # Dependências
    dependency(ax, 5.0, 9.0, 2.0, 6.8, label="cria")          # train → build_model
    dependency(ax, 6.5, 9.0, 6.5, 6.8, label="usa")           # train → loss
    dependency(ax, 8.0, 9.0, 10.8, 6.8, label="usa")          # train → dataloaders
    dependency(ax, 2.0, 5.6, 2.0, 3.8, label="instancia")     # build_model → Module
    dependency(ax, 10.8, 5.6, 10.8, 3.8, label="encapsula")   # dataloaders → Dataset
    association(ax, 4.8, 1.5, 3.75, 1.5, label="analisa")     # GradCAM → Module

    fig.savefig(os.path.join(OUT, "fig7_uml_collab.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG ARQ — Diagrama de arquitetura de software em camadas
# =====================================================================
def fig_arch():
    fig, ax = plt.subplots(figsize=(11.5, 11.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 16); ax.axis("off")
    ax.text(7.0, 15.5, "Arquitetura de software em camadas do Cervix Visual AI",
            ha="center", fontsize=12.5, fontweight="bold", color=AZUL)

    # paleta por camada
    AZ, AZl = "#1f4e79", "#dbe8f5"
    VD, VDl = "#1e7d4f", "#d8efe2"
    LR, LRl = "#c0561a", "#fbe6d6"
    RX, RXl = "#5b3a87", "#e7ddf2"
    GZ, GZl = "#555555", "#ededed"

    def band(ytop, title, comps, tcol, ccol, h=1.40):
        from matplotlib.patches import FancyBboxPatch
        # aba de título
        ax.add_patch(FancyBboxPatch((0.5, ytop - h), 2.15, h,
                     boxstyle="round,pad=0.01,rounding_size=0.06",
                     linewidth=1.4, edgecolor=tcol, facecolor=tcol, zorder=3))
        ax.text(1.575, ytop - h / 2, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color="white", zorder=4, wrap=True)
        # conteúdo
        ax.add_patch(FancyBboxPatch((2.85, ytop - h), 7.65, h,
                     boxstyle="round,pad=0.01,rounding_size=0.06",
                     linewidth=1.4, edgecolor=tcol, facecolor=ccol, zorder=2))
        ax.text(3.0, ytop - h / 2, comps, ha="left", va="center",
                fontsize=8.6, color="#111", zorder=4)

    layers = [
        ("Interface\n(CLI)",
         "main() / build_parser()\nprepare · demo · train · evaluate · predict · show", AZ, AZl),
        ("Orquestração",
         "run_demo() · train_model() · evaluate_checkpoint() · predict_image()", RX, RXl),
        ("Dados",
         "prepare_iarc_dataset() · load_manifest() · build_transforms()\n"
         "CervixImageDataset · create_dataloaders()", VD, VDl),
        ("Modelo",
         "build_model()  →  EfficientNet-B3/B4 · ResNet-34 · ConvNeXt-Tiny\n"
         "cabeça de 2 camadas · set_backbone_trainable()", LR, LRl),
        ("Treino &\nOtimização",
         "train_epoch() · mixup_batch() · clip de gradiente\n"
         "FocalLoss / LabelSmoothingBCE · AdamW + CosineAnnealingWR", RX, RXl),
        ("Inferência &\nCalibração",
         "predict_loader() — TTA 4 vistas · aggregate_case_predictions()\n"
         "optimize_threshold() · _calibrate_temperature()", AZ, AZl),
        ("Avaliação &\nMétricas",
         "binary_metrics() · bootstrap_confidence_intervals() · evaluate_splits()", VD, VDl),
        ("Explicabilidade\n& Saída",
         "GradCAM · generate_gradcam_*() · generate_plots()\n"
         "print_terminal_metrics() · artefatos .pt / .json / .csv / .png / .html", GZ, GZl),
    ]

    ytop0 = 14.45
    h = 1.40
    gap = 0.135
    tops = [ytop0 - i * (h + gap) for i in range(len(layers))]
    for yt, (title, comps, tcol, ccol) in zip(tops, layers):
        band(yt, title, comps, tcol, ccol, h)

    # seta de fluxo (top-down) na margem esquerda
    ybot = tops[-1] - h
    ax.add_patch(FancyArrowPatch((0.28, ytop0), (0.28, ybot),
                 arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color="#888"))
    ax.text(0.13, (ytop0 + ybot) / 2, "fluxo de controle (chamadas top-down)",
            rotation=90, ha="center", va="center", fontsize=7.8, color="#666", style="italic")

    # barra lateral: bibliotecas externas
    from matplotlib.patches import FancyBboxPatch
    sx, sw = 10.75, 2.9
    ax.add_patch(FancyBboxPatch((sx, ybot), sw, ytop0 - ybot,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.4, edgecolor="#2e6a9e", facecolor="#f4f8fc", zorder=2))
    ax.text(sx + sw / 2, ytop0 - 0.35, "Bibliotecas\nexternas", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=AZUL, zorder=4)
    libs = ["PyTorch /", "torchvision", "", "scikit-learn", "", "pandas / NumPy",
            "", "matplotlib /", "Plotly", "", "OpenCV / PIL"]
    ly = ytop0 - 1.3
    for lib in libs:
        ax.text(sx + sw / 2, ly, lib, ha="center", va="center", fontsize=8.6,
                color="#222", zorder=4)
        ly -= 0.92 if lib == "" else 0.50
    # conector pontilhado indicando dependência transversal
    ax.add_patch(FancyArrowPatch((10.5, (ytop0 + ybot) / 2), (sx, (ytop0 + ybot) / 2),
                 arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color=CINZA,
                 linestyle=(0, (5, 3))))
    ax.text((10.5 + sx) / 2, (ytop0 + ybot) / 2 + 0.18, "usa", ha="center",
            fontsize=7.5, color=CINZA, style="italic")

    fig.savefig(os.path.join(OUT, "fig_arch.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


for fn in (fig_arch, fig_class, fig_collab):
    fn()
    print("OK:", fn.__name__)
print("Diagramas UML gerados.")
