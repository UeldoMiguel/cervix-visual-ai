# -*- coding: utf-8 -*-
"""Gera diagramas (PNG) para o documento acadêmico Cervix Visual AI."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# Paleta
AZUL    = "#1f4e79"
AZUL_CL = "#dbe8f5"
VERDE   = "#1e7d4f"
VERDE_CL= "#d8efe2"
LARANJA = "#c0561a"
LARANJA_CL="#fbe6d6"
ROXO    = "#5b3a87"
ROXO_CL = "#e7ddf2"
CINZA   = "#444444"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc, ec, fs=10, bold=False, tc="#111111"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=CINZA, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, linewidth=1.6, color=color, linestyle=ls))


# =====================================================================
# FIG 1 — Pipeline ponta a ponta
# =====================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 14); ax.axis("off")

    ax.text(6, 13.5, "Visão geral do pipeline Cervix Visual AI",
            ha="center", fontsize=12.5, fontweight="bold", color=AZUL)

    # Etapa 1: Dados
    box(ax, 0.4, 12.0, 3.2, 1.0, "ZIP IARC\n(Cases - Images / Meta data)", AZUL_CL, AZUL, 9, True)
    box(ax, 4.4, 12.0, 3.2, 1.0, "prepare_iarc_dataset()\nmineração de texto → rótulo", AZUL_CL, AZUL, 9)
    box(ax, 8.4, 12.0, 3.2, 1.0, "manifest.csv\ncase_manifest.csv", AZUL_CL, AZUL, 9, True)
    arrow(ax, 3.6, 12.5, 4.4, 12.5); arrow(ax, 7.6, 12.5, 8.4, 12.5)

    # Etapa 2: split
    box(ax, 8.4, 10.3, 3.2, 1.0, "Divisão por PACIENTE\n70 / 15 / 15 estratificada", VERDE_CL, VERDE, 9, True)
    arrow(ax, 10.0, 12.0, 10.0, 11.3)
    box(ax, 4.4, 10.3, 3.2, 1.0, "DataLoaders + Aug.\nforte (treino)", VERDE_CL, VERDE, 9)
    arrow(ax, 8.4, 10.8, 7.6, 10.8)
    box(ax, 0.4, 10.3, 3.2, 1.0, "CervixImageDataset\nPIL→tensor, checagem NaN", VERDE_CL, VERDE, 9)
    arrow(ax, 4.4, 10.8, 3.6, 10.8)

    # Etapa 3: modelo
    box(ax, 0.4, 8.4, 3.2, 1.1, "EfficientNet-B3\n(backbone ImageNet)", LARANJA_CL, LARANJA, 9.5, True)
    arrow(ax, 2.0, 10.3, 2.0, 9.5)
    box(ax, 4.4, 8.4, 3.2, 1.1, "Cabeça 2 camadas\nDrop→FC256→ReLU→Drop→FC1", LARANJA_CL, LARANJA, 8.5)
    arrow(ax, 3.6, 8.95, 4.4, 8.95)
    box(ax, 8.4, 8.4, 3.2, 1.1, "Fine-tuning 2 fases\ncongela 5 épocas", LARANJA_CL, LARANJA, 9)
    arrow(ax, 7.6, 8.95, 8.4, 8.95)

    # Etapa 4: treino
    box(ax, 8.4, 6.5, 3.2, 1.1, "Focal Loss + pos_weight\nMixUp + clip de gradiente", ROXO_CL, ROXO, 9)
    arrow(ax, 10.0, 8.4, 10.0, 7.6)
    box(ax, 4.4, 6.5, 3.2, 1.1, "AdamW + Cosine\nWarmRestarts", ROXO_CL, ROXO, 9.5, True)
    arrow(ax, 8.4, 7.05, 7.6, 7.05)
    box(ax, 0.4, 6.5, 3.2, 1.1, "Seleção do modelo\n0,5·AUC+0,3·F1+0,2·BalAcc", ROXO_CL, ROXO, 8.5)
    arrow(ax, 4.4, 7.05, 3.6, 7.05)

    # Etapa 5: inferência/calibração
    box(ax, 0.4, 4.6, 3.2, 1.1, "TTA 4 vistas\n(orig, H, V, HV)", AZUL_CL, AZUL, 9.5, True)
    arrow(ax, 2.0, 6.5, 2.0, 5.7)
    box(ax, 4.4, 4.6, 3.2, 1.1, "Temperature scaling\n(LBFGS, T)", AZUL_CL, AZUL, 9.5)
    arrow(ax, 3.6, 5.15, 4.4, 5.15)
    box(ax, 8.4, 4.6, 3.2, 1.1, "Limiar clínico\nF1 s.a. Sens ≥ 0,80", AZUL_CL, AZUL, 9)
    arrow(ax, 7.6, 5.15, 8.4, 5.15)

    # Etapa 6: avaliação
    box(ax, 8.4, 2.7, 3.2, 1.1, "Métricas por CASO\n+ IC 95% bootstrap", VERDE_CL, VERDE, 9, True)
    arrow(ax, 10.0, 4.6, 10.0, 3.8)
    box(ax, 4.4, 2.7, 3.2, 1.1, "Agregação por paciente\n(média de probabilidades)", VERDE_CL, VERDE, 9)
    arrow(ax, 8.4, 3.25, 7.6, 3.25)
    box(ax, 0.4, 2.7, 3.2, 1.1, "Grad-CAM\nTP / TN / FP / FN", VERDE_CL, VERDE, 9.5, True)
    arrow(ax, 4.4, 3.25, 3.6, 3.25)

    # saída
    box(ax, 3.0, 0.7, 6.0, 1.1,
        "Artefatos: best_model.pt · *.json · CSV por divisão · gráficos PNG/HTML · painéis Grad-CAM",
        "#f2f2f2", CINZA, 9, True)
    arrow(ax, 2.0, 2.7, 4.0, 1.8); arrow(ax, 10.0, 2.7, 8.0, 1.8)

    # legenda de cores
    ax.text(0.4, 0.2, "Dados", color=AZUL, fontsize=8, fontweight="bold")
    ax.text(2.0, 0.2, "Pré-proc.", color=VERDE, fontsize=8, fontweight="bold")
    ax.text(4.0, 0.2, "Modelo", color=LARANJA, fontsize=8, fontweight="bold")
    ax.text(5.8, 0.2, "Treino", color=ROXO, fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 2 — Cabeça de classificação
# =====================================================================
def fig_head():
    fig, ax = plt.subplots(figsize=(11, 2.7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
    ax.text(6.5, 2.7, "Cabeça de classificação binária (sobre o backbone)",
            ha="center", fontsize=12, fontweight="bold", color=AZUL)
    stages = [
        ("Vetor de\ncaracterísticas\n(backbone)", AZUL_CL, AZUL),
        ("Dropout\np = 0,40", LARANJA_CL, LARANJA),
        ("Linear\n→ 256", ROXO_CL, ROXO),
        ("ReLU", "#eeeeee", CINZA),
        ("Dropout\np = 0,20", LARANJA_CL, LARANJA),
        ("Linear\n→ 1 (logit)", ROXO_CL, ROXO),
        ("Sigmoide\nP(alto grau)", VERDE_CL, VERDE),
    ]
    x = 0.3; w = 1.62; gap = 0.18; y = 0.9; h = 1.1
    for i, (t, fc, ec) in enumerate(stages):
        box(ax, x, y, w, h, t, fc, ec, 8.5, bold=(i in (0, 6)))
        if i < len(stages) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_head.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 3 — Fine-tuning em duas fases
# =====================================================================
def fig_finetune():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    ax.text(6, 3.7, "Transferência de aprendizado em duas fases",
            ha="center", fontsize=12, fontweight="bold", color=AZUL)

    # linha do tempo
    ax.add_patch(FancyArrowPatch((0.5, 0.6), (11.5, 0.6), arrowstyle="-|>",
                 mutation_scale=18, linewidth=1.8, color=CINZA))
    ax.text(11.5, 0.25, "épocas", fontsize=9, color=CINZA, ha="right")
    ax.axvline  # noqa
    for ep, lab in [(0.5, "0"), (4.0, "5"), (11.0, "30")]:
        ax.plot([ep, ep], [0.5, 0.7], color=CINZA, lw=1.2)
        ax.text(ep, 0.2, lab, ha="center", fontsize=9, color=CINZA)

    # Fase 1
    box(ax, 0.6, 1.7, 3.2, 1.4,
        "FASE 1 (épocas 1–5)\nBackbone CONGELADO\nTreina só a cabeça", LARANJA_CL, LARANJA, 9.5, True)
    # Fase 2
    box(ax, 4.2, 1.7, 7.2, 1.4,
        "FASE 2 (épocas 6–30)\nRede INTEIRA treinável (fine-tuning completo)\n"
        "LR backbone = 0,1 × LR cabeça   ·   AdamW + CosineAnnealingWarmRestarts",
        VERDE_CL, VERDE, 9.5, True)
    ax.plot([4.0, 4.0], [0.6, 3.1], color=CINZA, lw=1.0, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_finetune.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 4 — Divisão por paciente
# =====================================================================
def fig_split():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    ax.text(6, 3.7, "Divisão estratificada no nível do paciente (sem vazamento)",
            ha="center", fontsize=12, fontweight="bold", color=AZUL)

    box(ax, 0.4, 2.2, 2.4, 1.0, "Casos IARC\n(1 paciente = N imagens)", AZUL_CL, AZUL, 9, True)

    # barra proporcional
    total_w = 8.2; x0 = 3.4; y0 = 2.3; h = 0.9
    parts = [("TREINO 70%", 0.70, VERDE_CL, VERDE),
             ("VAL 15%", 0.15, LARANJA_CL, LARANJA),
             ("TESTE 15%", 0.15, ROXO_CL, ROXO)]
    xx = x0
    for lab, frac, fc, ec in parts:
        w = total_w * frac
        box(ax, xx, y0, w, h, lab, fc, ec, 9, True)
        xx += w
    arrow(ax, 2.8, 2.7, 3.4, 2.7)

    ax.text(6, 1.5,
            "Estratificação pelo alvo · unidade = patient_id · "
            "load_manifest() verifica que cada paciente pertence a um único split",
            ha="center", fontsize=9.5, color=CINZA)
    ax.text(6, 0.9,
            "Agregação final por caso: a probabilidade do paciente é a média das suas imagens",
            ha="center", fontsize=9.5, color=CINZA, style="italic")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_split.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 5 — TTA 4 vistas + calibração
# =====================================================================
def fig_tta():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(6, 3.9, "Inferência: TTA de 4 vistas, calibração e decisão",
            ha="center", fontsize=12, fontweight="bold", color=AZUL)

    views = ["Original", "Flip H", "Flip V", "Flip HV"]
    x = 0.5; w = 1.25; ystk = [3.0, 2.4, 1.8, 1.2]; h = 0.5
    for lab, yy in zip(views, ystk):
        box(ax, x, yy, w, h, lab, AZUL_CL, AZUL, 8.5)
        arrow(ax, x + w, yy + h / 2, 2.5, 2.35, color="#8aa6c2")

    # média
    box(ax, 2.5, 2.0, 1.9, 0.9, "Média das\n4 vistas", VERDE_CL, VERDE, 9, True)
    arrow(ax, 4.4, 2.45, 5.0, 2.45, color=CINZA)
    box(ax, 5.0, 2.0, 2.2, 0.9, "÷ Temperatura T\n(sigmoide)", LARANJA_CL, LARANJA, 9)
    arrow(ax, 7.2, 2.45, 7.8, 2.45, color=CINZA)
    box(ax, 7.8, 2.0, 3.4, 0.9, "Limiar clínico:\nmaximiza F1 com Sens ≥ 0,80", ROXO_CL, ROXO, 9, True)

    arrow(ax, 9.5, 2.0, 9.5, 1.5, color=CINZA)
    box(ax, 6.5, 0.5, 5.2, 1.0, "Decisão por CASO:\nNegativo/baixo grau  ou  Alto grau/câncer",
        "#f2f2f2", CINZA, 8.5, True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_tta.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


for fn in (fig_pipeline, fig_head, fig_finetune, fig_split, fig_tta):
    fn()
    print("OK:", fn.__name__)
print("Diagramas gerados.")
