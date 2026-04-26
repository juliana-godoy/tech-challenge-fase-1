from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "axes.titlesize":        13,
    "axes.titleweight":      "bold",
    "axes.labelsize":        11,
    "axes.labelweight":      "bold",
    "figure.facecolor":      "white",
    "axes.facecolor":        "#F8F9FA",
    "legend.fontsize":       9,
    "legend.title_fontsize": 10,
})

CORES = {
    "credit_card": "#2196F3",
    "boleto":      "#FF9800",
    "voucher":     "#4CAF50",
    "debit_card":  "#F44336",
    "not_defined": "#9E9E9E",
}

LABELS_PT = {
    "credit_card": "Cartão de Crédito",
    "boleto":      "Boleto",
    "voucher":     "Voucher",
    "debit_card":  "Cartão de Débito",
    "not_defined": "Não Definido",
}

SEG_CORES = {
    "Champions":      "#4CAF50",
    "Clientes Leais": "#2196F3",
    "Potencial":      "#FF9800",
    "Em Risco":       "#FF5722",
    "Perdidos":       "#9E9E9E",
}

SEG_DESC = {
    "Champions":      "Recentes, frequentes e alto valor",
    "Clientes Leais": "Frequência consistente e boa recência",
    "Potencial":      "Recentes, mas pouco frequentes",
    "Em Risco":       "Não compram há algum tempo",
    "Perdidos":       "Inativos há muito tempo, baixo valor",
}

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "dados"
OUTPUT_PATH = BASE_DIR / "graficos" / "comportamento_pagamentos"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("Carregando dados...")
orders    = pd.read_csv(f"{DATA_PATH}/olist_orders_dataset.csv",
                        parse_dates=["order_purchase_timestamp"])
payments  = pd.read_csv(f"{DATA_PATH}/olist_order_payments_dataset.csv")
customers = pd.read_csv(f"{DATA_PATH}/olist_customers_dataset.csv")

orders_cust      = orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
pay_per_order    = payments.groupby("order_id")["payment_value"].sum().reset_index()
orders_full      = orders_cust.merge(pay_per_order, on="order_id", how="left")
orders_delivered = orders_full[orders_full["order_status"] == "delivered"].copy()

print(f"  Pedidos: {len(orders):,}  |  "
      f"Clientes únicos: {orders_cust['customer_unique_id'].nunique():,}  |  "
      f"Receita total: R$ {payments['payment_value'].sum():,.2f}")


# Tópico 1 — Meios de Pagamento
print("\n[Tópico 1] Meios de Pagamento...")

pay_agg = (
    payments.groupby("payment_type")
    .agg(qtd=("order_id", "count"), receita=("payment_value", "sum"))
    .reset_index()
    .sort_values("receita", ascending=True)
)
pay_agg["tipo"]    = pay_agg["payment_type"].map(LABELS_PT).fillna(pay_agg["payment_type"])
pay_agg["pct_qtd"] = pay_agg["qtd"] / pay_agg["qtd"].sum() * 100
cores_list = [CORES.get(t, "#888") for t in pay_agg["payment_type"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Tópico 1 — Meios de Pagamento Mais Utilizados",
             fontsize=15, fontweight="bold", y=1.03)

bars1 = axes[0].barh(pay_agg["tipo"], pay_agg["receita"] / 1e6, color=cores_list, height=0.55)
axes[0].set_title("Receita Total por Meio de Pagamento")
axes[0].set_xlabel("Receita (R$ Milhões)")
axes[0].set_xlim(0, pay_agg["receita"].max() / 1e6 * 1.35)
for bar, val in zip(bars1, pay_agg["receita"] / 1e6):
    axes[0].text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                 f"R$ {val:.1f}M", va="center", fontsize=10, fontweight="bold")

bars2 = axes[1].barh(pay_agg["tipo"], pay_agg["pct_qtd"], color=cores_list, height=0.55)
axes[1].set_title("Participação por Volume de Pedidos (%)")
axes[1].set_xlabel("% de Pedidos")
axes[1].set_xlim(0, pay_agg["pct_qtd"].max() * 1.45)
for bar, pct, qtd in zip(bars2, pay_agg["pct_qtd"], pay_agg["qtd"]):
    axes[1].text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%  ({qtd:,.0f} pedidos)",
                 va="center", fontsize=9, fontweight="bold")

plt.tight_layout(pad=2.5)
plt.savefig(f"{OUTPUT_PATH}/meios_pagamento.png", dpi=150, bbox_inches="tight")

cc_row = pay_agg[pay_agg["payment_type"] == "credit_card"].iloc[0]
pct_cc = cc_row["qtd"] / pay_agg["qtd"].sum() * 100
print(f"  Cartão de crédito: {pct_cc:.1f}% das transações, R$ {cc_row['receita']/1e6:.1f}M em receita.")
plt.show()


# Tópico 2 — Parcelamento no Cartão de Crédito
print("\n[Tópico 2] Parcelamento no Cartão de Crédito...")

cc      = payments[payments["payment_type"] == "credit_card"].copy()
cc_inst = cc["payment_installments"].value_counts().sort_index()
cc_inst = cc_inst[cc_inst.index <= 12]
cc_avg  = (cc[cc["payment_installments"] <= 12]
           .groupby("payment_installments")["payment_value"].mean().reset_index())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Tópico 2 — Comportamento de Parcelamento (Cartão de Crédito)",
             fontsize=15, fontweight="bold", y=1.02)

bars = axes[0].bar(cc_inst.index.astype(str), cc_inst.values,
                   color=sns.color_palette("Blues_r", 12)[:len(cc_inst)], width=0.65)
axes[0].set_title("Frequência por Número de Parcelas")
axes[0].set_xlabel("Número de Parcelas")
axes[0].set_ylabel("Quantidade de Pedidos")
axes[0].set_ylim(0, cc_inst.max() * 1.22)
for bar, val in zip(bars, cc_inst.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + cc_inst.max() * 0.015,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

bars2 = axes[1].bar(cc_avg["payment_installments"].astype(str), cc_avg["payment_value"],
                    color=sns.color_palette("Oranges_r", 12)[:len(cc_avg)], width=0.65)
axes[1].set_title("Ticket Médio por Número de Parcelas")
axes[1].set_xlabel("Número de Parcelas")
axes[1].set_ylabel("Valor Médio (R$)")
axes[1].set_ylim(0, cc_avg["payment_value"].max() * 1.22)
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + cc_avg["payment_value"].max() * 0.015,
                 f"R${bar.get_height():.0f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")

plt.tight_layout(pad=2.5)
plt.savefig(f"{OUTPUT_PATH}/parcelamentos.png", dpi=150, bbox_inches="tight")

pct_1x  = cc_inst.get(1, 0) / cc_inst.sum() * 100
avg_12x = cc_avg[cc_avg["payment_installments"] == 12]["payment_value"].values
avg_1x  = cc_avg[cc_avg["payment_installments"] == 1]["payment_value"].values
print(f"  {pct_1x:.1f}% pagos à vista; 12x têm ticket ~{avg_12x[0]/avg_1x[0]:.1f}x maior.")
plt.show()


# Tópico 3 — Segmentação RFM
print("\n[Tópico 3] Segmentação RFM de Clientes...")

ref_date = orders_delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
rfm = (
    orders_delivered.groupby("customer_unique_id")
    .agg(
        recencia   =("order_purchase_timestamp", lambda x: (ref_date - x.max()).days),
        frequencia =("order_id", "count"),
        monetario  =("payment_value", "sum"),
    )
    .reset_index()
)
rfm["R"]   = pd.qcut(rfm["recencia"],  4, labels=[4, 3, 2, 1]).astype(int)
rfm["F"]   = pd.qcut(rfm["frequencia"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M"]   = pd.qcut(rfm["monetario"], 4, labels=[1, 2, 3, 4]).astype(int)
rfm["RFM"] = rfm["R"] + rfm["F"] + rfm["M"]

def segmento(row):
    if row["RFM"] >= 10: return "Champions"
    if row["RFM"] >= 8:  return "Clientes Leais"
    if row["RFM"] >= 6:  return "Potencial"
    if row["RFM"] >= 4:  return "Em Risco"
    return "Perdidos"

rfm["segmento"] = rfm.apply(segmento, axis=1)

seg_order = ["Champions", "Clientes Leais", "Potencial", "Em Risco", "Perdidos"]
seg_cts   = rfm["segmento"].value_counts().reindex(seg_order).dropna().astype(int)
seg_mon   = rfm.groupby("segmento")["monetario"].mean().reindex(seg_cts.index)
cores_seg = [SEG_CORES[s] for s in seg_cts.index]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Tópico 3 — Segmentação RFM (Recência · Frequência · Monetário)",
             fontsize=15, fontweight="bold", y=1.02)

bars = axes[0].bar(seg_cts.index, seg_cts.values, color=cores_seg, width=0.6)
axes[0].set_title("Quantidade de Clientes por Segmento")
axes[0].set_xlabel("Segmento")
axes[0].set_ylabel("Quantidade de Clientes")
axes[0].set_ylim(0, seg_cts.max() * 1.55)
axes[0].tick_params(axis="x", rotation=20)
for bar, val in zip(bars, seg_cts.values):
    pct = val / seg_cts.sum() * 100
    cx, top = bar.get_x() + bar.get_width() / 2, bar.get_height()
    axes[0].text(cx, top + seg_cts.max() * 0.02,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].text(cx, top + seg_cts.max() * 0.10,
                 f"({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color="#444444")

bars2 = axes[1].bar(seg_mon.index, seg_mon.values, color=cores_seg, width=0.6)
axes[1].set_title("Gasto Médio por Segmento RFM")
axes[1].set_xlabel("Segmento")
axes[1].set_ylabel("Gasto Médio (R$)")
axes[1].set_ylim(0, seg_mon.max() * 1.22)
axes[1].tick_params(axis="x", rotation=20)
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + seg_mon.max() * 0.02,
                 f"R${bar.get_height():.0f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

legend_patches = [
    mpatches.Patch(color=SEG_CORES[s], label=f"{s}: {SEG_DESC[s]}")
    for s in seg_cts.index
]
fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           ncol=3, title="Legenda dos Segmentos", title_fontsize=9,
           fontsize=8.5, framealpha=0.92, edgecolor="#cccccc")

plt.tight_layout(pad=2.5)
plt.subplots_adjust(bottom=0.30)
plt.savefig(f"{OUTPUT_PATH}/rfm_segmentacao.png", dpi=150, bbox_inches="tight")

champ_pct = seg_cts.get("Champions", 0) / seg_cts.sum() * 100
perd_pct  = seg_cts.get("Perdidos",  0) / seg_cts.sum() * 100
risco_pct = seg_cts.get("Em Risco",  0) / seg_cts.sum() * 100
print(f"  {champ_pct:.1f}% Champions; {perd_pct+risco_pct:.1f}% Em Risco ou Perdidos.")
plt.show()


# Tópico 4 — Recompra, Retenção e Monetização
print("\n[Tópico 4] Recompra e Retenção de Clientes...")

n_pedidos = (orders_cust.groupby("customer_unique_id")["order_id"]
             .count().reset_index(name="n_pedidos"))
total = len(n_pedidos)

grp      = n_pedidos["n_pedidos"].clip(upper=5).value_counts().sort_index()
labels_x = [str(k) if k < 5 else "5+" for k in grp.index]
vals     = grp.values.tolist()

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Tópico 4 — Recompra, Retenção e Monetização de Clientes",
             fontsize=15, fontweight="bold", y=1.02)

cmap = plt.cm.Blues
bars = ax.bar(range(len(labels_x)), vals, width=0.62,
              color=[cmap(0.3 + 0.65 * v / max(vals)) for v in vals],
              edgecolor="white", linewidth=1.2)
ax.plot(range(len(labels_x)), vals, color="steelblue", linewidth=1.5,
        marker="o", markersize=4, zorder=5)

ax.set_title("Distribuição de Clientes por Número de Pedidos Realizados")
ax.set_xlabel("Número de Pedidos por Cliente")
ax.set_ylabel("Quantidade de Clientes")
ax.set_xticks(range(len(labels_x)))
ax.set_xticklabels(labels_x, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.22)

for bar, val in zip(bars, vals):
    pct = val / total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + max(vals) * 0.012,
            f"{val:,.0f}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

recompra = (n_pedidos["n_pedidos"] >= 2).sum() / total * 100
ax.annotate(
    f"Taxa de recompra (≥ 2 pedidos): {recompra:.1f}%",
    xy=(0.98, 0.95), xycoords="axes fraction",
    ha="right", va="top", fontsize=11, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#E3F2FD",
              edgecolor="#2196F3", alpha=0.92),
)

plt.tight_layout(pad=2.5)
plt.savefig(f"{OUTPUT_PATH}/recompra_retencao.png", dpi=150, bbox_inches="tight")
print(f"  Taxa de recompra: {recompra:.1f}% — oportunidade para programas de fidelidade.")
plt.show()

print(f"\nGráficos salvos em: {OUTPUT_PATH}")
