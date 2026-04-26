import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "dados"
OUTPUT_PATH = BASE_DIR / "graficos" / "satisfacao_cliente"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize":   (12, 6),
    "axes.titlesize":   14,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format

# Carregamento
orders    = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
reviews   = pd.read_csv(DATA_PATH / "olist_order_reviews_dataset.csv")
items     = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")
products  = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")
customers = pd.read_csv(DATA_PATH / "olist_customers_dataset.csv")
payments  = pd.read_csv(DATA_PATH / "olist_order_payments_dataset.csv")
print("Arquivos carregados com sucesso.")

# Limpeza
for df in [orders, reviews, items, products, customers, payments]:
    df.drop_duplicates(inplace=True)

date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

reviews["review_creation_date"]    = pd.to_datetime(reviews["review_creation_date"],    errors="coerce")
reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce")
reviews["review_comment_title"]    = reviews["review_comment_title"].fillna("").astype(str).str.strip()
reviews["review_comment_message"]  = reviews["review_comment_message"].fillna("").astype(str).str.strip()
reviews["review_score"]            = pd.to_numeric(reviews["review_score"], errors="coerce")
reviews = reviews[reviews["review_score"].between(1, 5)].copy()

products["product_category_name"] = products["product_category_name"].fillna("categoria_desconhecida")
orders_delivered = orders[orders["order_status"] == "delivered"].copy()
reviews_latest   = (reviews.sort_values("review_answer_timestamp")
                            .drop_duplicates(subset="order_id", keep="last").copy())
print("Limpeza concluída.")

# Base analítica
order_value = (items.groupby("order_id", as_index=False)
               .agg(order_value=("price", "sum"),
                    freight_value=("freight_value", "sum"),
                    total_items=("order_item_id", "count")))

def moda_categoria(serie):
    moda = serie.mode()
    return moda.iloc[0] if not moda.empty else "categoria_desconhecida"

order_category = (
    items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    .groupby("order_id", as_index=False)
    .agg(main_category=("product_category_name", moda_categoria))
)

payment_summary = (payments.groupby("order_id", as_index=False)
                   .agg(payment_value=("payment_value", "sum"),
                        installments_mean=("payment_installments", "mean")))

base = (
    orders_delivered
    .merge(reviews_latest[["order_id", "review_score", "review_comment_title",
                            "review_comment_message", "review_creation_date"]],
           on="order_id", how="inner")
    .merge(customers[["customer_id", "customer_state", "customer_city"]], on="customer_id", how="left")
    .merge(order_value,       on="order_id", how="left")
    .merge(order_category,    on="order_id", how="left")
    .merge(payment_summary,   on="order_id", how="left")
)

# Feature engineering
base["purchase_month"] = base["order_purchase_timestamp"].dt.to_period("M").astype(str)
base["delivery_days"]  = (base["order_delivered_customer_date"] - base["order_purchase_timestamp"]).dt.days
base["approval_days"]  = (base["order_approved_at"] - base["order_purchase_timestamp"]).dt.days
base["delay_days"]     = (base["order_delivered_customer_date"] - base["order_estimated_delivery_date"]).dt.days
base["is_late"]        = np.where(base["delay_days"] > 0, 1, 0)
base["freight_ratio"]  = np.where(base["order_value"] > 0, base["freight_value"] / base["order_value"], np.nan)
base["has_comment"]    = np.where(base["review_comment_message"].str.len() > 0, 1, 0)

base["delay_bucket"] = pd.cut(
    base["delay_days"],
    bins=[-999, -1, 0, 3, 7, 999],
    labels=["Antecipado", "No prazo", "1-3 dias atraso", "4-7 dias atraso", "8+ dias atraso"]
)
base["satisfaction_group"] = pd.cut(
    base["review_score"], bins=[0, 2, 3, 5],
    labels=["Insatisfeito (1-2)", "Neutro (3)", "Satisfeito (4-5)"]
)

base_delivery = base.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date",
                                     "order_estimated_delivery_date", "review_score"]).copy()
base_month    = base.dropna(subset=["purchase_month", "review_score"]).copy()

# KPIs
share_satisfied  = (base["review_score"] >= 4).mean()
share_detractors = (base["review_score"] <= 2).mean()
print(f"\nNota média: {base['review_score'].mean():.2f}")
print(f"Satisfeitos (4-5): {share_satisfied:.1%}  |  Detratores (1-2): {share_detractors:.1%}")
print(f"Índice Líquido de Satisfação: {share_satisfied - share_detractors:.1%}")
print(f"Atraso: {base_delivery['is_late'].mean():.1%}  |  Entrega média: {base_delivery['delivery_days'].mean():.1f} dias")

# Gráfico 1 — Evolução mensal da satisfação
monthly = (
    base_month.groupby("purchase_month")
    .agg(avg_review=("review_score", "mean"),
         satisfied_share=("review_score", lambda s: (s >= 4).mean()),
         orders=("order_id", "nunique"))
    .reset_index()
)
monthly["purchase_date"] = pd.to_datetime(monthly["purchase_month"])
monthly = monthly[monthly["orders"] >= 300].sort_values("purchase_date")

mapa_meses = {"01":"Jan","02":"Fev","03":"Mar","04":"Abr","05":"Mai","06":"Jun",
              "07":"Jul","08":"Ago","09":"Set","10":"Out","11":"Nov","12":"Dez"}
monthly["purchase_label"] = monthly["purchase_date"].apply(
    lambda x: f"{mapa_meses[str(x.month).zfill(2)]}/{x.year}"
)

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(monthly["purchase_label"], monthly["avg_review"],
         marker="o", linewidth=2, color="#1f77b4", label="Nota média")
ax1.set_title("Evolução mensal da satisfação do cliente", fontweight="bold")
ax1.set_xlabel("Mês da compra", fontweight="bold")
ax1.set_ylabel("Nota média", fontweight="bold")
ax1.set_ylim(3.7, 4.6)
ax1.grid(True, linestyle="--", alpha=0.4)
for x, y in zip(monthly["purchase_label"], monthly["avg_review"]):
    ax1.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, -16), textcoords="offset points",
                 ha="center", va="top", fontsize=8, fontweight="bold", color="#1f77b4",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax1.tick_params(axis="x", rotation=45)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight("bold")

ax2 = ax1.twinx()
ax2.plot(monthly["purchase_label"], monthly["satisfied_share"],
         marker="s", linestyle="--", linewidth=2, color="#2ca02c", label="% satisfeitos")
ax2.set_ylabel("% satisfeitos", fontweight="bold")
ax2.set_ylim(0.675, 0.855)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for x, y in zip(monthly["purchase_label"], monthly["satisfied_share"]):
    ax2.annotate(f"{y:.1%}", xy=(x, y), xytext=(0, 12), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2ca02c",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
for label in ax2.get_yticklabels():
    label.set_fontweight("bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
for text in legend.get_texts():
    text.set_fontweight("bold")

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "evolucao_mensal_satisfacao.png", dpi=300, bbox_inches="tight")
plt.show()

# Gráfico 2 — Impacto do atraso na satisfação
delay_summary = (
    base_delivery.groupby("delay_bucket", observed=True)
    .agg(orders=("order_id", "nunique"), avg_review=("review_score", "mean"))
    .reset_index()
)
ordem_faixas = ["Antecipado", "No prazo", "1-3 dias atraso", "4-7 dias atraso", "8+ dias atraso"]
delay_summary["delay_bucket"] = pd.Categorical(delay_summary["delay_bucket"],
                                                categories=ordem_faixas, ordered=True)
delay_summary = delay_summary.sort_values("delay_bucket")

cores = ["#2ca02c", "#2ca02c", "#f1c40f", "#f1c40f", "#d62728"]
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(delay_summary["delay_bucket"].astype(str), delay_summary["avg_review"],
              color=cores, width=0.65)
ax.set_title("Impacto do atraso na satisfação do cliente", fontweight="bold")
ax.set_xlabel("Faixa de atraso da entrega", fontweight="bold")
ax.set_ylabel("Nota média", fontweight="bold")
ax.grid(True, axis="y", linestyle="--", alpha=0.4)

min_y, max_y = delay_summary["avg_review"].min(), delay_summary["avg_review"].max()
ax.set_ylim(min_y - 0.15, max_y + 0.15)
for bar, value in zip(bars, delay_summary["avg_review"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
            f"{value:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar, n_orders, value in zip(bars, delay_summary["orders"], delay_summary["avg_review"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value - 0.08,
            f"{n_orders:,} pedidos", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.tick_params(axis="x", rotation=20)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

plt.subplots_adjust(bottom=0.2, top=0.88)
plt.savefig(OUTPUT_PATH / "atraso_vs_satisfacao.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nGráficos salvos em: {OUTPUT_PATH}")
