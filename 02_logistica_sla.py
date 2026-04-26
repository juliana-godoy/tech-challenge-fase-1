from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "dados"
OUTPUT_PATH = BASE_DIR / "graficos" / "logistica_sla"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Carregamento
orders    = pd.read_csv(f"{DATA_PATH}/olist_orders_dataset.csv")
items     = pd.read_csv(f"{DATA_PATH}/olist_order_items_dataset.csv")
reviews   = pd.read_csv(f"{DATA_PATH}/olist_order_reviews_dataset.csv")
customers = pd.read_csv(f"{DATA_PATH}/olist_customers_dataset.csv")

cols_data = ['order_purchase_timestamp', 'order_approved_at',
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in cols_data:
    orders[col] = pd.to_datetime(orders[col])

# Tempos calculados
orders['tempo_entrega']  = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
orders['tempo_aprovacao']= (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600
orders['diff_entrega']   = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
orders['tempo_postagem'] = (orders['order_delivered_carrier_date'] - orders['order_approved_at']).dt.total_seconds() / 86400

# INSIGHT 1: Lead Time
plt.figure(figsize=(10, 5))
sns.histplot(orders['tempo_entrega'].dropna(), bins=50, kde=True, color='skyblue')
plt.title('Distribuição do Lead Time Total (Compra até Entrega)')
plt.xlabel('Dias')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/lead_time_distribuicao.png", dpi=150, bbox_inches="tight")
plt.show()

# INSIGHT 2: Atraso vs Review
df_atraso = pd.merge(
    orders[['order_id', 'order_delivered_customer_date', 'order_estimated_delivery_date']],
    reviews[['order_id', 'review_score']], on='order_id'
)
df_atraso['atrasado']      = df_atraso['order_delivered_customer_date'] > df_atraso['order_estimated_delivery_date']
df_atraso['status_entrega']= df_atraso['atrasado'].map({True: 'Atrasado', False: 'No Prazo'})

plt.figure(figsize=(8, 6))
sns.barplot(x='status_entrega', y='review_score', data=df_atraso, palette='viridis')
plt.title('Impacto do Atraso na Nota do Cliente')
plt.ylabel('Média Review Score')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/atraso_vs_review.png", dpi=150, bbox_inches="tight")
plt.show()

# INSIGHT 3: Desempenho por região
df_regiao = pd.merge(orders, customers[['customer_id', 'customer_state']], on='customer_id')
entrega_por_estado = df_regiao.groupby('customer_state')['tempo_entrega'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
entrega_por_estado.plot(kind='bar', color='salmon')
plt.title('Média de Dias para Entrega por Estado (SLA Real)')
plt.xlabel('Estado')
plt.ylabel('Dias Médios')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/entrega_por_estado.png", dpi=150, bbox_inches="tight")
plt.show()

# INSIGHT 4: Expectativa vs Realidade
plt.figure(figsize=(10, 6))
sns.kdeplot(orders['diff_entrega'], fill=True, color="green")
plt.axvline(0, color='red', linestyle='--')
plt.title('Performance de SLA: Dias de Antecipação (-) ou Atraso (+)', fontsize=14)
plt.xlabel('Dias em relação ao Prazo Estimado')
plt.ylabel('Densidade de Pedidos')
plt.xlim(-20, 10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/sla_performance.png", dpi=150, bbox_inches="tight")
plt.show()

# INSIGHT 5: Gargalo do Vendedor
plt.figure(figsize=(10, 5))
sns.boxplot(x=orders['tempo_postagem'], color='gold')
plt.title('Lead Time de Postagem (Eficiência do Vendedor)', fontsize=14)
plt.xlabel('Dias até a Postagem')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/lead_time_postagem.png", dpi=150, bbox_inches="tight")
plt.show()

# INSIGHT 6: Frete vs Valor do Produto por Estado
df_frete = (pd.merge(items, orders[['order_id', 'customer_id']], on='order_id')
              .merge(customers[['customer_id', 'customer_state']], on='customer_id'))
df_frete['pct_frete'] = (df_frete['freight_value'] / df_frete['price']) * 100
frete_est = df_frete.groupby('customer_state')['pct_frete'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
frete_est.plot(kind='bar', color='darkblue')
plt.title('Peso do Frete no Valor do Pedido (%) por Estado', fontsize=14)
plt.ylabel('% do Valor do Produto')
plt.xlabel('Estado do Cliente')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/frete_por_estado.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGráficos salvos em: {OUTPUT_PATH}")
