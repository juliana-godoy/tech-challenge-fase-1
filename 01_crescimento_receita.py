from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "dados"
OUTPUT_PATH = BASE_DIR / "graficos" / "crescimento_receita"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Carregamento
orders   = pd.read_csv(f"{DATA_PATH}/olist_orders_dataset.csv")
payments = pd.read_csv(f"{DATA_PATH}/olist_order_payments_dataset.csv")
products = pd.read_csv(f"{DATA_PATH}/olist_products_dataset.csv")
items    = pd.read_csv(f"{DATA_PATH}/olist_order_items_dataset.csv")
customers= pd.read_csv(f"{DATA_PATH}/olist_customers_dataset.csv")

orders_payments = pd.merge(orders, payments, on='order_id', how='inner')
orders_payments['order_purchase_timestamp'] = pd.to_datetime(orders_payments['order_purchase_timestamp'])
orders_payments['order_purchase_date']  = orders_payments['order_purchase_timestamp'].dt.date
orders_payments['order_purchase_month'] = orders_payments['order_purchase_timestamp'].dt.to_period('M')

daily_revenue = orders_payments.groupby('order_purchase_date')['payment_value'].sum().reset_index()
daily_revenue['order_purchase_date'] = pd.to_datetime(daily_revenue['order_purchase_date'])

monthly_revenue = orders_payments.groupby('order_purchase_month')['payment_value'].sum().reset_index()
monthly_revenue['order_purchase_month'] = monthly_revenue['order_purchase_month'].dt.to_timestamp()

def fmt_milhoes(x, pos):
    return f"R${x/1_000_000:,.1f}M"

formatter = mticker.FuncFormatter(fmt_milhoes)

# Receita diária
plt.figure(figsize=(15, 7))
sns.lineplot(x='order_purchase_date', y='payment_value', data=daily_revenue)
plt.title('Daily Revenue Over Time', fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Total Revenue (R$)', fontweight='bold')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/receita_diaria.png", dpi=150, bbox_inches="tight")
plt.show()

# Distribuição da receita diária
plt.figure(figsize=(10, 6))
sns.boxplot(y='payment_value', data=daily_revenue)
plt.title('Distribution of Daily Revenue', fontweight='bold')
plt.ylabel('Daily Revenue (R$)', fontweight='bold')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/distribuicao_receita_diaria.png", dpi=150, bbox_inches="tight")
plt.show()

# Receita mensal
plt.figure(figsize=(15, 7))
sns.lineplot(x='order_purchase_month', y='payment_value', data=monthly_revenue,
             marker='o', color='skyblue', linewidth=2.5)
plt.fill_between(monthly_revenue['order_purchase_month'], monthly_revenue['payment_value'],
                 color='skyblue', alpha=0.3)
plt.title('Receita Mensal ao Longo do Tempo', fontsize=18, fontweight='bold')
plt.xlabel('Mês', fontsize=14, fontweight='bold')
plt.ylabel('Total de Receita (Milhões R$)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().yaxis.set_major_formatter(formatter)
for _, row in monthly_revenue.iterrows():
    plt.text(row['order_purchase_month'], row['payment_value'],
             f"R${row['payment_value']/1_000_000:,.1f}M",
             color='black', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/receita_mensal.png", dpi=150, bbox_inches="tight")
plt.show()

# Top 10 categorias por receita
order_product_details = pd.merge(items, products, on='product_id', how='left')
top10 = (order_product_details.groupby('product_category_name')['price']
         .sum().reset_index().sort_values('price', ascending=False).head(10))

plt.figure(figsize=(12, 7))
ax = sns.barplot(x='product_category_name', y='price', data=top10,
                 palette='magma', hue='product_category_name', legend=False)
plt.title('Top 10 Produtos por Receita', fontsize=16, fontweight='bold')
plt.xlabel('Nome da Categoria do Produto', fontsize=14, fontweight='bold')
plt.ylabel('Receita Total (Milhões R$)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticklabels([l.get_text().replace('_', ' ') for l in ax.get_xticklabels()])
ax.yaxis.set_major_formatter(formatter)
for p in ax.patches:
    ax.annotate(f'R${p.get_height()/1_000_000:,.1f}M',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9),
                textcoords='offset points', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/top10_categorias.png", dpi=150, bbox_inches="tight")
plt.show()

# Receita por estado
customers_revenue = pd.merge(orders_payments, customers, on='customer_id', how='left')
revenue_by_state = (customers_revenue.groupby('customer_state')['payment_value']
                    .sum().reset_index().sort_values('payment_value', ascending=False))

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='payment_value', y='customer_state', data=revenue_by_state,
                 palette='YlGn_r', hue='customer_state', legend=False)
plt.title('Contribuição de Receita por Estado (Escala Logarítmica)', fontsize=16, fontweight='bold')
plt.xlabel('Receita Total (R$, Escala Logarítmica)', fontsize=14, fontweight='bold')
plt.ylabel('Estado', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(axis='x')
for p in ax.patches:
    width = p.get_width()
    plt.text(width * 1.02, p.get_y() + p.get_height() / 2,
             f'R${width:,.0f}', ha='left', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/receita_por_estado.png", dpi=150, bbox_inches="tight")
plt.show()

total_revenue  = daily_revenue['payment_value'].sum()
num_orders     = orders_payments['order_id'].nunique()
average_ticket = total_revenue / num_orders
print(f"Total Revenue: R${total_revenue:,.2f}")
print(f"Unique Orders: {num_orders:,}")
print(f"Average Ticket: R${average_ticket:,.2f}")
print(f"\nGráficos salvos em: {OUTPUT_PATH}")
