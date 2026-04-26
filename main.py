"""
Análise Olist — App principal
Executa os 4 módulos de análise em sequência.
Feche as janelas de cada gráfico para avançar ao próximo módulo.
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ("01_crescimento_receita.py",    "Crescimento e Receita"),
    ("02_logistica_sla.py",          "Logística e SLA"),
    ("03_comportamento_pagamentos.py","Comportamento e Pagamentos"),
    ("04_satisfacao_cliente.py",     "Satisfação do Cliente"),
]

print("=" * 55)
print("  ANÁLISE OLIST — INICIANDO")
print("=" * 55)

for i, (script, titulo) in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] {titulo}")
    print("-" * 55)
    path = os.path.join(BASE_DIR, script)
    result = subprocess.run([sys.executable, path], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n  ERRO em {script} (código {result.returncode}). Execução interrompida.")
        sys.exit(result.returncode)
    print(f"  Concluído: {script}")

print("\n" + "=" * 55)
print("  ANÁLISE COMPLETA")
print("  Gráficos em: graficos/")
print("=" * 55)
