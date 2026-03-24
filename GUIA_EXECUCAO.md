# Guia de Execução — Tech Challenge Fase 3

## Pré-requisitos

- **Python 3.10+** instalado (recomendado: 3.11)
- **VS Code** com extensão Jupyter instalada
- Os arquivos CSV (`flights.csv`, `airlines.csv`, `airports.csv`) já estão em `data/raw/`
- **WSL (Windows Subsystem for Linux)** ou terminal Linux/Mac

---

## Passo 1 — Criar o ambiente virtual e instalar dependências

Abra o **terminal WSL** no VS Code (Ctrl+`) e execute:

```bash
cd ~/TC3    # ou o caminho do projeto no seu WSL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Nota Windows PowerShell** (alternativa sem WSL):
> ```powershell
> cd C:\Users\Computador\Documents\TC3
> python -m venv venv
> .\venv\Scripts\Activate.ps1
> pip install -r requirements.txt
> ```

Confirme que tudo instalou com:

```bash
pip list
```

Você deve ver: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, plotly, kaleido, jupyter, joblib.

---

## Passo 2 — Executar o pipeline completo via script

O `main.py` executa tudo de uma vez: EDA + Classificação + Regressão + Clustering + PCA.

```bash
cd ~/TC3    # ou o caminho do projeto
source venv/bin/activate
python main.py
```

> **Windows PowerShell** (alternativa):
> ```powershell
> cd C:\Users\Computador\Documents\TC3
> .\venv\Scripts\Activate.ps1
> python main.py
> ```

### O que acontece:
1. Carrega e limpa os dados (~5.8M voos → ~5.3M após remover cancelados/desviados)
2. Cria features derivadas (período do dia, estação, feriados, congestionamento, etc.)
3. Gera gráficos de EDA (overview, temporal, correlação, tipos de atraso, estados, mapa geográfico)
4. Treina 3 classificadores com cross-validation (Random Forest, XGBoost, Logistic Regression)
5. Treina 2 regressores com cross-validation (Random Forest, XGBoost)
6. Executa KMeans + DBSCAN em perfis de aeroportos
7. Executa PCA nos dados numéricos
8. Executa Detecção de Anomalias (Isolation Forest)
9. Imprime resumo final com conclusões

### Tempo estimado: ~15-30 minutos
(depende do hardware; o gargalo é o treinamento de modelos)

### Resultados gerados:
| Arquivo | Descrição |
|---------|-----------|
| `outputs/figures/eda_overview.png` | Distribuição de atrasos, top aeroportos, companhias, dia da semana |
| `outputs/figures/eda_temporal.png` | Padrões por hora, mês, heatmap dia×hora |
| `outputs/figures/eda_correlation.png` | Matriz de correlação |
| `outputs/figures/eda_delay_types.png` | Composição dos tipos de atraso |
| `outputs/figures/eda_states.png` | Atrasos por estado (top 20) |
| `outputs/figures/mapa_atrasos.html` | Mapa geográfico interativo (Plotly) |
| `outputs/figures/mapa_atrasos.png` | Mapa geográfico estático |
| `outputs/figures/classification_results.png` | Métricas, ROC, matriz de confusão |
| `outputs/figures/feature_importance_clf.png` | Importância das features (classificação) |
| `outputs/figures/regression_results.png` | MAE/RMSE, real vs previsto, resíduos |
| `outputs/figures/clustering_results.png` | Elbow, silhouette, clusters PCA 2D, heatmap |
| `outputs/figures/pca_variance.png` | Variância explicada e loadings |
| `outputs/figures/anomaly_detection.png` | Score de anomalia, PCA 2D anomalias, distribuição |
| `outputs/models/best_classifier.pkl` | Melhor modelo de classificação serializado |
| `outputs/models/best_regressor.pkl` | Melhor modelo de regressão serializado |

---

## Passo 3 — Executar os notebooks (alternativa interativa)

Os notebooks permitem executar passo a passo e ver os resultados inline.

### Configurar o kernel do notebook

1. Abra qualquer notebook no VS Code
2. No canto superior direito, clique em **"Select Kernel"**
3. Escolha **"Python Environments"** → selecione:
   - **WSL/Linux**: `./venv/bin/python`
   - **Windows**: `.\venv\Scripts\python.exe`

### Ordem de execução dos notebooks

#### Notebook 1 — EDA
Abra `notebooks/01_EDA.ipynb` e execute todas as células (Ctrl+Shift+Enter ou "Run All").

**O que você verá**:
- Shape do dataset e estatísticas descritivas
- Distribuição da variável alvo (IS_DELAYED)
- Histogramas, boxplots, análise de nulos
- Top companhias e aeroportos por atraso
- Padrões temporais (hora, dia, mês, estação)
- Heatmap dia da semana × hora de partida
- Matriz de correlação
- Composição dos tipos de atraso
- Análise de features derivadas

#### Notebook 2 — Modelagem Supervisionada
Abra `notebooks/02_supervised.ipynb` e execute todas as células.

**O que você verá**:
- Verificação das features categóricas e numéricas
- Treinamento de 3 classificadores com métricas detalhadas
- Curvas ROC comparativas
- Matriz de confusão do melhor modelo
- Feature importance
- Tabela comparativa de classificação (com destaque para melhores métricas)
- Treinamento de 2 regressores
- Gráfico Real vs Previsto
- Distribuição de resíduos
- Tabela comparativa de regressão
- Análise crítica com interpretação e limitações

#### Notebook 3 — Modelagem Não Supervisionada
Abra `notebooks/03_unsupervised.ipynb` e execute todas as células.

**O que você verá**:
- Clusterização KMeans com método do cotovelo e silhouette
- Comparação com DBSCAN (detecção de outliers)
- Visualização PCA 2D dos clusters
- Heatmap de perfis médios dos clusters
- Perfil detalhado por cluster (tabela)
- Análise PCA completa com variância explicada
- Tabela de loadings colorida
- Detecção de Anomalias com Isolation Forest
- Análise crítica com interpretação e limitações

---

## Passo 4 — Coletar os resultados para apresentação

### Figuras
Todas estão em `outputs/figures/`. Copie-as para sua apresentação.

### Modelos salvos
- `outputs/models/best_classifier.pkl` — Para reusar o classificador
- `outputs/models/best_regressor.pkl` — Para reusar o regressor

### Métricas-chave para citar na apresentação

**Classificação** (valores preenchidos após execução):
- Melhor modelo: (nome aparecerá no console)
- AUC-ROC: (valor aparecerá no console)
- CV AUC-ROC (3-fold): média ± desvio
- Acurácia, Precision, Recall, F1

**Regressão**:
- Melhor modelo: (nome aparecerá no console)
- MAE: X minutos
- CV MAE (3-fold): média ± desvio
- RMSE: X minutos
- R²: X

**Clustering**:
- Número ótimo de clusters (KMeans): k
- Clusters DBSCAN: n
- Silhouette score

**PCA**:
- Componentes para 90% de variância: n

**Detecção de Anomalias**:
- % de voos anômalos identificados

---

## Resolução de Problemas

### "ModuleNotFoundError: No module named 'xgboost'"
```bash
source venv/bin/activate
pip install xgboost
```

### "FileNotFoundError: flights.csv"
Verifique que os 3 CSVs estão em `data/raw/`:
```bash
ls -la data/raw/
```

### Notebook não encontra os módulos `src`
Certifique-se de que a **primeira célula** do notebook contém:
```python
import sys
sys.path.insert(0, "..")
```
E que o kernel selecionado é o do venv do projeto.

### Kernel do notebook não aparece
```bash
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=tc3 --display-name "Python (TC3)"
```

### MemoryError durante treinamento
O dataset é grande (~5.8M linhas). Os scripts já fazem amostragem automática, mas se ocorrer problemas de memória, feche outros programas e tente novamente.

---

## Estrutura do Repositório

```
TC3/
├── data/
│   ├── raw/              # Dados brutos (flights.csv, airlines.csv, airports.csv)
│   └── processed/        # (reservado para dados processados)
├── notebooks/
│   ├── 01_EDA.ipynb              # Análise Exploratória (40 células)
│   ├── 02_supervised.ipynb       # Classificação + Regressão (14 células)
│   └── 03_unsupervised.ipynb     # Clustering + PCA (14 células)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Carregamento e limpeza
│   ├── feature_engineering.py    # Engenharia de features
│   ├── supervised.py             # Classificação e regressão
│   └── unsupervised.py           # KMeans e PCA
├── outputs/
│   ├── figures/          # Gráficos gerados
│   └── models/           # Modelos serializados (.pkl)
├── main.py               # Pipeline completo (orquestra tudo)
├── requirements.txt      # Dependências Python
├── README.md             # Documentação do projeto
└── GUIA_EXECUCAO.md      # Este guia
```
