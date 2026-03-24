# Tech Challenge Fase 3 — Flight Delays and Cancellations

## Descrição

Projeto de Machine Learning aplicado ao conjunto de dados público de voos domésticos nos EUA (2015), com o objetivo de analisar atrasos de voos e construir modelos preditivos (supervisionados) e exploratórios (não supervisionados).

O dataset contém informações detalhadas sobre mais de 5 milhões de voos, incluindo horários programados e reais, atrasos, companhias aéreas, aeroportos de origem e destino, entre outros.

## Estrutura do Projeto

```
TC3/
├── data/
│   ├── raw/                    # Dados brutos (CSVs originais)
│   │   ├── airlines.csv        # 14 companhias aéreas
│   │   ├── airports.csv        # ~320 aeroportos
│   │   └── flights.csv         # ~5.8M voos
│   └── processed/              # Dados processados (gerados em runtime)
├── notebooks/
│   ├── 01_EDA.ipynb            # Análise Exploratória de Dados
│   ├── 02_supervised.ipynb     # Modelagem Supervisionada
│   └── 03_unsupervised.ipynb   # Modelagem Não Supervisionada
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carregamento e limpeza dos dados
│   ├── feature_engineering.py  # Criação de variáveis derivadas
│   ├── supervised.py           # Pipeline supervisionado (Classificação + Regressão + Cross-Validation)
│   └── unsupervised.py         # Pipeline não supervisionado (KMeans + DBSCAN + PCA + Anomaly Detection)
├── outputs/
│   ├── figures/                # Gráficos gerados
│   └── models/                 # Modelos treinados salvos (.pkl)
├── main.py                     # Script principal — executa o pipeline completo
├── requirements.txt            # Dependências Python
└── README.md
```

## Etapas do Projeto

### 1. Exploração dos Dados (EDA)

- Estatísticas descritivas e distribuição de atrasos
- Análise de atrasos por companhia aérea, aeroporto, dia da semana e horário
- Heatmap temporal (dia da semana × hora de partida)
- Matriz de correlação entre variáveis numéricas
- Composição dos tipos de atraso (companhia, clima, sistema aéreo, segurança, aeronave)
- **Análise por estado** — ranking de atrasos por unidade federativa
- **Mapa geográfico interativo** (Plotly) — atrasos por aeroporto no mapa dos EUA
- **Análise de congestionamento** — impacto do volume de voos simultâneos em atrasos

### 2. Modelagem Supervisionada

- **Classificação** — Prever se um voo vai atrasar mais de 15 minutos
  - Algoritmos: Random Forest, XGBoost, Logistic Regression
  - Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
  - **Cross-validation 3-fold** para validação de robustez
  - Curvas ROC comparativas e matriz de confusão
  - Feature importance do melhor modelo

- **Regressão** — Prever o tempo de atraso em minutos (para voos atrasados)
  - Algoritmos: Random Forest Regressor, XGBoost Regressor
  - Métricas: MAE, RMSE, R²
  - **Cross-validation 3-fold** para validação de robustez
  - Gráfico Real vs Previsto e distribuição de resíduos

### 3. Modelagem Não Supervisionada

- **Clusterização (KMeans)** — Agrupamento de aeroportos por perfil operacional
  - Método do cotovelo e silhouette score para seleção de k
  - Visualização dos clusters via PCA 2D
  - Heatmap do perfil médio por cluster

- **DBSCAN (comparativo)** — Clusterização baseada em densidade
  - Detecção automática do número de clusters e outliers
  - Comparação com KMeans

- **PCA (Análise de Componentes Principais)** — Redução de dimensionalidade
  - Variância explicada acumulada
  - Loadings das componentes principais (PC1 e PC2)

- **Detecção de Anomalias (Isolation Forest)** — Identificar voos anômalos
  - Perfil comparativo: anomalias vs voos normais
  - Visualização em espaço PCA 2D
  - Distribuição de scores de anomalia

### 4. Feature Engineering

Variáveis derivadas criadas para enriquecer os modelos:

| Feature | Descrição |
|---------|-----------|
| `PERIODO_DIA` | Manhã, tarde, noite ou madrugada |
| `DEP_HOUR` | Hora de partida programada (0–23) |
| `ESTACAO` | Estação do ano (inverno, primavera, verão, outono) |
| `IS_WEEKEND` | Flag de fim de semana (sábado/domingo) |
| `IS_HOLIDAY` | Flag de feriado federal dos EUA |
| `NEAR_HOLIDAY` | Flag de proximidade a feriado (±1 dia) |
| `FLIGHTS_SAME_HOUR_ORIGIN` | Voos na mesma hora/aeroporto (proxy de congestionamento) |
| `ROUTE_POPULARITY` | Total de voos na mesma rota (popularidade) |
| `IS_DELAYED` | Target binário: atraso na chegada ≥ 15 minutos |

## Como Executar

```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate            # Linux / WSL / Mac
# .\venv\Scripts\Activate.ps1      # Windows PowerShell

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o pipeline completo (EDA + Supervisionado + Não Supervisionado)
python main.py
```

Os gráficos serão salvos em `outputs/figures/` e os modelos em `outputs/models/`.

Para explorar passo a passo, utilize os notebooks em `notebooks/`.

## Tecnologias

- **Python 3.10+**
- **pandas / numpy** — Manipulação de dados
- **matplotlib / seaborn** — Visualizações estáticas
- **Plotly** — Mapa geográfico interativo
- **scikit-learn** — Modelos de ML, métricas, PCA, KMeans, DBSCAN, Isolation Forest
- **XGBoost** — Gradient Boosting otimizado
- **joblib** — Serialização de modelos
- **kaleido** — Exportação de imagens Plotly

## Conclusões

- Atrasos são mais frequentes no final da tarde e à noite, e nos meses de verão (junho/julho)
- As companhias com maior índice de atraso variam significativamente
- É possível agrupar aeroportos em perfis distintos (hubs de alto tráfego vs regionais)
- DBSCAN complementa o KMeans identificando aeroportos outliers que não se encaixam em nenhum cluster
- Modelos de classificação conseguem identificar voos com risco elevado de atraso, validados por cross-validation
- A previsão exata de minutos de atraso é mais desafiadora (alta variabilidade)
- O congestionamento no aeroporto de origem é um preditor relevante de atrasos
- Isolation Forest detecta ~5% dos voos como anômalos — com perfil de atrasos extremos
- Mapa geográfico revela concentração de atrasos no Nordeste (NY, NJ) e Chicago

## Limitações

- Dados referentes apenas ao ano de 2015 (doméstico EUA) — sem garantia de generalização temporal
- Sem dados meteorológicos em tempo real ou eventos especiais
- Colunas detalhadas de tipo de atraso só existem para voos efetivamente atrasados (potencial data leakage)
- Amostragem utilizada no treino para viabilizar performance em datasets grandes
- KMeans assume clusters esféricos; DBSCAN mitiga parcialmente
- Desbalanceamento de classes (~73/27) pode gerar viés para classe majoritária

## Próximos Passos

- Incorporar dados meteorológicos externos (temperatura, precipitação, neve)
- Testar modelos de deep learning (redes neurais, LSTM para séries temporais)
- Criar dashboard interativo com Streamlit ou Plotly Dash
- Aplicar técnicas de balanceamento de classes (SMOTE, class_weight)
- Feature engineering avançado (atraso médio histórico por rota/aeroporto)
- Explorar aprendizado semi-supervisionado com pseudo-labels
- Análise de séries temporais para identificar tendências de atraso ao longo do ano

## Dataset

[Flight Delays and Cancellations — Kaggle / U.S. DOT](https://www.kaggle.com/usdot/flight-delays)

## Autores

[Seu nome / Grupo]
