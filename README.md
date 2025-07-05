# 🛒 Projeto de Mineração de Dados - E-commerce Brasileiro

## 📋 Descrição do Projeto

Este projeto analisa dados **reais** do e-commerce brasileiro usando o dataset da Olist, aplicando técnicas de **classificação**, **regressão** e **clustering** para extrair insights sobre o comportamento dos consumidores brasileiros.

## 🎯 Objetivos

- Analisar padrões de compra no e-commerce brasileiro
- Prever satisfação do cliente usando classificação
- Estimar preços de produtos através de regressão
- Segmentar estados por comportamento de compra
- Gerar insights acionáveis para o mercado brasileiro

## 📊 Dataset

**Nome**: Brazilian E-Commerce Public Dataset by Olist  
**Origem**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
**Registros**: 100.000+ pedidos reais  
**Período**: 2016-2018  
**Cobertura**: Todos os estados brasileiros  

### Principais Variáveis:
- `order_id`: ID do pedido
- `customer_state`: Estado do cliente
- `product_category_name`: Categoria do produto
- `price`: Preço do produto
- `freight_value`: Valor do frete
- `payment_type`: Tipo de pagamento
- `payment_installments`: Número de parcelas
- `review_score`: Avaliação (1-5)
- `order_status`: Status do pedido

## 🔧 Tecnologias Utilizadas

- **Python 3.8+**
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **matplotlib**: Visualizações
- **seaborn**: Gráficos estatísticos
- **scikit-learn**: Machine learning

## 📈 Técnicas Aplicadas

### 1. 🎯 Classificação (Random Forest)
- **Objetivo**: Prever satisfação do cliente
- **Target**: Cliente satisfeito (avaliação ≥ 4)
- **Features**: preço, frete, parcelas, valor total

### 2. 📊 Regressão Linear
- **Objetivo**: Prever preços dos produtos
- **Features**: categoria, estado, tipo pagamento, parcelas, frete

### 3. 🔍 Clustering (K-Means)
- **Objetivo**: Segmentar estados por comportamento
- **Features**: preço médio, frete médio, parcelas, avaliação
- **Resultado**: 3 grupos de estados

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Passo a Passo

1. **Baixe o dataset:**
   - Acesse: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
   - Baixe o arquivo ZIP
   - Extraia os arquivos CSV na pasta do projeto

2. **Execute a análise:**
```bash
python ecommerce_analysis.py
```

3. **Arquivos gerados:**
   - `analise_exploratoria.png`
   - `correlacao_variaveis.png`
   - `importancia_features.png`
   - `regressao_precos.png`
   - `clustering_estados.png`

## 📊 Principais Resultados

### Análise Exploratória
- **São Paulo**: 42% das vendas nacionais
- **Cartão de crédito**: 74% das transações
- **Satisfação média**: 4.1/5.0 (77% satisfeitos)
- **Ticket médio**: R$ 120,00

### Classificação - Satisfação
- **Acurácia**: ~80-85%
- **Principal fator**: Valor do frete
- **Insight**: Frete alto reduz satisfação

### Regressão - Preços
- **R² Score**: ~0.70-0.75
- **Principais fatores**: Categoria e estado
- **Insight**: Produtos de tecnologia são mais caros

### Clustering - Estados
- **Grupo 1**: SP, RJ, MG (mercados maduros)
- **Grupo 2**: Sul (mercados premium)
- **Grupo 3**: Norte/Nordeste (mercados emergentes)

## 📁 Estrutura do Projeto

```
ecommerce-data-mining-brasil/
├── data/
├── docs/
├── outputs/
├── .gitignore                   # Arquivos a ignorar
├── ecommerce_analysis.py        # Código principal
├── requirements.txt             # Dependências Python
└── README.md                    # Documentação principal  
```

## 🎓 Aplicações Práticas

1. **Segmentação de Marketing**: Campanhas por região
2. **Otimização de Preços**: Modelo de precificação dinâmica  
3. **Melhoria de Satisfação**: Redução de custos de frete
4. **Expansão Geográfica**: Priorização de mercados

## 👨‍💻 Autor

**João Carlos Jakaboski**