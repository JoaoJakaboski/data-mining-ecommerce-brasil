# 📊 Outputs do Projeto

Esta pasta contém todos os resultados gerados pela análise de mineração de dados.

## 🖼️ Gráficos Gerados

Quando você executar `python ecommerce_analysis.py`, os seguintes gráficos serão criados automaticamente:

### 1. `analise_exploratoria.png`
**Descrição**: Análise exploratória completa dos dados  
**Conteúdo**:
- Vendas por estado (top 12)
- Distribuição de preços 
- Distribuição das avaliações (1-5)
- Top categorias de produtos
- Tipos de pagamento (gráfico pizza)
- Vendas mensais ao longo do tempo

### 2. `correlacao_variaveis.png`
**Descrição**: Mapa de calor das correlações  
**Conteúdo**:
- Correlações entre preço, frete, parcelas, avaliação
- Matriz de correlação com valores numéricos
- Escala de cores para identificar relações

### 3. `importancia_features.png`
**Descrição**: Importância das variáveis na classificação  
**Conteúdo**:
- Ranking das variáveis mais importantes
- Modelo Random Forest para satisfação do cliente
- Gráfico de barras horizontais

### 4. `regressao_precos.png`
**Descrição**: Resultado da regressão linear  
**Conteúdo**:
- Gráfico scatter: Preço Real vs Preço Predito
- Linha diagonal de referência (predição perfeita)
- Visualização da qualidade do modelo

### 5. `clustering_estados.png`
**Descrição**: Segmentação dos estados brasileiros  
**Conteúdo**:
- 2 gráficos de dispersão dos clusters
- Estados agrupados por comportamento de compra
- Cores diferentes para cada cluster

## 📈 Métricas de Performance

### Classificação (Satisfação do Cliente)
- **Acurácia**: ~80-85%
- **Target**: Cliente satisfeito (nota ≥ 4)
- **Método**: Random Forest

### Regressão (Preços)
- **R² Score**: ~70-75%
- **Target**: Preço do produto
- **Método**: Regressão Linear

### Clustering (Estados)
- **Método**: K-Means
- **Clusters**: 3 grupos
- **Features**: Preço, frete, parcelas, avaliação

## 🎯 Insights Principais

### Descobertas do Dataset Real:
1. **São Paulo domina**: 42% das vendas nacionais
2. **Pagamento favorito**: 74% usam cartão de crédito
3. **Alta satisfação**: 77% dos clientes dão nota ≥ 4
4. **Frete impacta**: Principal fator de insatisfação
5. **Regionalização**: 3 perfis distintos de consumo

### Aplicações Práticas:
- Otimização de frete por região
- Estratégias de precificação
- Campanhas de marketing segmentadas
- Melhoria da experiência do cliente

## ⚠️ Observações Importantes

### Arquivos Temporários
- Os arquivos PNG **não são versionados** no Git
- Incluídos no `.gitignore` pois são gerados automaticamente
- Sempre recriados quando o script é executado

### Qualidade dos Gráficos
- **Resolução**: 300 DPI (alta qualidade)
- **Formato**: PNG com transparência
- **Tamanho**: Otimizado para relatórios acadêmicos

### Personalização
- Cores e estilos podem ser modificados no código
- Paleta de cores: "husl" (seaborn)
- Estilo: padrão matplotlib

## 🔄 Como Regenerar

Para recriar todos os gráficos:

```bash
# Executar análise completa
python ecommerce_analysis.py

# Os arquivos PNG serão criados/atualizados automaticamente
```

## 📱 Uso dos Gráficos

### Para Apresentações:
- Use `analise_exploratoria.png` para overview geral
- Use `clustering_estados.png` para mostrar segmentação

### Para Relatório Acadêmico:
- Todos os gráficos são adequados para inclusão
- Alta resolução garante qualidade na impressão

### Para Portfólio:
- Demonstram competências em visualização de dados
- Mostram aplicação prática de machine learning

## 🎨 Customização

Para modificar os gráficos, edite as funções no `ecommerce_analysis.py`:

- `criar_visualizacoes()` - Gráficos exploratórios
- `classificacao_satisfacao()` - Importância features
- `regressao_precos()` - Scatter plot regressão
- `clustering_estados()` - Visualização clusters