# 📊 Dados do Projeto

## Dataset Utilizado

**Nome**: Brazilian E-Commerce Public Dataset by Olist  
**Fonte**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
**Tamanho**: ~32 MB (compactado)  
**Registros**: 100.000+ pedidos reais  
**Período**: 2016-2018  

## Como Obter os Dados

### Passo 1: Acessar o Kaggle
1. Acesse: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Faça login no Kaggle (criar conta gratuita se necessário)
3. Clique no botão **"Download"** (32.43 MB)

### Passo 2: Extrair os Arquivos
1. Baixe o arquivo `brazilian-ecommerce.zip`
2. Extraia **todos os arquivos CSV** nesta pasta (`data/`)
3. Certifique-se de que os 6 arquivos principais estão aqui

## Arquivos Necessários

Após extrair, você deve ter estes arquivos nesta pasta:

```
data/
├── olist_customers_dataset.csv          (~12 MB, 99.441 clientes)
├── olist_order_items_dataset.csv        (~14 MB, 112.650 itens)
├── olist_order_payments_dataset.csv     (~5 MB, 103.886 pagamentos)
├── olist_order_reviews_dataset.csv      (~8 MB, 99.224 avaliações)
├── olist_orders_dataset.csv             (~5 MB, 99.441 pedidos)
├── olist_products_dataset.csv           (~1 MB, 32.951 produtos)
├── olist_sellers_dataset.csv            (~500 KB, 3.095 vendedores)
└── product_category_name_translation.csv (~2 KB, traduções)
```

## ⚠️ Importante

- Os arquivos CSV **NÃO são incluídos** no repositório Git devido ao tamanho
- Você **DEVE baixá-los** diretamente do Kaggle
- O script `ecommerce_analysis.py` só funciona com estes dados

## Verificação dos Dados

Execute este código Python para verificar se os dados estão corretos:

```python
import os
import pandas as pd

# Lista de arquivos necessários
arquivos_necessarios = [
    'olist_orders_dataset.csv',
    'olist_order_items_dataset.csv', 
    'olist_products_dataset.csv',
    'olist_customers_dataset.csv',
    'olist_order_reviews_dataset.csv',
    'olist_order_payments_dataset.csv'
]

print("🔍 Verificando arquivos da Olist...\n")

todos_ok = True
for arquivo in arquivos_necessarios:
    caminho = f'data/{arquivo}'
    if os.path.exists(caminho):
        tamanho = os.path.getsize(caminho) / 1024 / 1024  # MB
        try:
            df = pd.read_csv(caminho)
            print(f"✅ {arquivo}")
            print(f"   📊 {len(df):,} registros | 💾 {tamanho:.1f} MB")
        except Exception as e:
            print(f"❌ {arquivo} - ERRO ao ler: {e}")
            todos_ok = False
    else:
        print(f"❌ {arquivo} - ARQUIVO NÃO ENCONTRADO")
        todos_ok = False

if todos_ok:
    print(f"\n🎉 Todos os arquivos estão OK! Pronto para análise.")
else:
    print(f"\n⚠️ Alguns arquivos estão faltando. Baixe do Kaggle.")
```

## Estrutura dos Dados

### Principais Tabelas:
- **orders**: Informações dos pedidos (datas, status, cliente)
- **order_items**: Itens de cada pedido (produto, preço, frete)
- **customers**: Dados dos clientes (localização)
- **products**: Informações dos produtos (categoria, dimensões)
- **reviews**: Avaliações dos clientes (notas, comentários)
- **payments**: Dados de pagamento (tipo, parcelas, valor)

### Relacionamentos:
```
orders (order_id) ← order_items (order_id)
orders (customer_id) → customers (customer_id)
order_items (product_id) → products (product_id)
orders (order_id) ← reviews (order_id)
orders (order_id) ← payments (order_id)
```

## Problemas Comuns

### Erro: "File not found"
- **Solução**: Certifique-se de que os CSVs estão na pasta `data/`
- **Verifique**: Os nomes dos arquivos estão exatos (com underscores)

### Erro: "Permission denied"
- **Solução**: Feche o Excel se estiver com algum CSV aberto
- **Verifique**: Se tem permissão de escrita na pasta

### Arquivos corrompidos
- **Solução**: Baixe novamente do Kaggle
- **Verifique**: Se o download foi completo

## Sobre o Dataset

O dataset da Olist contém informações **reais e anonimizadas** de uma das maiores empresas de e-commerce do Brasil. Os dados foram disponibilizados publicamente para fins acadêmicos e de pesquisa.

**Características**:
- 🛒 E-commerce brasileiro real
- 📅 Período: Setembro 2016 - Outubro 2018
- 🌎 Cobertura: Todo o território nacional
- 🏪 Marketplace: Vendedores de diversos segmentos
- 📊 Dados estruturados e limpos