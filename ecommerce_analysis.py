# Projeto de Mineração de Dados - E-commerce Brasileiro
# Análise de dados REAIS da Olist
# Dataset: Brazilian E-Commerce Public Dataset by Olist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def verificar_arquivos():
    """
    Verifica se os arquivos da Olist estão presentes
    """
    arquivos_necessarios = [
        'data/olist_orders_dataset.csv',
        'data/olist_order_items_dataset.csv',
        'data/olist_products_dataset.csv',
        'data/olist_customers_dataset.csv',
        'data/olist_order_reviews_dataset.csv',
        'data/olist_order_payments_dataset.csv'
    ]

    arquivos_encontrados = []
    arquivos_faltando = []

    for arquivo in arquivos_necessarios:
        if os.path.exists(arquivo):
            arquivos_encontrados.append(arquivo)
        else:
            arquivos_faltando.append(arquivo)

    if arquivos_faltando:
        print("❌ Arquivos faltando:")
        for arquivo in arquivos_faltando:
            print(f"   - {arquivo}")
        print("\n📥 Para baixar os dados:")
        print("1. Acesse: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
        print("2. Baixe o dataset e extraia os arquivos CSV nesta pasta")
        return False

    print("✅ Todos os arquivos encontrados!")
    return True


def carregar_dados():
    """
    Carrega e processa os dados da Olist
    """
    print("📂 Carregando dados da Olist...")

    try:
        # Carregando as tabelas principais
        orders = pd.read_csv('data/olist_orders_dataset.csv')
        order_items = pd.read_csv('data/olist_order_items_dataset.csv')
        products = pd.read_csv('data/olist_products_dataset.csv')
        customers = pd.read_csv('data/olist_customers_dataset.csv')
        reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
        payments = pd.read_csv('data/olist_order_payments_dataset.csv')

        print(f"✅ Orders: {len(orders):,} registros")
        print(f"✅ Order Items: {len(order_items):,} registros")
        print(f"✅ Products: {len(products):,} registros")
        print(f"✅ Customers: {len(customers):,} registros")
        print(f"✅ Reviews: {len(reviews):,} registros")
        print(f"✅ Payments: {len(payments):,} registros")

        # Fazendo as junções das tabelas
        print("\n🔗 Integrando tabelas...")

        # Começando com orders e order_items
        df = orders.merge(order_items, on='order_id', how='inner')
        print(f"   Orders + Items: {len(df):,} registros")

        # Adicionando customers
        df = df.merge(customers[['customer_id', 'customer_state', 'customer_city']],
                      on='customer_id', how='left')
        print(f"   + Customers: {len(df):,} registros")

        # Adicionando products
        df = df.merge(products[['product_id', 'product_category_name', 'product_weight_g']],
                      on='product_id', how='left')
        print(f"   + Products: {len(df):,} registros")

        # Adicionando reviews
        df = df.merge(reviews[['order_id', 'review_score']],
                      on='order_id', how='left')
        print(f"   + Reviews: {len(df):,} registros")

        # Agregando payments por order_id
        payments_agg = payments.groupby('order_id').agg({
            'payment_type': 'first',
            'payment_installments': 'max',
            'payment_value': 'sum'
        }).reset_index()

        df = df.merge(payments_agg, on='order_id', how='left')
        print(f"   + Payments: {len(df):,} registros")

        # Limpeza dos dados
        print("\n🧹 Limpando dados...")

        # Removendo pedidos cancelados e sem review
        df = df[df['order_status'] == 'delivered']
        df = df.dropna(subset=['review_score'])
        print(f"   Após limpeza: {len(df):,} registros")

        # Removendo outliers de preço
        price_q99 = df['price'].quantile(0.99)
        df = df[df['price'] <= price_q99]
        print(f"   Após remover outliers: {len(df):,} registros")

        # Limitando sample para performance
        if len(df) > 15000:
            df = df.sample(n=15000, random_state=42)
            print(f"   Sample final: {len(df):,} registros")

        # Criando variáveis derivadas
        df['total_value'] = df['price'] + df['freight_value']
        df['order_date'] = pd.to_datetime(df['order_purchase_timestamp'])

        # Tradução das categorias principais para inglês
        category_translation = {
            'beleza_saude': 'health_beauty',
            'esporte_lazer': 'sports_leisure',
            'informatica_acessorios': 'computers_accessories',
            'moveis_decoracao': 'furniture_decor',
            'utilidades_domesticas': 'housewares',
            'relogios_presentes': 'watches_gifts',
            'telefonia': 'telephony',
            'automotivo': 'auto',
            'brinquedos': 'toys',
            'perfumaria': 'perfumery'
        }

        df['product_category_english'] = df['product_category_name'].map(category_translation)
        df['product_category_english'] = df['product_category_english'].fillna('others')

        print(f"✅ Dataset final: {len(df):,} registros x {len(df.columns)} colunas")

        return df

    except FileNotFoundError as e:
        print(f"❌ Arquivo não encontrado: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


def analise_exploratoria(df):
    """
    Análise exploratória dos dados
    """
    print("\n=== ANÁLISE EXPLORATÓRIA ===\n")

    print("📊 Informações básicas:")
    print(f"- Registros: {len(df):,}")
    print(f"- Período: {df['order_date'].min().strftime('%Y-%m-%d')} a {df['order_date'].max().strftime('%Y-%m-%d')}")
    print(f"- Estados: {df['customer_state'].nunique()}")
    print(f"- Categorias: {df['product_category_english'].nunique()}")
    print(f"- Clientes únicos: {df['customer_id'].nunique():,}")

    print(f"\n💰 Estatísticas financeiras:")
    print(f"- Preço médio: R$ {df['price'].mean():.2f}")
    print(f"- Frete médio: R$ {df['freight_value'].mean():.2f}")
    print(f"- Valor total médio: R$ {df['total_value'].mean():.2f}")
    print(f"- Parcelas médias: {df['payment_installments'].mean():.1f}")

    print(f"\n⭐ Satisfação:")
    print(f"- Nota média: {df['review_score'].mean():.2f}/5.0")
    satisfeitos = (df['review_score'] >= 4).mean() * 100
    print(f"- Clientes satisfeitos: {satisfeitos:.1f}%")

    print(f"\n🌎 Top 10 Estados:")
    print(df['customer_state'].value_counts().head(10))

    print(f"\n🛒 Top Categorias:")
    print(df['product_category_english'].value_counts().head(8))

    print(f"\n💳 Tipos de Pagamento:")
    print(df['payment_type'].value_counts())

    return df


def criar_visualizacoes(df):
    """
    Cria visualizações dos dados
    """
    print("\n=== GERANDO GRÁFICOS ===")

    # Figura 1: Análises principais
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Vendas por estado
    top_states = df['customer_state'].value_counts().head(12)
    top_states.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Vendas por Estado')
    axes[0, 0].set_xlabel('Estado')
    axes[0, 0].set_ylabel('Número de Vendas')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Distribuição de preços
    axes[0, 1].hist(df['price'], bins=50, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Distribuição de Preços')
    axes[0, 1].set_xlabel('Preço (R$)')
    axes[0, 1].set_ylabel('Frequência')

    # 3. Avaliações
    review_counts = df['review_score'].value_counts().sort_index()
    review_counts.plot(kind='bar', ax=axes[0, 2], color='orange')
    axes[0, 2].set_title('Distribuição das Avaliações')
    axes[0, 2].set_xlabel('Nota')
    axes[0, 2].set_ylabel('Quantidade')

    # 4. Categorias
    top_categories = df['product_category_english'].value_counts().head(8)
    top_categories.plot(kind='barh', ax=axes[1, 0], color='salmon')
    axes[1, 0].set_title('Top Categorias')
    axes[1, 0].set_xlabel('Vendas')

    # 5. Tipos de pagamento
    payment_counts = df['payment_type'].value_counts()
    axes[1, 1].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Tipos de Pagamento')

    # 6. Vendas ao longo do tempo
    monthly_sales = df.groupby(df['order_date'].dt.to_period('M')).size()
    monthly_sales.plot(ax=axes[1, 2], color='purple')
    axes[1, 2].set_title('Vendas Mensais')
    axes[1, 2].set_xlabel('Mês')
    axes[1, 2].set_ylabel('Vendas')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('outputs/analise_exploratoria.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figura 2: Mapa de correlação
    plt.figure(figsize=(10, 8))
    numeric_cols = ['price', 'freight_value', 'payment_installments', 'review_score', 'total_value']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlações entre Variáveis')
    plt.tight_layout()
    plt.savefig('outputs/correlacao_variaveis.png', dpi=300, bbox_inches='tight')
    plt.show()


def classificacao_satisfacao(df):
    """
    Classificação para prever satisfação do cliente
    """
    print("\n=== CLASSIFICAÇÃO: SATISFAÇÃO DO CLIENTE ===")

    # Target: cliente satisfeito (nota >= 4)
    df['cliente_satisfeito'] = (df['review_score'] >= 4).astype(int)

    print(f"📊 Distribuição:")
    satisfeitos = df['cliente_satisfeito'].sum()
    total = len(df)
    print(f"- Satisfeitos: {satisfeitos:,} ({satisfeitos / total * 100:.1f}%)")
    print(f"- Insatisfeitos: {total - satisfeitos:,} ({(total - satisfeitos) / total * 100:.1f}%)")

    # Features para classificação
    features = ['price', 'freight_value', 'payment_installments', 'total_value']
    X = df[features].fillna(0)
    y = df['cliente_satisfeito']

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modelo Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predições e avaliação
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Resultados:")
    print(f"- Acurácia: {accuracy:.4f}")
    print(f"\n📈 Relatório detalhado:")
    print(classification_report(y_test, y_pred, target_names=['Insatisfeito', 'Satisfeito']))

    # Importância das features
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Importância das Variáveis - Satisfação do Cliente')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig('outputs/importancia_features.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Feature mais importante: {feature_importance.iloc[0]['feature']}")

    return model, accuracy


def regressao_precos(df):
    """
    Regressão para prever preços
    """
    print("\n=== REGRESSÃO: PREDIÇÃO DE PREÇOS ===")

    # Preparando dados
    df_reg = df.copy()

    # Encoding das variáveis categóricas
    le_category = LabelEncoder()
    le_state = LabelEncoder()
    le_payment = LabelEncoder()

    df_reg['category_encoded'] = le_category.fit_transform(df_reg['product_category_english'].fillna('unknown'))
    df_reg['state_encoded'] = le_state.fit_transform(df_reg['customer_state'].fillna('unknown'))
    df_reg['payment_encoded'] = le_payment.fit_transform(df_reg['payment_type'].fillna('unknown'))

    # Features e target
    features = ['category_encoded', 'state_encoded', 'payment_encoded', 'payment_installments', 'freight_value']
    X = df_reg[features].fillna(0)
    y = df_reg['price']

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modelo de regressão
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predições e avaliação
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Resultados:")
    print(f"- R² Score: {r2:.4f}")
    print(f"- MSE: {mse:.2f}")
    print(f"- Preço médio real: R$ {y_test.mean():.2f}")

    # Gráfico predito vs real
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Preço Real (R$)')
    plt.ylabel('Preço Predito (R$)')
    plt.title('Regressão: Preço Real vs Predito')
    plt.tight_layout()
    plt.savefig('outputs/regressao_precos.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, r2


def clustering_estados(df):
    """
    Agrupamento de estados por comportamento de compra
    """
    print("\n=== CLUSTERING: SEGMENTAÇÃO DE ESTADOS ===")

    # Perfil por estado
    state_profile = df.groupby('customer_state').agg({
        'price': 'mean',
        'freight_value': 'mean',
        'payment_installments': 'mean',
        'review_score': 'mean',
        'order_id': 'count'
    }).round(2)

    state_profile.columns = ['preco_medio', 'frete_medio', 'parcelas_media', 'avaliacao_media', 'total_pedidos']

    # Filtrar estados com pelo menos 50 pedidos
    state_profile = state_profile[state_profile['total_pedidos'] >= 50]

    print(f"📊 Estados analisados: {len(state_profile)}")

    # Normalização
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(
        state_profile[['preco_medio', 'frete_medio', 'parcelas_media', 'avaliacao_media']])

    # K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    state_profile['cluster'] = clusters

    print(f"\n🎯 Perfil dos clusters:")
    for cluster in range(3):
        cluster_data = state_profile[state_profile['cluster'] == cluster]
        estados = ', '.join(cluster_data.index.tolist())
        print(f"\nCluster {cluster}: {estados}")
        print(f"  - Preço médio: R$ {cluster_data['preco_medio'].mean():.2f}")
        print(f"  - Frete médio: R$ {cluster_data['frete_medio'].mean():.2f}")
        print(f"  - Total pedidos: {cluster_data['total_pedidos'].sum():,}")

    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Preço vs Avaliação
    scatter1 = axes[0].scatter(state_profile['preco_medio'], state_profile['avaliacao_media'],
                               c=state_profile['cluster'], cmap='viridis', s=100)
    axes[0].set_xlabel('Preço Médio (R$)')
    axes[0].set_ylabel('Avaliação Média')
    axes[0].set_title('Clusters: Preço vs Avaliação')

    # Gráfico 2: Volume vs Frete
    scatter2 = axes[1].scatter(state_profile['total_pedidos'], state_profile['frete_medio'],
                               c=state_profile['cluster'], cmap='viridis', s=100)
    axes[1].set_xlabel('Total de Pedidos')
    axes[1].set_ylabel('Frete Médio (R$)')
    axes[1].set_title('Clusters: Volume vs Frete')

    plt.tight_layout()
    plt.savefig('outputs/clustering_estados.png', dpi=300, bbox_inches='tight')
    plt.show()

    return kmeans, state_profile


def main():
    """
    Função principal
    """
    print("🛒 PROJETO DE MINERAÇÃO DE DADOS - E-COMMERCE BRASILEIRO")
    print("📊 Dataset: Brazilian E-Commerce Public Dataset by Olist")
    print("🔗 Fonte: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n")

    # 1. Verificar arquivos
    if not verificar_arquivos():
        return

    # 2. Carregar dados
    print("\n1. Carregando dados...")
    df = carregar_dados()
    if df is None:
        return

    # 3. Análise exploratória
    print("\n2. Análise exploratória...")
    df = analise_exploratoria(df)

    # 4. Visualizações
    print("\n3. Criando visualizações...")
    criar_visualizacoes(df)

    # 5. Classificação
    print("\n4. Aplicando classificação...")
    rf_model, accuracy = classificacao_satisfacao(df)

    # 6. Regressão
    print("\n5. Aplicando regressão...")
    lr_model, r2 = regressao_precos(df)

    # 7. Clustering
    print("\n6. Aplicando clustering...")
    kmeans, state_profile = clustering_estados(df)

    # Resumo final
    print("\n" + "=" * 60)
    print("🎯 RESUMO DOS RESULTADOS")
    print("=" * 60)
    print(f"📊 Dataset: {len(df):,} registros reais")
    print(f"📅 Período: {df['order_date'].min().strftime('%Y-%m-%d')} a {df['order_date'].max().strftime('%Y-%m-%d')}")
    print(f"🌎 Estados: {df['customer_state'].nunique()}")
    print(f"🎯 Acurácia Classificação: {accuracy:.4f}")
    print(f"📈 R² Regressão: {r2:.4f}")
    print(f"🔍 Clusters: {len(state_profile)} estados em 3 grupos")
    print("=" * 60)
    print("✅ Análise concluída! Gráficos salvos em PNG.")

    return df, rf_model, lr_model, kmeans


if __name__ == "__main__":
    df, rf_model, lr_model, kmeans = main()