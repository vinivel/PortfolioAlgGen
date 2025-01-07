import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_portfolio_performance(weights, prices):
    """Calcula o retorno acumulado de um portfólio em percentual"""
    returns = prices.pct_change().fillna(0)
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    # Converter para variação percentual (descontando valor original)
    return (cumulative_returns - 1) * 100

def calculate_benchmark_performance(test_data):
    """Calcula o retorno do Ibovespa no período de teste em percentual"""
    try:
        ibov = yf.download('^BVSP',
                          start=test_data.index[0],
                          end=test_data.index[-1],
                          progress=False)['Close']

        ibov_returns = ibov.pct_change().fillna(0)
        ibov_performance = (1 + ibov_returns).cumprod()
        # Converter para variação percentual (descontando valor original)
        return (ibov_performance.reindex(test_data.index).ffill() - 1) * 100
    except Exception as e:
        print(f"Erro ao baixar dados do Ibovespa: {e}")
        return pd.Series(0.0, index=test_data.index)

def download_and_prepare_data():
    stocks = ['ABEV3.SA', 'B3SA3.SA', 'BBSE3.SA', 'BBDC4.SA', 'BBAS3.SA',
              'BPAC11.SA', 'CMIG4.SA', 'ELET3.SA', 'EMBR3.SA', 'ENEV3.SA',
              'EQTL3.SA', 'GGBR4.SA', 'ITSA4.SA', 'ITUB4.SA', 'JBSS3.SA',
              'RENT3.SA', 'PETR3.SA', 'PRIO3.SA', 'RADL3.SA', 'RDOR3.SA',
              'RAIL3.SA', 'SBSP3.SA', 'SUZB3.SA', 'UGPA3.SA', 'VALE3.SA',
              'VBBR3.SA', 'WEGE3.SA']

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 24 meses

    print("Downloading stock data...")
    df_closing_prices = pd.DataFrame()
    for symbol in stocks:
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not stock_data.empty:
                df_closing_prices[symbol] = stock_data['Close']
                print(f"Successfully downloaded {symbol}")
        except Exception as e:
            print(f"Error downloading {symbol}: {str(e)}")

    df_closing_prices = df_closing_prices.dropna(axis=1)

    # Dividir em períodos de treinamento e teste
    split_date = df_closing_prices.index[len(df_closing_prices)//2]
    train_data = df_closing_prices[:split_date]
    test_data = df_closing_prices[split_date:]

    return train_data, test_data

def optimize_portfolio_genetic(returns, n_assets=10):
    """Otimização por algoritmo genético simplificado"""
    n_total = returns.shape[1]
    best_sharpe = -np.inf
    best_weights = None

    print("Optimizing portfolio using Genetic Algorithm...")
    for i in range(1000):
        if i % 100 == 0:
            print(f"Iteration {i}/1000")

        selected_assets = np.random.choice(n_total, n_assets, replace=False)
        weights = np.zeros(n_total)
        random_weights = np.random.random(n_assets)
        weights[selected_assets] = random_weights / random_weights.sum()

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk != 0 else -np.inf

        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_weights = weights

    return best_weights

def optimize_portfolio_markowitz(returns, n_assets=10):
    """Otimização por Markowitz simplificada"""
    n_total = returns.shape[1]
    best_sharpe = -np.inf
    best_weights = None

    print("Optimizing portfolio using Markowitz approach...")
    for i in range(1000):
        if i % 100 == 0:
            print(f"Iteration {i}/1000")

        selected_assets = np.random.choice(n_total, n_assets, replace=False)
        weights = np.zeros(n_total)
        random_weights = np.random.random(n_assets)
        weights[selected_assets] = random_weights / random_weights.sum()

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk != 0 else -np.inf

        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_weights = weights

    return best_weights

def main():
    # Download e preparação dos dados
    print("Downloading and preparing data...")
    train_data, test_data = download_and_prepare_data()

    print("\nShape of training data:", train_data.shape)
    print("Shape of test data:", test_data.shape)

    # Calcular retornos para o período de treinamento
    train_returns = train_data.pct_change().dropna()

    # Otimização dos portfólios
    genetic_weights = optimize_portfolio_genetic(train_returns)
    markowitz_weights = optimize_portfolio_markowitz(train_returns)

    print("\nCalculating portfolio performances...")
    # Calcular desempenho no período de teste
    genetic_performance = calculate_portfolio_performance(genetic_weights, test_data)
    markowitz_performance = calculate_portfolio_performance(markowitz_weights, test_data)
    benchmark_performance = calculate_benchmark_performance(test_data)

    print("\nCreating performance DataFrame...")
    # Criar DataFrame com os desempenhos
    performance_df = pd.DataFrame(index=test_data.index)
    performance_df['Genetic Algorithm'] = genetic_performance
    performance_df['Markowitz'] = markowitz_performance
    performance_df['IBOV'] = benchmark_performance

    print("\nPerformance DataFrame head:")
    print(performance_df.head())
    print("\nPerformance DataFrame shape:", performance_df.shape)

    print("\nPlotting results...")
    # Plotar resultados
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    for column in performance_df.columns:
        plt.plot(performance_df.index, performance_df[column],
                label=column, linewidth=2)

    plt.title('Portfolio Performance - Test Period')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    print("\nSaving plot...")
    plt.savefig('portfolio_performance.png')
    plt.show()

    print("\nCalculating portfolio weights...")
    # Exibir composição dos portfólios
    portfolio_weights = pd.DataFrame({
        'Genetic Algorithm (%)': genetic_weights * 100,
        'Markowitz (%)': markowitz_weights * 100
    }, index=train_returns.columns)

    significant_positions = portfolio_weights[
        (portfolio_weights['Genetic Algorithm (%)'] > 0.1) |
        (portfolio_weights['Markowitz (%)'] > 0.1)
    ].round(2)

    print("\n=== Portfolio Compositions (>0.1%) ===")
    print(significant_positions)

    print("\nCalculating performance metrics...")
    # Calcular métricas de desempenho
    final_returns = pd.DataFrame({
        'Strategy': ['Genetic Algorithm', 'Markowitz', 'IBOV'],
        'Total Return (%)': [
            genetic_performance.iloc[-1],
            markowitz_performance.iloc[-1],
            benchmark_performance.iloc[-1]
        ]
    })

    print("\n=== Performance Metrics ===")
    print(final_returns.round(2))

    # Salvar resultados em arquivos
    print("\nSaving results to CSV files...")
    performance_df.to_csv('portfolio_performance.csv')
    significant_positions.to_csv('portfolio_compositions.csv')
    final_returns.to_csv('performance_metrics.csv')

    print("\nDone! All results have been saved.")

if __name__ == "__main__":
    np.random.seed(42)
    main()