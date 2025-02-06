import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Risk-free rate (annualized)
RISK_FREE_RATE = 0.05  # Exemplo: taxa livre de risco de 5% ao ano


# Função para calcular o fitness (Sharpe Ratio)
def fitness(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    return sharpe_ratio


# Inicializa a população para o Algoritmo Genético
def initialize_population(n_individuals, n_assets):
    return np.random.dirichlet(np.ones(n_assets), size=n_individuals)


# Seleciona os melhores indivíduos com base no score de fitness
def selection(population, returns, num_best=10):
    scores = np.array([fitness(ind, returns) for ind in population])
    best_indices = np.argsort(scores)[-num_best:]
    return population[best_indices]


# Realiza crossover entre dois pais para criar um filho
def crossover(p1, p2):
    alpha = np.random.uniform(0, 1, len(p1))
    child = alpha * p1 + (1 - alpha) * p2
    return child / np.sum(child)


# Aplica mutação em um indivíduo com alterações limitadas
def mutate(ind, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        ind += np.random.normal(0, 0.1, len(ind))
        ind = np.clip(ind, 0, None)  # Garante que não haja pesos negativos
        ind /= np.sum(ind)  # Normaliza os pesos para somarem 1
    return ind


# Algoritmo Genético para otimização de portfólio
def optimize_portfolio_genetic(returns, n_assets, n_individuals=100, generations=50):
    population = initialize_population(n_individuals, n_assets)
    for _ in range(generations):
        top_individuals = selection(population, returns)
        new_population = [mutate(crossover(top_individuals[np.random.randint(len(top_individuals))],
                                           top_individuals[np.random.randint(len(top_individuals))])) for _ in
                          range(n_individuals)]
        population = np.array(new_population)
    return selection(population, returns, num_best=1)[0]


# Otimização paralela usando Algoritmo Genético
def optimize_portfolio_parallel(returns, n_assets):
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(optimize_portfolio_genetic,
                               [(returns, n_assets, 100, 50) for _ in range(4)])
    return max(results, key=lambda w: fitness(w, returns))


# Otimização Markowitz usando o framework de média-variância
def markowitz_optimization(returns):
    n_assets = returns.shape[1]
    result = minimize(lambda w: -fitness(w, returns),
                      np.ones(n_assets) / n_assets,
                      method='SLSQP',
                      bounds=[(0, 1)] * n_assets,
                      constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}])
    return result.x


# Baixa os dados das ações do Yahoo Finance com tratamento de erros
def download_stock_data():
    stocks = ['ABEV3.SA', 'B3SA3.SA', 'BBSE3.SA', 'BBDC4.SA', 'BBAS3.SA', 'BPAC11.SA',
              'CMIG4.SA', 'ELET3.SA', 'EMBR3.SA', 'ENEV3.SA', 'EQTL3.SA', 'GGBR4.SA',
              'ITSA4.SA', 'ITUB4.SA', 'JBSS3.SA', 'RENT3.SA', 'PETR3.SA', 'PRIO3.SA',
              'RADL3.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SUZB3.SA', 'UGPA3.SA',
              'VALE3.SA', 'VBBR3.SA', 'WEGE3.SA']

    try:
        data = yf.download(stocks, period='5y', progress=False)['Close']
        return data.dropna()
    except Exception as e:
        print(f"Erro ao baixar dados das ações: {e}")
        return None


# Baixa os dados do índice IBOVESPA com tratamento de erros
def download_ibov():
    try:
        ibov = yf.download('^BVSP', period='5y', progress=False)['Close']
        return ibov.pct_change().dropna()
    except Exception as e:
        print(f"Erro ao baixar dados do IBOV: {e}")
        return None


# Plota os resultados com métricas adicionais de desempenho
def plot_results(performance_df):
    plt.figure(figsize=(12, 6))

    for column in performance_df.columns:
        plt.plot(performance_df.index, performance_df[column], label=column)

    plt.title('Desempenho do Portfólio vs IBOV')
    plt.xlabel('Data')
    plt.ylabel('Retorno Acumulado (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salva e exibe o gráfico
    plt.savefig('portfolio_performance.png')
    plt.show()


# Função principal para executar a análise
def main():
    np.random.seed(42)

    # Baixa os dados das ações e do IBOVESPA
    stock_data = download_stock_data()

    if stock_data is None:
        print("Falha ao baixar os dados das ações. Encerrando.")
        return

    ibov_returns = download_ibov()

    if ibov_returns is None:
        print("Falha ao baixar os dados do IBOV. Encerrando.")
        return

    # Calcula os retornos diários das ações
    returns = stock_data.pct_change().dropna()

    # Otimiza os portfólios usando Algoritmo Genético e Markowitz
    genetic_weights = optimize_portfolio_parallel(returns, len(stock_data.columns))
    markowitz_weights = markowitz_optimization(returns)

    # Calcula o desempenho acumulado dos portfólios e do IBOVESPA
    genetic_performance = (1 + (returns @ genetic_weights)).cumprod().squeeze() - 1
    markowitz_performance = (1 + (returns @ markowitz_weights)).cumprod().squeeze() - 1
    ibov_performance = (1 + ibov_returns).cumprod().squeeze() - 1

    # Cria um DataFrame para comparar o desempenho dos portfólios
    performance_df = pd.DataFrame({
        'Algoritmo Genético': genetic_performance,
        'Markowitz': markowitz_performance,
        'IBOV': ibov_performance
    })

    # Salva os resultados em arquivos CSV
    performance_df.to_csv('portfolio_performance.csv')

    pd.DataFrame({
        'Genetic': genetic_weights,
        'Markowitz': markowitz_weights
    }).to_csv('optimized_portfolios.csv', index=False)

    # Plota os resultados
    plot_results(performance_df)


if __name__ == "__main__":
    main()
