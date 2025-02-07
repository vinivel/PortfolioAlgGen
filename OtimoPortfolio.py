import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
import random


class PortfolioOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Otimizador de Carteira IBOV")
        self.root.geometry("900x700")

        self.create_widgets()
        self.set_defaults()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configurações de Entrada
        input_frame = ttk.LabelFrame(main_frame, text="Configurações", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        # File Selection
        ttk.Label(input_frame, text="Arquivo Excel com Tickers:").grid(row=0, column=0, sticky='w')
        self.file_entry = ttk.Entry(input_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Procurar", command=self.load_file).grid(row=0, column=2)

        # Número de Ativos
        ttk.Label(input_frame, text="Nº Máx. de Ativos:").grid(row=1, column=0, sticky='w')
        self.n_assets = ttk.Spinbox(input_frame, from_=2, to=50, width=8)
        self.n_assets.grid(row=1, column=1, sticky='w', padx=5)

        # Período de Teste
        ttk.Label(input_frame, text="Período de Teste:").grid(row=2, column=0, sticky='w')
        self.start_date = ttk.Entry(input_frame, width=12)
        self.start_date.grid(row=2, column=1, sticky='w', padx=5)
        ttk.Label(input_frame, text="até").grid(row=2, column=2)
        self.end_date = ttk.Entry(input_frame, width=12)
        self.end_date.grid(row=2, column=3, sticky='w', padx=5)

        # Periodicidade
        ttk.Label(input_frame, text="Granularidade:").grid(row=3, column=0, sticky='w')
        self.granularity = ttk.Combobox(input_frame, values=['1h', 'D', 'W', 'M'], width=5)
        self.granularity.grid(row=3, column=1, sticky='w', padx=5)

        # Rebalanceamento
        self.rebalance_var = tk.BooleanVar()
        ttk.Checkbutton(input_frame, text="Rebalanceamento", variable=self.rebalance_var,
                        command=self.toggle_rebalance).grid(row=4, column=0, sticky='w')
        self.rebalance_interval = ttk.Spinbox(input_frame, from_=1, to=365, width=8, state='disabled')
        self.rebalance_interval.grid(row=4, column=1, sticky='w', padx=5)

        # Botão de Execução
        ttk.Button(main_frame, text="Executar Otimização", command=self.run_optimization).pack(pady=10)

        # Área de Resultados
        self.results_frame = ttk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

    def set_defaults(self):
        self.n_assets.set(10)
        self.granularity.set('D')
        self.start_date.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.end_date.insert(0, datetime.now().strftime('%Y-%m-%d'))
        self.rebalance_interval.set(30)

    def toggle_rebalance(self):
        self.rebalance_interval.config(state='normal' if self.rebalance_var.get() else 'disabled')

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, filepath)

    def run_optimization(self):
        try:
            # Carregar tickers
            tickers = pd.read_excel(self.file_entry.get()).iloc[:, 0].tolist()

            # Download dos dados
            data = self.get_historical_data(tickers)

            # Selecionar melhores ativos
            selected_assets = self.select_assets(data)

            # Otimizar carteiras
            markowitz_weights = self.markowitz_optimization(selected_assets)
            genetic_weights = self.genetic_optimization(selected_assets)

            # Simular desempenho
            results = self.simulate_performance(data[selected_assets], markowitz_weights, genetic_weights)

            # Exibir resultados
            self.show_results(results)

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def get_historical_data(self, tickers):
        start_date = (datetime.strptime(self.start_date.get(), '%Y-%m-%d')
                      - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

        data = yf.download(
            [t + '.SA' for t in tickers],
            start=start_date,
            end=self.end_date.get(),
            interval=self.granularity.get(),
            group_by='ticker',
            auto_adjust=True
        )

        if data.empty:
            raise ValueError("Não foi possível obter dados para os tickers fornecidos")

        return data.swaplevel(axis=1)['Close'].dropna(axis=1)

    def select_assets(self, data):
        returns = np.log(data / data.shift(1)).dropna()
        performance = pd.DataFrame({
            'Retorno': returns.mean(),
            'Risco': returns.std(),
            'Sharpe': returns.mean() / returns.std()
        })
        return performance.nlargest(int(self.n_assets.get()), 'Sharpe').index.tolist()

    def markowitz_optimization(self, data):
        mu = mean_historical_return(data)
        S = CovarianceShrinkage(data).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe()
        return ef.clean_weights()

    def genetic_optimization(self, data, pop_size=100, generations=200):
        returns = data.pct_change().dropna()
        n_assets = len(data.columns)

        def fitness(weights):
            port_return = np.dot(returns.mean(), weights) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            return port_return / port_vol if port_vol != 0 else 0

        population = [np.random.dirichlet(np.ones(n_assets)) for _ in range(pop_size)]

        for _ in range(generations):
            scores = [fitness(ind) for ind in population]

            # Seleção
            selected = []
            for _ in range(pop_size):
                candidates = random.sample(range(pop_size), 3)
                selected.append(population[max(candidates, key=lambda x: scores[x])])

            # Crossover
            new_pop = []
            for i in range(0, pop_size, 2):
                p1, p2 = selected[i], selected[i + 1]
                child = (p1 + p2) / 2
                new_pop.extend([child, child])

            # Mutação
            for i in range(pop_size):
                if random.random() < 0.1:
                    idx = random.randint(0, n_assets - 1)
                    new_pop[i][idx] += random.uniform(-0.1, 0.1)
                    new_pop[i] /= new_pop[i].sum()

            population = new_pop

        best = max(population, key=fitness)
        return {data.columns[i]: best[i] for i in range(n_assets)}

    def simulate_performance(self, data, w_markowitz, w_genetic):
        test_data = data[self.start_date.get():self.end_date.get()]
        returns = test_data.pct_change().dropna()

        # Converter pesos para arrays
        markowitz_weights = np.array(list(w_markowitz.values()))
        genetic_weights = np.array(list(w_genetic.values()))

        # Calcular retornos
        returns['Markowitz'] = returns.dot(markowitz_weights)
        returns['Genetico'] = returns.dot(genetic_weights)

        # Adicionar IBOV
        ibov = yf.download('^BVSP', start=self.start_date.get(), end=self.end_date.get())['Close']
        returns['IBOV'] = ibov.pct_change().dropna()

        return (1 + returns[['Markowitz', 'Genetico', 'IBOV']]).cumprod()

    def show_results(self, results):
        # Limpar frame anterior
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Gráfico
        fig = plt.Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        results.plot(ax=ax)
        ax.set_title('Desempenho das Carteiras')
        ax.set_xlabel('Data')
        ax.set_ylabel('Retorno Acumulado')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Métricas
        metrics_frame = ttk.Frame(self.results_frame)
        metrics_frame.pack(fill=tk.X, pady=10)

        metrics = self.calculate_metrics(results)
        for i, (name, values) in enumerate(metrics.items()):
            ttk.Label(metrics_frame, text=name, font='bold').grid(row=0, column=i, padx=10)
            for j, (k, v) in enumerate(values.items()):
                ttk.Label(metrics_frame, text=f"{k}: {v:.2%}").grid(row=j + 1, column=i, sticky='w', padx=10)

    def calculate_metrics(self, results):
        metrics = {}
        for col in results.columns:
            returns = results[col].pct_change().dropna()
            peak = results[col].expanding().max()
            drawdown = (results[col] - peak) / peak

            metrics[col] = {
                'Retorno Total': results[col].iloc[-1] - 1,
                'Vol. Anual': returns.std() * np.sqrt(252),
                'Sharpe': returns.mean() / returns.std() * np.sqrt(252),
                'Drawdown Máx': drawdown.min(),
                'Retorno Médio': returns.mean()
            }
        return metrics


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioOptimizer(root)
    root.mainloop()
