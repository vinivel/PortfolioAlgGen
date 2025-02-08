import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import random
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Otimizador de Carteira IBOV")
        self.create_widgets()
        self.set_defaults()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Configurações", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        # Componentes da interface
        ttk.Label(input_frame, text="Arquivo Excel com Tickers:").grid(row=0, column=0, sticky='w')
        self.file_entry = ttk.Entry(input_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Procurar", command=self.load_file).grid(row=0, column=2)

        ttk.Label(input_frame, text="Nº Máx. de Ativos:").grid(row=1, column=0, sticky='w')
        self.n_assets = ttk.Spinbox(input_frame, from_=2, to=50, width=8)
        self.n_assets.grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(input_frame, text="Período de Teste:").grid(row=2, column=0, sticky='w')
        self.start_date = ttk.Entry(input_frame, width=12)
        self.start_date.grid(row=2, column=1, sticky='w', padx=5)
        ttk.Label(input_frame, text="até").grid(row=2, column=2)
        self.end_date = ttk.Entry(input_frame, width=12)
        self.end_date.grid(row=2, column=3, sticky='w', padx=5)

        ttk.Label(input_frame, text="Granularidade:").grid(row=3, column=0, sticky='w')
        self.granularity = ttk.Combobox(input_frame, values=['1h', '1d', '1wk', '1mo'], width=5)
        self.granularity.grid(row=3, column=1, sticky='w', padx=5)

        self.rebalance_var = tk.BooleanVar()
        ttk.Checkbutton(input_frame, text="Rebalanceamento", variable=self.rebalance_var,
                        command=self.toggle_rebalance).grid(row=4, column=0, sticky='w')
        self.rebalance_interval = ttk.Spinbox(input_frame, from_=1, to=365, width=8, state='disabled')
        self.rebalance_interval.grid(row=4, column=1, sticky='w', padx=5)

        ttk.Button(main_frame, text="Executar Otimização", command=self.safe_run_optimization).pack(pady=10)

        self.results_frame = ttk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

    # === MÉTODO AUSENTE ADICIONADO ===
    def toggle_rebalance(self):
        """Ativa/desativa o campo de rebalanceamento."""
        if self.rebalance_var.get():
            self.rebalance_interval.config(state='normal')
        else:
            self.rebalance_interval.config(state='disabled')

    def set_defaults(self):
        self.n_assets.set(10)
        self.granularity.set('1d')
        self.start_date.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.end_date.insert(0, datetime.now().strftime('%Y-%m-%d'))
        self.rebalance_interval.set(30)

    def load_file(self):
        try:
            filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
            if filepath:
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, filepath)
                pd.read_excel(filepath, engine='openpyxl').iloc[:, 0]  # Verificação rápida
        except Exception as e:
            self.show_error(f"Erro ao carregar arquivo:\n{str(e)}")

    def safe_run_optimization(self):
        try:
            self.run_optimization()
        except Exception as e:
            self.show_error(f"Erro na otimização:\n{str(e)}")

    def show_error(self, message):
        if self.root.winfo_exists():
            messagebox.showerror("Erro", message, parent=self.root)
        else:
            print("Erro após fechar janela:", message)

    def run_optimization(self):
        try:
            # Carregar tickers
            df = pd.read_excel(self.file_entry.get(), engine='openpyxl')
            if df.empty or df.columns[0].lower() != 'tickers':
                raise ValueError("Formato do arquivo inválido. Deve conter coluna 'Tickers'")

            tickers = df.iloc[:, 0].dropna().astype(str).tolist()
            data = self.get_historical_data(tickers)
            selected_assets = self.select_assets(data)

            returns = data[selected_assets].pct_change().dropna()
            weights_markowitz = self.markowitz_optimization(returns)
            weights_genetic = self.genetic_optimization(returns)

            results = self.simulate_performance(data[selected_assets], weights_markowitz, weights_genetic)
            self.show_results(results, {'Markowitz': weights_markowitz, 'Genetico': weights_genetic})

        except Exception as e:
            self.show_error(str(e))
            raise

    def get_historical_data(self, tickers):
        try:
            start_date = (datetime.strptime(self.start_date.get(), '%Y-%m-%d') -
                          timedelta(days=5 * 252)).strftime('%Y-%m-%d')

            data = yf.download(
                [t + '.SA' for t in tickers],
                start=start_date,
                end=self.end_date.get(),
                interval=self.granularity.get(),
                group_by='ticker',
                auto_adjust=True
            )['Close']

            if data.empty:
                raise ValueError("Nenhum dado encontrado para os tickers fornecidos")

            return data.dropna(axis=1)
        except Exception as e:
            self.show_error(f"Erro ao obter dados históricos:\n{str(e)}")
            raise

    def select_assets(self, data):
        returns = np.log(data / data.shift(1)).dropna()
        performance = pd.DataFrame({
            'Retorno': returns.mean(),
            'Risco': returns.std(),
            'Sharpe': returns.mean() / returns.std()
        })
        return performance.nlargest(int(self.n_assets.get()), 'Sharpe').index.tolist()

    def markowitz_optimization(self, returns):
        cov_matrix = returns.cov() * 252
        expected_returns = returns.mean() * 252

        n_assets = len(expected_returns)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        result = minimize(
            self._negative_sharpe,
            initial_weights,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x

    def _negative_sharpe(self, weights, expected_returns, cov_matrix):
        port_return = np.dot(weights, expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_volatility if port_volatility != 0 else 0

    def genetic_optimization(self, returns, pop_size=100, generations=200):
        n_assets = returns.shape[1]

        def fitness(weights):
            port_return = np.dot(returns.mean(), weights) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            return port_return / port_vol if port_vol != 0 else 0

        population = [np.random.dirichlet(np.ones(n_assets)) for _ in range(pop_size)]

        for _ in range(generations):
            scores = [fitness(ind) for ind in population]

            # Seleção por torneio
            selected = []
            for _ in range(pop_size):
                candidates = random.sample(range(pop_size), 3)
                selected.append(population[max(candidates, key=lambda x: scores[x])])

            # Crossover e mutação
            new_population = []
            for i in range(0, pop_size, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child = (parent1 + parent2) / 2
                new_population.extend([child, child])

            for i in range(pop_size):
                if random.random() < 0.1:
                    mutation_idx = random.randint(0, n_assets - 1)
                    new_population[i][mutation_idx] += random.uniform(-0.1, 0.1)
                    new_population[i] = np.clip(new_population[i], 0, 1)
                    new_population[i] /= new_population[i].sum()

            population = new_population

        return max(population, key=fitness)

    def simulate_performance(self, data, weights_markowitz, weights_genetic):
        test_data = data.loc[self.start_date.get():self.end_date.get()]
        returns = test_data.pct_change().dropna()

        returns['Markowitz'] = returns.dot(weights_markowitz)
        returns['Genetico'] = returns.dot(weights_genetic)

        ibov = yf.download('^BVSP', start=self.start_date.get(), end=self.end_date.get())['Close']
        returns['IBOV'] = ibov.pct_change().dropna()

        return (1 + returns[['Markowitz', 'Genetico', 'IBOV']]).cumprod()

    def show_results(self, results, weights):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Gráfico
        fig = plt.Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        results.plot(ax=ax)
        ax.set_title('Desempenho das Carteiras')
        ax.set_xlabel('Data')
        ax.set_ylabel('Retorno Acumulado')

        canvas = FigureCanvasTkAgg(fig, self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tabela de pesos
        metrics_frame = ttk.Frame(self.results_frame)
        metrics_frame.pack(fill=tk.X, pady=10)

        tree = ttk.Treeview(metrics_frame, columns=('Ativo', 'Markowitz', 'Genetico'), show='headings', height=5)
        tree.heading('Ativo', text='Ativo')
        tree.heading('Markowitz', text='Markowitz (%)')
        tree.heading('Genetico', text='Genetico (%)')

        assets = results.columns[:-3]
        for asset, w_m, w_g in zip(assets, weights['Markowitz'], weights['Genetico']):
            tree.insert('', 'end', values=(asset, f"{w_m * 100:.2f}", f"{w_g * 100:.2f}"))

        tree.pack(side=tk.LEFT, padx=10)

        # Métricas
        metrics_text = tk.Text(metrics_frame, height=10, width=40)
        metrics_text.pack(side=tk.LEFT, padx=10)

        for strategy in ['Markowitz', 'Genetico', 'IBOV']:
            stats = self.calculate_metrics(results[[strategy]])
            metrics_text.insert(tk.END, f"{strategy}:\n")
            for k, v in stats.items():
                metrics_text.insert(tk.END, f"  {k}: {v:.2%}\n")
            metrics_text.insert(tk.END, "\n")

    def calculate_metrics(self, data):
        returns = data.pct_change().dropna()
        cumulative = (1 + returns).cumprod()

        return {
            'Retorno Total': cumulative.iloc[-1] - 1,
            'Volatilidade Anual': returns.std() * np.sqrt(252),
            'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),
            'Drawdown Máximo': (cumulative / cumulative.cummax() - 1).min(),
            'Retorno Médio Mensal': returns.resample('ME').mean().mean()
        }

    def on_close(self):
        if messagebox.askokcancel("Sair", "Deseja realmente sair do programa?", parent=self.root):
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioOptimizer(root)
    root.mainloop()
