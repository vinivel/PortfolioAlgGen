import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime


class PortfolioOptimizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Portfolio Optimization Tool")
        self.setup_ui()

        self.historical_data = None
        self.training_data = None
        self.test_data = None
        self.ibov_data = None
        self.metrics = pd.DataFrame()
        self.selected_assets = []
        self.markowitz_weights = None
        self.ga_weights = None

    def setup_ui(self):
        """Initialize complete graphical interface"""
        self.frame = ttk.Frame(self.master, padding=20)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # File selection
        ttk.Button(self.frame, text="1. Select Ticker File",
                   command=self.load_tickers).grid(row=0, column=0, sticky=tk.W, pady=5)

        # Date inputs
        ttk.Label(self.frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W)
        self.start_date = ttk.Entry(self.frame, width=15)
        self.start_date.grid(row=1, column=1, sticky=tk.W)
        self.start_date.insert(0, "2015-01-01")

        ttk.Label(self.frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky=tk.W)
        self.end_date = ttk.Entry(self.frame, width=15)
        self.end_date.grid(row=2, column=1, sticky=tk.W)
        self.end_date.insert(0, datetime.now().strftime("%Y-%m-%d"))

        # Asset number input
        ttk.Label(self.frame, text="Max Assets in Portfolio:").grid(row=3, column=0, sticky=tk.W)
        self.n_assets = ttk.Spinbox(self.frame, from_=2, to=20, width=5)
        self.n_assets.grid(row=3, column=1, sticky=tk.W)
        self.n_assets.set(5)

        # Target return input
        ttk.Label(self.frame, text="Target Annual Return (%):").grid(row=4, column=0, sticky=tk.W)
        self.target_return = ttk.Spinbox(self.frame, from_=0, to=100, width=5)
        self.target_return.grid(row=4, column=1, sticky=tk.W)
        self.target_return.set(10)

        # Action buttons
        ttk.Button(self.frame, text="2. Download Data",
                   command=self.download_data).grid(row=5, column=0, pady=10, sticky=tk.W)
        ttk.Button(self.frame, text="3. Calculate Metrics",
                   command=self.calculate_metrics).grid(row=5, column=1, pady=10, sticky=tk.W)
        ttk.Button(self.frame, text="4. Run Optimization",
                   command=self.run_optimization).grid(row=6, column=0, pady=10, sticky=tk.W)
        ttk.Button(self.frame, text="5. Generate Report",
                   command=self.generate_report).grid(row=6, column=1, pady=10, sticky=tk.W)

        # Progress bar
        self.progress = ttk.Progressbar(self.frame, orient='horizontal',
                                        length=400, mode='determinate')
        self.progress.grid(row=7, columnspan=2, pady=20)

        # Console output
        self.console = tk.Text(self.frame, height=10, width=70)
        self.console.grid(row=8, columnspan=2)
        self.console.insert(tk.END, "Application Initialized...\n")

    def log_message(self, message):
        """Display messages in console"""
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.master.update_idletasks()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress['value'] = value
        self.master.update_idletasks()

    def load_tickers(self):
        """Load tickers from Excel file"""
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
            if not file_path:
                return

            raw_tickers = pd.read_excel(file_path).iloc[:, 0].tolist()
            self.tickers = [t + '.SA' if not t.endswith('.SA') else t for t in raw_tickers]
            self.log_message(f"Loaded {len(self.tickers)} tickers from file")
            self.update_progress(10)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tickers: {str(e)}")

    def download_data(self):
        """Download historical data from Yahoo Finance"""
        try:
            self.log_message("Downloading market data...")

            self.historical_data = yf.download(
                self.tickers,
                start=self.start_date.get(),
                end=self.end_date.get(),
                auto_adjust=True,
                multi_level_index=False
            )['Close'].ffill().dropna(axis=1)

            self.ibov_data = yf.download(
                '^BVSP',
                start=self.start_date.get(),
                end=self.end_date.get(),
                auto_adjust=True
            )['Close'].ffill()

            if self.historical_data.empty:
                raise ValueError("No data downloaded for selected tickers")

            split_idx = int(len(self.historical_data) * 0.8)
            self.training_data = self.historical_data.iloc[:split_idx]
            self.test_data = self.historical_data.iloc[split_idx:]

            self.log_message(f"Data downloaded: {len(self.historical_data)} trading days")
            self.update_progress(30)
        except Exception as e:
            messagebox.showerror("Download Error", str(e))

    def calculate_metrics(self):
        """Calculate financial metrics for asset selection"""
        try:
            self.log_message("Calculating performance metrics...")
            returns = np.log(self.training_data / self.training_data.shift(1)).dropna()

            self.metrics = pd.DataFrame({
                'Mean': returns.mean(),
                'Volatility': returns.std(),
                'Sharpe': returns.mean() / returns.std()
            }).sort_values('Sharpe', ascending=False)

            self.selected_assets = self.metrics.index[:int(self.n_assets.get())].tolist()
            self.log_message(f"Selected top {self.n_assets.get()} assets: {', '.join(self.selected_assets)}")
            self.update_progress(50)
        except Exception as e:
            messagebox.showerror("Calculation Error", str(e))

    def run_optimization(self):
        """Run both optimization methods"""
        try:
            self.log_message("Running Markowitz optimization...")
            self.markowitz_optimization()
            self.log_message("Running genetic algorithm optimization...")
            self.genetic_algorithm_optimization()
            self.update_progress(80)
        except Exception as e:
            messagebox.showerror("Optimization Error", str(e))

    def markowitz_optimization(self):
        """Traditional Markowitz optimization using linear algebra"""
        try:
            returns = self.training_data[self.selected_assets].pct_change().dropna()
            cov_matrix = returns.cov().values * 252
            expected_returns = returns.mean().values * 252
            target_return = float(self.target_return.get()) / 100

            n = len(expected_returns)
            ones = np.ones(n)
            cov_inv = np.linalg.pinv(cov_matrix)

            A = np.array([
                [expected_returns.T @ cov_inv @ expected_returns,
                 expected_returns.T @ cov_inv @ ones],
                [expected_returns.T @ cov_inv @ ones,
                 ones.T @ cov_inv @ ones]
            ])

            b = np.array([target_return, 1])

            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                A += np.eye(2) * 1e-6
                x = np.linalg.lstsq(A, b, rcond=None)[0]

            lambda1, lambda2 = x
            weights = (lambda1 * (cov_inv @ expected_returns) +
                       lambda2 * (cov_inv @ ones))
            weights /= weights.sum()
            weights = np.clip(weights, 0, 1)
            weights /= weights.sum()

            self.markowitz_weights = pd.Series(
                weights.round(4),
                index=self.selected_assets
            )
            self.log_message(f"Markowitz weights:\n{self.markowitz_weights}")
        except Exception as e:
            messagebox.showerror("Optimization Error", f"Markowitz failed: {str(e)}")

    def genetic_algorithm_optimization(self, population_size=50, generations=100):
        """Genetic algorithm implementation"""
        try:
            returns = self.training_data[self.selected_assets].pct_change().dropna()
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252

            def fitness(weights):
                ret = np.dot(weights, expected_returns)
                risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return ret - 1.5 * risk

            population = np.random.dirichlet(np.ones(len(self.selected_assets)), population_size)
            best_weights = None
            best_fitness = -np.inf

            for _ in range(generations):
                fitness_scores = [fitness(ind) for ind in population]
                max_idx = np.argmax(fitness_scores)

                if fitness_scores[max_idx] > best_fitness:
                    best_fitness = fitness_scores[max_idx]
                    best_weights = population[max_idx]

                parents = population[np.argsort(fitness_scores)[-2:]]
                offspring = np.zeros((population_size, len(self.selected_assets)))

                for i in range(population_size):
                    alpha = np.random.rand()
                    offspring[i] = alpha * parents[0] + (1 - alpha) * parents[1]

                    if np.random.rand() < 0.1:
                        mutation = np.random.normal(0, 0.1, size=len(self.selected_assets))
                        offspring[i] = np.clip(offspring[i] + mutation, 0, 1)
                        offspring[i] /= offspring[i].sum()

                population = offspring

            self.ga_weights = pd.Series(
                best_weights.round(4),
                index=self.selected_assets
            )
            self.log_message(f"Genetic algorithm weights:\n{self.ga_weights}")
        except Exception as e:
            messagebox.showerror("Optimization Error", f"Genetic Algorithm failed: {str(e)}")

    def generate_report(self):
        """Generate final analysis report"""
        try:
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 2)

            ax1 = fig.add_subplot(gs[0, :])
            self.plot_cumulative_returns(ax1)

            ax2 = fig.add_subplot(gs[1, 0])
            self.plot_drawdowns(ax2)

            ax3 = fig.add_subplot(gs[1, 1])
            self.plot_efficient_frontier(ax3)

            ax4 = fig.add_subplot(gs[2, :])
            self.plot_weight_comparison(ax4)

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.master)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.update_progress(100)
            self.log_message("Report generation complete")
        except Exception as e:
            messagebox.showerror("Reporting Error", str(e))

    def plot_cumulative_returns(self, ax):
        """Plot cumulative returns comparison"""
        try:
            markowitz_returns = self.test_data[self.selected_assets].pct_change().dot(self.markowitz_weights)
            ga_returns = self.test_data[self.selected_assets].pct_change().dot(self.ga_weights)
            ibov_returns = self.ibov_data.pct_change().reindex(markowitz_returns.index).fillna(0)

            cum_markowitz = (1 + markowitz_returns).cumprod() - 1
            cum_ga = (1 + ga_returns).cumprod() - 1
            cum_ibov = (1 + ibov_returns).cumprod() - 1

            ax.plot(cum_markowitz.index, cum_markowitz, label='Markowitz')
            ax.plot(cum_ga.index, cum_ga, label='Genetic Algorithm')
            ax.plot(cum_ibov.index, cum_ibov, label='IBOV')
            ax.set_title("Cumulative Returns Comparison")
            ax.set_ylabel("Return")
            ax.legend()
            ax.grid(True)
        except Exception as e:
            ax.text(0.5, 0.5, 'Error plotting cumulative returns', ha='center', va='center')
            self.log_message(f"Plotting error: {str(e)}")

    def plot_drawdowns(self, ax):
        """Plot maximum drawdowns"""
        try:
            markowitz_returns = self.test_data[self.selected_assets].pct_change().dot(self.markowitz_weights)
            cum_returns = (1 + markowitz_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max

            ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
            ax.set_title("Maximum Drawdown")
            ax.set_ylabel("Drawdown")
            ax.grid(True)
        except Exception as e:
            ax.text(0.5, 0.5, 'Error plotting drawdowns', ha='center', va='center')
            self.log_message(f"Drawdown plot error: {str(e)}")

    def plot_efficient_frontier(self, ax):
        """Plot efficient frontier"""
        try:
            returns, volatilities = self.calculate_efficient_frontier()

            # Validate and clean data
            valid_points = (volatilities > 0) & (~np.isnan(volatilities)) & (~np.isnan(returns))
            returns = returns[valid_points]
            volatilities = volatilities[valid_points]

            if len(returns) > 0:
                ax.plot(volatilities, returns, 'b-', linewidth=2)
                ax.scatter(volatilities, returns, c=returns / volatilities,
                           cmap='viridis', s=50)
                ax.set_title("Efficient Frontier")
                ax.set_xlabel("Annualized Volatility")
                ax.set_ylabel("Annualized Return")
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, 'No valid efficient frontier points',
                        ha='center', va='center')
                ax.set_title("Efficient Frontier (No Data)")
        except Exception as e:
            ax.text(0.5, 0.5, 'Error plotting efficient frontier', ha='center', va='center')
            self.log_message(f"Efficient frontier error: {str(e)}")

    def calculate_efficient_frontier(self):
        """Calculate efficient frontier points"""
        try:
            returns = self.training_data[self.selected_assets].pct_change().dropna()
            cov_matrix = returns.cov().values * 252
            expected_returns = returns.mean().values * 252
            n = len(expected_returns)
            ones = np.ones(n)
            cov_inv = np.linalg.pinv(cov_matrix)

            target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 50)
            frontier_vol = []
            frontier_ret = []

            for ret in target_returns:
                try:
                    A = np.array([
                        [expected_returns.T @ cov_inv @ expected_returns,
                         expected_returns.T @ cov_inv @ ones],
                        [expected_returns.T @ cov_inv @ ones,
                         ones.T @ cov_inv @ ones]
                    ])
                    b = np.array([ret, 1])

                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    lambda1, lambda2 = x

                    weights = (lambda1 * (cov_inv @ expected_returns) +
                               lambda2 * (cov_inv @ ones))
                    weights /= weights.sum()

                    vol = np.sqrt(weights.T @ cov_matrix @ weights)
                    frontier_vol.append(vol)
                    frontier_ret.append(ret)
                except Exception as e:
                    self.log_message(f"Skipping frontier point: {str(e)}")
                    continue

            return np.array(frontier_ret), np.array(frontier_vol)
        except Exception as e:
            self.log_message(f"Efficient frontier calculation failed: {str(e)}")
            return np.array([]), np.array([])

    def plot_weight_comparison(self, ax):
        """Plot portfolio weight comparison"""
        try:
            width = 0.35
            x = np.arange(len(self.selected_assets))

            ax.bar(x - width / 2, self.markowitz_weights, width, label='Markowitz', alpha=0.8)
            ax.bar(x + width / 2, self.ga_weights, width, label='Genetic Algorithm', alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(self.selected_assets, rotation=45, ha='right')
            ax.set_title("Portfolio Weight Comparison")
            ax.legend()
            ax.grid(True)
        except Exception as e:
            ax.text(0.5, 0.5, 'Error plotting weights', ha='center', va='center')
            self.log_message(f"Weight plot error: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioOptimizer(root)
    root.mainloop()
