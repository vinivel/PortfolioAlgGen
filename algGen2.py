import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Risk-free rate (annualized)
RISK_FREE_RATE = 0.05  # Example: 5% annual risk-free rate

# Predefined list of assets (Brazilian stocks)
STOCKS = [
    'BPAC11.SA', 'EQTL3.SA', 'SUZB3.SA', 'PETR3.SA', 'B3SA3.SA', 'ITSA4.SA',
    'ITUB4.SA', 'PRIO3.SA', 'VALE3.SA', 'WEGE3.SA', 'RDOR3.SA', 'BBAS3.SA',
    'ENEV3.SA', 'JBSS3.SA', 'VBBR3.SA', 'ABEV3.SA', 'BBDC4.SA', 'UGPA3.SA',
    'SBSP3.SA', 'EMBR3.SA', 'GGBR4.SA', 'RADL3.SA', 'ELET3.SA', 'CMIG4.SA',
    'BBSE3.SA', 'RENT3.SA', 'RAIL3.SA'
]


# Function to calculate fitness (Sharpe Ratio)
def fitness(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    return sharpe_ratio


# Markowitz optimization function
def markowitz_optimization(returns):
    n_assets = returns.shape[1]
    result = minimize(lambda w: -fitness(w, returns),
                      np.ones(n_assets) / n_assets,
                      method='SLSQP',
                      bounds=[(0, 1)] * n_assets,
                      constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}])
    return result.x


# Download stock data from Yahoo Finance
def download_stock_data(start_date, end_date):
    valid_tickers = []
    all_data = pd.DataFrame()

    for ticker in STOCKS:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
            if not data.empty:
                valid_tickers.append(ticker)
                all_data[ticker] = data
            else:
                print(f"Warning: No data available for {ticker} (empty dataset).")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    if all_data.empty:
        messagebox.showerror("Error", "No valid stock data could be downloaded.")
        return None

    print(f"Successfully downloaded data for {len(valid_tickers)} tickers.")
    return all_data


# Plot results and save the graph
def plot_results(performance_df):
    plt.figure(figsize=(12, 6))

    for column in performance_df.columns:
        plt.plot(performance_df.index, performance_df[column], label=column)

    plt.title('Portfolio Performance vs IBOV')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save and display the plot
    plt.savefig('portfolio_performance.png')
    plt.show()


# Main function for portfolio optimization and evaluation
def optimize_portfolio():
    # Reset progress bar
    progress['value'] = 0

    # Get user inputs
    start_date = start_date_entry.get_date()
    end_date = end_date_entry.get_date()

    # Validate inputs
    if not start_date or not end_date:
        messagebox.showwarning("Input Error", "Please select both start and end dates.")
        return

    # Adjust training period (252 days before the start date)
    training_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=365)

    # Download stock data for training and evaluation periods
    progress['value'] += 10
    stock_data = download_stock_data(training_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if stock_data is None:
        return

    # Split into training and evaluation periods
    training_data = stock_data[:start_date]
    evaluation_data = stock_data[start_date:end_date]

    # Calculate daily returns for training and evaluation periods
    training_returns = training_data.pct_change().dropna()
    evaluation_returns = evaluation_data.pct_change().dropna()

    # Optimize portfolio using Markowitz method on training data
    progress['value'] += 30
    markowitz_weights = markowitz_optimization(training_returns)

    # Evaluate portfolio performance during the evaluation period
    progress['value'] += 30
    markowitz_performance = (1 + (evaluation_returns @ markowitz_weights)).cumprod().squeeze() - 1

    # Download IBOV data for comparison
    ibov_data = \
    yf.download('^BVSP', start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)[
        'Close']

    if ibov_data is None or ibov_data.empty:
        messagebox.showwarning("Error", "Failed to download IBOV data.")
        return

    ibov_returns = ibov_data.pct_change().dropna()
    ibov_performance = (1 + ibov_returns).cumprod().squeeze() - 1

    # Create a DataFrame for performance comparison
    performance_df = pd.DataFrame({
        'Markowitz': markowitz_performance,
        'IBOV': ibov_performance[start_date:]
    })

    # Plot results and display table with weights and performance metrics
    plot_results(performance_df)


# Create the Tkinter interface
root = tk.Tk()
root.title("Portfolio Optimization")
root.geometry("800x600")

tk.Label(root, text="Start Date:").pack(pady=5)
start_date_entry = DateEntry(root, width=12, background='darkblue',
                             foreground='white', borderwidth=2)
start_date_entry.pack(pady=5)

tk.Label(root, text="End Date:").pack(pady=5)
end_date_entry = DateEntry(root, width=12, background='darkblue',
                           foreground='white', borderwidth=2)
end_date_entry.pack(pady=5)

progress_label = tk.Label(root, text="Progress:")
progress_label.pack(pady=10)

progress = ttk.Progressbar(root, orient=tk.HORIZONTAL,
                           length=300, mode='determinate')
progress.pack(pady=10)

optimize_button = tk.Button(root,
                            text="Optimize Portfolio",
                            command=optimize_portfolio)
optimize_button.pack(pady=20)

table_frame = tk.Frame(root)
table_frame.pack(fill="both", expand=True)

root.mainloop()
