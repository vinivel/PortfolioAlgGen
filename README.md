# PortfolioAlgGen

#### **General Description**
This code is designed to analyze and optimize investment portfolios using financial data from the Brazilian stock market (B3). It employs two primary methods for portfolio optimization: a genetic algorithm and a simplified Markowitz approach. Additionally, it compares the performance of these strategies against the Ibovespa benchmark.

The script is structured into distinct phases, making it easier to execute, from downloading financial data to plotting the results and exporting the final information to CSV files.

---

### **Code Structure**

#### **1. Imported Libraries**
- **Financial and Data Manipulation**:
  - `yfinance`: Used to download historical financial data for listed stocks.
  - `pandas`: For handling tabular datasets.
  - `numpy`: For numerical computations and randomization.
- **Dates and Visualization**:
  - `datetime` and `timedelta`: For date manipulations.
  - `matplotlib` and `seaborn`: For plotting performance charts.

#### **2. Key Functions**
Below is the description of the main functions implemented in the code:

1. **`calculate_portfolio_performance(weights, prices)`**:
   - **Purpose**: Calculates the cumulative return of a portfolio, given a set of weights (allocations) and a historical price dataset. The return is cumulative and converted into a percentage.
   - **Inputs**:
     - `weights`: Vector containing the allocated weights for each asset.
     - `prices`: DataFrame with historical closing prices for assets.
   - **Output**: Cumulative percentage return of the portfolio.

2. **`calculate_benchmark_performance(test_data)`**:
   - **Purpose**: Calculates the benchmark performance (Ibovespa) during the testing period.
   - **Input**:
     - `test_data`: DataFrame containing closing price data for the test period.
   - **Output**: Cumulative percentage return of the Ibovespa for the same test period.

3. **`download_and_prepare_data()`**:
   - **Purpose**: Downloads historical closing price data for a predefined list of stocks and splits the data into training and testing periods.
   - **Outputs**:
     - `train_data`: Data for the training period.
     - `test_data`: Data for the testing period.

4. **`optimize_portfolio_genetic(returns, n_assets=10)`**:
   - **Purpose**: Performs portfolio optimization using a simplified genetic algorithm. This method tests various weight combinations to find the optimal portfolio based on the return/risk ratio (Sharpe Ratio).
   - **Parameters**:
     - `returns`: Historical asset returns.
     - `n_assets`: Number of selected assets in the portfolio (default: 10).
   - **Output**: Optimized portfolio weights.

5. **`optimize_portfolio_markowitz(returns, n_assets=10)`**:
   - **Purpose**: Implements portfolio optimization based on Markowitz's theory, maximizing the Sharpe Ratio.
   - **Inputs/Outputs**: Same as the genetic algorithm optimization.

6. **`main()`**:
   - The entry-point function that orchestrates the script:
     - Downloads and prepares financial data.
     - Performs optimization using both Genetic and Markowitz methods.
     - Calculates portfolio and benchmark performances.
     - Generates visualizations and saves results to CSV files.

---

#### **3. Execution Flow**

1. **Preprocessing Financial Data**:
   - Downloads up to 24 months of stock closing prices.
   - Splits the data into:
     - Training period: Used for portfolio optimization.
     - Testing period: Used for strategy evaluation.

2. **Portfolio Optimization**:
   - Two methods are employed:
     - Genetic Algorithm.
     - Markowitz's Approach.

3. **Performance Calculation**:
   - Evaluates:
     - The cumulative returns of the optimized portfolios.
     - Benchmark (Ibovespa) performance.

4. **Visualization and Export**:
   - Plots the performances over time.
   - Saves performance metrics, portfolio allocations, and other results to CSV files.

---

#### **4. Metrics and Outputs**

- **Calculated Metrics**:
  - Cumulative Returns (%) for:
    - Genetic Algorithm strategy.
    - Markowitz strategy.
    - IBOV (benchmark).
  - Portfolio Weights (allocations) for each strategy.
  - Sharpe Ratios to identify optimal portfolios.

- **Generated Files**:
  - **CSV Files**:
    - `portfolio_performance.csv`: Accumulated returns for each portfolio and the benchmark.
    - `portfolio_compositions.csv`: Portfolio allocations (in percentage).
    - `performance_metrics.csv`: Total returns for each strategy in percentage.
  - **Image File**:
    - `portfolio_performance.png`: Line chart comparing cumulative performances of each portfolio.

---

#### **5. Key Functions Explained**

1. **Portfolio Performance Calculations**:
   The function `calculate_portfolio_performance` computes portfolio returns by multiplying daily price changes with asset allocations and accumulating the returns over time.

2. **Benchmark Performance**:
   The function `calculate_benchmark_performance` downloads Ibovespa (^BVSP) data and calculates its cumulative return over the test period.

3. **Data Preparation**:
   Stock price data is downloaded for a curated list of assets from `yfinance`. Then, the data is split into training and testing subsets to evaluate performances.

4. **Optimization Methods**:
   - **Genetic Algorithm**:
     - Randomly selects assets.
     - Generates weight combinations.
     - Optimizes for the highest Sharpe Ratio.
   - **Markowitz**:
     - Similar logic but follows a direct Sharpe Ratio estimation, without evolving generations like in Genetic Algorithm.

5. **Results Visualization**:
   - Matplotlib and seaborn are used to plot performance charts showing cumulative returns over time for the Genetic Algorithm, Markowitz, and IBOV strategies.

---

#### **6. Generated Outputs**
- The entire results analysis is saved locally:
  - **Performance Metrics (CSV)**: Cumulative returns and weights.
  - **Portfolio Composition (CSV)**: Stocks and their weight allocations.
  - **Performance Plot**: A `.png` file with the performance graph.

---

#### **7. Considerations and Improvements**
- **Dependencies**:
  - The code relies on `yfinance` for downloading stock market data, which can sometimes fail if API or connection issues occur.
- **Data Gaps**:
  - Missing or incomplete asset data may affect the portfolio's accuracy. Handling or interpolation methods are recommended for robust results.
- **Sharpe Ratio**:
  - Both optimization methods rely on maximizing Sharpe Ratios. Ensure this is an appropriate risk/reward metric for your financial goals.

---

### **Conclusion**
This script implements a systematic approach for portfolio analysis, optimization, and evaluation. It is highly useful for investors and analysts aiming to build optimized portfolios using widely adopted financial strategies while benchmarking their performance against a market index.

