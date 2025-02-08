# Portfolio Optimization Tool v0.9

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional portfolio optimization system combining Modern Portfolio Theory with evolutionary algorithms. Designed for financial analysis and quantitative research.

![App Screenshot](https://via.placeholder.com/800x400.png?text=GUI+Preview+Here)

## 🚀 Features
- **Dual Optimization** - Markowitz + Genetic Algorithm approaches
- **Brazil Market Focus** - Auto-.SA suffix handling for B3 tickers
- **Performance Metrics** - Sharpe Ratio, Max Drawdown, Cumulative Returns
- **Interactive Visualizations** - Embedded matplotlib charts in Tkinter
- **Data Pipeline** - Automated train/test split (80/20 period)

## 📦 Installation
Clone repo
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
text

## 🖥️ Usage
1. **Prepare Input File** (`input.xlsx`):
Tickers
PETR4.SA
VALE3.SA
ITUB4.SA

The Excel file must have the tickers specified in the first column, without title. 
2. In case of brazilian assets, .SA termination must be entered in the asset names in Excel file.

3. **Run Application**:
python portfolio_optimizer.py
text

4. **Workflow**:
Load Tickers → 2. Set Dates → 3. Download Data →
Calculate Metrics → 5. Run Optimizations → 6. Generate Report
text

## ⚙️ Technical Implementation
**Core Components**:
graph TD
A[Tkinter GUI] --> B[Data Manager]
B --> C[YFinance API]
B --> D[Optimization Engine]
D --> E[Markowitz Solver]
D --> F[Genetic Algorithm]
E --> G[Visualization System]
F --> G
text

**Key Formulas**:
- **Portfolio Return**: 
μ = Σ(w_i * r_i)
text
- **Portfolio Risk**: 
σ = √(wᵀΣw)
text
- **Efficient Frontier**: Solved via Lagrange multipliers

## 🛡️ Error Handling
| Error Type              | Solution                      |
|-------------------------|-------------------------------|
| Missing Yahoo Data       | Auto-retry with backoff       |
| Singular Covariance      | Pseudo-inverse + Regularization |
| Invalid Weights          | Auto-normalization            |

## 📅 Roadmap
- **v1.0** (Q4 2024): Transaction cost modeling
- **v1.1** (Q1 2025): Cloud data integration
- **v2.0** (2025): Machine learning forecasting

## 💻 Supported OS
| OS         | Status   |
|------------|----------|
| Windows 10 | ✅ Verified |
| Ubuntu 22  | ✅ Verified |
| macOS 13   | ⚠️ Beta    |

## 📜 License
MIT License - see [LICENSE](LICENSE) file

---

**Disclaimer**: This tool is for educational purposes only. Past performance ≠ future results. Consult financial professionals before making investment decisions.