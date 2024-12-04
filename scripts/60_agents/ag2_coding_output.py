#%%
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the cryptocurrencies
cryptos = ['ETH-USD', 'SOL-USD']

# Fetch the historical price data for the year 2023
data = yf.download(cryptos, start='2023-01-01', end='2023-12-31')

# Get the closing prices
close_prices = data['Close']

# Calculate YTD price change
ytd_price_changes = close_prices.iloc[-1] - close_prices.iloc[0]
ytd_price_changes_percent = (ytd_price_changes / close_prices.iloc[0]) * 100

# Plotting
plt.figure(figsize=(10, 6))
ytd_price_changes_percent.plot(kind='bar', color=['blue', 'orange'])
plt.title('YTD Price Change Percentage of ETH and SOL in 2023')
plt.ylabel('Price Change (%)')
plt.xticks(ticks=range(len(cryptos)), labels=['Ethereum (ETH)', 'Solana (SOL)'], rotation=0)
plt.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.savefig('eth_sol_ytd_change.png')
plt.show()
# %%
