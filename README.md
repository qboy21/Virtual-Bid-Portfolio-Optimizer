# Virtual-Bid-Portfolio-Optimizer


<b>Background</b>
Wholesale electricity markets in the US operate with a two settlement mechanism that consists of a real-time (RT) and day-ahead (DA) market. In both markets, participants make supply offers and demand bids to deliver or consume power at a particular location (i.e. node on the electrical grid) and time interval (e.g. hour of day). The RT market accepts bids on a rolling basis for immediate delivery, while the DA market accepts bids during a fixed time window every day for each hour of the following day (the “operating day”). The markets also allow for purely financial participation via virtual bidding, in which participants arbitrage price differences across the RT and DA markets at a particular node. These bids are either virtual supply offers (INCs), with which power is sold in the DA market and purchased in the RT market, or virtual demand bids (DECs), with which power is purchased in the DA market and sold in the RT market.

<b>Goals</b>
The goal of this challenge is to develop a simple virtual bidding strategy based on Harry Markowitz’s Modern Portfolio Theory, and to backtest the strategy on some provided historical data. You will be provided with skeleton Python code for your solution, with stubs for all of the functions that must be implemented to complete a working solution, as well as the historical price data you should use for your backtest. Your task is to write a program that will execute the strategy and backtest it over the last year of data, outputting the bids produced by the strategy during the backtest period. Although the skeleton code is written in Python, you are welcome to use any language of your choice provided you can find a package with an optimization capability similar to the one from SciPy used in the skeleton.

<b>Mean-Variance Optimization for Virtual Bidding</b>

Modern Portfolio Theory (MPT) is a classic mathematical framework from the field of quantitative finance for constructing portfolios of assets that maximize expected returns for a given level of risk. In MPT, risk is measured using the variance of the portfolio return. The “efficient” portfolio (i.e. the portfolio with the greatest return) for a given risk level is found by minimizing the following expression:
𝑤𝑇Σ𝑤 − 𝑞×𝑅𝑇𝑤
where
● 𝑤 is the vector of portfolio weights, which can be positive or negative, and ∑ 𝑤𝑖 = 1. 𝑖

● Σ is the covariance matrix for the returns of the assets in the portfolio.
  
● 𝑞 ≥ 0 is the “risk tolerance” factor, where 0 gives a portfolio with minimal risk and larger values give riskier portfolios.

● 𝑅 is the vector of expected returns.

● 𝑤𝑇Σ𝑤 is the variance of the portfolio return.

● 𝑅𝑇𝑤 is the expected return of the portfolio.

In the typical setting, the assets in the portfolio would be financial assets such as stocks or bonds, but we can apply the same framework to virtual bidding in electricity markets by taking power delivered at a certain node and hour of day to be the “asset” and using the difference (or spread) between the DA and RT prices as the “return”. The “weights” then become the quantities of our bids (in MW) with the sign determining the direction (buy/sell or DEC/INC). Rather than summing to 1, the weights instead sum to a maximum volume (in MW) that we wish to trade on any day.
Concretely, in the virtual bidding setting:

● 𝑤 is the vector of bid quantities, with one entry per node and hour combination, which
can be positive or negative, and ∑ |𝑤𝑖| ≤ 𝑉, where 𝑉 is the daily volume limit. 𝑖

● 𝑅 is the vector of expected “DART spreads” or differences between DA and RT prices at each location and hour.

● Σ is the covariance matrix for the DART spreads.

Note that for simplicity we’ll assume that our strategy is “self-scheduling”: it will always take whatever price the market offers for a bid, and the entire quantity of the bid will clear. This is analogous to a market order in other financial markets.
Your solution should use fixed values of 500 for the risk tolerance factor, 𝑞, and 100 for the daily volume limit, 𝑉. These are defined as constants for you in the skeleton code.

Simplifying the Covariance Matrix

One of the biggest challenges with mean-variance portfolio optimization is coming up with good estimates for the covariance matrix, Σ. To simplify this step, for our strategy we’ll estimate only the diagonal entries of the covariance matrix, which represent the variance of the DART spread at a single node and hour combination. We’ll use 0 for all other entries, which is equivalent to the assumption that DART spreads are not correlated across nodes and hours. In general this is a poor assumption but it works reasonably well in practice and vastly simplifies the problem.

Constrained Optimization with SciPy

To perform the minimization of the above mathematical expression, the skeleton solution uses the optimization package included with SciPy, specifically its minimization routine of an arbitrary objective function with an inequality constraint. Part of your task is to define the objective function corresponding to this expression as well as a function describing the constraint over the bid quantities, which must take a non-negative value if and only if the constraint is satisfied. The objective function should accept as arguments the bid quantities (𝑤), expected spreads (𝑅), and spread variance estimates (diagonal entries of Σ), while the constraint function should accept the bid quantities (𝑤) and maximum daily volume limit (𝑉).
Point-In-Time Mean and Variance Estimates

To estimate the expected DART spreads and their variances, we’ll take the sample mean and variance observed over all days in our historical data. Concretely, to estimate the spread and spread variance for node 𝑖 and hour 𝑗, we will take the sample mean and variance, respectively, of historical spread observations for node 𝑖 and hour 𝑗.
To produce an accurate backtest, however, we need to censor any data that would not have been observable at the time we would have placed our bids. Since virtual bids are placed in the day-ahead market, we must exclude any observations later than one day prior to the operating day for which we are bidding. Concretely, when calculating the expected spreads and spread variances for bids for operating day 𝑑, exclude all observations after day 𝑑 − 2 (although technically some observations for day 𝑑 − 1 are available in time for trading, it’s simpler to use the same cutoff for all hours). So when estimating the spread/variance for hour 23 at node ARKANSAS.HUB on operating 2022-12-10, the last observation in our sample should be hour 23 at node ARKANSAS.HUB on operating day 2022-12-08.

Data

The input data consists of historical price data across 8 nodes from 2021 to 2022. It is provided in a CSV file called “prices.csv” with the following schema:
