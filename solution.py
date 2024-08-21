import datetime
import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp  # type: ignore[import-untyped]
import tqdm
import matplotlib.pyplot as plt

FloatArray = npt.NDArray[np.float_]

RISK_TOLERANCE = 500
DAILY_VOLUME_LIMIT = 100


def compute_expected_spread(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the expected spread for each node and hour for the operating day, by taking the mean of observed
    spread values available as of 24 hours prior to the beginning of the operating day.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :return: data frame of expected spread values [operating_day, hour_beginning, node, expected_dart_spread]
    """
    # Calculate the expected DART spread and sort the dataframe
    spread_df = (price_df[["operating_day", "hour_beginning", "node"]]
                 .assign(expected_dart_spread=price_df.da_price - price_df.rt_price,)
                 .sort_values(["operating_day", "hour_beginning", "node"])
                 )
    
    # Group by hour_beginning and node, then calculate the mean
    mean_df = (spread_df.set_index("operating_day")
               .groupby(["hour_beginning", "node"])["expected_dart_spread"]
               .expanding()
               .mean()
               .reset_index()
               )
    # Adjust the operating day (excluding all observations after d-2)
    return mean_df.assign(operating_day=mean_df.operating_day + pd.Timedelta(2, "days"))


def compute_spread_variance(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the estimated spread variance for each node and hour for the operating day, by taking the sample variance
    of observed spread values available as of 24 hours prior to the beginning of the operating day.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :return: data frame of estimated spread variance values [operating_day, hour_beginning, node, dart_spread_var]
    """
    # Calculate the expected DART spread and sort the dataframe
    spread_df = (price_df[["operating_day", "hour_beginning", "node"]]
                 .assign(expected_dart_spread=price_df.da_price - price_df.rt_price,)
                 .sort_values(["operating_day", "hour_beginning", "node"])
                 )
    
    # Group by hour_beginning and node, calculate the variance and rename the column
    var_df = (spread_df.set_index("operating_day")
              .groupby(["hour_beginning", "node"])["expected_dart_spread"]
              .expanding()
              .var()
              .reset_index()
              .rename(columns={"expected_dart_spread":"dart_spread_var"})
              )

    # Adjust the operating day (excluding all observations after d-2)
    return var_df.assign(operating_day=var_df.operating_day + pd.Timedelta(2, "days"))


def get_daily_expected_spread_vectors(expected_spread_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the expected spread data frame into a data frame with one row per operating day, where the index is
    the operating day and the elements of each row are the expected spread values for all node and hour combinations
    on that day.

    :param expected_spread_df: data frame of expected spread values
        [operating_day, hour_beginning, node, expected_dart_spread]
    :return: data frame of expected spread vectors with operating day as index
    """
    # Reshape the dataframe and set index to operating_day creating a NultiIndex for the column levels and corresponding values
    return expected_spread_df.pivot(index="operating_day", columns=["hour_beginning", "node"], values="expected_dart_spread")


def get_daily_spread_variance_vectors(spread_var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the spread variance data frame into a data frame with one row per operating day, where the index is
    the operating day and the elements of each row are the estimated spread variance values for all node and hour
    combinations on that day (i.e. the diagonal entries of the covariance matrix).

    :param spread_var_df: data frame of expected spread values
        [operating_day, hour_beginning, node, dart_spread_var]
    :return: data frame of expected spread vectors with operating day as index
    """
    # Reshape the dataframe and set index to operating_day creating a NultiIndex for the column levels and corresponding values
    return spread_var_df.pivot(index="operating_day", columns=["hour_beginning", "node"], values="dart_spread_var")


def portfolio_objective_fn(bid_mw: FloatArray, expected_spread: FloatArray, spread_variance: FloatArray,) -> float:
    """
    The objective function to minimize in the portfolio optimizer. This should also use the RISK_TOLERANCE constant
    defined above.

    :param bid_mw: array containing the bid quantities (in MW) for the daily portfolio
    :param expected_spread: array containing the expected spread values for the day
    :param spread_variance: array containing the estimated spread variance values for the day (i.e. the diagonal
        entries of the covariance matrix)
    :return: objective function value to minimize
    """
    # Total Risk minus Risk-Adjusted Expected Return: Squaring bid_mw reflects the financial risk that increases quadratically with position size (non-linear)
    return np.dot(spread_variance, bid_mw ** 2) - RISK_TOLERANCE * np.dot(expected_spread, bid_mw)


def mw_constraint_fn(bid_mw: FloatArray, max_total_mw: float) -> float:
    """
    The constraint function which must take a non-negative value if and only if the constraint is satisfied.

    :param bid_mw: array containing the bid quantities (in MW) for the daily portfolio
    :param max_total_mw: the maximum number of total MW that can be traded in a day
    :return: constraint function value which must be non-negative iff the constraint is satisfied
    """
    # Ensure that the total absolute value of the bid quantities does not exceed a specified maximum allowable total
    return max_total_mw - np.abs(bid_mw).sum()


def get_bids_from_daily_portfolios(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a data frame of daily portfolios to a data frame of bids. Also removes any bids smaller than 0.1 MW.

    :param portfolio_df: data frame of daily bid quantities with operating day as the index
    :return: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    """
    # Stack the DataFrame to create a single column of bid quantities
    bid_df = portfolio_df.stack(level=[0, 1],future_stack=True).rename("bid_mw").reset_index()

    # Assign bid type and bid magnitude, and filter out small bids
    bid_df = (bid_df.assign(bid_type=np.where(bid_df["bid_mw"] > 0, "INC", "DEC"),
                            bid_mw=bid_df["bid_mw"].abs()).loc[lambda df: df["bid_mw"] >= 0.1])

    return bid_df


def compute_total_pnl(price_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """
    Compute the total PnL over all operating days in the bid data frame

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param bid_df: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    :return: the total PnL
    """
    # Merge the price data and bid data on operating_day and hour_beginning
    bid_pnl = pd.merge(price_df, bid_df, on=["operating_day", "hour_beginning"])

    # Compute the PnL for each bid (INC or DEC)
    bid_pnl = bid_pnl.assign(pnl=np.where(bid_pnl["bid_type"] == "INC",
                                        bid_pnl["bid_mw"] * (bid_pnl["da_price"] - bid_pnl["rt_price"]),
                                        bid_pnl["bid_mw"] * (bid_pnl["rt_price"] - bid_pnl["da_price"])))

    # Aggregate daily PnL
    daily_pnl = bid_pnl.groupby("operating_day")["pnl"].sum().reset_index().sort_values("operating_day")

    # Compute cumulative PnL
    daily_pnl = daily_pnl.assign(cumpnl=daily_pnl["pnl"].cumsum())

    return daily_pnl


def generate_daily_bids(price_df: pd.DataFrame, first_operating_day: t.Union[str, datetime.date], last_operating_day: t.Union[str, datetime.date],) -> pd.DataFrame:
    """
    Generate bids for the date range, computing the expected DART spreads and estimated variances from
    the price data and limiting each daily portfolio to a maximum size in MW.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param first_operating_day: first operating day for which to generate bids
    :param last_operating_day: last operating day for which to generate bids
    :return: data frame of bids [operating_day, hour_beginning, node, bid_type, bid_mw]
    """
    expected_spread = compute_expected_spread(price_df)
    spread_variance = compute_spread_variance(price_df)

    daily_expected_spread = get_daily_expected_spread_vectors(expected_spread)
    daily_spread_variance = get_daily_spread_variance_vectors(spread_variance)

    portfolios = []
    for day in tqdm.tqdm(pd.date_range(first_operating_day, last_operating_day)):
        result = sp.optimize.minimize(
            portfolio_objective_fn,
            np.zeros(len(daily_expected_spread.columns)),
            args=(daily_expected_spread.loc[day].values, daily_spread_variance.loc[day].values),
            constraints={
                "type": "ineq",
                "fun": mw_constraint_fn,
                "args": [DAILY_VOLUME_LIMIT],
            },
        )
        portfolios.append(
            pd.DataFrame(
                result.x[None, :],
                columns=daily_expected_spread.columns,
                index=pd.Index([day], name="operating_day")
            )
        )

    return get_bids_from_daily_portfolios(pd.concat(portfolios))


def load_price_data(path: str) -> pd.DataFrame:
    """
    Load historical price data

    :param path: path to a CSV of the price data
    :return: data frame of historical prices [operating_day, hour_beginning, node, da_price, rt_price]
    """
    return pd.read_csv(path, parse_dates=["operating_day"])


def calculate_sharpe_ratio(pnl_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio.

    :param pnl_df: DataFrame with columns 'operating_day' and 'pnl'
    :param risk_free_rate: The risk-free rate, default is 0.0
    :return: Sharpe ratio
    """
    # Compute daily returns from pnl values
    daily_returns = pnl_df.pnl.pct_change().dropna()
    
    # Calculate mean return and standard deviation of the daily returns
    mean_returns = daily_returns.mean() * 365
    return_volatility = daily_returns.std() * np.sqrt(365)

    # Calculate Sharpe ratio
    sharpe_ratio = (mean_returns- risk_free_rate) / return_volatility 
 
    return sharpe_ratio

def plot_pnl(pnl_df: pd.DataFrame) -> None:
    """
    Plot daily PnL as bars and cumulative PnL as a line.

    :param pnl_df: DataFrame with columns 'operating_day', 'pnl', and 'cumpnl'
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot daily PnL as bars
    ax1.bar(pnl_df['operating_day'], pnl_df['pnl'], color=np.where(pnl_df['pnl'] >= 0, 'g', 'r'), alpha=0.6)
    ax1.set_xlabel('Operating Day')
    ax1.set_ylabel('Daily PnL')
    ax1.set_title('Daily Profit and Loss')

    # Plot cumulative PnL as a line
    ax2.plot(pnl_df['operating_day'], pnl_df['cumpnl'], color='b', marker='o', linestyle='-')
    ax2.set_xlabel('Operating Day')
    ax2.set_ylabel('Cumulative PnL')
    ax2.set_title('Cumulative Profit and Loss')
    
    plt.show()
    
    
def tune_risk_tolerance(price_df: pd.DataFrame,risk_free_rate: float = 0.0,tolerance_range: t.Tuple[int, int] = (100, 1000),step_size: int = 50,) -> float:
    """
    Optimize the RISK_TOLERANCE factor to maximize the risk-adjusted return based on the Sharpe ratio.

    :param price_df: historical price data [operating_day, hour_beginning, node, da_price, rt_price]
    :param risk_free_rate: The risk-free rate, default is 0.0
    :param tolerance_range: Range of tolerance values to test
    :param step_size: Step size for testing tolerance values
    :return: Optimized RISK_TOLERANCE factor
    """
    best_sharpe_ratio = float("-inf")
    optimal_tolerance = 0
    
    for tolerance in range(*tolerance_range, step_size):
        global RISK_TOLERANCE  # Ensure we modify the global variable
        RISK_TOLERANCE = tolerance

        sharpe_ratio = calculate_sharpe_ratio(pnl_df, risk_free_rate=risk_free_rate)
        
        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            optimal_tolerance = tolerance
            
    return optimal_tolerance


if __name__ == "__main__":
    price_df = load_price_data("prices.csv")

    print(f"Generating bids for 2022 with daily limit of 100 MW...")
    bid_df = generate_daily_bids(price_df, "2022-01-01", "2022-12-31")
    pnl_df = compute_total_pnl(price_df, bid_df)

    bid_df.to_csv("bids.csv", index=False)
    print(f"The strategy made ${pnl_df.pnl.sum():.2f}")
    
    sharpe_ratio = calculate_sharpe_ratio(pnl_df, risk_free_rate=0.04434)
    print(f"The Sharpe ratio is {sharpe_ratio:.2f}")
    
    optimal_risk_tolerance = tune_risk_tolerance(price_df, risk_free_rate=0.04434)
    print(f"Optimal risk tolerance factor is {optimal_risk_tolerance}")
    
    plot_pnl(pnl_df)
    
