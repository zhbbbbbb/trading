import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cumulative return
def cumul_ret(rets) -> float:
    try:
        return (rets.fillna(0) + 1).prod() - 1
    except:
        pass
    rets_copy = [ret + 1 for ret in rets if ret]
    return np.prod(rets_copy) - 1.0

# annualized return
def annual_ret(rets) -> float:
    rets_copy = rets.fillna(0)
    try:
        totalSeconds = abs((rets_copy.index[-1] - rets_copy.index[0]).total_seconds())
    except:
        totalSeconds = abs((pd.Timestamp(rets_copy.index[-1]) - pd.Timestamp(rets_copy.index[0])).total_seconds())
    scalar = (365.0 * 24.0 * 60.0 * 60.0) / totalSeconds
    return (1.0 + cumul_ret(rets)) ** scalar - 1.0

# stdev considering only negative returns
def risk(rets, downside:bool=True, annualize:bool=True) -> float:
    if annualize:
        try:
            totalSeconds = abs((rets.index[-1] - rets.index[0]).total_seconds())
        except:
            totalSeconds = abs((pd.Timestamp(rets.index[-1]) - pd.Timestamp(rets.index[0])).total_seconds())
        scalar = (365.0 * 24.0 * 60.0 * 60.0) / (totalSeconds / len(rets))

    if downside:
        rets = [ret for ret in rets if ret < 0.0]
    if downside and not rets:
        raise Exception("No downside risk.")

    if annualize:
        return np.std(rets) * np.sqrt(scalar)
    else:
        return np.std(rets)

# calculate annualized sharpe ratio
def sharpe(rets, rf:float = 0.0, downside:bool = False) -> float:
    return (annual_ret(rets) - rf) / risk(rets, downside=downside, annualize=True)

def max_drawdown(rets):
    rets = [np.log(ret+1.0) for ret in rets]
    currentSum, maxDrawdown= rets[0], rets[0]
    for i in range(1, len(rets)):
        currentSum = min(rets[i], rets[i] + currentSum)
        maxDrawdown = min(maxDrawdown, currentSum)
    return np.exp(maxDrawdown) - 1.0

# compute the path (growth) of investment using rets
def path(rets):
    rets_copy = rets.copy()
    rets_copy[0] += 1.0
    for i in range(1, len(rets_copy)):
        rets_copy[i] = rets_copy[i-1] * (1+rets_copy[i])
    return rets_copy

# plot the path (growth) of investment using rets
def plot_path(rets, rets_asset=None, rf = 0.0, downside: bool = True, daily_xticks: bool = True, segments: int = 3, show=True):
    wealth = [v for v in path(rets)]
    foo = lambda x : x.split(' ')[0] if daily_xticks else lambda x : x
    plt.plot(wealth, linewidth=1, label='Strategy', color='Blue')

    if rets_asset is not None:
        wealth_asset = [value for value in path(rets_asset)]
        plt.plot(wealth_asset, linewidth=1, label='Holding the asset', color='Black')
        plt.legend(loc='best')

    if daily_xticks:
        try:
            plt.xticks(ticks=range(len(wealth))[::len(wealth)//segments-1], labels = rets.index[::len(rets)//segments-1].map(foo))
        except:
            plt.xticks(ticks=range(len(wealth))[::len(wealth)//segments-1], labels = rets.index[::len(rets)//segments-1].map(lambda x: foo(str(x.date()))))
    else:
        plt.xticks(ticks=range(len(wealth))[::len(wealth)//segments-1], labels = rets.index[::len(rets)//segments-1])

    print(f"cumulative return: {cumul_ret(rets)*100:.2f}%", end='')
    if rets_asset is not None:
        print(f'\t{cumul_ret(rets_asset)*100:.2f}%')
    else:
        print()
    print(f"    annual return: {annual_ret(rets)*100:.2f}%", end='')
    if rets_asset is not None:
        print(f'\t{annual_ret(rets_asset)*100:.2f}%')
    else:
        print()
    print(f"annual volatility: {risk(rets, downside=downside)*100:.2f}%", end='')
    if rets_asset is not None:
        print(f'\t{risk(rets_asset, downside=downside)*100:.2f}%')
    else:
        print()
    print(f"           sharpe: {sharpe(rets, rf=rf, downside = downside):.2f}", end='')
    if rets_asset is not None:
        print(f'\t\t{sharpe(rets_asset, rf=rf, downside = downside):.2f}')
    else:
        print()
    print(f"     max drawdown: {max_drawdown(rets)*100:.2f}%", end='')
    if rets_asset is not None:
        print(f'\t{max_drawdown(rets_asset)*100:.2f}%')
    else:
        print()

    # text = f"cumulative return: {cumul_ret(rets)*100:.2f}%" + "\n" + \
    #        f"annual return:        {annual_ret(rets)*100:.2f}%" + "\n" + \
    #        f"annual volatility:    {risk(rets, downside=downside)*100:.2f}%" + "\n" + \
    #        f"sharpe:                  {sharpe(rets, rf=rf, downside = downside):.2f}" + "\n" + \
    #        f"max drawdown:     {max_drawdown(rets)*100:.2f}%"
    # plt.text(0, -0.5, text, fontsize=12, fontweight='medium')

    if show:
        plt.show()


def mean_absolute_deviations(X):
    mean = np.mean(X)
    return np.abs(X - mean).mean()
