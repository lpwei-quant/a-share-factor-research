"""合成因子的分年度表现与多空尾部月份诊断。

用法: python diagnose.py
输入: results/composite_group_returns.csv、results/ic_monthly.csv（由 research.py 生成）
输出: results/composite_yearly.csv、results/long_short_worst_months.csv

要回答的问题: 合成因子年化 ICIR 超过 2.5, 为什么多空组合夏普不到 1?
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "results"
N_WORST = 3


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(12)


def main():
    rets = pd.read_csv(OUT / "composite_group_returns.csv", index_col=0, parse_dates=True)
    ic = pd.read_csv(OUT / "ic_monthly.csv", index_col=0, parse_dates=True)["composite"]
    cols = {"1": "第1组", "5": "第5组", "universe": "全市场等权", "long_short": "多空", "excess_top": "第5组超额"}
    rets = rets[list(cols)].rename(columns=cols)

    # 分年度: 各组合当年累计收益与当年月均 IC（2026 年截至 8 月）
    yearly = rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly["合成IC均值"] = ic.groupby(ic.index.year).mean()
    yearly.index.name = "年份"
    yearly.to_csv(OUT / "composite_yearly.csv", encoding="utf-8-sig")

    # 多空最差的月份: 亏损来自多头下跌还是空头上涨
    ls = rets["多空"]
    worst = rets.loc[ls.nsmallest(6).index, ["第1组", "第5组", "全市场等权", "多空"]]
    worst.insert(0, "当月合成IC", ic.reindex(worst.index))
    worst.index.name = "调仓日"
    worst.to_csv(OUT / "long_short_worst_months.csv", encoding="utf-8-sig")

    trimmed = ls.drop(ls.nsmallest(N_WORST).index)
    pd.set_option("display.width", 200)
    print(yearly.round(4))
    print(worst.round(4))
    print(f"各组年化波动: 第1组 {rets['第1组'].std() * np.sqrt(12):.3f}, 第5组 {rets['第5组'].std() * np.sqrt(12):.3f}")
    print(f"多空与全市场等权月收益相关系数: {ls.corr(rets['全市场等权']):.3f}")
    print(f"多空夏普: 全样本 {sharpe(ls):.2f}, 剔除最差 {N_WORST} 个月 {sharpe(trimmed):.2f}")
    print(f"最差 {N_WORST} 个月合计拖累（对数收益）: {np.log1p(ls.nsmallest(N_WORST)).sum():.3f}")


if __name__ == "__main__":
    main()
