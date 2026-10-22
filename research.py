"""A 股量价因子研究: 因子构建、IC 分析、分层回测。

用法: python research.py [数据目录]
输入: fetch_data.py 生成的 daily_hfq.parquet、stock_basic.csv、industry_csrc.csv
输出: results/ 下的 csv、summary.json 与 figures/*.png

研究约定（在查看结果前确定）:
- 每月最后一个交易日收盘调仓, 持有到下月最后一个交易日收盘
- 合成因子 = 反转、低波动、低换手、非流动性 四个中性化因子等权
- 交易成本按单边 0.15% 计, 作用于每期组合换手
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
FIRST_REBAL = "2019-01-01"
COST = 0.0015
N_GROUPS = 5
MIN_STOCKS = int(sys.argv[2]) if len(sys.argv) > 2 else 300   # 截面股票数下限
FACTORS = {
    "rev_20": "20日反转",
    "mom_12_1": "12-1月动量",
    "vol_20": "20日低波动",
    "turn_20": "20日低换手",
    "illiq_20": "Amihud非流动性",
    "size": "小市值",
}
COMPOSITE_PARTS = ["rev_20", "vol_20", "turn_20", "illiq_20"]
SPLIT = "2023-01-01"


def load():
    daily = pd.read_parquet(DATA_DIR / "daily_hfq.parquet")
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    basic = pd.read_csv(DATA_DIR / "stock_basic.csv", dtype=str).fillna("")
    ind = pd.read_csv(DATA_DIR / "industry_csrc.csv", dtype=str).fillna("")
    wide = {c: daily.pivot(index="date", columns="code", values=c)
            for c in ["close", "amount", "turn", "volume"]}
    # baostock 代码形如 sh.600000, 行情代码为 600000
    industry = ind.assign(code=ind["code"].str[3:]).set_index("code")["industry"].str[:3].replace("", "NA")
    ipo = pd.to_datetime(basic.assign(code=basic["code"].str[3:]).set_index("code")["ipoDate"])
    return wide, industry, ipo


def build_factors(w):
    close = w["close"].ffill()                      # 停牌/退市后沿用最后价格
    traded = w["volume"] > 0                        # 停牌日无行情记录, pivot 后为 NaN
    ret = w["close"].pct_change(fill_method=None).where(traded)
    amount = w["amount"].where(traded)
    turn = w["turn"].where(traded & (w["turn"] > 0))
    cap = (amount / (turn / 100)).rolling(20, min_periods=10).median()  # 流通市值近似

    f = {
        "rev_20": -(close / close.shift(20) - 1),
        "mom_12_1": close.shift(20) / close.shift(250) - 1,
        "vol_20": -ret.rolling(20, min_periods=15).std(),
        "turn_20": -turn.rolling(20, min_periods=15).mean(),
        "illiq_20": np.log((ret.abs() / amount).rolling(20, min_periods=15).mean() * 1e9 + 1e-12),
        "size": -np.log(cap),
    }
    valid_days = ret.notna().rolling(20).sum()
    eligible = traded & (valid_days >= 15)
    return f, close, eligible, np.log(cap)


def rebal_dates(index):
    """各月最后一个交易日; 最后一个日期只用于计算上一期的持有收益。"""
    s = pd.Series(index, index=index)
    ends = s.groupby([index.year, index.month]).max()
    return [d for d in ends if d >= pd.Timestamp(FIRST_REBAL)]


def winsor_z(x):
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    x = x.clip(med - 5 * mad, med + 5 * mad)
    return (x - x.mean()) / x.std()


def neutralize(y, logcap, industry, with_size=True):
    X = pd.get_dummies(industry.reindex(y.index).fillna("NA"), dtype=float)
    if with_size:
        X["logcap"] = winsor_z(logcap.reindex(y.index))
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    resid = pd.Series(y.values - X.values @ beta, index=y.index)
    return (resid - resid.mean()) / resid.std()


def cross_sections(f, close, eligible, logcap, industry, ipo, dates, min_cap_pct=0.0, jumps_cum=None,
                   lag=0):
    """返回 {date: DataFrame(因子..., composite, fwd_ret[1..6])}

    jumps_cum: 单日 |收益| > 21% 的累计次数; 传入时剔除持有期内出现此类跳空（多为长期停牌后复牌）的股票。
    lag: 成交延后的交易日数; 1 表示调仓日收盘计算因子、次一交易日收盘成交。
    """
    days = close.index
    trade_day = {d: days[min(days.get_loc(d) + lag, len(days) - 1)] for d in dates}
    out = {}
    for i, t in enumerate(dates[:-1]):
        age_ok = ((t - ipo.reindex(close.columns)).dt.days >= 365).values
        ok = eligible.loc[t] & age_ok
        codes = ok[ok].index
        lc = logcap.loc[t, codes].dropna()
        if min_cap_pct > 0:
            lc = lc[lc >= lc.quantile(min_cap_pct)]
        raw = pd.DataFrame({k: v.loc[t, lc.index] for k, v in f.items()}).dropna()
        if len(raw) < MIN_STOCKS:
            continue
        df = pd.DataFrame(index=raw.index)
        for k in FACTORS:
            z = winsor_z(raw[k])
            df[k] = neutralize(z, lc, industry, with_size=(k != "size"))
        df["composite"] = df[COMPOSITE_PARTS].mean(axis=1)
        if jumps_cum is not None:
            jumped = jumps_cum.loc[dates[i + 1], df.index] - jumps_cum.loc[t, df.index] > 0
            df = df[~jumped]
        for h in range(1, 7):
            if i + h < len(dates):
                end, start = trade_day[dates[i + h]], trade_day[dates[i + h - 1]]
                df[f"fwd_{h}"] = close.loc[end, df.index] / close.loc[start, df.index] - 1
        out[t] = df
    return out


def ic_table(cs, cols):
    rows = {}
    for t, df in cs.items():
        rows[t] = {c: df[c].corr(df["fwd_1"], method="spearman") for c in cols}
    return pd.DataFrame(rows).T


def ic_summary(ic):
    n = ic.count()
    return pd.DataFrame({
        "IC均值": ic.mean(), "IC标准差": ic.std(), "ICIR": ic.mean() / ic.std(),
        "年化ICIR": ic.mean() / ic.std() * np.sqrt(12), "t值": ic.mean() / ic.std() * np.sqrt(n),
        "IC>0占比": (ic > 0).mean(), "期数": n,
    })


def ic_decay(cs, col):
    out = {}
    for h in range(1, 7):
        vals = [df[col].corr(df[f"fwd_{h}"], method="spearman") for df in cs.values() if f"fwd_{h}" in df]
        out[f"滞后{h}月"] = np.nanmean(vals)
    return pd.Series(out)


def group_backtest(cs, col):
    """五分组等权回测, 扣除换手成本。返回各组净收益序列与换手率。"""
    rets, turnover, prev = {}, {}, {}
    for t, df in cs.items():
        d = df[[col, "fwd_1"]].dropna()
        g = pd.qcut(d[col].rank(method="first"), N_GROUPS, labels=range(1, N_GROUPS + 1))
        r, to = {}, {}
        for k in range(1, N_GROUPS + 1):
            members = d.index[g == k]
            w = pd.Series(1 / len(members), index=members)
            pw = prev.get(k, pd.Series(dtype=float))
            traded = w.sub(pw, fill_value=0).abs().sum()
            to[k] = traded / 2
            r[k] = d.loc[members, "fwd_1"].mean() - COST * traded
            prev[k] = w
        r["universe"] = d["fwd_1"].mean()
        rets[t], turnover[t] = r, to
    rets = pd.DataFrame(rets).T
    rets["long_short"] = rets[N_GROUPS] - rets[1]
    rets["excess_top"] = rets[N_GROUPS] - rets["universe"]
    return rets, pd.DataFrame(turnover).T


def perf(r):
    nav = (1 + r).cumprod()
    n = r.count()
    return {
        "年化收益": nav.iloc[-1] ** (12 / n) - 1,
        "年化波动": r.std() * np.sqrt(12),
        "夏普": r.mean() / r.std() * np.sqrt(12),
        "最大回撤": (nav / nav.cummax() - 1).min(),
        "月胜率": (r > 0).mean(),
    }


def plot(ic, rets, path_ic, path_nav):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in ic.columns:
        ax.plot(ic.index, ic[c].cumsum(), label=FACTORS.get(c, "合成因子"),
                lw=2.4 if c == "composite" else 1.2)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("各因子累计 Rank IC（行业市值中性化后）")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path_ic, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.get_cmap("RdYlGn")
    for k in range(1, N_GROUPS + 1):
        ax.plot(rets.index, (1 + rets[k]).cumprod(), color=cmap((k - 1) / (N_GROUPS - 1)), label=f"第{k}组")
    ax.plot(rets.index, (1 + rets["universe"]).cumprod(), color="#333", ls="--", label="全市场等权")
    ax.plot(rets.index, (1 + rets["long_short"]).cumprod(), color="#1f4e79", lw=2, label="多空（5-1）")
    ax.set_yscale("log")
    ax.set_title("合成因子五分组净值（月度调仓，扣单边 0.15% 成本）")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path_nav, dpi=150)
    plt.close(fig)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    wide, industry, ipo = load()
    f, close, eligible, logcap = build_factors(wide)
    dates = rebal_dates(close.index)
    cols = list(FACTORS) + ["composite"]

    cs = cross_sections(f, close, eligible, logcap, industry, ipo, dates)
    ic = ic_table(cs, cols)
    summ = ic_summary(ic)
    decay = pd.DataFrame({c: ic_decay(cs, c) for c in cols})
    rets, to = group_backtest(cs, "composite")

    group_perf = pd.DataFrame({k: perf(rets[k]) for k in list(range(1, N_GROUPS + 1)) +
                               ["universe", "long_short", "excess_top"]}).T
    sub = {}
    for name, mask in {"2019-2022": rets.index < SPLIT, "2023-2026.08": rets.index >= SPLIT}.items():
        sub[name] = {"合成IC均值": ic.loc[ic.index < SPLIT if name.startswith("2019") else ic.index >= SPLIT,
                                         "composite"].mean(),
                     **{f"多空{k}": v for k, v in perf(rets.loc[mask, "long_short"]).items()},
                     **{f"多头超额{k}": v for k, v in perf(rets.loc[mask, "excess_top"]).items()}}

    # 稳健性: 剔除流通市值最小的 30% 股票后重跑
    cs_big = cross_sections(f, close, eligible, logcap, industry, ipo, dates, min_cap_pct=0.3)
    ic_big = ic_summary(ic_table(cs_big, cols))
    rets_big, _ = group_backtest(cs_big, "composite")

    # 稳健性: 剔除持有期内出现单日跳空 > 21% 的股票-月份
    # 用前向填充后的价格计算, 复牌日相对停牌前最后收盘价的跳空才会被计入
    jumps_cum = ((close.pct_change(fill_method=None).abs() > 0.21) & (wide["volume"] > 0)).cumsum()
    cs_jump = cross_sections(f, close, eligible, logcap, industry, ipo, dates, jumps_cum=jumps_cum)
    ic_jump = ic_summary(ic_table(cs_jump, ["composite"]))
    rets_jump, _ = group_backtest(cs_jump, "composite")
    dropped = np.mean([len(cs[t]) - len(cs_jump[t]) for t in cs_jump])

    # 稳健性: 次一交易日收盘成交, 排除“用收盘价算因子又按同一收盘价成交”的乐观假设
    cs_lag = cross_sections(f, close, eligible, logcap, industry, ipo, dates, lag=1)
    ic_lag = ic_summary(ic_table(cs_lag, cols))
    rets_lag, _ = group_backtest(cs_lag, "composite")

    summ.to_csv(OUT / "ic_summary.csv", encoding="utf-8-sig")
    ic.to_csv(OUT / "ic_monthly.csv", encoding="utf-8-sig")
    decay.to_csv(OUT / "ic_decay.csv", encoding="utf-8-sig")
    group_perf.to_csv(OUT / "composite_group_perf.csv", encoding="utf-8-sig")
    rets.to_csv(OUT / "composite_group_returns.csv", encoding="utf-8-sig")
    ic_big.to_csv(OUT / "ic_summary_ex_small30.csv", encoding="utf-8-sig")
    plot(ic, rets, OUT / "figures" / "cumulative_ic.png", OUT / "figures" / "composite_groups.png")

    summary = {
        "调仓期数": len(cs),
        "区间": [str(min(cs))[:10], str(max(cs))[:10]],
        "平均截面股票数": float(np.mean([len(d) for d in cs.values()])),
        "合成因子各组平均单边换手": to.mean().round(4).to_dict(),
        "合成因子分组绩效": group_perf.round(4).to_dict(orient="index"),
        "分段检验": {k: {a: round(b, 4) for a, b in v.items()} for k, v in sub.items()},
        "稳健性_剔除小市值30%": {
            "合成IC均值": round(ic_big.loc["composite", "IC均值"], 4),
            "合成年化ICIR": round(ic_big.loc["composite", "年化ICIR"], 4),
            "多空": {k: round(v, 4) for k, v in perf(rets_big["long_short"]).items()},
            "多头超额": {k: round(v, 4) for k, v in perf(rets_big["excess_top"]).items()},
        },
        "稳健性_剔除跳空": {
            "每期平均剔除股票数": round(float(dropped), 2),
            "合成IC均值": round(ic_jump.loc["composite", "IC均值"], 4),
            "合成年化ICIR": round(ic_jump.loc["composite", "年化ICIR"], 4),
            "多空": {k: round(v, 4) for k, v in perf(rets_jump["long_short"]).items()},
        },
        "稳健性_T+1成交": {
            "各因子IC均值": ic_lag["IC均值"].round(4).to_dict(),
            "合成年化ICIR": round(ic_lag.loc["composite", "年化ICIR"], 4),
            "多空": {k: round(v, 4) for k, v in perf(rets_lag["long_short"]).items()},
            "多头超额": {k: round(v, 4) for k, v in perf(rets_lag["excess_top"]).items()},
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summ.round(4))
    print(decay.round(4))
    print(group_perf.round(4))
    print(json.dumps(summary["分段检验"], ensure_ascii=False, indent=1))
    print(json.dumps(summary["稳健性_剔除小市值30%"], ensure_ascii=False, indent=1))
    print(json.dumps(summary["稳健性_剔除跳空"], ensure_ascii=False, indent=1))
    print(json.dumps(summary["稳健性_T+1成交"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
