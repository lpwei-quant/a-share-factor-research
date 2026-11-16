"""A 股量价因子研究: 因子构建、IC 分析、分层回测。

用法: python research.py [数据目录]
输入: fetch_data.py 生成的 daily_hfq.parquet、stock_basic.csv, fetch_meta.py 生成的 industry.csv、st_intervals.csv
输出: results/ 下的 csv、summary.json 与 figures/*.png

研究约定（在查看结果前确定）:
- 每月最后一个交易日收盘调仓, 持有到下月最后一个交易日收盘
- 合成因子 = 反转、低波动、低换手、非流动性 四个中性化因子等权
- 交易成本按单边 0.15% 计, 作用于每期组合换手
- 股票池: 上市满一年、调仓日非 ST、非停牌、未涨跌停（涨停买不进、跌停卖不出）
- ST 状态: 深市用交易所简称变更记录; 沪市主板用 5% 涨跌幅限制推断, 并在深市主板上检验推断的准确率
- 2026-07-06 起主板风险警示股票涨跌幅限制由 5% 调整为 10%: 此后 ST 股按 10% 判断涨跌停,
  沪市 ST 状态无法再从价格推断, 沿用规则变更前最后一个交易日的状态
- 换手按持仓漂移计算: 上期持仓先按持有期收益变动权重, 再与本期目标权重比较
- 已退市股票在退市前照常进入样本, 退市后按最后成交价计
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
CHINEXT_20PCT = "2020-08-24"    # 创业板涨跌幅限制由 10% 放宽到 20% 的首个交易日
ST_10PCT = "2026-07-06"         # 主板风险警示股票涨跌幅限制由 5% 调整为 10% 的首个交易日


def load():
    daily = pd.read_parquet(DATA_DIR / "daily_hfq.parquet")
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    basic = pd.read_csv(DATA_DIR / "stock_basic.csv", dtype=str).fillna("")
    basic["code"] = basic["code"].str[3:]            # baostock 代码形如 sh.600000
    ind = pd.read_csv(DATA_DIR / "industry.csv", dtype=str).fillna("")
    st_sz = pd.read_csv(DATA_DIR / "st_intervals.csv", dtype={"code": str}, parse_dates=["start", "end"])
    wide = {c: daily.pivot(index="date", columns="code", values=c)
            for c in ["close", "high", "low", "amount", "turn", "volume"]}
    industry = ind.set_index("code")["industry"].replace("", "NA")   # 查不到行业的股票中性化时归入 NA
    names = ind.set_index("code")["name"]                            # 下载当日（2026-11）的简称
    ipo = pd.to_datetime(basic.set_index("code")["ipoDate"])
    listed = set(basic.loc[basic["status"] == "1", "code"])           # 样本期末仍在市的股票
    return wide, industry, ipo, listed, st_sz, names


def st_from_records(index, columns, st_sz):
    """深市: 按交易所简称变更记录还原的 ST 区间 [start, end)。"""
    st = pd.DataFrame(False, index=index, columns=columns)
    for code, start, end in st_sz.itertuples(index=False):
        if code in st.columns:
            st.loc[(index >= start) & (index < end), code] = True
    return st


def st_from_limits(w, traded):
    """主板: 用 5% 涨跌幅限制推断 ST 状态。

    进入: 10 个交易日内两次收在 5% 涨跌停附近（4.4%–5.6%, 低价股涨跌停价取整后会偏离 5%）,
          且期间没有超过 5.7% 的涨跌;
    退出: 出现超过 5.7% 的单日涨跌（说明已恢复 10% 限制）。
    2026-07-06 起 ST 股票同样适用 10% 限制, 价格不再能区分 ST, 此后沿用变更前最后一个交易日的状态。
    """
    pct = w["pctChg"].where(traded).abs()
    at_edge = (w["close"] >= w["high"]) | (w["close"] <= w["low"])
    hit5 = traded & at_edge & (pct >= 4.4) & (pct <= 5.6)
    big = pct > 5.7
    enter = (hit5.rolling(10, min_periods=1).sum() >= 2) & (big.rolling(10, min_periods=1).sum() == 0)
    state = pd.DataFrame(np.nan, index=pct.index, columns=pct.columns)
    state = state.mask(big, 0.0).mask(enter, 1.0).ffill().fillna(0.0) == 1.0
    after = pct.index >= ST_10PCT
    if after.any() and (~after).any():
        vals = state.to_numpy().copy()
        vals[after] = vals[~after][-1]
        state = pd.DataFrame(vals, index=state.index, columns=state.columns)
    main_board = pct.columns.str.startswith(("60", "00"))
    state.loc[:, ~main_board] = False
    return state


def limit_pct(index, columns):
    """各股票各交易日的涨跌幅限制（%）: 科创板 20, 创业板自 2020-08-24 起 20, 其余 10。"""
    lim = pd.DataFrame(10.0, index=index, columns=columns)
    lim.loc[:, columns.str.startswith("68")] = 20.0
    lim.loc[index >= CHINEXT_20PCT, columns.str.startswith("30")] = 20.0
    return lim


def build_factors(w, st_sz):
    close = w["close"].ffill()                      # 停牌/退市后沿用最后价格
    traded = w["close"].notna() & (w["volume"] > 0)  # 停牌日与退市后无行情记录
    # 当日涨跌幅（%）: 相对上一个有行情交易日的后复权收盘价, 用于识别涨跌停与推断 ST
    w = {**w, "pctChg": close.pct_change(fill_method=None) * 100}
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
    # ST: 深市用交易所记录, 沪市主板用涨跌幅推断; 科创板 ST 无法识别
    st_inferred = st_from_limits(w, traded)
    st_exact = st_from_records(close.index, close.columns, st_sz)
    is_sz = close.columns.str.startswith(("00", "30"))
    st = st_inferred.copy()
    st.loc[:, is_sz] = st_exact.loc[:, is_sz]
    # 收盘价封在涨停（跌停）价: 收盘 = 最高（最低）且涨跌幅达到限制; 2026-07-06 前 ST 股票限制为 5%
    # 涨跌停价按分取整, 低价股实际涨跌幅会偏离名义限制, 因此留出容差
    before = pd.Series(close.index < ST_10PCT, index=close.index)
    st5 = st.mul(before, axis=0).astype(bool)
    lim = limit_pct(close.index, close.columns).mask(st5, 5.0)
    tol = pd.DataFrame(0.3, index=close.index, columns=close.columns).mask(st5, 0.6)
    at_limit = traded & (((w["close"] >= w["high"]) & (w["pctChg"] >= lim - tol)) |
                         ((w["close"] <= w["low"]) & (w["pctChg"] <= -(lim - tol))))
    masks = {
        "eligible": traded & (valid_days >= 15),    # 当日有交易, 且近 20 日至少 15 日有行情
        "st": st,
        "at_limit": at_limit,
        "st_inferred": st_inferred,
        "st_exact": st_exact,
    }
    return f, close, masks, np.log(cap)


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


def cross_sections(f, close, masks, logcap, industry, ipo, dates, min_cap_pct=0.0, jumps_cum=None,
                   lag=0, drop_st=True, drop_limit=True, only=None):
    """返回 {date: DataFrame(因子..., composite, fwd_ret[1..6])}

    drop_st / drop_limit: 是否剔除调仓日为 ST、成交日收盘涨跌停的股票（关闭后用于对照）。
    only: 只保留给定代码集合, 用于“仅在市股票”的幸存者偏差对照。

    jumps_cum: 单日 |收益| > 21% 的累计次数; 传入时剔除持有期内出现此类跳空（多为长期停牌后复牌）的股票。
    lag: 成交延后的交易日数; 1 表示调仓日收盘计算因子、次一交易日收盘成交。
    """
    days = close.index
    trade_day = {d: days[min(days.get_loc(d) + lag, len(days) - 1)] for d in dates}
    out = {}
    for i, t in enumerate(dates[:-1]):
        age_ok = ((t - ipo.reindex(close.columns)).dt.days >= 365).values
        ok = masks["eligible"].loc[t] & age_ok
        if drop_st:
            ok &= ~masks["st"].loc[t]
        if drop_limit:
            ok &= ~masks["at_limit"].loc[trade_day[t]]
        if only is not None:
            ok &= ok.index.isin(only)
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
    """五分组等权回测, 扣除换手成本。返回各组净收益序列与换手率。

    换手 = 本期目标权重与上期持仓漂移后权重之差的一半; 成本按单边 COST 作用于买卖总额。
    """
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
            grown = w * (1 + d.loc[members, "fwd_1"])          # 持有期结束时的实际权重
            prev[k] = grown / grown.sum()
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


def st_inference_check(masks, dates):
    """在深市主板上, 用交易所记录检验沪市 ST 的识别方法（只看调仓日、可交易的股票）。

    2026-07-06 前检验“5% 涨跌幅推断”; 之后检验“沿用变更前状态”。
    """
    cols = masks["st_exact"].columns[masks["st_exact"].columns.str.startswith("00")]
    out = {}
    for name, ds in {"规则变更前_价格推断": [d for d in dates if d < pd.Timestamp(ST_10PCT)],
                     "规则变更后_沿用状态": [d for d in dates if d >= pd.Timestamp(ST_10PCT)]}.items():
        if not ds:
            continue
        ok = masks["eligible"].loc[ds, cols]
        truth, guess = masks["st_exact"].loc[ds, cols] & ok, masks["st_inferred"].loc[ds, cols] & ok
        hit = (truth & guess).values.sum()
        out[name] = {
            "调仓期数": len(ds),
            "交易所记录ST数": int(truth.values.sum()),
            "识别为ST数": int(guess.values.sum()),
            "召回率": round(hit / truth.values.sum(), 4),
            "精确率": round(hit / guess.values.sum(), 4),
        }
    return out


def st_name_check(masks, names, listed):
    """沪市主板仍在市股票: 样本期末沿用的 ST 状态与下载当日（2026-11）简称是否一致。"""
    st_end = masks["st"].iloc[-1]
    cols = [c for c in st_end.index if c.startswith("60") and c in listed and c in names.index]
    by_name = names.reindex(cols).str.upper().str.contains("ST")
    by_state = st_end.reindex(cols)
    return {
        "沪市主板在市股票数": len(cols),
        "期末判为ST": int(by_state.sum()),
        "当前简称含ST": int(by_name.sum()),
        "两者一致": int((by_state == by_name).sum()),
        "判为ST但简称不含ST": int((by_state & ~by_name).sum()),
        "简称含ST但未判为ST": int((~by_state & by_name).sum()),
    }


def brief(cs):
    """一组对照实验的摘要: 截面股票数、合成因子 IC 与分组表现。"""
    ic = ic_summary(ic_table(cs, ["composite"]))
    rets, _ = group_backtest(cs, "composite")
    return {
        "平均截面股票数": round(float(np.mean([len(d) for d in cs.values()])), 1),
        "合成IC均值": round(ic.loc["composite", "IC均值"], 4),
        "合成年化ICIR": round(ic.loc["composite", "年化ICIR"], 4),
        "全市场等权年化": round(perf(rets["universe"])["年化收益"], 4),
        "多空": {k: round(v, 4) for k, v in perf(rets["long_short"]).items()},
        "多头超额": {k: round(v, 4) for k, v in perf(rets["excess_top"]).items()},
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
    wide, industry, ipo, listed, st_sz, names = load()
    f, close, masks, logcap = build_factors(wide, st_sz)
    dates = rebal_dates(close.index)
    cols = list(FACTORS) + ["composite"]
    base = (f, close, masks, logcap, industry, ipo, dates)

    cs = cross_sections(*base)
    ic = ic_table(cs, cols)
    summ = ic_summary(ic)
    decay = pd.DataFrame({c: ic_decay(cs, c) for c in cols})
    rets, to = group_backtest(cs, "composite")

    keys = list(range(1, N_GROUPS + 1)) + ["universe", "long_short", "excess_top"]
    group_perf = pd.DataFrame({k: perf(rets[k]) for k in keys}).T
    yearly = rets[[1, N_GROUPS, "universe", "long_short", "excess_top"]].groupby(rets.index.year).apply(
        lambda x: (1 + x).prod() - 1)
    yearly["合成IC均值"] = ic["composite"].groupby(ic.index.year).mean()
    sub = {}
    for name, mask in {"2019-2022": rets.index < SPLIT, "2023-2026.08": rets.index >= SPLIT}.items():
        sub[name] = {"合成IC均值": ic.loc[mask, "composite"].mean(),
                     **{f"多空{k}": v for k, v in perf(rets.loc[mask, "long_short"]).items()},
                     **{f"多头超额{k}": v for k, v in perf(rets.loc[mask, "excess_top"]).items()}}

    # 稳健性: 剔除流通市值最小的 30% 股票后重跑
    cs_big = cross_sections(*base, min_cap_pct=0.3)
    ic_big = ic_summary(ic_table(cs_big, cols))
    # 稳健性: 次一交易日收盘成交, 排除“用收盘价算因子又按同一收盘价成交”的乐观假设
    cs_lag = cross_sections(*base, lag=1)
    ic_lag = ic_summary(ic_table(cs_lag, cols))
    # 稳健性: 剔除持有期内出现单日跳空 > 21% 的股票-月份（多为长期停牌后复牌）
    jumps_cum = ((close.pct_change(fill_method=None).abs() > 0.21) & (wide["volume"] > 0)).cumsum()
    cs_jump = cross_sections(*base, jumps_cum=jumps_cum)
    # 对照: 逐项放松股票池处理, 量化每一项对结论的影响
    cs_st = cross_sections(*base, drop_st=False)
    cs_lim = cross_sections(*base, drop_limit=False)
    cs_surv = cross_sections(*base, only=listed)

    summ.to_csv(OUT / "ic_summary.csv", encoding="utf-8-sig")
    ic.to_csv(OUT / "ic_monthly.csv", encoding="utf-8-sig")
    decay.to_csv(OUT / "ic_decay.csv", encoding="utf-8-sig")
    group_perf.to_csv(OUT / "composite_group_perf.csv", encoding="utf-8-sig")
    rets.to_csv(OUT / "composite_group_returns.csv", encoding="utf-8-sig")
    ic_big.to_csv(OUT / "ic_summary_ex_small30.csv", encoding="utf-8-sig")
    plot(ic, rets, OUT / "figures" / "cumulative_ic.png", OUT / "figures" / "composite_groups.png")

    n_st = float(np.mean([(masks["eligible"].loc[t] & masks["st"].loc[t]).sum() for t in cs]))
    n_lim = float(np.mean([(masks["eligible"].loc[t] & masks["at_limit"].loc[t]).sum() for t in cs]))
    summary = {
        "调仓期数": len(cs),
        "区间": [str(min(cs))[:10], str(max(cs))[:10]],
        "样本股票数": int(close.shape[1]),
        "其中已退市": int(close.shape[1] - len(listed & set(close.columns))),
        "日频记录数": int(wide["close"].notna().values.sum()),
        "ST推断检验": st_inference_check(masks, list(cs)),
        "沪市ST期末简称核对": st_name_check(masks, names, listed),
        "平均截面股票数": float(np.mean([len(d) for d in cs.values()])),
        "每期平均ST股票数": round(n_st, 1),
        "每期平均涨跌停股票数": round(n_lim, 1),
        "合成因子各组平均单边换手": to.mean().round(4).to_dict(),
        "合成因子分组绩效": group_perf.round(4).to_dict(orient="index"),
        "分年度": yearly.round(4).to_dict(orient="index"),
        "分段检验": {k: {a: round(b, 4) for a, b in v.items()} for k, v in sub.items()},
        "稳健性_剔除小市值30%": brief(cs_big),
        "稳健性_T+1成交": {"各因子IC均值": ic_lag["IC均值"].round(4).to_dict(), **brief(cs_lag)},
        "稳健性_剔除跳空": brief(cs_jump),
        "对照_基准": brief(cs),
        "对照_不剔除ST": brief(cs_st),
        "对照_不剔除涨跌停": brief(cs_lim),
        "对照_仅在市股票": brief(cs_surv),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summ.round(4))
    print(decay.round(4))
    print(group_perf.round(4))
    print(yearly.round(4))
    for k in ["样本股票数", "其中已退市", "日频记录数", "平均截面股票数", "每期平均ST股票数", "每期平均涨跌停股票数",
              "ST推断检验", "沪市ST期末简称核对", "合成因子各组平均单边换手"]:
        print(k, json.dumps(summary[k], ensure_ascii=False))
    for k in ["分段检验", "稳健性_剔除小市值30%", "稳健性_T+1成交", "稳健性_剔除跳空",
              "对照_基准", "对照_不剔除ST", "对照_不剔除涨跌停", "对照_仅在市股票"]:
        print(k, json.dumps(summary[k], ensure_ascii=False))


if __name__ == "__main__":
    main()
