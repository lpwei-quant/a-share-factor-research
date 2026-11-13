"""下载沪深 A 股日频行情（后复权，含已退市股票），保存为 parquet。

用法:
    python fetch_data.py [数据目录]            # 下载, 中断后重跑会续传
    python fetch_data.py [数据目录] --merge    # 合并分片为 daily_hfq.parquet

- 股票列表来自 baostock（含上市日期、退市日期），数据目录已有 stock_basic.csv 时直接复用。
  2019 年以后退市的股票保留在列表中，退市前的行情照常进入样本。
- 价格、成交量、成交额全部来自新浪财经（AKShare stock_zh_a_daily）。
  新浪有已退市股票的价格和成交额, 但没有换手率, 这部分股票的换手率从腾讯财经
  （stock_zh_a_hist_tx）补齐。腾讯的后复权价格与新浪不一致, 因此只取它的换手率, 不取价格。
- 单进程限速请求, 请勿改成并发: 数据源有频率限制, 并发会导致 IP 被限流或封禁。
- 按 100 只一个分片落盘, 分片用 .lock 文件认领。中断后先删除没有对应 .parquet 的 .lock 文件再重跑。
"""
import sys
import time
import warnings
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

# AKShare 请求未设超时, 连接挂起会让进程永久阻塞; 这里给所有请求补上默认超时
_request = requests.Session.request
requests.Session.request = lambda self, *a, **kw: _request(self, *a, **{"timeout": 30, **kw})
warnings.filterwarnings("ignore", category=UserWarning)   # 新浪对退市股票的“无股本数据”提示

START, END = "20180101", "20260930"
CHUNK, INTERVAL = 100, 0.3
HAS_DIR = len(sys.argv) > 1 and not sys.argv[1].startswith("--")
DATA_DIR = Path(sys.argv[1]) if HAS_DIR else Path(__file__).parent / "data"
COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount", "turn"]


def load_universe():
    basic_path = DATA_DIR / "stock_basic.csv"
    if not basic_path.exists():
        import baostock as bs

        bs.login()
        rs, out = bs.query_stock_basic(), []
        while rs.error_code == "0" and rs.next():
            out.append(rs.get_row_data())
        bs.logout()
        basic = pd.DataFrame(out, columns=rs.fields)
        basic = basic[(basic["type"] == "1") & basic["code"].str.match(r"^(sh\.60|sh\.68|sz\.00|sz\.30)")]
        basic = basic[(basic["outDate"] == "") | (basic["outDate"] >= "2019-01-01")]
        basic = basic[basic["ipoDate"] <= "2026-09-30"]
        basic.to_csv(basic_path, index=False, encoding="utf-8-sig")
    basic = pd.read_csv(basic_path, dtype=str).fillna("")
    return basic["code"].str.replace(".", "").tolist()          # sh.600000 -> sh600000


def retry(func):
    """网络失败时退避重试; 成功返回 DataFrame（无数据为空表）, 连续失败返回 None。"""
    for i in range(3):
        try:
            df = func()
        except requests.RequestException:
            time.sleep(20 * (i + 1))
            continue
        except (KeyError, ValueError, IndexError):
            time.sleep(INTERVAL)
            return pd.DataFrame()                               # 数据源没有这只股票
        time.sleep(INTERVAL)
        return df if df is not None else pd.DataFrame()
    return None


def fetch_one(symbol):
    df = retry(lambda: ak.stock_zh_a_daily(symbol=symbol, start_date=START, end_date=END, adjust="hfq"))
    if df is None or df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["turn"] = df["turnover"] * 100                           # 新浪换手率为小数, 统一为百分比
    if df["turn"].isna().all():                                 # 已退市: 新浪无股本数据, 用腾讯换手率补齐
        tx = retry(lambda: ak.stock_zh_a_hist_tx(symbol=symbol, start_date=START, end_date=END, adjust="hfq"))
        if tx is not None and not tx.empty:
            tx["date"] = pd.to_datetime(tx["date"])
            df["turn"] = df["date"].map(tx.set_index("date")["turnover"] * 100)
    df["code"] = symbol[2:]
    return df[COLUMNS]


def merge():
    parts = sorted((DATA_DIR / "chunks").glob("*.parquet"))
    daily = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    daily.to_parquet(DATA_DIR / "daily_hfq.parquet", index=False)
    print(f"合并 {len(parts)} 个分片: {daily['code'].nunique()} 只, {len(daily)} 行", flush=True)


def main():
    (DATA_DIR / "chunks").mkdir(parents=True, exist_ok=True)
    if "--merge" in sys.argv:
        return merge()
    codes = load_universe()
    n_chunks = -(-len(codes) // CHUNK)
    print(f"股票池 {len(codes)} 只, 共 {n_chunks} 个分片", flush=True)

    streak, pauses, failed, empty = 0, 0, [], []
    for idx in range(n_chunks):
        path = DATA_DIR / "chunks" / f"{idx:03d}.parquet"
        try:
            open(path.with_suffix(".lock"), "x").close()        # 认领分片; 已被认领或已完成则跳过
        except FileExistsError:
            continue
        frames = []
        for symbol in codes[idx * CHUNK:(idx + 1) * CHUNK]:
            df = fetch_one(symbol)
            if df is None:
                failed.append(symbol)
                streak += 1
                if streak >= 5:                                 # 连续失败说明被限流, 暂停而不是继续请求
                    pauses += 1
                    if pauses > 3:
                        path.with_suffix(".lock").unlink()
                        sys.exit("数据源持续拒绝请求, 已停止。稍后重跑会从未完成的分片继续。")
                    print(f"连续失败, 暂停 10 分钟（第 {pauses} 次）", flush=True)
                    time.sleep(600)
                    streak = 0
                continue
            streak = 0
            if df.empty:
                empty.append(symbol)
            else:
                frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
        print(f"分片 {idx + 1}/{n_chunks} 完成, 累计失败 {len(failed)}, 无行情 {len(empty)}", flush=True)

    pd.Series(failed + empty, name="code").to_csv(DATA_DIR / "missing_codes.csv", index=False)
    print(f"下载结束: 失败 {len(failed)} 只, 无行情 {len(empty)} 只; 接着运行 --merge", flush=True)


if __name__ == "__main__":
    main()
