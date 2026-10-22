"""下载沪深 A 股日频行情（后复权），保存为 parquet。

用法: python fetch_data.py [数据目录]
- 股票列表与证监会行业来自 baostock，若数据目录已有 stock_basic.csv / industry_csrc.csv 则直接复用。
- 日线来自新浪财经（AKShare stock_zh_a_daily）。新浪不提供已退市股票, 返回空数据时直接跳过。
- 单进程限速请求，网络失败退避；数据源有频率限制，请勿改成并发。
- 按分片落盘，中断后重跑会跳过已完成的分片。
"""
import json
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import requests
from akshare.utils.demjson import JSONDecodeError as SinaDecodeError

# AKShare 请求新浪时未设超时, 连接挂起会让进程永久阻塞; 这里给所有请求补上默认超时
_request = requests.Session.request
requests.Session.request = lambda self, *a, **kw: _request(self, *a, **{"timeout": 30, **kw})

START, END = "20180101", "20260930"
CHUNK, INTERVAL = 100, 0.5
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"


def load_universe():
    basic_path, ind_path = DATA_DIR / "stock_basic.csv", DATA_DIR / "industry_csrc.csv"
    if not basic_path.exists():
        import baostock as bs

        def rows(rs):
            out = []
            while rs.error_code == "0" and rs.next():
                out.append(rs.get_row_data())
            return pd.DataFrame(out, columns=rs.fields)

        bs.login()
        basic, industry = rows(bs.query_stock_basic()), rows(bs.query_stock_industry())
        bs.logout()
        basic = basic[(basic["type"] == "1") & basic["code"].str.match(r"^(sh\.60|sh\.68|sz\.00|sz\.30)")]
        basic = basic[(basic["outDate"] == "") | (basic["outDate"] >= "2019-01-01")]
        basic.to_csv(basic_path, index=False, encoding="utf-8-sig")
        industry.to_csv(ind_path, index=False, encoding="utf-8-sig")
    basic = pd.read_csv(basic_path, dtype=str).fillna("")
    # 新浪不提供已退市股票的行情, 只下载在市股票
    basic = basic[(basic["ipoDate"] <= "2026-09-30") & (basic["status"] == "1")]
    return basic["code"].str.replace(".", "").tolist()


def fetch_one(symbol):
    """成功返回 DataFrame（无数据时为空）, 网络连续失败返回 None。"""
    for i in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, start_date=START, end_date=END, adjust="hfq")
        except (SinaDecodeError, json.JSONDecodeError, KeyError, ValueError):
            time.sleep(INTERVAL)
            return pd.DataFrame()          # 退市或无行情
        except Exception:
            time.sleep(20 * (i + 1))
            continue
        time.sleep(INTERVAL)
        df["code"] = symbol[2:]
        df["turn"] = df["turnover"] * 100  # 新浪换手率为小数, 统一为百分比
        return df.drop(columns="turnover")
    return None


def main():
    (DATA_DIR / "chunks").mkdir(parents=True, exist_ok=True)
    codes = load_universe()
    print(f"股票池 {len(codes)} 只", flush=True)

    streak, pauses, failed = 0, 0, []
    for idx, s in enumerate(range(0, len(codes), CHUNK)):
        path = DATA_DIR / "chunks" / f"{idx:03d}.parquet"
        if path.exists():
            continue
        frames = []
        for j, code in enumerate(codes[s:s + CHUNK], 1):
            if j % 25 == 0:
                print(f"  分片 {idx + 1} 进度 {j}/{CHUNK}", flush=True)
            df = fetch_one(code)
            if df is None:
                failed.append(code)
                streak += 1
                if streak >= 10:
                    pauses += 1
                    if pauses > 3:
                        sys.exit("数据源持续拒绝请求, 已停止。稍后重跑会从未完成的分片继续。")
                    print(f"连续失败, 暂停 10 分钟（第 {pauses} 次）", flush=True)
                    time.sleep(600)
                    streak = 0
            else:
                streak = 0
                frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
        print(f"分片 {idx + 1}/{-(-len(codes) // CHUNK)} 完成, 累计失败 {len(failed)}", flush=True)

    daily = pd.concat([pd.read_parquet(p) for p in sorted((DATA_DIR / "chunks").glob("*.parquet"))],
                      ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_parquet(DATA_DIR / "daily_hfq.parquet", index=False)
    pd.Series(failed, name="code").to_csv(DATA_DIR / "failed_codes.csv", index=False)
    print(f"完成: {daily['code'].nunique()} 只, {len(daily)} 行, 失败 {len(failed)} 只", flush=True)


if __name__ == "__main__":
    main()
