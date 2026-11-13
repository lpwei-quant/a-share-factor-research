"""下载行业分类与深市 ST 历史区间。

用法: python fetch_meta.py [数据目录]
输出:
- industry.csv      证监会行业（门类-大类, 下载当日的静态分类）, 来自东方财富数据中心, 含已退市股票; 查不到的股票在回测中归入 NA
- st_intervals.csv  深市股票被 ST / *ST 的起止日期, 由深交所“简称变更”记录还原
                    （沪市没有找到带日期的公开记录, 在 research.py 中用 5% 涨跌幅限制推断）
"""
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
END = "2026-09-30"


def fetch_industry():
    """东方财富数据中心的公司资料表: 证监会行业（门类-大类）, 含部分已退市公司。"""
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    params = {"reportName": "RPT_F10_BASIC_ORGINFO", "columns": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,INDUSTRYCSRC1",
              "pageNumber": 1, "pageSize": 500, "source": "HSF10", "client": "PC"}
    rows, pages = [], None
    while pages is None or params["pageNumber"] <= pages:
        for i in range(5):                           # 连接被断开时退避重试
            try:
                result = requests.get(url, params=params, timeout=20,
                                      headers={"User-Agent": "Mozilla/5.0"}).json()["result"]
                break
            except (requests.RequestException, ValueError, KeyError):
                time.sleep(20 * (i + 1))
        else:
            sys.exit("行业列表下载失败, 请稍后重试")
        pages = result["pages"]
        rows += result["data"]
        params["pageNumber"] += 1
        time.sleep(1.5)
    df = pd.DataFrame(rows)
    df = df[df["SECUCODE"].str.endswith((".SH", ".SZ"), na=False)]
    df = df.rename(columns={"SECURITY_CODE": "code", "SECURITY_NAME_ABBR": "name", "INDUSTRYCSRC1": "industry"})
    df["industry"] = df["industry"].fillna("")
    return df[["code", "name", "industry"]].drop_duplicates("code")


def is_st(name):
    return "ST" in str(name).upper()


def fetch_st_intervals():
    """按时间顺序回放每只股票的简称变更: 改成带 ST 的简称时区间开始, 改回不带 ST 时结束。"""
    chg = ak.stock_info_sz_change_name(symbol="简称变更")
    chg["变更日期"] = pd.to_datetime(chg["变更日期"])
    chg["证券代码"] = chg["证券代码"].astype(str).str.zfill(6)
    out = []
    for code, g in chg.sort_values("变更日期").groupby("证券代码"):
        start = pd.Timestamp("1990-01-01") if is_st(g["变更前简称"].iloc[0]) else None
        for _, r in g.iterrows():
            if is_st(r["变更后简称"]) and start is None:
                start = r["变更日期"]
            elif not is_st(r["变更后简称"]) and start is not None:
                out.append((code, start, r["变更日期"]))
                start = None
        if start is not None:
            out.append((code, start, pd.Timestamp(END) + pd.Timedelta(days=1)))
    return pd.DataFrame(out, columns=["code", "start", "end"])   # ST 状态覆盖 [start, end)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ind = fetch_industry()
    ind.to_csv(DATA_DIR / "industry.csv", index=False, encoding="utf-8-sig")
    print(f"行业: {len(ind)} 只, {ind['industry'].nunique()} 个行业, 无行业 {(ind['industry'] == '').sum()} 只")
    st = fetch_st_intervals()
    st.to_csv(DATA_DIR / "st_intervals.csv", index=False, encoding="utf-8-sig")
    print(f"深市 ST 区间: {len(st)} 段, 涉及 {st['code'].nunique()} 只股票")


if __name__ == "__main__":
    main()
