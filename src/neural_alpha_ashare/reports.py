from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

STYLE = """
:root{color-scheme:dark;--bg:#091018;--panel:#111c28;--line:#233448;--text:#e7eef7;
--muted:#8ea2b7;--green:#38d996;--amber:#ffbf69}*{box-sizing:border-box}body{margin:0;
font-family:Inter,"Microsoft YaHei",system-ui,sans-serif;background:var(--bg);color:var(--text)}
main{max-width:1180px;margin:auto;padding:32px 22px}.hero{display:flex;
justify-content:space-between;
gap:20px;align-items:end;margin-bottom:24px}h1{margin:.2rem 0;font-size:clamp(28px,5vw,48px)}
.kicker,.muted{color:var(--muted)}nav a{color:var(--text);margin-left:16px}.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}.card{background:var(--panel);
border:1px solid var(--line);border-radius:14px;padding:18px}.value{font-size:26px;margin-top:8px}
table{width:100%;border-collapse:collapse;margin-top:14px}th,td{text-align:right;padding:10px;
border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
.ok{color:var(--green)}
.warn{color:var(--amber)}footer{color:var(--muted);margin-top:30px;font-size:13px}@media(max-width:650px){
.hero{display:block}nav a{margin:0 14px 0 0}.table-wrap{overflow:auto}}
"""


def _page(title: str, data_date: str, body: str) -> str:
    navigation = (
        '<a href="index.html">首页</a><a href="daily.html">日报</a><a href="weekly.html">周报</a>'
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{STYLE}</style></head><body><main><header class="hero"><div><div class="kicker">
NeuralAlpha · PIT Machine Learning</div><h1>{html.escape(title)}</h1><div class="muted">
数据截止：{html.escape(data_date)}</div></div><nav>{navigation}</nav></header>{body}<footer>
研究用途 · t 日收盘后产生信号 · 最早 t+1 成交 · 不连接券商自动下单</footer></main></body></html>"""


def _cards(metrics: dict[str, Any]) -> str:
    if not metrics:
        return '<section class="card"><span class="muted">暂无回测指标</span></section>'
    labels = {
        "total_return": "累计收益",
        "annual_return": "年化收益",
        "annual_volatility": "年化波动",
        "sharpe": "夏普",
        "max_drawdown": "最大回撤",
        "rank_ic_mean": "平均 Rank IC",
        "rank_ic_ir": "ICIR",
    }
    cards = []
    for key, label in labels.items():
        if key not in metrics:
            continue
        value = metrics[key]
        formatted = (
            f"{value:.2%}"
            if key in {"total_return", "annual_return", "annual_volatility", "max_drawdown"}
            else f"{value:.3f}"
        )
        cards.append(
            f'<article class="card"><div class="muted">{label}</div>'
            f'<div class="value">{formatted}</div></article>'
        )
    return f'<section class="grid">{"".join(cards)}</section>'


def _prediction_table(predictions: pl.DataFrame, rows: int = 20) -> str:
    if predictions.is_empty():
        return '<section class="card"><span class="muted">暂无预测</span></section>'
    latest = predictions["trade_date"].max()
    top = (
        predictions.filter(pl.col("trade_date") == latest)
        .sort("model_alpha", descending=True)
        .head(rows)
    )
    body = "".join(
        f"<tr><td>{html.escape(str(record['symbol']))}</td><td>{record['model_rank']}</td>"
        f"<td>{record['model_alpha']:.4f}</td>"
        f"<td>{'是' if record.get('eligible', True) else '否'}</td></tr>"
        for record in top.to_dicts()
    )
    return f"""<section class="card"><h2>最新横截面</h2><div class="table-wrap"><table>
<thead><tr><th>证券</th><th>排名</th><th>Alpha</th><th>可选</th></tr></thead><tbody>{body}</tbody>
</table></div></section>"""


def write_reports(
    docs_dir: str | Path,
    title: str,
    data_date: str,
    predictions: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    universe_quality: str = "UNKNOWN",
) -> list[Path]:
    target = Path(docs_dir)
    target.mkdir(parents=True, exist_ok=True)
    predictions = predictions if predictions is not None else pl.DataFrame()
    metrics = metrics or {}
    status_class = "ok" if universe_quality == "PIT" else "warn"
    status = (
        '<section class="card"><h2>审计状态</h2><p>历史证券目录：'
        f'<span class="{status_class}">{html.escape(universe_quality)}</span></p>'
        '<p class="muted">DEGRADED 表示数据源没有覆盖整段历史的当时目录快照，'
        "结果不得冒充无幸存者偏差。</p></section>"
    )
    daily_body = _prediction_table(predictions) + status
    weekly_body = _cards(metrics) + status
    index_body = _cards(metrics) + _prediction_table(predictions, 10) + status
    pages = {
        "index.html": _page(title, data_date, index_body),
        "daily.html": _page(f"{title} · 日报", data_date, daily_body),
        "weekly.html": _page(f"{title} · 周报", data_date, weekly_body),
    }
    paths = []
    for name, content in pages.items():
        path = target / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    (target / "report.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "data_date": data_date,
                "universe_quality": universe_quality,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
