#!/usr/bin/env python3
"""从 arXiv 抓取一个小型真实语料，缓存到磁盘供评测离线使用。

为什么必须缓存
--------------
如果评测时现场抓取，会出现三个问题：
1. 网络抖动会让"同一份评测"跑出不同结果，**丧失可复现性**；
2. arXiv 会不断新增论文，语料随时间漂移，历史指标无法对照；
3. 离线环境完全跑不了。

所以：抓取与评测分离。本脚本负责抓取并落盘（含清单），
评测只读缓存；缓存缺失时**明确报错**而不是偷偷联网。

语料设计：为什么选同一个小领域
------------------------------
故意集中抓 `cs.IR`（信息检索）里与 LLM 相关的论文 —— 它们**词汇高度重叠**。
这让检索任务有区分度：若语料是杂乱主题，词面方法也能轻松满分，
消融实验就测不出差异（所有方法都 1.0，没有信号）。

用法
----
    python scripts/fetch_corpus.py                 # 抓取并缓存
    python scripts/fetch_corpus.py --max 40        # 限制条数
    python scripts/fetch_corpus.py --check         # 只检查缓存状态
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

for _name in ("stdout", "stderr"):
    _s = getattr(sys, _name, None)
    if _s is not None and hasattr(_s, "buffer"):
        enc = getattr(_s, "encoding", None) or "utf-8"
        try:
            setattr(sys, _name, io.TextIOWrapper(_s.buffer, encoding=enc, errors="replace"))
        except Exception:
            pass

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_API = "http://export.arxiv.org/api/query"

#: 集中在一个小领域，制造词汇重叠 —— 否则消融实验没有区分度
DEFAULT_QUERY = 'cat:cs.IR AND abs:"large language model"'

CORPUS_PATH = Path(__file__).resolve().parent.parent / "eval" / "dataset" / "corpus.json"


@dataclass
class Paper:
    doc_id: str          # arXiv id，例如 2401.12345v1
    title: str
    abstract: str
    authors: list[str]
    published: str
    categories: list[str]
    url: str

    @property
    def content(self) -> str:
        """用于检索与证据绑定的正文。

        标题与摘要拼接：标题提供高信号短语，摘要提供上下文。
        二者之间用句号分隔，保证分段器能正确切开（否则标题会被并进第一句，
        导致标题查询命中时给出的证据片段包含无关前缀）。
        """
        return f"{self.title}。{self.abstract}"


def _fetch(query: str, start: int, max_results: int, retries: int = 3) -> bytes:
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "copilot-paper/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - 网络异常种类多，统一重试
            last = exc
            wait = 3 * (attempt + 1)
            print(f"  第 {attempt + 1} 次失败（{exc}），{wait}s 后重试…", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"抓取失败：{last}")


def parse_entries(xml_bytes: bytes) -> list[Paper]:
    root = ET.fromstring(xml_bytes)
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        # http://arxiv.org/abs/2401.12345v1 -> 2401.12345v1
        doc_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        authors = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in entry.findall(f"{ATOM}author")
        ]
        cats = [
            c.get("term", "")
            for c in entry.findall(f"{ATOM}category")
            if c.get("term")
        ]
        if not doc_id or not title or not abstract:
            continue
        papers.append(
            Paper(
                doc_id=doc_id,
                title=title,
                abstract=abstract,
                authors=[a for a in authors if a],
                published=(entry.findtext(f"{ATOM}published") or "").strip(),
                categories=cats,
                url=raw_id,
            )
        )
    return papers


def fetch_corpus(query: str, target: int, page: int = 50) -> tuple[list[Paper], list[str]]:
    """分页抓取直到达到 target 条（或没有更多）。

    返回 ``(papers, 实际使用的查询列表)`` —— 查询列表进清单，便于复现。
    """
    seen: dict[str, Paper] = {}
    queries_used: list[str] = []
    start = 0
    while len(seen) < target:
        want = min(page, target - len(seen))
        print(f"  抓取 start={start} max_results={want} …")
        xml_bytes = _fetch(query, start, want)
        queries_used.append(query)
        batch = parse_entries(xml_bytes)
        if not batch:
            break
        new = 0
        for p in batch:
            if p.doc_id not in seen:
                seen[p.doc_id] = p
                new += 1
        if new == 0:
            break
        start += len(batch)
        time.sleep(3)  # arXiv 礼貌间隔
    return list(seen.values()), queries_used


def save(papers: list[Paper], query: str, queries_used: list[str]) -> None:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": {
            "source": "arXiv API",
            "endpoint": ARXIV_API,
            "query": query,
            "queries_used": sorted(set(queries_used)),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(papers),
            "note": "评测只读本缓存；语料漂移会破坏可复现性，故不现场抓取。",
        },
        "papers": [asdict(p) for p in papers],
    }
    CORPUS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def check() -> int:
    if not CORPUS_PATH.is_file():
        print(f"缓存不存在：{CORPUS_PATH}")
        print("请先运行：python scripts/fetch_corpus.py")
        return 1
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    m = data["manifest"]
    print(f"缓存路径 : {CORPUS_PATH}")
    print(f"条数     : {m['count']}")
    print(f"抓取时间 : {m['fetched_at']}")
    print(f"查询     : {m['query']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 arXiv 语料并缓存")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--max", type=int, default=60, dest="target")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        return check()

    print(f"查询：{args.query}\n目标：{args.target} 条")
    papers, used = fetch_corpus(args.query, args.target)
    if not papers:
        print("未抓到任何论文", file=sys.stderr)
        return 1
    save(papers, args.query, used)
    print(f"\n已保存 {len(papers)} 条 -> {CORPUS_PATH}")
    for p in papers[:3]:
        print(f"  [{p.doc_id}] {p.title[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
