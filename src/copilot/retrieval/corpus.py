"""语料加载与切分。

评测**只读缓存**，不联网
------------------------
语料由 ``scripts/fetch_corpus.py`` 抓取并落盘。评测阶段若现场抓取，
网络抖动与 arXiv 的持续新增会让同一份评测跑出不同结果，
**丧失可复现性**。因此这里只读缓存，缺失时明确报错。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from copilot.evidence.models import Document
from copilot.retrieval.base import Chunk
from copilot.textseg import iter_segments

__all__ = ["CorpusPaper", "load_corpus", "documents_from_corpus", "chunk_document"]

def find_corpus() -> Path | None:
    """定位语料缓存文件，找不到返回 ``None``。

    刻意用「**从包文件向上逐级查找**」而不是数 ``parents[N]``：
    本项目按 Agent 模块把源码分散在多个文件夹里，包的所在深度
    **会随结构调整而变化**。数层级的写法每搬一次家就断一次
    （实测已经断过一次），而向上查找对搬家免疫。

    也支持用环境变量 ``COPILOT_CORPUS`` 显式指定，便于测试与部署。
    """
    env = os.environ.get("COPILOT_CORPUS")
    if env:
        return Path(env)
    for base in Path(__file__).resolve().parents:
        candidate = base / "eval" / "dataset" / "corpus.json"
        if candidate.is_file():
            return candidate
        # 语料目录挂在**某个阶段**下面（如 05-format/eval/dataset/），
        # 而本模块在共享层 00-core 里 —— 只向上找永远到不了，
        # 必须再往下一层看一眼。
        for staged in sorted(base.glob("*/eval/dataset/corpus.json")):
            if staged.is_file():
                return staged
    return None


#: 启动时解析一次。找不到就是 None —— 具体报错留给 load_corpus 说清楚
DEFAULT_CORPUS = find_corpus()


@dataclass(frozen=True)
class CorpusPaper:
    doc_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: str
    url: str

    @property
    def content(self) -> str:
        """标题 + 摘要，用句号分隔。

        分隔符不能省：否则标题会被并进摘要的第一句，
        导致标题查询命中时给出的证据片段带上无关前缀。
        """
        return f"{self.title}。{self.abstract}"


def load_corpus(path: Path | None = None) -> tuple[list[CorpusPaper], dict]:
    """读取缓存的语料，返回 ``(papers, manifest)``。

    缓存缺失时抛 ``FileNotFoundError`` 并给出修复命令 ——
    **不静默联网**，否则可复现性会被悄悄破坏。
    """
    target = path or DEFAULT_CORPUS or find_corpus()
    if target is None or not target.is_file():
        raise FileNotFoundError(
            f"语料缓存不存在：{target}\n"
            "请先运行：python scripts\\fetch_corpus.py\n"
            "（或设置环境变量 COPILOT_CORPUS 指向语料文件）"
        )
    data = json.loads(target.read_text(encoding="utf-8"))
    papers = [
        CorpusPaper(
            doc_id=p["doc_id"],
            title=p["title"],
            abstract=p["abstract"],
            authors=tuple(p.get("authors", ())),
            published=p.get("published", ""),
            url=p.get("url", ""),
        )
        for p in data["papers"]
    ]
    return papers, data.get("manifest", {})


def documents_from_corpus(
    papers: list[CorpusPaper], *, include_title: bool = True
) -> list[Document]:
    """转成 P1 的 Document（共用同一套 hash 算法，保证校验链一致）。

    ``include_title=False`` 时**只索引摘要**。这是评测设计的一部分，
    不是省事：如果把标题也索引进去，用标题作查询就会**逐字命中**，
    所有方法都能拿满分，消融实验完全没有区分度。
    排除标题后，检索必须依靠标题与摘要之间的**主题词汇重叠**，
    词面方法与语义方法才会拉开差距。
    """
    out: list[Document] = []
    for p in papers:
        content = p.content if include_title else p.abstract
        out.append(Document.create(p.doc_id, content))
    return out


def chunk_document(
    doc: Document,
    *,
    max_chars: int = 320,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """按句子打包成定长片段，保留精确偏移。

    参数取舍：

    - ``max_chars`` 小 → 片段更聚焦、检索更精确，但上下文变少，
      可能丢掉需要跨句理解的证据；
    - ``overlap_sentences`` > 0 → 缓解"答案被切在边界上"的问题，
      代价是索引变大、结果里有重复内容。
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars 必须为正，当前 {max_chars}")
    if overlap_sentences < 0:
        raise ValueError(f"overlap_sentences 不能为负，当前 {overlap_sentences}")

    sentences = list(iter_segments(doc.content))
    if not sentences:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(sentences):
        start_i = i
        cur_start = sentences[i][0]
        cur_end = sentences[i][1]
        # 尽量多装句子，但不超 max_chars
        j = i
        while j + 1 < len(sentences):
            nxt_end = sentences[j + 1][1]
            if nxt_end - cur_start > max_chars:
                break
            j += 1
            cur_end = nxt_end

        text = doc.content[cur_start:cur_end]
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                char_start=cur_start,
                char_end=cur_end,
                text=text,
                source_hash=doc.content_hash,
            )
        )
        if j + 1 >= len(sentences):
            break
        # 回退若干句形成重叠，但必须前进，否则死循环
        i = max(j + 1 - overlap_sentences, start_i + 1)

    return chunks
