"""最小 OpenAI 兼容客户端（模型接入层）。

为什么不用 ``openai`` SDK
-------------------------
本系统只用一个接口（``POST /chat/completions``）。
一个 SDK 换来的是：额外的依赖、额外的版本兼容问题、额外的隐式重试行为。

而 ``httpx`` 已经因为 LangGraph 装上了 —— **用现成的，不新增依赖**。

配置全部来自环境变量
--------------------
**密钥绝不写在代码里，也绝不进日志。**

==========================  ==============================  ============================
环境变量                     作用                            缺省
==========================  ==============================  ============================
``COPILOT_LLM_API_KEY``     密钥（**必填**）                 回退 ``OPENAI_API_KEY``
``COPILOT_LLM_BASE_URL``    接口根地址，如 ``…/v1``          回退 ``OPENAI_BASE_URL``
``COPILOT_LLM_MODEL``       模型名                           ``gpt-6.1-sol``
==========================  ==============================  ============================

``LlmConfig`` 的 ``__repr__`` 经过脱敏 —— 打印配置不会泄露密钥。
这是刻意的：**调试时最常做的事就是打印配置**。

典型用法::

    from copilot.llm import LlmConfig, chat

    cfg = LlmConfig.from_env()                 # 从环境变量装配
    texts = chat([{"role": "user", "content": "…"}], config=cfg)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "LlmConfig",
    "LlmError",
    "chat",
    "ping",
    "load_env",
    "Transport",
    "GetTransport",
    "DEFAULT_MODEL",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
]

#: 主用环境变量名（回退到 OPENAI_* 是为了兼容常见的既有配置）
ENV_API_KEY = "COPILOT_LLM_API_KEY"
ENV_BASE_URL = "COPILOT_LLM_BASE_URL"
ENV_MODEL = "COPILOT_LLM_MODEL"

DEFAULT_MODEL = "gpt-6.1-sol"


class LlmError(RuntimeError):
    """模型调用失败。

    **消息里绝不包含密钥** —— 调用方可以直接把这个消息写进报告。
    """


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class LlmConfig:
    """一次模型调用的全部配置。"""

    base_url: str
    #: ``repr=False`` + 自定义 ``__repr__``：**打印配置不泄露密钥**
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: float = 60.0
    #: 失败重试次数（仅对超时 / 429 / 5xx 重试）
    max_retries: int = 2

    def __repr__(self) -> str:
        # 只显示"有没有配"，**不显示密钥的任何片段** ——
        # 调试时最常做的事就是打印配置，所以脱敏必须在这里做。
        return (
            f"LlmConfig(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key=<已设置 {len(self.api_key)} 字符>, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens})"
        )

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LlmConfig:
        """从环境变量装配配置。

        密钥缺失时抛 ``LlmError``，消息里**只说缺哪个变量**，不回显任何值。

        若配置写在 ``.env`` 里，先调用一次 :func:`load_env`。
        """
        api_key = _env(ENV_API_KEY, "OPENAI_API_KEY")
        if not api_key:
            raise LlmError(
                f"未配置模型密钥：请设置环境变量 {ENV_API_KEY}"
                f"（或 OPENAI_API_KEY），可写在 .env 里（.env 已在 .gitignore 中）"
            )
        base_url = (_env(ENV_BASE_URL, "OPENAI_BASE_URL") or "").rstrip("/")
        if not base_url:
            raise LlmError(f"未配置接口地址：请设置环境变量 {ENV_BASE_URL}（或 OPENAI_BASE_URL）")

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model or _env(ENV_MODEL, "OPENAI_MODEL") or DEFAULT_MODEL,
            temperature=0.3 if temperature is None else temperature,
            max_tokens=1024 if max_tokens is None else max_tokens,
            timeout=60.0 if timeout is None else timeout,
        )


def _env_candidates() -> list[Any]:
    """``.env`` 的候选位置，按优先级排列。

    为什么要列一串而不是只认一个：

    - 用户可能从**仓库根**、**某个阶段目录**或**任意目录**运行；
    - 包被 `pip install` 后，``__file__`` 在 ``site-packages`` 里，
      "往上走"根本走不到仓库 —— 所以当前工作目录必须排第一；
    - 源码开发时（editable 安装）``__file__`` 在 ``src/copilot/``，
      往上两级就是仓库根，这一条覆盖"在别的目录里跑"的情况。
    """
    from pathlib import Path

    cands = [Path.cwd() / ".env"]
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / ".env"
        if cand not in cands:
            cands.append(cand)
    return cands


def load_env(path: str | None = None) -> bool:
    """从 ``.env`` 加载环境变量，返回是否成功。

    **为什么是独立的模块级函数而不是 import 副作用**：
    库不应该"被 import 就读文件"。显式调用一次，行为可预期。

    不传 ``path`` 时按 :func:`_env_candidates` 的顺序找第一个存在的 ``.env``。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv 是核心依赖
        return False

    if path is not None:
        return load_dotenv(dotenv_path=path, override=False)

    for cand in _env_candidates():
        if cand.is_file():
            return load_dotenv(dotenv_path=str(cand), override=False)
    return False


#: 传输层签名：``(url, headers, payload, timeout) -> (status_code, json_or_text)``
Transport = Callable[[str, dict[str, str], dict[str, Any], float], "tuple[int, Any]"]

#: GET 传输层签名：``(url, headers, timeout) -> (status_code, json_or_text)``
GetTransport = Callable[[str, dict[str, str], float], "tuple[int, Any]"]


def _default_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float):
    import httpx

    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        return response.status_code, response.json()
    except Exception:  # noqa: BLE001 - 非 JSON 响应（网关错误页等）
        return response.status_code, response.text


def _default_get(url: str, headers: dict[str, str], timeout: float):
    import httpx

    response = httpx.get(url, headers=headers, timeout=timeout)
    try:
        return response.status_code, response.json()
    except Exception:  # noqa: BLE001
        return response.status_code, response.text


def _extract_texts(data: Any) -> list[str]:
    """从 ``/chat/completions`` 响应里取出全部候选文本。"""
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            out.append(content.strip())
    return out


def chat(
    messages: Sequence[dict[str, str]],
    *,
    config: LlmConfig,
    n: int = 1,
    transport: Transport | None = None,
) -> list[str]:
    """调用 ``/chat/completions``，返回候选文本列表。

    ``transport`` 用于测试注入 —— **测试不应该真的联网**。
    """
    url = f"{config.base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if n > 1:
        payload["n"] = n

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    send = transport or _default_transport

    last_error = "未知错误"
    for attempt in range(1 + max(0, config.max_retries)):
        try:
            status, data = send(url, headers, payload, config.timeout)
        except Exception as exc:  # noqa: BLE001 - 网络层异常统一成 LlmError
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < config.max_retries:
                time.sleep(min(2.0 * (attempt + 1), 6.0))
                continue
            raise LlmError(f"模型调用失败（{last_error}）") from None

        if status == 200:
            texts = _extract_texts(data)
            if texts:
                return texts
            raise LlmError("模型返回成功但没有内容（choices 为空）")

        # 只对可恢复的状态重试
        if status in (408, 429, 500, 502, 503, 504) and attempt < config.max_retries:
            last_error = f"HTTP {status}"
            time.sleep(min(2.0 * (attempt + 1), 6.0))
            continue

        detail = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("code") or "")[:200]
            elif err:
                detail = str(err)[:200]
        elif isinstance(data, str):
            detail = data[:200]
        raise LlmError(f"模型调用失败：HTTP {status}" + (f"（{detail}）" if detail else ""))

    raise LlmError(f"模型调用失败（{last_error}）")


def ping(config: LlmConfig, *, transport: GetTransport | None = None) -> list[str]:
    """探活：列出该端点可用的模型 id。

    单独留一个"能确认端点是否通"的最小调用 —— 排查连接问题时先跑它，
    不要把问题混进业务逻辑里。

    **注意这是 GET**（``/models``）。早期实现误用 POST，
    结果一路 404 —— 排查了半天才发现是方法错了。
    """
    url = f"{config.base_url}/models"
    headers = {"Authorization": f"Bearer {config.api_key}"}
    send = transport or _default_get
    status, data = send(url, headers, config.timeout)
    if status != 200 or not isinstance(data, dict):
        raise LlmError(f"探活失败：HTTP {status}")
    return [str(m.get("id")) for m in (data.get("data") or []) if m.get("id")]
