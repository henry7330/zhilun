"""模型接入层测试（**不联网**）。

核心验收：
1. **密钥永远不出现会被打印的地方**（repr、异常消息）；
2. 缺配置时给**可操作**的报错（说清缺哪个变量），而不是空指针；
3. ``chat`` 的响应解析、重试、错误映射都正确；
4. ``ping`` 用的是 **GET**（早期实现误用 POST 一路 404）。

所有网络调用都用注入的假 transport，**测试不依赖网络**。
"""

from __future__ import annotations

import json

import pytest

from copilot.llm import (
    DEFAULT_MODEL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    LlmConfig,
    LlmError,
    chat,
    ping,
)

# ─────────────────────── 夹具 ───────────────────────


def _cfg(**kw) -> LlmConfig:
    base = dict(base_url="https://example.test/v1", api_key="sk-test-secret-value", model="m1")
    base.update(kw)
    return LlmConfig(**base)


@pytest.fixture
def clean_env(monkeypatch):
    """清掉所有相关环境变量，保证测试不受本机配置影响。"""
    for name in (ENV_API_KEY, ENV_BASE_URL, ENV_MODEL,
                 "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ─────────────────── ★ 密钥不外泄 ───────────────────


def test_repr_does_not_leak_key() -> None:
    """★ 调试时最常做的事就是打印配置 —— 所以脱敏必须做在 repr 里。"""
    cfg = _cfg(api_key="sk-super-secret-1234")
    text = repr(cfg)
    assert "sk-super-secret" not in text
    assert "1234" not in text
    assert "已设置" in text


def test_repr_keeps_useful_fields() -> None:
    text = repr(_cfg())
    assert "example.test" in text
    assert "m1" in text


# ─────────────────── 配置装配 ───────────────────


def test_from_env_requires_key(clean_env) -> None:
    clean_env.setenv(ENV_BASE_URL, "https://x.test/v1")
    with pytest.raises(LlmError) as ei:
        LlmConfig.from_env()
    assert ENV_API_KEY in str(ei.value), "报错必须说清缺哪个变量"


def test_from_env_requires_base_url(clean_env) -> None:
    clean_env.setenv(ENV_API_KEY, "k")
    with pytest.raises(LlmError) as ei:
        LlmConfig.from_env()
    assert ENV_BASE_URL in str(ei.value)


def test_from_env_reads_all_fields(clean_env) -> None:
    clean_env.setenv(ENV_API_KEY, "k1")
    clean_env.setenv(ENV_BASE_URL, "https://x.test/v1/")
    clean_env.setenv(ENV_MODEL, "model-x")
    cfg = LlmConfig.from_env()
    assert cfg.api_key == "k1"
    assert cfg.base_url == "https://x.test/v1"      # 末尾斜杠被去掉
    assert cfg.model == "model-x"


def test_from_env_falls_back_to_openai_names(clean_env) -> None:
    clean_env.setenv("OPENAI_API_KEY", "k2")
    clean_env.setenv("OPENAI_BASE_URL", "https://y.test/v1")
    cfg = LlmConfig.from_env()
    assert cfg.api_key == "k2"
    assert cfg.base_url == "https://y.test/v1"


def test_from_env_default_model(clean_env) -> None:
    clean_env.setenv(ENV_API_KEY, "k")
    clean_env.setenv(ENV_BASE_URL, "https://x.test/v1")
    assert LlmConfig.from_env().model == DEFAULT_MODEL


def test_from_env_explicit_overrides_env(clean_env) -> None:
    clean_env.setenv(ENV_API_KEY, "k")
    clean_env.setenv(ENV_BASE_URL, "https://x.test/v1")
    clean_env.setenv(ENV_MODEL, "from-env")
    assert LlmConfig.from_env(model="from-arg").model == "from-arg"


# ─────────────────── chat ───────────────────


def _ok(payload: dict):
    return lambda url, headers, body, timeout: (200, payload)


def _reply(*texts: str) -> dict:
    return {"choices": [{"message": {"content": t}} for t in texts]}


def test_chat_returns_texts() -> None:
    cfg = _cfg()
    got = chat([{"role": "user", "content": "hi"}], config=cfg, transport=_ok(_reply("第一段", "第二段")))
    assert got == ["第一段", "第二段"]


def test_chat_sends_auth_and_model() -> None:
    captured: dict = {}

    def fake(url, headers, body, timeout):
        captured.update(url=url, headers=headers, body=body)
        return 200, _reply("ok")

    chat([{"role": "user", "content": "x"}], config=_cfg(), transport=fake)
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert captured["body"]["model"] == "m1"


def test_chat_passes_n_for_multiple_candidates() -> None:
    captured: dict = {}

    def fake(url, headers, body, timeout):
        captured["body"] = body
        return 200, _reply("a", "b")

    chat([{"role": "user", "content": "x"}], config=_cfg(), n=2, transport=fake)
    assert captured["body"]["n"] == 2


def test_chat_strips_empty_choices() -> None:
    payload = {"choices": [{"message": {"content": "  "}}, {"message": {"content": "ok"}}]}
    assert chat([{"role": "user", "content": "x"}], config=_cfg(), transport=_ok(payload)) == ["ok"]


def test_chat_raises_on_empty_choices() -> None:
    with pytest.raises(LlmError, match="没有内容"):
        chat([{"role": "user", "content": "x"}], config=_cfg(), transport=_ok({"choices": []}))


def test_chat_reports_auth_error_detail() -> None:
    payload = {"error": {"message": "invalid api key"}}
    with pytest.raises(LlmError) as ei:
        chat(
            [{"role": "user", "content": "x"}],
            config=_cfg(),
            transport=lambda *a: (401, payload),
        )
    msg = str(ei.value)
    assert "401" in msg
    assert "invalid api key" in msg


def test_chat_error_message_never_contains_key() -> None:
    """★ 异常消息会被写进报告 —— 里面绝不能有密钥。"""
    cfg = _cfg(api_key="sk-must-not-appear")
    with pytest.raises(LlmError) as ei:
        chat([{"role": "user", "content": "x"}], config=cfg, transport=lambda *a: (500, {"error": "boom"}))
    assert "sk-must-not-appear" not in str(ei.value)


def test_chat_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky(url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 503, {"error": "busy"}
        return 200, _reply("终于成功")

    cfg = _cfg(max_retries=2, timeout=30)
    assert chat([{"role": "user", "content": "x"}], config=cfg, transport=flaky) == ["终于成功"]
    assert calls["n"] == 2


def test_chat_gives_up_after_retries() -> None:
    calls = {"n": 0}

    def always_503(url, headers, body, timeout):
        calls["n"] += 1
        return 503, {"error": "busy"}

    with pytest.raises(LlmError, match="503"):
        chat([{"role": "user", "content": "x"}], config=_cfg(max_retries=1), transport=always_503)
    assert calls["n"] == 2   # 首次 + 1 次重试


def test_chat_does_not_retry_on_400(monkeypatch) -> None:
    """4xx 是请求本身有问题，重试没有意义，只会浪费钱。"""
    calls = {"n": 0}

    def bad_request(url, headers, body, timeout):
        calls["n"] += 1
        return 400, {"error": "bad"}

    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(LlmError, match="400"):
        chat([{"role": "user", "content": "x"}], config=_cfg(max_retries=2), transport=bad_request)
    assert calls["n"] == 1


def test_chat_wraps_network_exception(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def boom(*a):
        raise ConnectionError("网络断了")

    with pytest.raises(LlmError, match="ConnectionError"):
        chat([{"role": "user", "content": "x"}], config=_cfg(max_retries=0), transport=boom)


# ─────────────────── ping（GET！） ───────────────────


def test_ping_uses_get_and_lists_models() -> None:
    """★ 早期实现用 POST 打 /models，一路 404 —— 这条测试锁住它。"""
    seen: dict = {}

    def fake_get(url, headers, timeout):
        seen["url"] = url
        seen["headers"] = headers
        return 200, {"data": [{"id": "a"}, {"id": "b"}]}

    models = ping(_cfg(), transport=fake_get)
    assert models == ["a", "b"]
    assert seen["url"].endswith("/models")
    assert "Authorization" in seen["headers"]


def test_ping_raises_on_non_200() -> None:
    with pytest.raises(LlmError, match="探活失败"):
        ping(_cfg(), transport=lambda url, h, t: (404, {}))


def test_ping_skips_entries_without_id() -> None:
    models = ping(_cfg(), transport=lambda url, h, t: (200, {"data": [{"id": "a"}, {}, {"id": ""}]}))
    assert models == ["a"]
