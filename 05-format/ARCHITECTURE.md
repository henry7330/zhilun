# 路径与环境：两条必须知道的坑

> **这个文件不描述架构。** 架构在根 `README.md` 的 §2–§4 与 §15–§17；
> 每个阶段的设计在自己的 `README.md`。
>
> 这里只讲**工程环境**上两件会咬人的事。

---

# 一、⚠️ 仓库不能放在含中文的路径下

这不是洁癖，是一个**会静默失败**的具体 bug。

## 现象

把仓库放在 `E:\学习\agent\论文辅助系统` 这类含中文的路径下：

```
$ pip install -e .
Successfully installed zhilun-0.1.0        ← 看起来成功了

$ python -c "import copilot"
ModuleNotFoundError: No module named 'copilot'   ← 但就是找不到
```

**没有任何错误提示**，只有一个"找不到"。

## 根因（已实测确认）

1. `pip` 把可编辑安装路径写进 `.pth` 文件时用 **UTF-8** 编码：

   ```
   _editable_impl_zhilun.pth:
   E:\学习\agent\论文辅助系统\src        ← 24 个非 ASCII 字节
   ```

2. 而 CPython 的 `site.py` 读 `.pth` 用的是 **locale 编码**：

   ```python
   # Lib/site.py:184
   f = io.TextIOWrapper(io.open_code(fullname), encoding="locale")
   # Lib/site.py:199
   if not dircase in known_paths and os.path.exists(dir):
       sys.path.append(dir)      # ← 路径解成乱码 -> exists 为假 -> 静默跳过
   ```

3. 中文区域下 locale = **cp936**，UTF-8 字节按 GBK 解码 → 路径不存在 →
   `sys.path` 不加入 `src`，**且不报错**。

## 解法：把仓库放在英文路径下

```
E:\dev\zhilun          ✅
D:\code\zhilun         ✅
E:\学习\agent\...      ❌ 可编辑安装会静默失效
```

实测对照（同一份代码、同样的命令）：

| 路径 | `pip install -e .` | `import copilot` |
|---|---|---|
| `E:\dev\zhilun-probe` | 成功 | ✅ 成功 |
| `E:\学习\agent\论文辅助系统` | "成功" | ❌ 失败 |

## 已经试过但无效的绕法

| 方案 | 结果 |
|---|---|
| `PYTHONUTF8=1` | ❌ 无效 |
| ASCII junction + pip 安装 | ❌ **pip 会把 junction 解析回中文真实路径** |
| junction + 手工写 `.pth` | ⚠️ 可用，但 junction 被递归删除时有删掉真实内容的风险 |
| 非 editable `pip install .` | ⚠️ 能导入，但**改源码不生效**，需反复重装 |
| 手工把 `.pth` 写成 `\uXXXX` 转义 | ⚠️ 曾经可用，但**已废弃** —— 换路径、换环境都要重跑脚本 |

**结论：别绕，直接放英文路径。** 本次已将仓库从 `E:\学习\agent\论文辅助系统`
迁到 `E:\dev\zhilun`。

## 验证方法

必须能从**任意目录**导入（不要依赖 `pytest` 的 `pythonpath`）：

```powershell
cd C:\
python -c "import copilot; print(copilot.__file__)"
```

---

# 二、依赖为什么分组装

核心校验逻辑（格式规则、证据绑定、统计）**不需要向量模型**。
先把这些用最小依赖跑通，避免一开始就被重依赖拖住
（`sentence-transformers` 会拉入数 GB 的 `torch`）。

| 组 | 内容 | 何时需要 |
|---|---|---|
| （默认） | langgraph / rank-bm25 / numpy / scipy / httpx / python-dotenv | 任何模块 |
| `dev` | pytest / ruff | 跑测试与格式化 |
| `ui` | streamlit | 本地交互页 |
| `dense` | model2vec | 稠密检索（可选，会拉模型） |
| `pdf` | pypdf | 解析 PDF 版查重报告 |

```powershell
python -m pip install -e ".[dev]"        # 最常用
python -m pip install -e ".[dev,ui]"     # 加交互页
```

---

# 三、开工前的三条自检

```powershell
# ① Python ≥ 3.11，依赖齐，import 正常，六个阶段注册
python 05-format\scripts\check_env.py

# ② 全部测试（714 项）
python -m pytest

# ③ 一条命令看效果（不需要模型）
python -m copilot --demo
```

三条都过再动手改代码。
