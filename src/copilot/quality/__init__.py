"""阶段内节点的实现：格式规则、文本一致性、报告解析、改写、降AI。

这些包**不对外作为服务**，它们被 E / F 两个阶段调用::

    E 格式规整   → quality/format/ + quality/review/ + delivery/export/ + delivery/prcheck/
    F 降重降AI   → quality/report_parse/ + quality/rewrite/ + quality/deai/

对外入口在 ``copilot.format_stage`` 与 ``copilot.revise``。

## 为什么这些包叫 quality 而阶段叫 format/revise

历史上它们都在一个"质检"阶段里，后来链路拆成「格式规整」与「降重降AI」两个阶段，
但节点实现没有搬家 —— **节点是节点，阶段是阶段**，两者不必同名。
"""
