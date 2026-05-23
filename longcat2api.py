#!/usr/bin/env python3
"""
LongCat → OpenAI API 兼容层 (v3 — oversea端点，无需Cookie)

逆向结论（v3更新）:
- 无登录端点: POST /api/v1/chat-completion-oversea-V2
- 无需Cookie、无需mtgsig、无需session-create
- 请求体: {content, agentId, messages, reasonEnabled, searchEnabled, regenerate}
- SSE格式: event.type = create/content/reason/finish
- event.content = 增量文本, event.finalContentX = 完整文本
- lastOne=true 标记结束
- tokenInfo 包含 usage 信息

架构参考 ds2api:
- SSE解析管道 → LineResult
- 微批缓冲 16chars/10ms
- Hook模式流式引擎
- reasoning_content 标准字段
- tool_calls XML检测
- Cookie池（可选，用于登录模式增强）
- 空输出重试
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional, Sequence

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# ============================================================
# 1. 配置
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("longcat2api")

LONGCAT_BASE = "https://longcat.chat"
API_BASE = f"{LONGCAT_BASE}/api/v1"

# oversea端点 — 无需Cookie
OVERSEA_ENDPOINT = f"{API_BASE}/chat-completion-oversea-V2"

# 国内登录端点 — 需要Cookie
CN_ENDPOINT = f"{API_BASE}/chat-completion-V2"

# 微批缓冲
BATCH_MIN_CHARS = 16
BATCH_MAX_WAIT_MS = 10
BATCH_FLUSH_ON_NEWLINE = True
BATCH_FLUSH_ON_FINISH = True

# 空输出重试
MAX_EMPTY_RETRIES = 2

# 429限流自动重试
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_DELAY = 30  # 首次等待30秒，指数递增

# Keepalive
KEEPALIVE_INTERVAL = 15

# 默认Cookie
DEFAULT_COOKIE = os.environ.get("LONGCAT_COOKIE", "")

# 模式选择: oversea(默认免登录) / cn(需Cookie)
DEFAULT_MODE = os.environ.get("LONGCAT_MODE", "oversea")

# API Key认证 — OpenAI格式 Bearer token
API_KEY = os.environ.get("LONGCAT_API_KEY", "")

# ============================================================
# 2. 数据结构
# ============================================================

class ContentType(str, Enum):
    THINKING = "thinking"
    TEXT = "text"


@dataclass
class ContentPart:
    text: str
    type: ContentType = ContentType.TEXT


@dataclass
class LineResult:
    parsed: bool = False
    stop: bool = False
    content_filter: bool = False
    error_message: str = ""
    parts: list[ContentPart] = field(default_factory=list)
    next_type: ContentType = ContentType.TEXT
    raw_event_type: str = ""
    conversation_id: str = ""
    message_id: int = 0
    usage: dict = field(default_factory=dict)
    final_content: str = ""  # finalContentX


@dataclass
class ParsedToolCall:
    name: str
    input: dict = field(default_factory=dict)
    id: str = ""


@dataclass
class CollectResult:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    content_filter: bool = False
    error_message: str = ""
    usage: dict = field(default_factory=dict)
    final_content: str = ""


@dataclass
class Turn:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    content_filter: bool = False
    error_message: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StandardRequest:
    model: str = "longcat-default"
    agent_id: str = "1"
    final_prompt: str = ""
    thinking_enabled: bool = False
    search_enabled: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: list[dict] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    tool_choice: str = "auto"
    raw_messages: list[dict] = field(default_factory=list)
    stream: bool = False
    mode: str = "oversea"  # oversea or cn

    def oversea_payload(self) -> dict:
        """构建oversea端点请求体"""
        # 简化messages格式 — 每次只发当前用户消息
        return {
            "content": self.final_prompt,
            "agentId": self.agent_id,
            "messages": [
                {
                    "role": "user",
                    "events": [{"type": "userMsg", "content": self.final_prompt, "status": "FINISHED"}],
                    "chatStatus": "FINISHED",
                    "messageId": 1,
                    "idType": "custom",
                },
                {
                    "role": "assistant",
                    "events": [],
                    "chatStatus": "LOADING",
                    "messageId": 2,
                    "idType": "custom",
                },
            ],
            "reasonEnabled": 1 if self.thinking_enabled else 0,
            "searchEnabled": 1 if self.search_enabled else 0,
            "regenerate": 0,
        }

    def cn_payload(self, conversation_id: str) -> dict:
        """构建国内登录端点请求体"""
        return {
            "conversationId": conversation_id,
            "agentId": self.agent_id,
            "content": self.final_prompt,
            "reasonEnabled": 1 if self.thinking_enabled else 0,
            "searchEnabled": 1 if self.search_enabled else 0,
            "parentMessageId": 0,
        }


# ============================================================
# 3. SSE解析器 — Oversea格式
# ============================================================

def parse_oversea_sse_line(raw: str) -> tuple[Optional[dict], bool, bool]:
    """解析一行SSE"""
    line = raw.strip()
    if not line or line.startswith((":")) or line.startswith("event:"):
        return None, False, False
    if not line.startswith("data:"):
        return None, False, False
    data_str = line[5:].strip()
    if data_str == "[DONE]":
        return None, True, True
    try:
        chunk = json.loads(data_str)
        return chunk, False, True
    except (json.JSONDecodeError, ValueError):
        return None, False, False


def parse_oversea_content_line(
    raw: str,
    thinking_enabled: bool,
    current_type: ContentType = ContentType.TEXT,
) -> LineResult:
    """
    解析oversea SSE行 → LineResult
    格式: {"event": {"type": "content", "content": "xxx"}, "lastOne": false, ...}
    """
    result = LineResult()
    chunk, is_done, is_valid = parse_oversea_sse_line(raw)

    if not is_valid:
        return result

    if is_done:
        result.parsed = True
        result.stop = True
        return result

    if not isinstance(chunk, dict):
        return result

    # 检查HTTP错误（oversea端点可能返回HTTP 200但body中code!=0）
    if chunk.get("code") is not None and chunk.get("code") != 0:
        code = chunk.get("code")
        msg = chunk.get("message", "Unknown error")
        if code == 429:
            result.parsed = True
            result.stop = True
            result.error_message = f"429: {msg}"
            return result
        result.parsed = True
        result.stop = True
        result.error_message = f"Error {code}: {msg}"
        return result

    event = chunk.get("event", {})
    if not isinstance(event, dict):
        return result

    event_type = event.get("type", "")
    result.raw_event_type = event_type
    result.conversation_id = chunk.get("conversationId", "")
    result.message_id = chunk.get("messageId", 0)

    # 结束
    if event_type == "finish":
        result.parsed = True
        result.stop = True
        result.final_content = event.get("finalContentX", "")
        finish_type = event.get("finishType", "")
        if finish_type == "sensitive":
            result.content_filter = True
        # usage
        usage = event.get("usage", {})
        token_info = chunk.get("tokenInfo", {})
        result.usage = {
            "prompt_tokens": usage.get("inputTokens", token_info.get("promptTokens", 0)),
            "completion_tokens": usage.get("outputTokens", token_info.get("completionTokens", 0)),
            "total_tokens": usage.get("totalTokens", token_info.get("totalTokens", 0)),
        }
        return result

    # 创建事件
    if event_type == "create":
        result.parsed = True
        return result

    # 内容
    if event_type == "content":
        content = event.get("content", "")
        if content:
            result.parsed = True
            result.parts.append(ContentPart(text=content, type=ContentType.TEXT))
            # 不改变next_type——让调用者根据后续事件决定状态转换
            # 如果下一个事件是think/reason/summary/create，说明还在推理阶段
            # 如果下一个事件是finish，说明这是最终回复
        return result

    # 推理（oversea端点用reason，CN端点用think）
    if event_type == "reason":
        content = event.get("content", "")
        if content:
            result.parsed = True
            result.parts.append(ContentPart(text=content, type=ContentType.THINKING))
            result.next_type = ContentType.THINKING
        return result

    # CN端点的think事件 = 推理内容
    if event_type == "think":
        content = event.get("content", "")
        if content:
            result.parsed = True
            result.parts.append(ContentPart(text=content, type=ContentType.THINKING))
            result.next_type = ContentType.THINKING
        return result

    # CN端点的summary事件 = Thinker阶段结束标记，忽略内容
    if event_type == "summary":
        result.parsed = True
        return result

    # 搜索结果
    if event_type in ("common_search", "general_search", "local_life_search"):
        # 搜索结果作为text输出
        content = event.get("content", "")
        if isinstance(content, str) and content:
            result.parsed = True
            result.parts.append(ContentPart(text=content, type=ContentType.TEXT))
        elif isinstance(content, list) and content:
            texts = []
            for item in content:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    snippet = item.get("snippet", item.get("content", ""))
                    if title:
                        texts.append(title)
                    if snippet:
                        texts.append(snippet)
            if texts:
                result.parsed = True
                result.parts.append(ContentPart(text="\n".join(texts), type=ContentType.TEXT))
        return result

    # 错误
    if event_type == "event_error":
        result.parsed = True
        result.stop = True
        result.error_message = event.get("message", event.get("content", "Unknown error"))
        return result

    # 其他
    return result


# ============================================================
# 4. StreamAccumulator
# ============================================================

class StreamAccumulator:
    def __init__(self):
        self.raw_text = ""
        self.visible_text = ""
        self.thinking = ""

    def apply(self, parsed: LineResult) -> list[ContentPart]:
        for part in parsed.parts:
            if part.type == ContentType.THINKING:
                self.thinking += part.text
            else:
                self.raw_text += part.text
                self.visible_text += part.text
        return parsed.parts


# ============================================================
# 5. BatchBuffer
# ============================================================

@dataclass
class BatchItem:
    content: str
    content_type: ContentType
    is_newline: bool = False


class BatchBuffer:
    def __init__(self, min_chars=16, max_wait_ms=10, flush_on_newline=True, flush_on_finish=True):
        self.min_chars = min_chars
        self.max_wait_ms = max_wait_ms
        self.flush_on_newline = flush_on_newline
        self.flush_on_finish = flush_on_finish
        self._items: list[BatchItem] = []
        self._buffered_chars = 0
        self._last_flush_time = time.monotonic()

    def add(self, text: str, content_type: ContentType) -> list[list[BatchItem]]:
        flushes = []
        if not text:
            return flushes
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line:
                self._items.append(BatchItem(content=line, content_type=content_type))
                self._buffered_chars += len(line)
            if i < len(lines) - 1:
                self._items.append(BatchItem(content="\n", content_type=content_type, is_newline=True))
                if self.flush_on_newline:
                    flushes.append(self._flush())
        if self._should_flush():
            flushes.append(self._flush())
        return flushes

    def finish(self) -> list[list[BatchItem]]:
        if self.flush_on_finish and self._items:
            return [self._flush()]
        return []

    def _should_flush(self):
        if self._buffered_chars >= self.min_chars:
            return True
        elapsed_ms = (time.monotonic() - self._last_flush_time) * 1000
        return elapsed_ms >= self.max_wait_ms and bool(self._items)

    def _flush(self) -> list[BatchItem]:
        items = self._items
        self._items = []
        self._buffered_chars = 0
        self._last_flush_time = time.monotonic()
        return items


# ============================================================
# 6. ToolCall
# ============================================================

TOOL_CALLS_RE = re.compile(r"<tool_calls>\s*(.*?)\s*</tool_calls>", re.DOTALL)
INVOKE_RE = re.compile(r"<invoke\s+name\s*=\s*\"([^\"]+)\"\s*>(.*?)</invoke>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter\s+name\s*=\s*\"([^\"]+)\"\s*>(.*?)</parameter>", re.DOTALL)


def detect_tool_calls(text: str, tool_names: list[str] | None = None) -> list[ParsedToolCall]:
    if not text or "<tool_calls>" not in text:
        return []
    calls = []
    for tc_match in TOOL_CALLS_RE.finditer(text):
        for inv_match in INVOKE_RE.finditer(tc_match.group(1)):
            func_name = inv_match.group(1)
            if tool_names and func_name not in tool_names:
                continue
            params = {}
            for pm in PARAM_RE.finditer(inv_match.group(2)):
                try:
                    params[pm.group(1)] = json.loads(pm.group(2).strip())
                except:
                    params[pm.group(1)] = pm.group(2).strip()
            calls.append(ParsedToolCall(name=func_name, input=params, id=f"call_{uuid.uuid4().hex[:24]}"))
    return calls


def strip_tool_calls_markup(text: str) -> str:
    if "<tool_calls>" not in text:
        return text
    return TOOL_CALLS_RE.sub("", text).strip()


def format_openai_tool_calls(calls: list[ParsedToolCall]) -> list[dict]:
    return [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.input, ensure_ascii=False)}}
        for c in calls
    ]


# ============================================================
# 7. 非流式收集
# ============================================================

async def collect_stream_oversea(response: httpx.Response, thinking_enabled: bool) -> CollectResult:
    text_parts, thinking_parts = [], []
    current_type = ContentType.THINKING if thinking_enabled else ContentType.TEXT
    usage, final_content = {}, ""
    content_filter, error_message = False, ""

    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        # 先检查是否是429等错误JSON（不是SSE格式）
        if line.startswith("{"):
            try:
                j = json.loads(line)
                if j.get("code") is not None and j.get("code") != 0:
                    return CollectResult(error_message=f"{j.get('code')}: {j.get('message', 'Unknown')}")
            except:
                pass
            continue  # 非SSE JSON，跳过

        result = parse_oversea_content_line(line, thinking_enabled, current_type)
        current_type = result.next_type
        if result.stop:
            if result.content_filter:
                content_filter = True
            if result.error_message:
                error_message = result.error_message
            usage = result.usage
            final_content = result.final_content
            break
        if not result.parsed:
            continue
        for part in result.parts:
            if part.type == ContentType.THINKING:
                thinking_parts.append(part.text)
            else:
                text_parts.append(part.text)

    raw_text = "".join(text_parts)
    raw_thinking = "".join(thinking_parts)

    tool_calls = []
    if raw_text:
        detected = detect_tool_calls(raw_text)
        if detected:
            tool_calls = detected
            raw_text = strip_tool_calls_markup(raw_text)

    return CollectResult(text=raw_text, thinking=raw_thinking, tool_calls=tool_calls,
                         content_filter=content_filter, error_message=error_message,
                         usage=usage, final_content=final_content)


def build_turn(result: CollectResult, thinking_enabled: bool) -> Turn:
    finish_reason = "stop"
    if result.tool_calls:
        finish_reason = "tool_calls"
    elif result.content_filter:
        finish_reason = "content_filter"

    u = result.usage
    return Turn(
        text=result.text, thinking=result.thinking if thinking_enabled else "",
        tool_calls=result.tool_calls, finish_reason=finish_reason,
        content_filter=result.content_filter, error_message=result.error_message,
        prompt_tokens=u.get("prompt_tokens", 0),
        completion_tokens=u.get("completion_tokens", 0),
        total_tokens=u.get("total_tokens", 0),
    )


# ============================================================
# 8. 请求标准化
# ============================================================

MODEL_ALIASES = {
    "longcat-default": ("1", False, False),
    "longcat-reason": ("1", True, False),
    "longcat-search": ("1", False, True),
    "longcat-reason-search": ("1", True, True),
    "longcat": ("1", False, False),
    "longcat-r1": ("1", True, False),
    "longcat-pro": ("1", True, True),
}


def resolve_model(model: str) -> tuple[str, bool, bool]:
    if model not in MODEL_ALIASES:
        raise ValueError(f"Unknown model: {model}. Available: {', '.join(MODEL_ALIASES.keys())}")
    return MODEL_ALIASES[model]


def build_prompt_from_messages(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
        if not content:
            continue
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        elif role == "tool":
            parts.append(f"[Tool Result: {msg.get('name', 'tool')}]\n{content}")
        else:
            parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def normalize_openai_request(body: dict) -> StandardRequest:
    model = body.get("model", "longcat-default")
    agent_id, thinking, search = resolve_model(model)
    if body.get("reason_enabled") is not None:
        thinking = bool(body["reason_enabled"])
    if body.get("search_enabled") is not None:
        search = bool(body["search_enabled"])

    messages = body.get("messages", [])
    final_prompt = build_prompt_from_messages(messages)

    tools = body.get("tools", [])
    tool_names = [t.get("function", {}).get("name", "") for t in tools if t.get("function", {}).get("name")]

    tool_choice = body.get("tool_choice", "auto")
    if isinstance(tool_choice, dict):
        forced_name = tool_choice.get("function", {}).get("name", "")
        tool_choice = f"forced:{forced_name}" if forced_name else "auto"

    return StandardRequest(
        model=model, agent_id=agent_id, final_prompt=final_prompt,
        thinking_enabled=thinking, search_enabled=search,
        temperature=body.get("temperature"), max_tokens=body.get("max_tokens"),
        tools=tools, tool_names=tool_names, tool_choice=tool_choice,
        raw_messages=messages, stream=body.get("stream", False),
        mode=DEFAULT_MODE,
    )


# ============================================================
# 9. OpenAI格式化输出
# ============================================================

def build_chat_completion(completion_id: str, model: str, turn: Turn, created: int | None = None) -> dict:
    if created is None:
        created = int(time.time())
    message = {"role": "assistant", "content": turn.text}
    if turn.thinking:
        message["reasoning_content"] = turn.thinking
    if turn.tool_calls:
        message["tool_calls"] = format_openai_tool_calls(turn.tool_calls)
        message["content"] = None
    return {
        "id": completion_id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": turn.finish_reason}],
        "usage": {"prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens, "total_tokens": turn.total_tokens},
    }


def build_chat_stream_chunk(completion_id: str, model: str, delta: dict, finish_reason: str | None = None, created: int | None = None) -> dict:
    if created is None:
        created = int(time.time())
    choice = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [choice]}


# ============================================================
# 10. Cookie池（可选，CN模式用）
# ============================================================

class CookiePool:
    def __init__(self):
        self._cookies: OrderedDict[str, dict] = OrderedDict()
        self._lock = asyncio.Lock()

    def add(self, cookie: str, label: str = "") -> None:
        if not cookie or not cookie.strip():
            return
        cookie = cookie.strip()
        if cookie not in self._cookies:
            self._cookies[cookie] = {"label": label or f"cookie-{len(self._cookies)}", "last_used": 0, "error_count": 0, "active": True}

    def remove(self, cookie: str) -> None:
        self._cookies.pop(cookie, None)

    async def acquire(self) -> str | None:
        async with self._lock:
            if not self._cookies:
                return None
            active = [(c, i) for c, i in self._cookies.items() if i["active"]]
            if not active:
                for i in self._cookies.values():
                    i["active"], i["error_count"] = True, 0
                active = list(self._cookies.items())
            active.sort(key=lambda x: (x[1]["error_count"], x[1]["last_used"]))
            cookie, info = active[0]
            info["last_used"] = time.time()
            return cookie

    async def report_error(self, cookie: str) -> None:
        async with self._lock:
            if cookie in self._cookies:
                self._cookies[cookie]["error_count"] += 1
                if self._cookies[cookie]["error_count"] >= 3:
                    self._cookies[cookie]["active"] = False

    async def report_success(self, cookie: str) -> None:
        async with self._lock:
            if cookie in self._cookies:
                self._cookies[cookie]["error_count"] = 0

    def list_cookies(self) -> list[dict]:
        return [{"label": i["label"], "active": i["active"], "errors": i["error_count"]} for i in self._cookies.values()]


cookie_pool = CookiePool()
if DEFAULT_COOKIE:
    cookie_pool.add(DEFAULT_COOKIE, "default")


# ============================================================
# 11. 上游客户端
# ============================================================

OVERSEA_HEADERS = {
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "X-Client-Language": "zh",
    "Accept": "text/event-stream,application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Origin": LONGCAT_BASE,
    "Referer": f"{LONGCAT_BASE}/t",
}

CN_HEADERS = {
    **OVERSEA_HEADERS,
}


class LongCatClient:
    def __init__(self, cookie: str = "", mode: str = "oversea"):
        self.cookie = cookie
        self.mode = mode

    def _headers(self) -> dict:
        h = dict(CN_HEADERS if self.mode == "cn" else OVERSEA_HEADERS)
        if self.cookie:
            h["Cookie"] = self.cookie
        return h

    @property
    def endpoint(self) -> str:
        return CN_ENDPOINT if self.mode == "cn" else OVERSEA_ENDPOINT

    async def create_session(self, agent_id: str = "1") -> dict:
        """仅CN模式需要"""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.post(f"{API_BASE}/session-create", headers=self._headers(), json={"agentId": agent_id})
            data = resp.json()
            if data.get("code") != 0:
                raise Exception(f"session-create failed: {data.get('message', '')}")
            return data.get("data", {})

    async def chat_completion_raw(self, std_req: StandardRequest, conversation_id: str = "") -> tuple[httpx.Response, httpx.AsyncClient]:
        """发送聊天请求"""
        if self.mode == "oversea":
            payload = std_req.oversea_payload()
        else:
            if not conversation_id:
                raise Exception("CN mode requires conversation_id")
            payload = std_req.cn_payload(conversation_id)

        client = httpx.AsyncClient(timeout=180, follow_redirects=True)
        req = client.build_request("POST", self.endpoint, headers=self._headers(), json=payload)
        resp = await client.send(req, stream=True)
        return resp, client


# ============================================================
# 12. 流式引擎
# ============================================================

@dataclass
class ConsumeConfig:
    thinking_enabled: bool = False
    tool_names: list[str] = field(default_factory=list)
    keepalive_interval: float = KEEPALIVE_INTERVAL
    idle_timeout: float = 120.0


@dataclass
class ParsedDecision:
    content_seen: bool = False
    stop: bool = False


class ConsumeHooks:
    def __init__(self, on_parsed=None, on_keepalive=None, on_finalize=None, on_context_done=None):
        self.on_parsed = on_parsed or (lambda _: ParsedDecision())
        self.on_keepalive = on_keepalive or (lambda: None)
        self.on_finalize = on_finalize or (lambda *_: None)
        self.on_context_done = on_context_done or (lambda: None)


async def consume_sse_oversea(response: httpx.Response, config: ConsumeConfig, hooks: ConsumeHooks) -> None:
    current_type = ContentType.THINKING if config.thinking_enabled else ContentType.TEXT
    stop_reason = "stop"
    scanner_err = None

    try:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            result = parse_oversea_content_line(line, config.thinking_enabled, current_type)
            current_type = result.next_type
            if not result.parsed and not result.stop:
                continue
            decision = hooks.on_parsed(result)
            if result.stop or decision.stop:
                if result.error_message:
                    stop_reason = "error"
                elif result.content_filter:
                    stop_reason = "content_filter"
                break
    except Exception as e:
        scanner_err = e
        stop_reason = "error"
    finally:
        hooks.on_finalize(stop_reason, scanner_err)


# ============================================================
# 13. FastAPI应用
# ============================================================

app = FastAPI(title="LongCat2API", version="3.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- API Key 认证中间件 (OpenAI 格式) ---
@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    # 未设置 API_KEY 则跳过认证
    if not API_KEY:
        return await call_next(request)

    # 健康检查不需要认证
    if request.url.path == "/health":
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    else:
        token = ""

    if token != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
        )

    return await call_next(request)


@app.get("/v1/models")
async def list_models():
    seen = set()
    models = []
    for mid, (aid, think, search) in MODEL_ALIASES.items():
        if mid not in seen:
            seen.add(mid)
            models.append({"id": mid, "object": "model", "owned_by": "longcat", "thinking": think, "search": search})
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(400, "Invalid JSON")

    try:
        std_req = normalize_openai_request(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    # 选择模式
    mode = std_req.mode
    cookie = ""

    if mode == "cn":
        cookie = await cookie_pool.acquire()
        if not cookie:
            # CN模式没Cookie，自动降级到oversea
            mode = "oversea"
            logger.info("No cookie for CN mode, falling back to oversea")

    client = LongCatClient(cookie=cookie, mode=mode)

    if std_req.stream:
        return StreamingResponse(
            _stream_chat(client, std_req, completion_id, created, mode),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    else:
        return await _non_stream_chat(client, std_req, completion_id, created, mode, cookie)


async def _non_stream_chat(client, std_req, completion_id, created, mode, cookie="") -> dict:
    conversation_id = ""

    if mode == "cn":
        try:
            session = await client.create_session(std_req.agent_id)
            conversation_id = session.get("conversationId", "")
            if not conversation_id:
                raise Exception("No conversationId")
        except Exception as e:
            if cookie:
                await cookie_pool.report_error(cookie)
            raise HTTPException(500, f"Session failed: {e}")

    http_client = None
    for attempt in range(MAX_EMPTY_RETRIES + 1):
        # 429限流自动重试（指数退避）
        for rate_attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            resp, http_client = await client.chat_completion_raw(std_req, conversation_id)
            try:
                is_rate_limited = False

                if resp.status_code == 429:
                    is_rate_limited = True
                elif resp.status_code == 401:
                    if cookie:
                        await cookie_pool.report_error(cookie)
                    raise HTTPException(401, "Cookie expired")
                elif resp.status_code != 200:
                    body_text = await resp.aread()
                    raise Exception(f"Upstream {resp.status_code}: {body_text.decode()[:200]}")

                # oversea端点可能HTTP 200但body中code!=0（如429限流）
                body_bytes = await resp.aread()
                body_text = body_bytes.decode("utf-8", errors="replace")

                # 检查是否是错误JSON（非SSE格式）
                if body_text.strip().startswith("{"):
                    try:
                        j = json.loads(body_text.strip())
                        if j.get("code") is not None and j.get("code") != 0:
                            code = j.get("code")
                            msg = j.get("message", "Unknown error")
                            if code == 429:
                                is_rate_limited = True
                            else:
                                raise Exception(f"Upstream error {code}: {msg}")
                    except json.JSONDecodeError:
                        pass

                if is_rate_limited:
                    await http_client.aclose()
                    http_client = None
                    if rate_attempt < RATE_LIMIT_MAX_RETRIES:
                        delay = RATE_LIMIT_BASE_DELAY * (2 ** rate_attempt)
                        logger.warning(f"429 rate limited, retrying in {delay}s (attempt {rate_attempt+1}/{RATE_LIMIT_MAX_RETRIES})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise HTTPException(429, "Rate limited by upstream after retries. Please retry later.")

                # 非429，跳出重试循环，进入SSE解析
                break

            except HTTPException:
                if http_client:
                    await http_client.aclose()
                raise
            except Exception as e:
                if http_client:
                    await http_client.aclose()
                if cookie:
                    await cookie_pool.report_error(cookie)
                raise HTTPException(500, f"Chat error: {e}")

        # 解析SSE行 — 先收集所有content/think/reason事件
        all_content_events = []  # 记录所有content事件及其文本
        text_parts, thinking_parts = [], []
        current_type = ContentType.THINKING if std_req.thinking_enabled else ContentType.TEXT
        usage, final_content = {}, ""
        content_filter, error_message = False, ""

        for line in body_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            result = parse_oversea_content_line(line, std_req.thinking_enabled, current_type)
            current_type = result.next_type
            if result.stop:
                if result.content_filter:
                    content_filter = True
                if result.error_message:
                    error_message = result.error_message
                usage = result.usage
                final_content = result.final_content
                break
            for part in result.parts:
                if part.type == ContentType.THINKING:
                    thinking_parts.append(part.text)
                else:
                    # content事件暂时收集，后面后处理
                    all_content_events.append(part.text)

        # 推理模式后处理
        # 纯推理模式(thinking=True, search=False)：只有最后一个content是真正回复，其余(开场白)归入thinking
        # 搜索模式(search=True)：content全部是回复，不做后处理
        if std_req.thinking_enabled and not std_req.search_enabled and len(all_content_events) > 1:
            # 最后一个content → TEXT，其余 → THINKING
            for c in all_content_events[:-1]:
                thinking_parts.append(c)
            text_parts.append(all_content_events[-1])
        elif std_req.thinking_enabled and not std_req.search_enabled and len(all_content_events) == 1:
            # 只有一个content，可能是真正的回复（oversea），也可能是开场白
            # 使用final_content来判断——如果finish事件有finalContentX，用那个
            if final_content:
                text_parts.append(final_content)
                # 原先收集的content归入thinking
                for c in all_content_events:
                    thinking_parts.append(c)
            else:
                # 没有final_content，就用最后一个content
                text_parts.append(all_content_events[-1])
        elif std_req.search_enabled:
            # 搜索模式：content全部是回复
            text_parts.extend(all_content_events)
        else:
            # 非推理模式，全部是TEXT
            text_parts.extend(all_content_events)

        raw_text = "".join(text_parts)
        raw_thinking = "".join(thinking_parts)
        tool_calls = detect_tool_calls(raw_text) if raw_text else []
        if tool_calls:
            raw_text = strip_tool_calls_markup(raw_text)

        result = CollectResult(text=raw_text, thinking=raw_thinking, tool_calls=tool_calls,
                               content_filter=content_filter, error_message=error_message,
                               usage=usage, final_content=final_content)
        turn = build_turn(result, std_req.thinking_enabled)

        # 429错误
        if "429" in turn.error_message:
            if http_client:
                await http_client.aclose()
            raise HTTPException(429, turn.error_message)
        if turn.error_message:
            if http_client:
                await http_client.aclose()
            raise HTTPException(502, turn.error_message)

        # 空输出重试
        if not turn.text and not turn.thinking and not turn.tool_calls and not turn.content_filter:
            if http_client:
                await http_client.aclose()
                http_client = None
            if attempt < MAX_EMPTY_RETRIES:
                logger.warning(f"Empty output, retrying ({attempt+1}/{MAX_EMPTY_RETRIES})")
                continue

        if cookie:
            await cookie_pool.report_success(cookie)
        if http_client:
            await http_client.aclose()
        return build_chat_completion(completion_id, std_req.model, turn, created)

    if http_client:
        await http_client.aclose()
    return build_chat_completion(completion_id, std_req.model, Turn(), created)


async def _stream_chat(client, std_req, completion_id, created, mode) -> AsyncIterator[str]:
    conversation_id = ""

    if mode == "cn":
        try:
            session = await client.create_session(std_req.agent_id)
            conversation_id = session.get("conversationId", "")
            if not conversation_id:
                yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": "Error: No conversationId"}, "stop", created))
                yield "data: [DONE]\n\n"
                return
        except Exception as e:
            yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": f"Error: {e}"}, "stop", created))
            yield "data: [DONE]\n\n"
            return

    # 429限流自动重试 + 真正逐行流式消费
    resp = None
    http_client = None
    body_text_for_retry = None  # 如果429检查失败，缓存body用于重试后解析

    for rate_attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        resp, http_client = await client.chat_completion_raw(std_req, conversation_id)

        if resp.status_code == 429:
            await http_client.aclose()
            http_client = None
            if rate_attempt < RATE_LIMIT_MAX_RETRIES:
                delay = RATE_LIMIT_BASE_DELAY * (2 ** rate_attempt)
                logger.warning(f"429 stream, retrying in {delay}s (attempt {rate_attempt+1}/{RATE_LIMIT_MAX_RETRIES})")
                await asyncio.sleep(delay)
                continue
            else:
                yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": "Rate limited after retries. Please retry later."}, "stop", created))
                yield "data: [DONE]\n\n"
                return

        if resp.status_code == 401:
            yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": "Error: 401 Unauthorized"}, "stop", created))
            yield "data: [DONE]\n\n"
            await http_client.aclose()
            return

        if resp.status_code != 200:
            body_text = await resp.aread()
            await http_client.aclose()
            yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": f"Error: upstream {resp.status_code}"}, "stop", created))
            yield "data: [DONE]\n\n"
            return

        # HTTP 200 — 可能是真正的SSE流，也可能是oversea端点的429 JSON
        # 全量读取body（429检查+解析都需要）
        body_bytes = await resp.aread()
        body_text = body_bytes.decode("utf-8", errors="replace")

        # 检查是否是429错误JSON（非SSE格式）
        if body_text.strip().startswith("{"):
            try:
                j = json.loads(body_text.strip())
                if j.get("code") is not None and j.get("code") != 0:
                    if j.get("code") == 429:
                        # 限流，重试
                        await http_client.aclose()
                        http_client = None
                        if rate_attempt < RATE_LIMIT_MAX_RETRIES:
                            delay = RATE_LIMIT_BASE_DELAY * (2 ** rate_attempt)
                            logger.warning(f"429 stream (body), retrying in {delay}s (attempt {rate_attempt+1}/{RATE_LIMIT_MAX_RETRIES})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": "Rate limited after retries."}, "stop", created))
                            yield "data: [DONE]\n\n"
                            return
                    else:
                        await http_client.aclose()
                        yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": f"Error: {j.get('message')}"}, "stop", created))
                        yield "data: [DONE]\n\n"
                        return
            except json.JSONDecodeError:
                pass  # 不是错误JSON，是SSE格式

        # 成功获取流式数据，跳出重试循环
        break
    else:
        # 重试耗尽
        return

    # ========== 流式消费（从body_text逐行解析并模拟流式输出） ==========
    first_chunk_sent = False
    finish_reason = "stop"
    usage_info = {}

    # 解析所有SSE行
    all_content_texts = []
    thinking_texts = []
    usage_data = {}
    final_text = ""
    error_msg = ""
    current_type = ContentType.THINKING if std_req.thinking_enabled else ContentType.TEXT

    for line_text in body_text.split("\n"):
        line_text = line_text.strip()
        if not line_text:
            continue
        result = parse_oversea_content_line(line_text, std_req.thinking_enabled, current_type)
        current_type = result.next_type
        if result.stop:
            if result.error_message:
                error_msg = result.error_message
            usage_data = result.usage
            final_text = result.final_content
            break
        for part in result.parts:
            if part.type == ContentType.THINKING:
                thinking_texts.append(part.text)
            else:
                all_content_texts.append(part.text)

    # 推理模式后处理
    # 纯推理模式(thinking=True, search=False)：只有最后一个content是最终回复，其余(开场白)归入thinking
    # 搜索模式(search=True)：content全部是回复，不做后处理
    if std_req.thinking_enabled and not std_req.search_enabled and len(all_content_texts) > 1:
        for c in all_content_texts[:-1]:
            thinking_texts.append(c)
        final_content_text = all_content_texts[-1]
    elif std_req.thinking_enabled and not std_req.search_enabled and len(all_content_texts) == 1:
        if final_text:
            for c in all_content_texts:
                thinking_texts.append(c)
            final_content_text = final_text
        else:
            final_content_text = all_content_texts[-1]
    elif std_req.search_enabled:
        # 搜索模式：content全部是回复，拼接输出
        final_content_text = "".join(all_content_texts)
    elif not std_req.thinking_enabled:
        final_content_text = "".join(all_content_texts)
    else:
        final_content_text = final_text

    # 逐块输出thinking（推理模式）
    for t in thinking_texts:
        delta = {}
        if not first_chunk_sent:
            first_chunk_sent = True
            delta["role"] = "assistant"
        delta["reasoning_content"] = t
        yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, delta, None, created))
        # 模拟流式延迟
        await asyncio.sleep(0.01)

    # 输出最终content
    if final_content_text:
        delta = {}
        if not first_chunk_sent:
            first_chunk_sent = True
            delta["role"] = "assistant"
        delta["content"] = final_content_text
        yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, delta, None, created))

    # finish chunk
    if error_msg and "429" in error_msg:
        yield _sse_chunk(build_chat_stream_chunk(completion_id, std_req.model, {"content": "Rate limited"}, "stop", created))
    finish_chunk = build_chat_stream_chunk(completion_id, std_req.model, {}, finish_reason, created)
    pt = usage_data.get("prompt_tokens", 0)
    ct = usage_data.get("completion_tokens", max(len(final_content_text) // 4, 1))
    finish_chunk["usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
    yield _sse_chunk(finish_chunk)
    yield "data: [DONE]\n\n"

    await http_client.aclose()
    return


def _sse_chunk(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ============================================================
# 14. 管理端点
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "service": "longcat2api", "version": "3.0.0", "mode": DEFAULT_MODE,
            "cookies": len(cookie_pool._cookies), "active_cookies": sum(1 for c in cookie_pool._cookies.values() if c["active"])}


@app.post("/cookie")
async def set_cookie(request: Request):
    body = await request.json()
    cookie = body.get("cookie", "")
    label = body.get("label", "")
    if not cookie:
        raise HTTPException(400, "cookie field required")
    cookie_pool.add(cookie, label)
    return {"status": "ok", "message": f"Cookie added (label: {label or 'auto'}", "pool_size": len(cookie_pool._cookies)}


@app.delete("/cookie")
async def remove_cookie(request: Request):
    body = await request.json()
    cookie = body.get("cookie", "")
    if not cookie:
        raise HTTPException(400, "cookie field required")
    cookie_pool.remove(cookie)
    return {"status": "ok", "message": "Cookie removed"}


@app.get("/cookies")
async def list_cookies():
    return {"cookies": cookie_pool.list_cookies()}


@app.get("/test")
async def test_api():
    """测试oversea端点"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                OVERSEA_ENDPOINT,
                headers=OVERSEA_HEADERS,
                json={"content": "ping", "agentId": "1",
                      "messages": [{"role": "user", "events": [{"type": "userMsg", "content": "ping", "status": "FINISHED"}], "chatStatus": "FINISHED", "messageId": 1, "idType": "custom"},
                                   {"role": "assistant", "events": [], "chatStatus": "LOADING", "messageId": 2, "idType": "custom"}],
                      "reasonEnabled": 0, "searchEnabled": 0, "regenerate": 0},
            )
            if resp.status_code == 429:
                return {"status": "rate_limited", "message": "Upstream rate limit, API is reachable"}
            if resp.status_code == 200:
                return {"status": "ok", "message": "Oversea endpoint working"}
            return {"status": "error", "message": f"Status {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
