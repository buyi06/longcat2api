# LongCat2API v3

LongCat AI → OpenAI API 兼容层。将 [longcat.chat](https://longcat.chat) 的聊天接口转换为 OpenAI 格式的 API。

> v3: 发现**免登录端点**，无需Cookie即可使用！

## ✨ 特性

- **无需Cookie**: 使用 `oversea-V2` 端点，免登录直接调用
- **完全 OpenAI 兼容**: `/v1/chat/completions` + `/v1/models`
- **流式 + 非流式**: 完整支持 SSE 流式和非流式两种模式
- **推理/思考**: `reasoning_content` 标准字段输出
- **工具调用**: `<tool_calls>` XML 检测 → OpenAI `tool_calls` 格式
- **微批缓冲**: 16字符/10ms 累积后刷新
- **Cookie池**: 可选，用于国内CN模式（需登录）
- **空输出重试**: 自动重试空响应（最多2次）
- **模型别名**: 10个（含DeepSeek兼容别名）
- **Keepalive**: 15秒心跳

## 🔑 逆向发现

| 项目 | 详情 |
|------|------|
| **免登录端点** | `POST /api/v1/chat-completion-oversea-V2` |
| **国内登录端点** | `POST /api/v1/chat-completion-V2`（需Cookie） |
| **认证** | oversea端点无需任何认证；CN端点需美团SSO Cookie |
| **无需session-create** | oversea端点直接发消息即可 |
| **SSE格式** | `event.type = create/content/reason/finish` |
| **请求体** | `{content, agentId, messages, reasonEnabled, searchEnabled, regenerate}` |

## 🚀 启动

```bash
# 默认免登录模式（oversea）
cd /root/longcat && source venv/bin/activate
uvicorn longcat2api:app --host 0.0.0.0 --port 8000

# 可选：国内登录模式（需Cookie）
export LONGCAT_MODE=cn
export LONGCAT_COOKIE="你的美团SSO Cookie"
uvicorn longcat2api:app --host 0.0.0.0 --port 8000
```

## 📡 API 端点

### Chat Completions

```bash
# 非流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat-default","messages":[{"role":"user","content":"你好"}],"stream":false}'

# 流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat-default","messages":[{"role":"user","content":"你好"}],"stream":true}'

# 推理模式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat-reason","messages":[{"role":"user","content":"9.11和9.9哪个大？"}],"stream":true}'

# 搜索模式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat-search","messages":[{"role":"user","content":"今天新闻"}],"stream":true}'
```

### 输出格式

**推理输出（非流式）:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "正式回复",
      "reasoning_content": "思考过程..."
    }
  }]
}
```

**推理输出（流式）:**
```
data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"思考片段"}}]}
data: {"choices":[{"delta":{"content":"回复片段"}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}
data: [DONE]
```

### 模型列表

| 模型ID | 推理 | 搜索 |
|--------|------|------|
| `longcat-default` | ❌ | ❌ |
| `longcat-reason` | ✅ | ❌ |
| `longcat-search` | ❌ | ✅ |
| `longcat-reason-search` | ✅ | ✅ |
| `deepseek-chat` | ❌ | ❌ |
| `deepseek-reasoner` / `deepseek-r1` | ✅ | ❌ |

### 管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/cookie` | POST | 添加Cookie（CN模式用） |
| `/cookie` | DELETE | 移除Cookie |
| `/cookies` | GET | Cookie池状态 |
| `/test` | GET | 测试oversea端点连通性 |

## 🆚 版本变化

| 特性 | v1 | v2 | v3 |
|------|----|----|-----|
| 认证 | Cookie必须 | Cookie池 | **无需Cookie** |
| 端点 | chat-completion-V2 | chat-completion-V2 | **oversea-V2** |
| SSE格式 | TEXT/REASON事件 | 同v1 | **event.type格式** |
| session-create | 必须 | 必须 | **不需要** |
| 推理输出 | `<arg_key>` 标签 | reasoning_content | reasoning_content |
| 工具调用 | ❌ | ✅ | ✅ |
| 429限流处理 | ❌ | ❌ | ✅ 正确报错 |

## ⚠️ 限流说明

oversea端点有IP级限流（频繁请求会触发429），建议：
- 不要过于频繁地调用
- 多IP轮转可提高吞吐
- CN模式（需Cookie）可能限流更宽松
