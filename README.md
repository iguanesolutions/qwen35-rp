# qwen35-rp

Qwen35 Reverse Proxy is a lightweight HTTP reverse proxy that automatically adjusts sampling parameters (temperature, top_p) based on whether a thinking or non-thinking model is being used. It sits between your application and the backend LLM server (e.g., vLLM).

## Installation

Requirements: Go 1.24.2 or later

```bash
go build -o qwen35-rp .
```

## Configuration

Configure the proxy using command-line flags or environment variables:

| Flag | Environment Variable | Default | Description |
|------|---------------------|---------|-------------|
| `-listen` | `QWEN35RP_LISTEN` | `0.0.0.0` | IP address to listen on |
| `-port` | `QWEN35RP_PORT` | `9000` | Port to listen on |
| `-target` | `QWEN35RP_TARGET` | `http://127.0.0.1:8000` | Backend target URL |
| `-loglevel` | `QWEN35RP_LOGLEVEL` | `INFO` | Log level (DEBUG, INFO, WARN, ERROR) |
| `-thinking-model` | `QWEN35RP_THINKING_MODEL_NAME` | (required) | Name of the thinking model |
| `-no-thinking-model` | `QWEN35RP_NO_THINKING_MODEL_NAME` | (required) | Name of the non-thinking model |

## How It Works

1. Client sends a request with a model name in the request body
2. Proxy inspects the `model` field to determine if it's a thinking or non-thinking model
3. Proxy sets appropriate sampling parameters:
   - If thinking model: `temperature=1.0`, `top_p=0.95`, `extra_body.thinking=true`
   - If non-thinking model: `temperature=0.6`, `top_p=0.95`, `extra_body.thinking=false`
4. Request is forwarded to the backend server
5. Response is streamed back to the client

## License

MIT License - see [LICENSE](LICENSE) file for details.