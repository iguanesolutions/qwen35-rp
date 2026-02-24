# qwen35-rp

Qwen 3.5 Reverse Proxy is a lightweight HTTP reverse proxy that automatically adjusts sampling parameters (temperature, top_p, etc.) and thinking mode based on whether a thinking or instant mode is being used. It sits between your application and the backend LLM server serving Qwen 3.5 (e.g., vLLM). The proxy supports systemd integration with `type=notify` and structured logging compatible with journald.

## Core Functionality

This proxy's primary purpose is to:

1. **Accept requests for two virtual model names** (configured via `-thinking-model` and `-no-thinking-model`), rejecting all other model names with HTTP 400
2. **Set appropriate sampling parameters** automatically based on the virtual model (official Qwen-recommended values from [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8#using-qwen35-via-the-chat-completions-api)):
   - **Thinking mode**: `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
   - **Instant mode**: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
3. **Configure thinking mode** by setting `chat_template_kwargs.enable_thinking`:
   - `enable_thinking=true` for thinking mode
   - `enable_thinking=false` for instant mode
4. **Rewrite the model name** to the actual backend model name (e.g., `Qwen/Qwen3.5-397B-A17B-FP8`) before forwarding to vLLM
5. **Fix vLLM response bugs** where non-thinking, non-streaming responses incorrectly place content in `reasoning_content` or `reasoning` fields instead of `content`

## Installation

Requirements: Go 1.24.2 or later

```bash
go build -o qwen35-rp .
```

## Usage

```bash
./qwen35-rp \
  -target "http://127.0.0.1:8000" \
  -served-model "Qwen/Qwen3.5-397B-A17B-FP8" \
  -thinking-model "qwen-thinking" \
  -no-thinking-model "qwen-no-thinking"
```

Or using environment variables:

```bash
export QWEN35RP_TARGET="http://127.0.0.1:8000"
export QWEN35RP_SERVED_MODEL_NAME="Qwen/Qwen3.5-397B-A17B-FP8"
export QWEN35RP_THINKING_MODEL_NAME="qwen-thinking"
export QWEN35RP_NO_THINKING_MODEL_NAME="qwen-no-thinking"
./qwen35-rp
```

## Configuration

Configure the proxy using command-line flags or environment variables:

| Flag | Environment Variable | Default | Description |
|------|---------------------|---------|-------------|
| `-listen` | `QWEN35RP_LISTEN` | `0.0.0.0` | IP address to listen on |
| `-port` | `QWEN35RP_PORT` | `9000` | Port to listen on |
| `-target` | `QWEN35RP_TARGET` | `http://127.0.0.1:8000` | Backend target URL |
| `-loglevel` | `QWEN35RP_LOGLEVEL` | `INFO` | Log level (COMPLETE, DEBUG, INFO, WARN, ERROR) |
| `-served-model` | `QWEN35RP_SERVED_MODEL_NAME` | (required) | Backend model name to use in outgoing requests |
| `-thinking-model` | `QWEN35RP_THINKING_MODEL_NAME` | (required) | Name of the thinking model (incoming request identifier) |
| `-no-thinking-model` | `QWEN35RP_NO_THINKING_MODEL_NAME` | (required) | Name of the non-thinking model (incoming request identifier) |
| `-enforce-sampling-params` | `QWEN35RP_ENFORCE_SAMPLING_PARAMS` | `false` | Enforce sampling parameters, overriding client-provided values |

### Enforce Sampling Parameters

By default, the proxy only sets sampling parameters if they are not already present in the request. When `-enforce-sampling-params` is enabled, the proxy will **always override** client-provided sampling parameters with the predefined values for the detected mode.

## Request Routing

- **`POST /v1/chat/completions`**: Transformed (sampling params + thinking mode applied)
- **`POST /v1/completions`**: Transformed (sampling params + thinking mode applied)
- **All other paths**: Passed through unchanged to the backend

## Health Check

- **`GET /health`**: Returns `{"status":"healthy"}` for Docker health checks

## Log Levels

The proxy supports the following log levels:

| Level | Description |
|-------|-------------|
| `COMPLETE` | Most verbose - includes full HTTP request/response dumps |
| `DEBUG` | Debug information including parameter application details |
| `INFO` | General operational information |
| `WARN` | Warning messages |
| `ERROR` | Error messages only |

When set to `COMPLETE`, the proxy will log full HTTP request and response bodies, which is useful for debugging but very verbose.

⚠️ **Privacy Warning**: LLM requests often contain sensitive or personal data (conversation history, personal information, confidential content). The `COMPLETE` log level will expose all this data in plaintext. Only enable it in secure, non-production environments or ensure logs are properly secured and retained temporarily.

## systemd Integration

The proxy includes native systemd support for production deployments:

- **Type**: `notify` - The proxy signals readiness to systemd automatically
- **Status Updates**: Sends periodic status updates to systemd showing processed request counts
- **Graceful Shutdown**: Properly signals systemd when stopping
- **Journald Logging**: Structured logging output is compatible with journald

Example systemd unit file:

```ini
[Unit]
Description=Qwen 3.5 Reverse Proxy
After=network.target

[Service]
Type=notify
User=qwen35-rp
Group=qwen35-rp
ExecStart=/usr/bin/qwen35-rp -served-model "Qwen/Qwen3.5-397B-A17B-FP8" -thinking-model "qwen-thinking" -no-thinking-model "qwen-instant"
Restart=on-failure
Environment=QWEN35RP_LOGLEVEL=INFO

[Install]
WantedBy=multi-user.target
```

⚠️ **Security Best Practice**: Always run the proxy under a dedicated, unprivileged user account (e.g., `qwen35-rp`). Never run as root. Create the user with:
```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin qwen35-rp
sudo chown qwen35-rp:qwen35-rp /usr/bin/qwen35-rp
```

## Graceful Shutdown

The server supports graceful shutdown with a 3-minute timeout to allow in-flight requests to complete. Send `SIGINT` or `SIGTERM` to initiate shutdown. When running under systemd, the proxy will automatically signal the service manager when ready and during shutdown.

## License

MIT License - see [LICENSE](LICENSE) file for details.