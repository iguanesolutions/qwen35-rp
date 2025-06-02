ARG go_build_image=golang
ARG go_build_tag=1.24-bookworm

ARG app_image=debian
ARG app_tag=bookworm-slim

FROM ${go_build_image}:${go_build_tag} AS go_build
RUN mkdir -p /build
WORKDIR /build
ADD . /build
RUN go build -v -trimpath -ldflags "-s -w" -o qwen3-rp .

FROM ${app_image}:${app_tag}
RUN apt update && apt install -y ca-certificates curl ffmpeg
COPY --from=go_build /build/qwen3-rp /usr/bin/qwen3-rp

EXPOSE 9000

ENTRYPOINT ["/usr/bin/qwen3-rp"]
