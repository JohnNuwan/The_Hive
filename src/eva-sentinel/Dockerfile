FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src/eva-sentinel/target/release/eva-sentinel /usr/local/bin/eva-sentinel
RUN chmod +x /usr/local/bin/eva-sentinel
CMD ["/usr/local/bin/eva-sentinel"]
