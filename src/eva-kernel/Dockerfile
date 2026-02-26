FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src/eva-kernel/target/release/eva-kernel /usr/local/bin/eva-kernel
RUN chmod +x /usr/local/bin/eva-kernel
CMD ["/usr/local/bin/eva-kernel"]
