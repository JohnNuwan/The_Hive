# --- Build Stage ---
FROM rust:1.80 AS builder

WORKDIR /app
COPY src/eva-kernel/ .
RUN cargo build --release

# --- Runtime Stage ---
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/eva-kernel /usr/local/bin/eva-kernel
RUN chmod +x /usr/local/bin/eva-kernel

EXPOSE 8080
CMD ["/usr/local/bin/eva-kernel"]
