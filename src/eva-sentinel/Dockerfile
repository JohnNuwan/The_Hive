# Multi-stage build for Sentinel Rust Core
FROM rust:latest as builder

WORKDIR /app
COPY src/eva-sentinel /app/eva-sentinel
RUN rm -f /app/eva-sentinel/rust-toolchain.toml
WORKDIR /app/eva-sentinel

# Build Sentinel binary
RUN cargo build --release

# Final runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/eva-sentinel/target/release/eva-sentinel /usr/local/bin/eva-sentinel

# Set execute permissions
RUN chmod +x /usr/local/bin/eva-sentinel

# Exposed port for potential future API
EXPOSE 8200

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8200/health || exit 0 
    # Note: Health endpoint logic to be implemented later in Rust

CMD ["/usr/local/bin/eva-sentinel"]
