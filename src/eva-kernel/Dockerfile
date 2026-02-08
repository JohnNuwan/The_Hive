# Build Stage
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/eva-kernel /app

# Build the kernel
RUN cargo build --release

# Run Stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/eva-kernel .

# Ensure mnt/tablet exists for the constitution
RUN mkdir -p /mnt/tablet

CMD ["./eva-kernel"]
