# --- Build Stage ---
FROM rust:nightly AS builder

# Installer les dépendances de compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers du projet
COPY src/eva-kernel/ .

# Compiler le projet
RUN cargo build --release

# --- Runtime Stage ---
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le binaire depuis l'étape de construction
COPY --from=builder /app/target/release/eva-kernel /usr/local/bin/eva-kernel

RUN chmod +x /usr/local/bin/eva-kernel

# Port exposé (Axum)
EXPOSE 8080

CMD ["/usr/local/bin/eva-kernel"]

