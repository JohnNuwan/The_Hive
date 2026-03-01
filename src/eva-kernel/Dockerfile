# --- Build Stage ---
FROM rust:1.75-slim-bookworm AS builder

# Installer les dépendances de compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers de définition (depuis le context .)
COPY src/eva-kernel/Cargo.toml src/eva-kernel/Cargo.lock ./

# Créer un projet dummy pour mettre en cache les dépendances
# On crée le dossier src attendu par Cargo
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

# Copier le code source réel
COPY src/eva-kernel/src ./src

# Compiler le vrai projet
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

