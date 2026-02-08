# ═══════════════════════════════════════════════════════════════════════════════
# THE HIVE — Nexus Frontend (Multi-stage Build)
# ═══════════════════════════════════════════════════════════════════════════════

FROM node:20-slim AS builder

WORKDIR /app

# Copier les fichiers de dépendances
COPY package.json package-lock.json* ./

# Installer TOUTES les dépendances (y compris devDependencies pour tsc/vite)
ENV NODE_ENV=development
RUN npm install

# Copier le code source (respecte le .dockerignore local)
COPY . .

# Build : utiliser le chemin complet vers tsc dans node_modules
RUN ./node_modules/.bin/tsc && ./node_modules/.bin/vite build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Serve statique avec Nginx
# ─────────────────────────────────────────────────────────────────────────────

FROM nginx:stable-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
