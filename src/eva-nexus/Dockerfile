# ═══════════════════════════════════════════════════════════════════════════════
# THE HIVE — Nexus Frontend (Multi-stage Build)
# ═══════════════════════════════════════════════════════════════════════════════

FROM nginx:stable-alpine

COPY dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN chmod -R a+rX /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
