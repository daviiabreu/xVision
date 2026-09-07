ARG VITE_API_URL=""

FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM nginx:1.27-alpine AS runtime

RUN printf '%s\n' \
    'server {' \
    '  listen 3000;' \
    '  root /usr/share/nginx/html;' \
    '  client_max_body_size 128m;' \
    '  add_header X-Content-Type-Options nosniff always;' \
    '  add_header X-Frame-Options DENY always;' \
    '  add_header Referrer-Policy no-referrer always;' \
    "  add_header Content-Security-Policy \"default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; img-src 'self' data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; connect-src 'self'\" always;" \
    '  resolver 127.0.0.11 valid=30s ipv6=off;' \
    '  location /health { default_type text/plain; return 200 "ok"; }' \
    '  location /v1/ {' \
    '    set $api_upstream http://api:8080;' \
    '    proxy_pass $api_upstream;' \
    '    proxy_http_version 1.1;' \
    '    proxy_set_header X-Forwarded-For $remote_addr;' \
    '    proxy_request_buffering off;' \
    '    proxy_read_timeout 300s;' \
    '  }' \
    '  location /docs {' \
    '    set $api_upstream http://api:8080;' \
    '    proxy_pass $api_upstream;' \
    '    proxy_hide_header X-Frame-Options;' \
    '    add_header X-Frame-Options SAMEORIGIN always;' \
    "    add_header Content-Security-Policy \"default-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'self'\" always;" \
    '  }' \
    '  location = /openapi.json {' \
    '    set $api_upstream http://api:8080;' \
    '    proxy_pass $api_upstream;' \
    '  }' \
    '  location = /bruno.zip {' \
    '    set $api_upstream http://api:8080;' \
    '    proxy_pass $api_upstream;' \
    '  }' \
    '  location / { try_files $uri $uri/ /index.html; }' \
    '}' > /etc/nginx/conf.d/default.conf \
 && rm -f /etc/nginx/conf.d/default.conf.default \
 && nginx -t

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 3000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3000/health >/dev/null || exit 1
