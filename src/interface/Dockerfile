# O endereco da API entra no pacote no build, porque o Vite substitui `import.meta.env` ali.
# Ele e declarado uma vez antes do primeiro `FROM` para valer nos dois estagios: o de build assa
# o valor no JavaScript, e o de execucao precisa dele para a `connect-src` da CSP.
ARG VITE_API_URL=http://localhost:8080

FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM nginx:1.27-alpine AS runtime

ARG VITE_API_URL

# `add_header` de um nivel so vale enquanto o nivel de baixo nao declara nenhum, e sem `always`
# ele fica de fora de resposta que nao seja 2xx ou 3xx. Por isso `default_type` no /health, e nao
# outro `add_header`. A CSP e a mitigacao que faltava ao token guardado em `localStorage`, e a
# `connect-src` precisa da origem da API, que so se conhece no build.
RUN printf '%s\n' \
    'server {' \
    '  listen 3000;' \
    '  root /usr/share/nginx/html;' \
    '  add_header X-Content-Type-Options nosniff always;' \
    '  add_header X-Frame-Options DENY always;' \
    '  add_header Referrer-Policy no-referrer always;' \
    "  add_header Content-Security-Policy \"default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; img-src 'self' data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; connect-src 'self' ${VITE_API_URL}\" always;" \
    '  location /health { default_type text/plain; return 200 "ok"; }' \
    '  location / { try_files $uri $uri/ /index.html; }' \
    '}' > /etc/nginx/conf.d/default.conf \
 && rm -f /etc/nginx/conf.d/default.conf.default \
 && nginx -t

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 3000
