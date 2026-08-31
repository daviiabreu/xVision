#!/usr/bin/env python3
"""
Baixa os clipes de "cortes de finalização" do Playbook xG (Footstats) e salva os
metadados de cada chute (minuto, time, pé/perna, zona, resultado etc.) em CSV/JSONL.

Reproduz em Python o fluxo que a própria página já oferece (botão "cortes-copa2026.txt"
+ `curl -K`), só que baixando os vídeos direto e guardando os metadados junto — a página
só gera a lista de URLs, sem metadados, porque o botão existe pra `curl`, não pra análise.

AUTENTICAÇÃO
------------
O login do Playbook xG é só federado (Google/Entra, OAuth2 + PKCE) — não existe login por
senha, então este script não tenta automatizar o OAuth. Em vez disso, ele reaproveita a
sessão que você já tem no navegador:

  1. Faça login normalmente em https://playbook-xg.footstats.com.br/viewer/cortes-sessao.html
  2. Abra o DevTools (F12) -> Console e rode:
         copy(localStorage.getItem('copa-tokens'))
     (isso copia o JSON de tokens pra área de transferência)
  3. Cole o conteúdo em um arquivo, ex: copa-tokens.json
  4. Rode este script apontando pra esse arquivo:
         python baixar_cortes_footstats.py --token-file copa-tokens.json --out ./cortes

O script cuida do refresh automático (usa o refresh_token salvo nesse mesmo arquivo,
igual ao auth.js faz no navegador) enquanto o refresh_token continuar válido, e reescreve
o arquivo de tokens a cada refresh.

USO
---
    python baixar_cortes_footstats.py --token-file copa-tokens.json --out ./cortes
    python baixar_cortes_footstats.py --token-file copa-tokens.json --out ./cortes --match-id 237053
    python baixar_cortes_footstats.py --token-file copa-tokens.json --out ./cortes --metadata-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ---- Config do ambiente, espelhando viewer/js/config.js ----
API_BASE = "https://v2290v403k.execute-api.us-east-1.amazonaws.com"
COGNITO_REGION = "us-east-1"
CLIENT_ID = "344rpae5kn4pip3gflu5rsvcl7"
COGNITO_IDP = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/"

# Rótulos legíveis pro resultado do chute (RL em cortes-sessao.html)
RESULT_LABELS = {
    "gol": "Gol",
    "on": "No gol",
    "off": "Fora",
    "block": "Bloqueada",
    "wood": "Trave",
    "na": "—",
}


class AuthError(Exception):
    pass


def load_tokens(token_file: Path) -> dict:
    return json.loads(token_file.read_text(encoding="utf-8"))


def save_tokens(token_file: Path, tokens: dict) -> None:
    token_file.write_text(json.dumps(tokens, indent=2), encoding="utf-8")


def refresh_tokens(refresh_token: str) -> dict:
    """Troca o refresh_token por um id_token novo, igual ao refresh() do auth.js."""
    r = requests.post(
        COGNITO_IDP,
        headers={
            "content-type": "application/x-amz-json-1.1",
            "x-amz-target": "AWSCognitoIdentityProviderService.InitiateAuth",
        },
        json={
            "AuthFlow": "REFRESH_TOKEN_AUTH",
            "ClientId": CLIENT_ID,
            "AuthParameters": {"REFRESH_TOKEN": refresh_token},
        },
        timeout=30,
    )
    data = r.json()
    if not r.ok or "AuthenticationResult" not in data:
        raise AuthError(
            "Refresh token expirou ou é inválido — refaça o login no navegador e "
            "gere um novo copa-tokens.json. Detalhe: " + json.dumps(data)[:300]
        )
    res = data["AuthenticationResult"]
    return {
        "id_token": res["IdToken"],
        "access_token": res["AccessToken"],
        "refresh_token": refresh_token,  # Cognito só reenvia um novo às vezes
        "expiresAt": int(time.time() * 1000) + (res.get("ExpiresIn", 3600) - 60) * 1000,
    }


def get_id_token(token_file: Path) -> str:
    tokens = load_tokens(token_file)
    if tokens.get("expiresAt", 0) > time.time() * 1000:
        return tokens["id_token"]
    if not tokens.get("refresh_token"):
        raise AuthError(f"Token expirado e sem refresh_token em {token_file}. Refaça o login.")
    tokens = refresh_tokens(tokens["refresh_token"])
    save_tokens(token_file, tokens)
    return tokens["id_token"]


def fetch_clips(id_token: str) -> list[dict]:
    r = requests.get(
        f"{API_BASE}/clips",
        headers={"authorization": f"Bearer {id_token}"},
        timeout=60,
    )
    if r.status_code == 401:
        raise AuthError("401 da API — token inválido/expirado. Refaça o login no navegador.")
    r.raise_for_status()
    return r.json()


def hhmmss(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFD", str(value))
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return value


@dataclass
class Shot:
    match_id: str
    cut_id: str
    url: str | None
    home: str
    away: str
    competition: str
    venue: str
    date: str
    minute: int
    team: str  # nome do time que chutou (derivado de t: 0=home, 1=away)
    foot: str  # campo "b" da API (pé/perna da finalização)
    play_type: str  # campo "o" da API (tipo/origem da jogada)
    result: str  # gol / on / off / block / wood / na
    result_label: str
    zone: str  # ex.: "área"
    x: float
    y: float
    in_seconds: float
    out_seconds: float
    duration_seconds: float

    def file_name(self) -> str:
        team_slug = slugify(self.team)
        return f"{self.match_id}_{self.minute:03d}_{team_slug}_{self.result}_{self.cut_id}.mp4"


def parse_shots(raw_clips: list[dict]) -> list[Shot]:
    shots: list[Shot] = []
    for c in raw_clips:
        s = c.get("shot") or {}
        team_side = s.get("t", 0)
        team = c.get("home") if team_side == 0 else c.get("away")
        in_s = float(c.get("in") or 0)
        out_s = float(c.get("out") or 0)
        result = s.get("r") or "na"
        shots.append(
            Shot(
                match_id=c["mid"],
                cut_id=c["id"],
                url=c.get("url"),
                home=c.get("home", ""),
                away=c.get("away", ""),
                competition=c.get("comp", ""),
                venue=c.get("venue", ""),
                date=c.get("date", ""),
                minute=int(s.get("m") or 0),
                team=team or "",
                foot=s.get("b") or "—",
                play_type=s.get("o") or "—",
                result=result,
                result_label=RESULT_LABELS.get(result, result),
                zone=s.get("z") or "—",
                x=s.get("x", 50),
                y=s.get("y", 50),
                in_seconds=in_s,
                out_seconds=out_s,
                duration_seconds=round(out_s - in_s, 2),
            )
        )
    return shots


METADATA_FIELDNAMES = [
    "match_id", "cut_id", "date", "competition", "venue", "home", "away",
    "minute", "team", "foot", "play_type", "result", "result_label", "zone",
    "x", "y", "in_seconds", "out_seconds", "duration_seconds", "clip_url", "file_name",
]


def shot_row(s: Shot) -> dict:
    return {
        "match_id": s.match_id, "cut_id": s.cut_id, "date": s.date,
        "competition": s.competition, "venue": s.venue, "home": s.home, "away": s.away,
        "minute": s.minute, "team": s.team, "foot": s.foot, "play_type": s.play_type,
        "result": s.result, "result_label": s.result_label, "zone": s.zone,
        "x": s.x, "y": s.y, "in_seconds": s.in_seconds, "out_seconds": s.out_seconds,
        "duration_seconds": s.duration_seconds, "clip_url": s.url or "",
        "file_name": s.file_name() if s.url else "",
    }


def write_metadata_file(shots: list[Shot], csv_path: Path, jsonl_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv, \
         jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        for s in shots:
            row = shot_row(s)
            writer.writerow(row)
            f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metadata(shots: list[Shot], out_dir: Path) -> tuple[Path, Path]:
    """Escreve o CSV/JSONL geral em out_dir e, além dele, uma cópia por partida
    dentro de out_dir/<match_id>/ — pra pasta do jogo carregar clipe e metadado juntos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cortes_metadados.csv"
    jsonl_path = out_dir / "cortes_metadados.jsonl"
    write_metadata_file(shots, csv_path, jsonl_path)

    by_match: dict[str, list[Shot]] = {}
    for s in shots:
        by_match.setdefault(s.match_id, []).append(s)
    for match_id, match_shots in by_match.items():
        match_dir = out_dir / match_id
        write_metadata_file(
            match_shots,
            match_dir / "cortes_metadados.csv",
            match_dir / "cortes_metadados.jsonl",
        )

    return csv_path, jsonl_path


def download_clip(shot: Shot, out_dir: Path, session: requests.Session) -> str:
    dest = out_dir / shot.match_id / shot.file_name()
    if dest.exists() and dest.stat().st_size > 0:
        return f"skip (já existe): {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with session.get(shot.url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return f"ok: {dest}"


def download_all(shots: list[Shot], out_dir: Path, workers: int) -> None:
    downloadable = [s for s in shots if s.url]
    print(f"Baixando {len(downloadable)} clipes com {workers} workers...")
    with requests.Session() as session, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_clip, s, out_dir, session): s for s in downloadable}
        done = 0
        for fut in as_completed(futures):
            shot = futures[fut]
            done += 1
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001 - log e segue pros outros clipes
                msg = f"ERRO ({shot.cut_id}): {e}"
            print(f"[{done}/{len(downloadable)}] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-file", required=True, type=Path, help="JSON com id_token/refresh_token (ver instruções no topo do arquivo)")
    parser.add_argument("--out", required=True, type=Path, help="Diretório de saída para vídeos e metadados")
    parser.add_argument("--match-id", help="Baixa só os cortes de uma partida (mid)")
    parser.add_argument("--metadata-only", action="store_true", help="Só gera CSV/JSONL, não baixa os .mp4")
    parser.add_argument("--workers", type=int, default=4, help="Downloads simultâneos (padrão: 4)")
    args = parser.parse_args()

    try:
        id_token = get_id_token(args.token_file)
        raw_clips = fetch_clips(id_token)
    except AuthError as e:
        print(f"Erro de autenticação: {e}", file=sys.stderr)
        sys.exit(1)

    shots = parse_shots(raw_clips)
    if args.match_id:
        shots = [s for s in shots if s.match_id == args.match_id]

    if not shots:
        print("Nenhum corte encontrado para os filtros informados.")
        return

    csv_path, jsonl_path = write_metadata(shots, args.out)
    print(f"Metadados salvos em:\n  {csv_path}\n  {jsonl_path}")
    print(f"Total de cortes: {len(shots)} | com vídeo: {sum(1 for s in shots if s.url)}")

    if not args.metadata_only:
        download_all(shots, args.out, args.workers)


if __name__ == "__main__":
    main()
