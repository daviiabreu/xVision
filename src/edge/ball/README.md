# Rastreamento da bola (`ball`)

Frente de visão da borda responsável por **detectar e rastrear a bola**. A velocidade depende da
homografia (px → metros do gramado), que é outra frente, e por isso o pacote `velocity/` ainda é um
espaço reservado.

## Estrutura

```text
ball/
├── tracking/
│   ├── detector.py    # YOLO atrás da interface BallDetector + filtro de forma/tamanho
│   ├── tracker.py     # BallTracker: seleção por movimento, gate de salto, interpolação de lacunas
│   └── preprocess.py  # pré-processamento próprio da bola (esqueleto, ainda por implementar)
└── velocity/          # iniciado depois da homografia
```

O detector devolve **todos** os candidatos a bola do quadro (já filtrados por forma e tamanho); a
`BallTracker` escolhe entre eles por **continuidade de movimento** e fecha a trajetória.

## Pré-requisitos

O detector real usa o YOLO (Ultralytics), que é uma dependência **opcional** (extra `ball`):

```bash
cd src
uv sync --extra ball
```

Os pesos `yolov8n.pt` são baixados automaticamente pelo Ultralytics no primeiro uso (ou coloque um
arquivo de pesos próprio e aponte para ele em `DetectorConfig.weights`). Pesos e vídeos **não são
versionados** (ver `.gitignore`): mantenha-os fora do controle de versão.

O peso ajustado no SoccerNet é declarado como artefato, do mesmo jeito que os de jogadores e de
campo, e não vai no git:

```bash
uv run xvision artifact get models/manifests/ball/ball-yolov8n-0.1.0.toml
```

O comando falha dizendo o diretório exato sob `../var/models/` onde o arquivo deve ficar, e depois
que ele estiver lá o mesmo comando confere o sha256 e imprime o caminho do entrypoint. Ele não
baixa sozinho de propósito: peso derivado de dado restrito não sai por download automático.

Este peso não é derivado de dado restrito, veio do SoccerNet, então ele está no registro de
pacotes do projeto. Baixe para a pasta que o comando acima indicou:

```bash
set -a; . ../.env; set +a
curl -sf -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://git.inteli.edu.br/api/v4/projects/29510/packages/generic/ball-yolov8n/0.1.0/ball_yolov8n.pt" \
  -o ../var/models/<pasta-que-o-comando-disse>/ball_yolov8n.pt
```

O endereço também está no campo `download` do manifesto. Rode o `artifact get` de novo depois de
baixar: é ele que confere o sha256 do que chegou.

## Rodar no vídeo (script de avaliação)

O script `models/ball/track_ball_clip.py` roda o detector + a `BallTracker` num clipe e grava um
vídeo anotado (bola em vermelho, posição interpolada em círculo, candidatos reprovados em cinza):

```bash
python models/ball/track_ball_clip.py \
  "entrada.mkv" "saida_anotada.mp4" \
  --start 35 --end 50 --imgsz 1280 --conf 0.15 --weights yolov8n.pt
```

Parâmetros:

| Flag | Padrão | Para que serve |
|---|---|---|
| `--imgsz` | 1280 | resolução de entrada do YOLO; bola pequena pede valor alto |
| `--conf` | 0.15 | confiança mínima da detecção |
| `--start` / `--end` | 0 / fim | janela do clipe em segundos |
| `--weights` | `yolov8n.pt` | caminho dos pesos |

Ao final, imprime um resumo, por exemplo:

```text
frames=900 detected=496 (55%) interpolated=125 missing=279 cobertura=69% rejeitados_forma=341 -> saida_anotada.mp4
```

## Usar em código

```python
from edge.ball.tracking import (
    BallObservation, BallTracker, DetectorConfig, YoloBallDetector,
)

detector = YoloBallDetector.from_config(
    DetectorConfig(weights="yolov8n.pt", imgsz=1280, conf=0.15)
)
tracker = BallTracker()

for index, frame in enumerate(frames):           # frames: quadros BGR (ex.: OpenCV)
    candidates = detector.detect_candidates(frame)
    observation = BallObservation(index, index / fps, candidates)
    for point in tracker.update(observation):
        # point.source ∈ {"detected", "interpolated", "missing"}
        # point.center, point.box, point.confidence
        ...
for point in tracker.flush():                     # fecha lacunas pendentes no fim do stream
    ...
```

O rastreamento não precisa do YOLO: a `BallTracker` recebe `BallObservation`s, então dá para testar a
lógica com detecções de mentira, sem GPU.

## Testes

```bash
uv run pytest tests/edge     # inclui os testes do rastreador e do filtro de forma
uv run ruff check edge
uv run pyright edge
```

## Notas

- **Sem `pitch_xy` aqui:** a trajetória sai em pixels; a conversão para metros e a velocidade dependem
  da homografia (outra frente).
- **Baseline sem ajuste fino:** o YOLO do COCO detecta a bola de forma intermitente em transmissão; o
  filtro de forma/tamanho e a seleção por movimento reduzem falsos positivos (ex.: a marca do pênalti),
  mas o recall só melhora de verdade com ajuste fino em dados de futebol.
- Detalhes e resultados na doc da Sprint 2: `Rastreamento da bola`.
