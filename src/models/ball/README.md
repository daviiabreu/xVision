# Treino do detector da bola (fine tuning)

Pipeline para ajustar o **YOLOv8n** à detecção da bola em vídeo de transmissão, usando o **SoccerNet
Tracking** como dado de treino. O baseline sem ajuste fino detecta a bola de forma intermitente (ver a
doc da Sprint 2); este pipeline é o passo para melhorar o recall da bola perto do contato.

Nada aqui roda no runtime da borda — é ferramental de treino. Dados, pesos e resultados **não são
versionados** (ver `.gitignore`).

## Por que o SoccerNet Tracking serve

| Critério | Avaliação |
|---|---|
| Domínio | **Ideal.** Câmera principal de transmissão, 1080p a 25 fps — o mesmo domínio dos nossos clipes. |
| Rótulo da bola | **Sim.** Caixas da bola por quadro (tracklets), além de jogadores, goleiros e árbitros. |
| Volume | 201 sequências de ~30 s (57 treino / 49 teste / challenge sem rótulo). Suficiente para fine tuning. |
| Uso extra | Os tracklets também servem para avaliar o nosso rastreador (trocas de ID, lacunas) mais adiante. |

Ressalvas: (1) é **MOTChallenge**, não YOLO — o conversor abaixo resolve; (2) o split `challenge` **não
tem rótulo** (só submissão); (3) a bola é minúscula e às vezes não anotada quando invisível; (4)
**precisa de GPU** para treinar.

> **Sobre o NDA/senha:** o download do **tracking** (`train`/`test`) sai **sem senha** — o comando
> oficial não seta nenhuma. O termo (NDA) da página do SoccerNet vale para os **Broadcast Videos** (os
> 500 jogos em `.mkv`), não para a tarefa de tracking. Se algum split pedir senha (HTTP 403), aí sim
> preencha o NDA em <https://www.soccer-net.org/> e passe `--password`.

## Um comando (orquestrador)

`run.py` encadeia as três etapas. **O treino precisa de GPU** — em CPU, no volume do SoccerNet, é
inviável; então rode as duas primeiras etapas localmente e o treino numa GPU:

```bash
uv sync --extra training
python src/models/ball/run.py --stages download,convert   # local
python src/models/ball/run.py --stages train --device 0   # numa GPU (Colab/Kaggle/servidor)
# ou tudo de uma vez numa GPU:
python src/models/ball/run.py --epochs 100 --device 0
```

As etapas individuais continuam disponíveis (abaixo), úteis para reexecutar ou depurar cada parte.

### GPU: PyTorch com CUDA

No **Linux**, `uv sync --extra training` já traz o PyTorch com CUDA (o wheel padrão do PyPI inclui
CUDA), então numa máquina com a RTX 4080 costuma funcionar direto. Confirme antes de treinar:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # tem que dar True
```

Se der `False` (no Windows o wheel padrão é CPU, ou num Linux sem o driver NVIDIA adequado), instale
a build CUDA — ex.: CUDA 12.4:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

O `train_ball.py` imprime no início se enxergou a GPU e avisa se cair para CPU.

## Passo a passo (etapas separadas)

Instale o extra de treino:

```bash
uv sync --extra training
```

### 1. Baixar (precisa da senha do SoccerNet)

```bash
python src/models/ball/download_soccernet.py --out data/soccernet --split train test
# tracking baixa sem senha; use --password so se algum split retornar 403 (ai preencha o NDA)
```

### 2. Converter MOTChallenge → YOLO

```bash
python src/models/ball/soccernet_to_yolo.py \
  --src data/soccernet/tracking --out data/ball-yolo --train train --val test
```

Gera `data/ball-yolo/{images,labels}/{train,val}` e `data.yaml` (uma classe: `ball`). Por padrão mantém
só quadros com bola; `--keep-background` inclui quadros sem bola como negativos.

### 3. Treinar (GPU)

```bash
python src/models/ball/train_ball.py --data data/ball-yolo/data.yaml --epochs 100 --device 0
```

Sai um `best.pt` em `runs/ball/...`. Para usá-lo no rastreamento, aponte `DetectorConfig(weights=...)`
para esse arquivo — a interface `BallDetector` não muda.

### 4. Comparar com o baseline

Rode o script de avaliação da frente de rastreamento com os pesos novos e compare a cobertura contra o
baseline sem ajuste fino:

```bash
python src/models/ball/track_ball_clip.py entrada.mkv saida.mp4 \
  --weights runs/ball/yolov8n-ball/weights/best.pt --start 2 --end 15 --imgsz 1280
```

## Onde fica a lógica

A conversão MOT→YOLO (descobrir a bola no `gameinfo.ini`, normalizar caixas) está em
`edge.ball.dataset` — funções puras, cobertas por `src/tests/edge/test_dataset_soccernet.py`.
Os scripts daqui só orquestram download, travessia de diretórios e treino.
