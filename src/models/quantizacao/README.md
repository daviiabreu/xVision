# Quantização e Avaliação de Modelos

Scripts para exportar modelos YOLO para TensorRT FP16 e avaliar os resultados,
seja na Jetson com o engine compilado ou em outro PC (ex: A4000) usando o `.fp16.onnx`.

---

## Estrutura de pastas

```
quantizacao/
├── quantizacao.py      # Exporta .pt → engine TensorRT FP16  (rodar na Jetson)
├── avaliar.py          # Valida com vídeo ou dataset usando .engine (rodar na Jetson)
├── avaliar_fp16.py     # Valida com vídeo ou dataset usando .fp16.onnx (rodar em qualquer PC)
│
├── ball.pt             # ← coloque aqui os modelos .pt (necessário para quantizacao.py)
├── player.pt
├── keypoints.pt
│
├── ball/               # ← criado automaticamente pelo quantizacao.py
│   ├── ball.onnx
│   ├── ball.fp16.onnx  # ← copie este para o PC de validação
│   └── ball.engine     # ← só roda na Jetson onde foi gerado
├── player/
│   ├── player.fp16.onnx
│   └── player.engine
├── keypoints/
│   ├── keypoints.fp16.onnx
│   └── keypoints.engine
│
├── dataset/            # ← coloque aqui os datasets para validação
│   ├── ball/
│   │   └── data.yaml
│   ├── player/
│   │   └── data.yaml
│   └── keypoints/
│       └── data.yaml
│
└── resultados/         # ← criado automaticamente pelos scripts de avaliação
    ├── ball_video/
    │   ├── predicoes.mp4
    │   └── velocidade.json
    ├── ball_validar/
    │   ├── metricas.json
    │   └── validacao/
    └── ball_fp16_validar/
        ├── metricas.json
        └── validacao/
```

Os nomes dos modelos são: `ball`, `player` e `keypoints`.

---

## 1. Exportar modelos - rodar na Jetson

Coloque os arquivos `.pt` na raiz desta pasta, depois rode:

```bash
# Exportar todos os modelos de uma vez
python3 -u quantizacao.py

# Exportar apenas um modelo
python3 -u quantizacao.py --modelo ball
python3 -u quantizacao.py --modelo player
python3 -u quantizacao.py --modelo keypoints
```

Cada modelo gera três arquivos dentro da sua subpasta:
- `.onnx` — pesos em FP32
- `.fp16.onnx` — pesos em FP16 (use este para validar fora da Jetson)
- `.engine` — engine TRT compilado, só roda na Jetson onde foi gerado

**Opção extra:** `--workspace 2` aumenta o limite de memória do TensorRT para 2 GiB
(padrão é 1 GiB).

---

## 2. Avaliar com engine - rodar na Jetson

```bash
# Inferência em vídeo
python3 -u avaliar.py video --modelo ball --video video.mp4

# Validação com dataset
python3 -u avaliar.py validar --modelo ball
```

O engine é carregado de `<modelo>/<modelo>.engine`.
O dataset é buscado automaticamente em `dataset/<modelo>/`.

---

## 3. Avaliar com .fp16.onnx - rodar em qualquer PC com GPU

Use quando o dataset for grande demais para baixar na Jetson, ou para comparar
métricas FP16 vs FP32 fora dela.

**Pré-requisito:** instale o ONNX Runtime com suporte a GPU:

```bash
pip install onnxruntime-gpu
```

**Copie o `.fp16.onnx` da Jetson para o PC** (mantendo a estrutura de pastas):

```
ball/ball.fp16.onnx
player/player.fp16.onnx
keypoints/keypoints.fp16.onnx
```

Depois rode:

```bash
# Inferência em vídeo
python3 -u avaliar_fp16.py video --modelo ball --video video.mp4

# Validação com dataset
python3 -u avaliar_fp16.py validar --modelo ball
```

O dataset é buscado automaticamente em `dataset/<modelo>/`, da mesma forma que o `avaliar.py`.
Os resultados ficam em `resultados/<modelo>_fp16_<modo>/`.

> **Por que o `.fp16.onnx` e não o `.onnx` ou recompilar o engine em outra maquina?**
> O `.onnx` contém pesos FP32 — as métricas seriam idênticas ao `.pt` original.
> O engine TRT não é portável entre GPUs: um engine gerado na Jetson não roda
> em outra maquina e vice-versa. O `.fp16.onnx` contém os pesos já convertidos para FP16
> e roda via ONNX Runtime em qualquer GPU com CUDA, permitindo medir o impacto
> real da quantização.

---

## Estrutura esperada do dataset

```
dataset/
└── ball/
    ├── data.yaml
    ├── images/
    │   └── val/
    │       └── frame001.jpg
    └── labels/
        └── val/
            └── frame001.txt
```

Exemplo de `data.yaml`:

```yaml
path: /caminho/absoluto/para/dataset/ball
train: images/train
val: images/val
names:
  0: ball
```

> Use o `path` absoluto no `.yaml` para evitar problemas de caminho relativo.