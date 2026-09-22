# SPDX-License-Identifier: GPL-3.0-or-later
"""Executor do T-DEED para o modulo `shot` do xVision, num processo separado.

Este arquivo importa o codigo do T-DEED (Xarles et al., CVPRW 2024), que e GPL-3.0, e por isso e
distribuido sob a mesma licenca, fora de `src/`. O produto conversa com ele so por arquivo: recebe
os quadros em `.npy` e devolve as probabilidades por passo em JSON. Nenhum tipo, funcao ou objeto
atravessa a fronteira, e e isso que mantem `src/` na licenca dele.

Uso, a partir da raiz do repositorio:

    python external/tdeed/run.py --source var/external/T-DEED --checkpoint <peso> \
        --frames quadros.npy --saida probabilidades.json

`quadros.npy` e um uint8 (N, altura, largura, 3) em RGB, na taxa que o modelo espera.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CONFIG = "config/SoccerNetBall/SoccerNetBall_challenge1.json"
CLASSES = (
    "BACKGROUND",
    "PASS",
    "DRIVE",
    "HEADER",
    "HIGH PASS",
    "OUT",
    "CROSS",
    "THROW IN",
    "SHOT",
    "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE",
    "FREE KICK",
    "GOAL",
)
CLASSES_DO_PRE_TREINO = 18


def main() -> None:
    argumentos = _argumentos()
    sys.path.insert(0, str(argumentos.source.resolve()))

    import numpy as np
    import timm
    import torch

    criar = timm.create_model

    def sem_download(*posicionais: object, **nomeados: object) -> object:
        nomeados["pretrained"] = False
        return criar(*posicionais, **nomeados)

    # O T-DEED pede o RegNetY da ImageNet a cada construcao. Todo peso dele e sobrescrito pelo
    # checkpoint logo depois, entao baixar seria so uma dependencia de rede a mais na borda.
    timm.create_model = sem_download

    from model.model import TDEEDModel
    from model.modules import FC2Layers

    dispositivo = argumentos.device
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads((argumentos.source / CONFIG).read_text(encoding="utf-8"))
    args = argparse.Namespace(**{**config, "crop_dim": None})

    inicio = time.perf_counter()
    modelo = TDEEDModel(device=dispositivo, args=args)
    # O checkpoint publicado tem duas cabecas: as 13 classes do SoccerNet Ball e as 18 do
    # pre-treino no SoccerNet. `update_pred_head` do repositorio chama `.cuda()`, entao a troca e
    # feita aqui, para funcionar tambem sem GPU.
    modelo._model._pred_fine = FC2Layers(
        modelo._model._feat_dim, [len(CLASSES), CLASSES_DO_PRE_TREINO]
    )
    modelo._model._double_head = True
    estado = torch.load(argumentos.checkpoint, map_location="cpu", weights_only=False)
    modelo._model.load_state_dict(estado, strict=True)
    modelo._model.to(dispositivo)
    modelo._model.eval()
    carga = time.perf_counter() - inicio

    quadros = np.load(argumentos.frames)
    if quadros.ndim != 4 or quadros.shape[-1] != 3 or quadros.dtype != np.uint8:
        raise SystemExit(
            f"quadros {quadros.shape} {quadros.dtype}: o esperado e uint8 (N, H, W, 3)"
        )
    passos = quadros.transpose(0, 3, 1, 2)
    tamanho = int(args.clip_len)
    probabilidades = np.zeros((passos.shape[0], len(CLASSES)), dtype=np.float32)
    inicio = time.perf_counter()
    for comeco in range(0, passos.shape[0], tamanho):
        pedaco = passos[comeco : comeco + tamanho]
        util = pedaco.shape[0]
        if util < tamanho:
            # Completa repetindo o ultimo quadro. A parte inventada e descartada logo abaixo.
            repetido = np.repeat(pedaco[-1:], tamanho - util, axis=0)
            pedaco = np.concatenate([pedaco, repetido])
        _, saida = modelo.predict(
            torch.from_numpy(pedaco), use_amp=dispositivo == "cuda"
        )
        probabilidades[comeco : comeco + util] = saida[0][:util, : len(CLASSES)]
    inferencia = time.perf_counter() - inicio

    argumentos.saida.write_text(
        json.dumps(
            {
                "classes": list(CLASSES),
                "probabilities": probabilidades.round(6).tolist(),
                "device": dispositivo,
                "torch": torch.__version__,
                "load_s": round(carga, 3),
                "infer_s": round(inferencia, 3),
            }
        ),
        encoding="utf-8",
    )


def _argumentos() -> argparse.Namespace:
    leitor = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    leitor.add_argument("--source", type=Path, required=True, help="checkout do T-DEED")
    leitor.add_argument("--checkpoint", type=Path, required=True)
    leitor.add_argument("--frames", type=Path, required=True)
    leitor.add_argument("--saida", type=Path, required=True)
    leitor.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    return leitor.parse_args()


if __name__ == "__main__":
    main()
