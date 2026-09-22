# Executor do T-DEED

Este diretório tem licença própria, GPL-3.0 (ver `LICENSE`), diferente do resto do repositório.
O `run.py` importa o código do [T-DEED](https://github.com/arturxe2/T-DEED), que é GPL-3.0, e
por isso é distribuído sob a mesma licença.

O módulo `shot` do produto (`src/edge/modules/shot/tdeed.py`) não importa nada daqui. Ele grava os
quadros num `.npy`, chama `run.py` em outro processo e lê as probabilidades de volta em JSON. Os
dois programas conversam só por arquivo e linha de comando, e é isso que mantém `src/` na licença
dele.

## Instalação

A partir da raiz do repositório:

```bash
git clone https://github.com/arturxe2/T-DEED.git var/external/T-DEED
git -C var/external/T-DEED checkout d953b4de8a2ad941c565e1683c5b07255cbe644e
cd src && uv sync --all-extras
```

O peso é o checkpoint `SoccerNetBall_test/checkpoint_best.pt` publicado pelos autores, na pasta do
Google Drive indicada no manifesto `src/models/manifests/shot/tdeed-soccernetball-test-1.0.0.toml`.
Ele vai para o diretório que o manifesto calcula a partir do sha256:

```bash
mkdir -p var/models/5553acb409b22d59828fea2b43d946ab3b10e3661c036974459286d1a73bbbe9
cp <download>/SoccerNetBall_test/checkpoint_best.pt \
  var/models/5553acb409b22d59828fea2b43d946ab3b10e3661c036974459286d1a73bbbe9/
```

O `load()` do módulo confere o sha256 e recusa arquivo diferente.

## Licença do peso

Os autores publicam o código como GPL-3.0 e não declaram licença para os pesos. Silêncio não é
permissão. O peso não é versionado neste repositório e cada instalação baixa da fonte original. Usar
o peso no produto entregue ao parceiro continua sendo uma decisão a registrar com ele e com a
orientação.
