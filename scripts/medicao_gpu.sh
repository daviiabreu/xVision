#!/usr/bin/env bash
# Monta o g02 do zero e roda a medicao de type_of_play e occupied_area nas dobras 0 a 2.
# Uso: bash medicao_gpu.sh <pasta com 0005 e 0006> <pasta com os clipes por partida> [reservado]
# Com `reservado`, faz a passada unica nas dobras 3 e 4 com o modelo ja treinado. So uma vez.
set -euo pipefail

TABELAS=$(realpath "${1:?pasta com 0005_match_shots_1390.csv e 0006_match_shots_xg_1390.csv}")
CORTES=$(realpath "${2:?pasta com os clipes, uma subpasta por partida}")
MODO=${3:-ajuste}
BRANCH=feat/insumos-type-of-play-occupied-area

[ -d g02 ] || git clone https://git.inteli.edu.br/graduacao/2026-2a/t12/g02.git
cd g02
git fetch origin && git switch "$BRANCH" && git pull --ff-only

mkdir -p data/footstats/db
cp "$TABELAS"/0005_match_shots_1390.csv "$TABELAS"/0006_match_shots_xg_1390.csv data/footstats/db/
ln -sfn "$CORTES" data/footstats/cortes

cd src
uv sync --frozen --all-extras

baixa() {  # repositorio, arquivo, revisao, diretorio
  [ -f "../var/models/$4/$2" ] || uv run --frozen hf download "$1" "$2" --revision "$3" --local-dir "../var/models/$4"
}
baixa xvisionm11/football-player-detection football-player-detection.pt v1.0.0 fac37e36b0dc6f605ace4b134107d134396734a99254931aee2acbc2768a0256
baixa xvisionm11/yolo-football-ball-detection yolo-football-ball-detection.pt v0.1.0 935b75a4a9b8c29f353a0c202bb984f299e47c55107857bd4882ee62c43018bf
baixa xvisionm11/pitch-keypoints-yolo11n-pose best.pt v0.1.0 167fe1bb7fd29c39f8b2434129d75b73b16a8a5f86d66fd539372481fa07c787
baixa xvisionm11/yolov8n-pose yolov8n-pose.pt v8.4.0 d5d6a7a1de6b0a9609d99bb1e6ab6bee5b5d2128e3c35a1c119262c16c3dda2f

if [ ! -d ../var/external/T-DEED ]; then
  git clone https://github.com/arturxe2/T-DEED.git ../var/external/T-DEED
  git -C ../var/external/T-DEED checkout d953b4de8a2ad941c565e1683c5b07255cbe644e
fi
TDEED=../var/models/5553acb409b22d59828fea2b43d946ab3b10e3661c036974459286d1a73bbbe9
mkdir -p "$TDEED"
[ -f "$TDEED/checkpoint_best.pt" ] || uvx --from gdown gdown 1b7REo4_Cota8SWPSmtdSEfbVNXNgziX6 -O "$TDEED/checkpoint_best.pt"

for m in players/football-player-detection-1.0.0 ball/yolo-football-ball-detection-0.1.0 \
  field/pitch-keypoints-yolo11n-pose-0.1.0 pose/yolov8n-pose-8.4.0 shot/tdeed-soccernetball-test-1.0.0; do
  uv run --frozen xvision artifact get "models/manifests/$m.toml"
done
uv run --frozen xvision data check

cp config/pipeline.models.toml config/pipeline.experimento.toml
cat >> config/pipeline.experimento.toml <<'EOF'

[features.occupied_area]
impl = "edge.features.occupied_area:OccupiedArea"

[features.marked]
impl = "edge.features.marked:Marked"
EOF
uv run --frozen python - <<'EOF'
from pathlib import Path
texto = Path("config/pipeline.experimento.toml").read_text()
ancora = "[modules.field]\n"
depois = texto.index(ancora) + len(ancora)
opcoes = 'propagation_steps = 8\nsafe_projection = "visible"\n'
Path("config/pipeline.experimento-campo.toml").write_text(texto[:depois] + opcoes + texto[depois:])
EOF

mede() {  # nome, config
  mkdir -p "../var/runs/$1"
  uv run --frozen --all-extras xvision run --all --dobra ajuste --config "config/$2" --out "../var/runs/$1"
  local execucao
  execucao=$(ls -td "../var/runs/$1"/*/ | head -1)
  uv run --frozen xvision eval "$execucao" --dobra ajuste --bootstrap 1000 | tee "../var/runs/$1/eval.txt"
  for feature in type_of_play occupied_area; do
    uv run --frozen xvision treino "$execucao" --feature "$feature" \
      --saida "../var/runs/$1/$feature.json" | tee -a "../var/runs/$1/treino.txt"
  done
}
if [ "$MODO" = reservado ]; then
  MODELO=../var/runs/experimento-campo/type_of_play.json
  [ -f "$MODELO" ] || { echo "falta $MODELO: rode a medicao nas dobras 0 a 2 antes"; exit 1; }
  uv run --frozen python - "$MODELO" <<'EOF'
import sys
from pathlib import Path
texto = Path("config/pipeline.experimento-campo.toml").read_text()
ancora = 'impl = "edge.features.type_of_play:TypeOfPlay"\n'
assert ancora in texto
texto = texto.replace(ancora, ancora + f'model = "{sys.argv[1]}"\n', 1)
Path("config/pipeline.final.toml").write_text(texto)
EOF
  mkdir -p ../var/runs/final
  uv run --frozen --all-extras xvision run --all --dobra reservado --config config/pipeline.final.toml \
    --out ../var/runs/final
  execucao=$(ls -td ../var/runs/final/*/ | head -1)
  uv run --frozen xvision eval "$execucao" --dobra reservado --bootstrap 1000 | tee ../var/runs/final/eval.txt
  echo "resultado final em $(realpath ../var/runs/final)/eval.txt"
  exit 0
fi

mede experimento pipeline.experimento.toml
mede experimento-campo pipeline.experimento-campo.toml

echo "resultados em $(realpath ../var/runs)/experimento e experimento-campo"
