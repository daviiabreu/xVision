"""
App de anotação manual dos clipes de cortes de finalização.

Pra cada clip da amostra, a pessoa marca:
  1. o frame exato do momento da finalização (calibra o "frame extractor" do
     pipeline de detecção — ver scripts/baixar_cortes_footstats.py e a
     discussão de design da abordagem A);
  2. se a área do chute estava "lotada" (crowded) ou não — vira o gabarito
     usado depois pra medir a acurácia do pipeline automático.

Uso:
    streamlit run scripts/anotar_clips.py

Lê scripts/cortes/cortes_metadados.csv (gerado por baixar_cortes_footstats.py).
Escreve:
  - scripts/cortes/fila_anotacao.csv   (amostra sorteada uma única vez, fixa)
  - scripts/cortes/anotacoes.csv       (respostas; reabrir o app retoma de onde parou)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

CORTES_DIR = Path(__file__).parent / "cortes"
METADATA_CSV = CORTES_DIR / "cortes_metadados.csv"
QUEUE_CSV = CORTES_DIR / "fila_anotacao.csv"
ANNOTATIONS_CSV = CORTES_DIR / "anotacoes.csv"
SAMPLE_SIZE = 40
SEED = 42

ANNOTATION_COLUMNS = [
    "cut_id", "match_id", "shot_frame_index", "shot_frame_seconds",
    "crowded_manual", "annotated_at", "status",
]


@st.cache_data
def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV, dtype={"cut_id": str, "match_id": str})
    return df[df["clip_url"].notna() & (df["clip_url"] != "")]


def build_or_load_queue(df: pd.DataFrame) -> pd.DataFrame:
    if QUEUE_CSV.exists():
        return pd.read_csv(QUEUE_CSV, dtype={"cut_id": str, "match_id": str})

    rng = np.random.default_rng(SEED)
    results = df["result"].unique()
    per_result = max(1, SAMPLE_SIZE // len(results))
    picked = []
    for r in results:
        group = df[df["result"] == r]
        n = min(per_result, len(group))
        idx = rng.choice(group.index, size=n, replace=False)
        picked.append(df.loc[idx])
    sample = pd.concat(picked)
    if len(sample) < SAMPLE_SIZE:
        remaining = df.drop(sample.index)
        extra_n = min(SAMPLE_SIZE - len(sample), len(remaining))
        extra_idx = rng.choice(remaining.index, size=extra_n, replace=False)
        sample = pd.concat([sample, df.loc[extra_idx]])

    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    queue = sample[["cut_id", "match_id"]]
    QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE_CSV, index=False)
    return queue


def load_annotations() -> dict[str, dict]:
    if not ANNOTATIONS_CSV.exists():
        return {}
    df = pd.read_csv(ANNOTATIONS_CSV, dtype={"cut_id": str, "match_id": str})
    return {row["cut_id"]: row.to_dict() for _, row in df.iterrows()}


def save_annotations(annotations: dict[str, dict]) -> None:
    df = pd.DataFrame(annotations.values(), columns=ANNOTATION_COLUMNS)
    df.to_csv(ANNOTATIONS_CSV, index=False)


@st.cache_data(show_spinner="Carregando frames do clip...")
def get_frames(mp4_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def prefetch_frames(mp4_path: str) -> None:
    try:
        get_frames(mp4_path)  # popula o st.cache_data pra quando o usuário avançar
    except Exception:
        pass  # prefetch é best-effort; erro aqui não deve travar nada


def trigger_prefetch_next(queue: pd.DataFrame, df: pd.DataFrame, pointer: int) -> None:
    if pointer + 1 >= len(queue):
        return
    next_cut_id = queue.iloc[pointer + 1]["cut_id"]
    if "prefetched" not in st.session_state:
        st.session_state.prefetched = set()
    if next_cut_id in st.session_state.prefetched:
        return
    next_meta = df[df["cut_id"] == next_cut_id]
    if next_meta.empty:
        return
    next_row = queue.iloc[pointer + 1]
    next_mp4 = CORTES_DIR / next_row["match_id"] / next_meta.iloc[0]["file_name"]
    if not next_mp4.exists():
        return
    st.session_state.prefetched.add(next_cut_id)
    threading.Thread(target=prefetch_frames, args=(str(next_mp4),), daemon=True).start()


def main() -> None:
    st.set_page_config(page_title="Anotação de cortes", layout="centered")
    st.title("Anotação de finalização e lotação da área")

    if not METADATA_CSV.exists():
        st.error(f"Não encontrei {METADATA_CSV}. Rode baixar_cortes_footstats.py primeiro.")
        return

    df = load_metadata()
    queue = build_or_load_queue(df)

    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    annotations = st.session_state.annotations

    if "pointer" not in st.session_state:
        pointer = 0
        for i, cut_id in enumerate(queue["cut_id"]):
            if cut_id not in annotations:
                pointer = i
                break
        else:
            pointer = len(queue)
        st.session_state.pointer = pointer

    total = len(queue)
    pointer = st.session_state.pointer

    st.progress(min(pointer, total) / total)
    st.caption(f"{min(pointer, total)} / {total} anotados")

    if pointer >= total:
        st.success("Todos os clipes da amostra foram anotados.")
        return

    trigger_prefetch_next(queue, df, pointer)

    row = queue.iloc[pointer]
    cut_id, match_id = row["cut_id"], row["match_id"]
    meta = df[df["cut_id"] == cut_id].iloc[0]

    st.subheader(f"{meta['home']} x {meta['away']} — {meta['minute']}'")
    st.write(
        f"**Time:** {meta['team']} · **Pé/parte:** {meta['foot']} · "
        f"**Jogada:** {meta['play_type']} · **Resultado:** {meta['result_label']} · "
        f"**Zona:** {meta['zone']}"
    )

    def skip_clip(reason: str) -> None:
        annotations[cut_id] = {
            "cut_id": cut_id, "match_id": match_id,
            "shot_frame_index": None, "shot_frame_seconds": None,
            "crowded_manual": None, "annotated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": f"skipped ({reason})",
        }
        save_annotations(annotations)
        st.session_state.pointer += 1
        st.rerun()

    mp4_path = CORTES_DIR / match_id / meta["file_name"]
    if not mp4_path.exists():
        st.error(f"Vídeo não encontrado: {mp4_path}")
        if st.button("Pular este clip"):
            skip_clip("arquivo ausente")
        return

    frames, fps = get_frames(str(mp4_path))
    if not frames:
        st.error(f"Não consegui ler nenhum frame de {mp4_path} (arquivo corrompido?).")
        if st.button("Pular este clip"):
            skip_clip("vídeo corrompido")
        return

    existing = annotations.get(cut_id)
    default_frame = (
        int(existing["shot_frame_index"]) if existing is not None
        else max(0, len(frames) - 1)
    )

    st.video(str(mp4_path))
    st.caption("Assista/pause aqui pra achar a hora certa, depois ajuste o frame exato embaixo.")

    frame_idx = st.slider(
        "Frame exato da finalização", 0, len(frames) - 1,
        value=min(default_frame, len(frames) - 1),
        key=f"frame_slider_{cut_id}",
    )
    frame_idx = min(frame_idx, len(frames) - 1)
    st.image(frames[frame_idx], width='stretch')
    st.caption(f"t = {frame_idx / fps:.2f}s de {len(frames) / fps:.2f}s ({len(frames)} frames)")

    default_crowded = "Sim"
    if existing is not None and not pd.isna(existing.get("crowded_manual")):
        default_crowded = "Sim" if existing["crowded_manual"] else "Não"
    crowded = st.radio(
        "A área do chute estava lotada de jogadores?",
        ["Sim", "Não"],
        index=0 if default_crowded == "Sim" else 1,
        horizontal=True,
    )

    col_prev, col_save = st.columns(2)
    with col_prev:
        if st.button("Anterior", disabled=pointer == 0, width='stretch'):
            st.session_state.pointer -= 1
            st.rerun()
    with col_save:
        if st.button("Salvar e próximo", type="primary", width='stretch'):
            annotations[cut_id] = {
                "cut_id": cut_id,
                "match_id": match_id,
                "shot_frame_index": frame_idx,
                "shot_frame_seconds": round(frame_idx / fps, 3),
                "crowded_manual": crowded == "Sim",
                "annotated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "done",
            }
            save_annotations(annotations)
            st.session_state.pointer += 1
            st.rerun()


if __name__ == "__main__":
    main()
