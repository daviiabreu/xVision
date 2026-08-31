"""
Dashboard de inspeção visual do "crowded" — mostra, para cada clip anotado, o
frame do chute com as DUAS âncoras candidatas desenhadas ao mesmo tempo (bola
detectada de verdade e o fallback por cor de time) e a contagem em metros reais
de campo (via homografia), lado a lado com o rótulo manual.

Uso:
    streamlit run scripts/avaliar_crowded.py

Lê scripts/cortes/anotacoes.csv (gabarito) e scripts/cortes/cortes_metadados.csv.
Roda a detecção uma vez por clip e guarda em cache de sessão — o slider de raio
só recalcula a contagem, sem re-rodar os modelos.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from crowded_pipeline import (
    CORTES_DIR,
    RADIUS_M_DEFAULT,
    THRESHOLD_DEFAULT,
    BothAnchors,
    analyze_both_anchors,
    load_annotated_clips,
    load_ball_model,
    load_field_model,
    load_player_model,
)

PLAYER_COLOR = (56, 189, 248)
BALL_ANCHOR_COLOR = (250, 204, 21)
FALLBACK_ANCHOR_COLOR = (251, 146, 60)


@st.cache_resource
def get_models() -> tuple[YOLO, YOLO, YOLO | None]:
    return load_player_model(), load_field_model(), load_ball_model()


@st.cache_data(show_spinner="Detectando jogadores, bola e campo...")
def analyze_for_display(_player_model: YOLO, _field_model: YOLO, _ball_model, mp4_path: str, frame_index: int):
    result = analyze_both_anchors(_player_model, _field_model, mp4_path, frame_index, ball_model=_ball_model)
    if isinstance(result, str):
        return None, None, result
    frame, analysis = result
    return frame, analysis, None


def count_within(anchor: tuple[float, float] | None, positions: list, radius_cm: float) -> int | None:
    if anchor is None:
        return None
    ax, ay = anchor
    return sum(
        1 for pos in positions
        if pos is not None and ((pos[0] - ax) ** 2 + (pos[1] - ay) ** 2) ** 0.5 <= radius_cm
    )


def draw_both_anchors(frame: np.ndarray, a: BothAnchors, radius_px_ball: float, radius_px_fallback: float) -> np.ndarray:
    img = frame.copy()

    inside = set()
    for i, d in enumerate(a.dets):
        cx, cy = d.center
        if a.ball_anchor_px is not None:
            ax, ay = a.ball_anchor_px
            if ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 <= radius_px_ball:
                inside.add(i)
        if a.fallback_anchor_px is not None:
            ax, ay = a.fallback_anchor_px
            if ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 <= radius_px_fallback:
                inside.add(i)

    for i, d in enumerate(a.dets):
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                      PLAYER_COLOR, 3 if i in inside else 1)

    if a.ball_anchor_px is not None:
        ax, ay = a.ball_anchor_px
        cv2.drawMarker(img, (int(ax), int(ay)), BALL_ANCHOR_COLOR, cv2.MARKER_STAR, 28, 3)
        cv2.circle(img, (int(ax), int(ay)), int(radius_px_ball), BALL_ANCHOR_COLOR, 2)
    if a.fallback_anchor_px is not None:
        ax, ay = a.fallback_anchor_px
        cv2.drawMarker(img, (int(ax), int(ay)), FALLBACK_ANCHOR_COLOR, cv2.MARKER_CROSS, 28, 3)
        cv2.circle(img, (int(ax), int(ay)), int(radius_px_fallback), FALLBACK_ANCHOR_COLOR, 2)

    return img


def main() -> None:
    st.set_page_config(page_title="Inspeção crowded", layout="wide")
    st.title("Inspeção visual: bola real x fallback por cor")
    st.caption(
        "★ amarelo = bola detectada (modelo dedicado, com fallback pro combinado) · "
        "✚ laranja = par mais próximo entre times · raio em metros reais de campo (via homografia)"
    )

    clips = load_annotated_clips()
    player_model, field_model, ball_model = get_models()
    st.sidebar.caption(f"Modelo dedicado de bola: {'carregado' if ball_model is not None else 'ausente'}")

    st.sidebar.header("Parâmetros")
    radius_m = st.sidebar.slider("Raio real (metros)", 0.5, 12.0, RADIUS_M_DEFAULT, 0.5)
    threshold = st.sidebar.slider("Threshold de contagem", 1, 8, THRESHOLD_DEFAULT)
    filtro = st.sidebar.radio(
        "Mostrar", ["Marcados crowded (manual)", "Marcados não-crowded (manual)", "Todos"], index=0,
    )
    radius_cm = radius_m * 100

    analyses: dict[str, tuple[np.ndarray, BothAnchors]] = {}
    skips = {}
    for _, row in clips.iterrows():
        mp4_path = CORTES_DIR / row["match_id"] / row["file_name"]
        frame, analysis, skip_reason = analyze_for_display(
            player_model, field_model, ball_model, str(mp4_path), row["shot_frame_index"],
        )
        if analysis is None:
            skips[row["cut_id"]] = skip_reason
            continue
        analyses[row["cut_id"]] = (frame, analysis)

    if filtro == "Marcados crowded (manual)":
        clips_mostradas = clips[clips["crowded_manual"] == True]  # noqa: E712
    elif filtro == "Marcados não-crowded (manual)":
        clips_mostradas = clips[clips["crowded_manual"] == False]  # noqa: E712
    else:
        clips_mostradas = clips

    # métrica agregada: mesma prioridade do pipeline real (bola primeiro, fallback só sem bola)
    preds = []
    for _, row in clips.iterrows():
        item = analyses.get(row["cut_id"])
        if item is None:
            continue
        _, a = item
        anchor_pitch = a.ball_anchor_pitch if a.ball_anchor_pitch is not None else a.fallback_anchor_pitch
        if anchor_pitch is None or not a.has_homography:
            continue
        count = count_within(anchor_pitch, a.player_positions, radius_cm)
        preds.append({"pred": count >= threshold, "manual": bool(row["crowded_manual"])})

    if preds:
        pdf = pd.DataFrame(preds)
        tp = int(((pdf["pred"]) & (pdf["manual"])).sum())
        tn = int(((~pdf["pred"]) & (~pdf["manual"])).sum())
        fp = int(((pdf["pred"]) & (~pdf["manual"])).sum())
        fn = int(((~pdf["pred"]) & (pdf["manual"])).sum())
        acc = (tp + tn) / len(pdf)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Acurácia", f"{acc:.1%}")
        c2.metric("Verdadeiro positivo", tp)
        c3.metric("Verdadeiro negativo", tn)
        c4.metric("Falso positivo", fp)
        c5.metric("Falso negativo", fn)
        st.caption(f"{len(pdf)} clipes válidos pra métrica (bola quando há, fallback senão; raio={radius_m}m, threshold={threshold})")

    st.caption(f"{len(analyses)}/{len(clips)} clipes com pelo menos uma âncora calculável ({len(skips)} pulados)")
    st.divider()

    for _, row in clips_mostradas.iterrows():
        cut_id = row["cut_id"]
        if cut_id not in analyses:
            continue
        frame, a = analyses[cut_id]

        # raio em pixels só pra desenhar — escala pela altura mediana (não temos escala px/cm direta sem
        # reprojetar o círculo pela homografia inversa, o que é overkill só pra visualização).
        px_per_m_proxy = a.median_height / 1.8  # ~1.8m de altura média de jogador
        radius_px_ball = radius_m * px_per_m_proxy
        radius_px_fallback = radius_px_ball

        count_ball = count_within(a.ball_anchor_pitch, a.player_positions, radius_cm) if a.has_homography else None
        count_fallback = (
            count_within(a.fallback_anchor_pitch, a.player_positions, radius_cm) if a.has_homography else None
        )

        img = draw_both_anchors(frame, a, radius_px_ball, radius_px_fallback)

        col_img, col_info = st.columns([3, 1])
        with col_img:
            st.image(img, width="stretch")
        with col_info:
            st.markdown(f"### {cut_id}")
            st.write(f"**{row['home']} x {row['away']}** — {row['minute']}'")
            st.write(f"Manual: **{'crowded' if row['crowded_manual'] else 'não crowded'}**")
            st.write(f"Homografia: **{'sim' if a.has_homography else 'não'}**")
            if count_ball is not None:
                pred = "crowded" if count_ball >= threshold else "não crowded"
                st.write(f"★ bola: **{count_ball}** jogadores em {radius_m}m → {pred}")
            else:
                st.write("★ bola: não detectada nesse frame")
            if count_fallback is not None:
                pred = "crowded" if count_fallback >= threshold else "não crowded"
                st.write(f"✚ fallback: **{count_fallback}** jogadores em {radius_m}m → {pred}")
            else:
                st.write("✚ fallback: sem par entre times")
        st.divider()

    if skips:
        with st.expander(f"{len(skips)} clipes pulados na detecção"):
            for cut_id, reason in skips.items():
                st.write(f"- `{cut_id}`: {reason}")


if __name__ == "__main__":
    main()
