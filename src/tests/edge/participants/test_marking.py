"""O que a decisao de marcacao promete, no vocabulario de quem le a saida."""

from __future__ import annotations

import pytest

from edge.participants.marking import (
    MarkingCandidate,
    MarkingError,
    candidates_from_frame,
    marked_keys,
    pair_markings,
    teams_from_identities,
)
from edge.participants.types import Box, FramePerception, PlayerDetection


def candidate(key: int, team: str, x: float, y: float = 0.0) -> MarkingCandidate:
    return MarkingCandidate(key=key, team=team, position=(x, y))  # type: ignore[arg-type]


def test_par_sai_quando_os_dois_sao_o_mais_proximo_um_do_outro():
    pares = pair_markings(
        [candidate(1, "team_a", 0.0), candidate(2, "team_b", 2.0)],
        radius=3.0,
    )
    assert pares[0].team_a_key == 1
    assert pares[0].team_b_key == 2
    assert pares[0].distance == pytest.approx(2.0)


def test_adversario_fora_do_raio_nao_vira_par():
    assert pair_markings([candidate(1, "team_a", 0.0), candidate(2, "team_b", 9.0)]) == ()


def test_reciprocidade_recusa_o_triangulo():
    """O zagueiro tem o atacante como mais proximo, e o atacante tem outro zagueiro."""
    pares = pair_markings(
        [
            candidate(1, "team_a", 0.0),
            candidate(2, "team_b", 1.0),
            candidate(3, "team_a", 1.5),
        ],
        radius=5.0,
    )
    # 2 esta mais perto de 3 do que de 1, entao o par de 1 nao fecha.
    assert [(p.team_a_key, p.team_b_key) for p in pares] == [(3, 2)]


def test_um_zagueiro_nao_marca_a_area_inteira():
    """O caso que o limiar puro erra: um contra tres sai como um par so."""
    pares = pair_markings(
        [
            candidate(1, "team_a", 0.0),
            candidate(2, "team_b", 1.0),
            candidate(3, "team_b", 2.0),
            candidate(4, "team_b", 2.5),
        ],
        radius=3.0,
    )
    assert len(pares) == 1
    assert pares[0].team_b_key == 2


def test_cada_jogador_entra_em_no_maximo_um_par():
    pares = pair_markings(
        [
            candidate(1, "team_a", 0.0),
            candidate(2, "team_b", 1.0),
            candidate(3, "team_a", 20.0),
            candidate(4, "team_b", 21.0),
        ],
        radius=3.0,
    )
    chaves = [k for p in pares for k in (p.team_a_key, p.team_b_key)]
    assert sorted(chaves) == [1, 2, 3, 4]


def test_empate_exato_desempata_pela_menor_chave():
    """Sem desempate o par mudaria com a ordem das caixas."""
    quadro = [
        candidate(1, "team_a", 0.0),
        candidate(9, "team_b", 0.0, 2.0),
        candidate(5, "team_b", 0.0, -2.0),
    ]
    escolhido = pair_markings(quadro, radius=3.0)[0].team_b_key
    assert escolhido == 5
    assert pair_markings(list(reversed(quadro)), radius=3.0)[0].team_b_key == escolhido


def test_time_sozinho_no_quadro_nao_produz_par_e_nao_estoura():
    assert pair_markings([candidate(1, "team_a", 0.0), candidate(2, "team_a", 1.0)]) == ()
    assert pair_markings([]) == ()


def test_saida_ordenada_por_chave_do_time_a():
    pares = pair_markings(
        [
            candidate(7, "team_a", 0.0),
            candidate(8, "team_b", 1.0),
            candidate(2, "team_a", 30.0),
            candidate(3, "team_b", 31.0),
        ],
        radius=3.0,
    )
    assert [p.team_a_key for p in pares] == [2, 7]


def test_chave_repetida_no_quadro_e_erro():
    with pytest.raises(MarkingError, match="duas vezes"):
        pair_markings([candidate(1, "team_a", 0.0), candidate(1, "team_b", 1.0)])


def test_raio_nao_positivo_e_erro():
    with pytest.raises(MarkingError, match="positivo"):
        pair_markings([candidate(1, "team_a", 0.0)], radius=0.0)


def test_marcado_e_quem_aparece_em_algum_par():
    pares = pair_markings(
        [
            candidate(1, "team_a", 0.0),
            candidate(2, "team_b", 1.0),
            candidate(3, "team_a", 50.0),
        ],
        radius=3.0,
    )
    assert marked_keys(pares) == frozenset({1, 2})
    assert 3 not in marked_keys(pares)


def test_distancia_e_euclidiana_nos_dois_eixos():
    pares = pair_markings(
        [candidate(1, "team_a", 0.0, 0.0), candidate(2, "team_b", 3.0, 4.0)],
        radius=5.0,
    )
    assert pares[0].distance == pytest.approx(5.0)


# --- leitura do identities.json e montagem dos candidatos ---------------------------------


def identidade(track_id: int, team: str | None, status: str = "available") -> dict[str, object]:
    return {"track_id": track_id, "team": team, "status": status}


def test_so_entra_track_com_equipe_medida():
    equipes = teams_from_identities(
        {
            "tracks": [
                identidade(1, "team_a"),
                identidade(2, "team_b"),
                identidade(3, "team_a", status="unavailable"),
                identidade(4, None, status="not_applicable"),
            ]
        }
    )
    assert equipes == {1: "team_a", 2: "team_b"}


def test_documento_sem_tracks_nao_estoura():
    assert teams_from_identities({}) == {}


def test_team_fora_do_vocabulario_fica_de_fora():
    """Equipe fora do vocabulario nao pode virar adversario de ninguem."""
    assert teams_from_identities({"tracks": [identidade(1, "team_c")]}) == {}


def deteccao(
    track_id: int | None,
    role: str,
    x1: float = 0.0,
    x2: float = 10.0,
    y2: float = 100.0,
) -> PlayerDetection:
    return PlayerDetection(
        box=Box(x1=x1, y1=0.0, x2=x2, y2=y2),
        confidence=0.9,
        role=role,
        track_id=track_id,
    )


def quadro(*detections: PlayerDetection) -> FramePerception:
    return FramePerception(frame_index=0, timestamp_s=0.0, players=detections)


def test_posicao_e_o_contato_com_o_gramado_e_nao_o_centro():
    cands = candidates_from_frame(
        quadro(deteccao(1, "player", x1=0.0, x2=10.0, y2=100.0)),
        {1: "team_a"},
    )
    assert cands[0].position == (5.0, 100.0)


def test_goleiro_e_arbitro_nao_entram_na_conta():
    cands = candidates_from_frame(
        quadro(deteccao(1, "player"), deteccao(2, "goalkeeper"), deteccao(3, "referee")),
        {1: "team_a", 2: "team_b", 3: "team_b"},
    )
    assert [c.key for c in cands] == [1]


def test_jogador_sem_track_id_ou_sem_equipe_fica_de_fora():
    cands = candidates_from_frame(
        quadro(deteccao(None, "player"), deteccao(9, "player"), deteccao(1, "player")),
        {1: "team_a"},
    )
    assert [c.key for c in cands] == [1]


def test_projecao_troca_a_unidade_da_posicao():
    """Com homografia o ponto sai em metros; sem ela, em pixel."""
    cands = candidates_from_frame(
        quadro(deteccao(1, "player", x1=0.0, x2=200.0, y2=1000.0)),
        {1: "team_a"},
        project=lambda ponto: (ponto[0] / 100, ponto[1] / 100),
    )
    assert cands[0].position == (1.0, 10.0)
