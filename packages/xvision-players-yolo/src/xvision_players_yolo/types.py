"""A ponte entre o contrato congelado da percepcao e o que o YOLO devolve.

`Box`, `PlayerDetection` e `FramePerception` moram em `contracts.perception` e sao o acordo entre
as pontas do sistema. Este pacote nao redeclara nenhum dos tres: redeclarar criava um segundo
vocabulario com os mesmos nomes e menos campos, e o runner monta `Perception` a partir do
primeiro, entao o segundo so existiria para ser convertido de volta.

O que sobra aqui e o que o contrato nao tem porque nao e dele: o vocabulario de papel que este
detector reconhece, a traducao de nome de classe do peso para esse vocabulario, os dois calculos
de geometria que a anotacao usa, e a serializacao para JSON, que existe porque o artefato da v0 e
arquivo em disco e dataclass nao atravessa.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Literal, TypeAlias

from contracts.perception import Box, FramePerception, PlayerDetection

Role: TypeAlias = Literal["player", "goalkeeper", "referee"]

NAME_TO_ROLE: Mapping[str, Role] = {
    "player": "player",
    "jogador": "player",
    "goalkeeper": "goalkeeper",
    "goleiro": "goalkeeper",
    "referee": "referee",
    "arbitro": "referee",
    "árbitro": "referee",
    "juiz": "referee",
}
"""Papel por nome de classe, e nao por indice.

O indice era o acoplamento silencioso entre este pacote e o peso. O detector nascera com o mapa
`{1: goalkeeper, 2: player, 3: referee}` do peso pre-treinado do `roboflow/sports`, e o
`train_player_detector.py` do repositorio treina um modelo de uma classe so, `jogador`, no indice
0. Alimentar um com o peso do outro nao levantava nada: o indice 0 caia no `continue` do filtro e
o clipe inteiro saia com zero deteccoes, sem erro, sem aviso e com o JSON bem formado. Nome bate
ou nao bate, e quando nao bate `roles_from_names` diz qual peso veio e o que ele traz dentro.
"""

IGNORED_NAMES = frozenset({"ball", "bola", "sports ball"})
"""Classes que o peso traz e esta especie nao reporta: a bola e do modulo `ball`."""


class UnknownClassNames(ValueError):
    """Nenhuma classe do peso corresponde a um papel conhecido."""


def roles_from_names(names: Mapping[int, str]) -> dict[int, Role]:
    """Traduz o `model.names` do peso carregado para o vocabulario de papeis.

    Classe desconhecida que nao seja a bola nao aborta: peso de terceiro traz classe que nao
    interessa, e recusar o peso inteiro por causa dela seria pior do que ignora-la. O que aborta e
    o peso em que *nenhuma* classe vira papel, porque ai a saida seria vazia por construcao.
    """
    mapa: dict[int, Role] = {
        indice: NAME_TO_ROLE[nome.strip().lower()]
        for indice, nome in names.items()
        if nome.strip().lower() in NAME_TO_ROLE
    }
    if not mapa:
        conhecidos = ", ".join(sorted(NAME_TO_ROLE))
        trazidas = ", ".join(f"{i}={n!r}" for i, n in sorted(names.items())) or "nenhuma"
        raise UnknownClassNames(
            f"o peso declara as classes {trazidas}, e nenhuma delas e um papel conhecido "
            f"({conhecidos}). Um peso assim detecta e o filtro descarta tudo, entao o clipe sairia "
            f"com zero jogadores e sem erro nenhum"
        )
    return mapa


def as_role(value: str | None) -> Role | None:
    """Le o `role` de uma deteccao do contrato de volta para o vocabulario deste detector.

    `PlayerDetection.role` e `str | None` porque o contrato serve toda especie de percepcao e nao
    pode fechar a lista de papeis que cada uma reconhece. Quem consome aqui precisa de `Role`, e a
    conversao tem que ser uma so: espalhar `cast` pelos modulos transformava papel desconhecido em
    `KeyError` no meio do desenho do quadro. Papel que este detector nao reconhece devolve `None`,
    e cada chamador decide se ignora a deteccao ou se pinta com a cor de desconhecido.
    """
    if value is None:
        return None
    return NAME_TO_ROLE.get(value.strip().lower())


def ground_point(box: Box) -> tuple[float, float]:
    """Centro inferior da caixa, aproximacao do contato com o gramado."""
    return ((box.x1 + box.x2) / 2, box.y2)


def clip_box(box: Box, width: int, height: int) -> Box:
    """Limita a caixa as dimensoes do quadro."""
    return Box(
        x1=min(max(box.x1, 0.0), float(width)),
        y1=min(max(box.y1, 0.0), float(height)),
        x2=min(max(box.x2, 0.0), float(width)),
        y2=min(max(box.y2, 0.0), float(height)),
    )


def player(
    box: Box,
    confidence: float,
    role: Role,
    track_id: int | None = None,
) -> PlayerDetection:
    """Monta a deteccao conferindo o que o contrato aceita e nao cobra.

    `PlayerDetection` e congelada e sem validacao de proposito: ela e o acordo, e cobrar faixa de
    valor la obrigaria toda especie a calibrar confianca. Aqui e a borda deste detector, onde a
    faixa e conhecida, entao a conferencia mora aqui.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence {confidence!r} fora de [0, 1]")
    if track_id is not None and track_id < 0:
        raise ValueError(f"track_id {track_id!r} e negativo")
    return PlayerDetection(box=box, confidence=confidence, role=role, track_id=track_id)


def frame_to_dict(frame: FramePerception) -> dict[str, object]:
    """Serializa o quadro para o artefato em disco.

    Sai `players` e nao `detections` porque a chave passou a ser a do contrato, e junto vieram
    `team`, `jersey_number` e `pitch_xy`, que este detector ainda nao preenche e reporta como
    `null` em vez de omitir: campo ausente e ambiguo entre nao medido e medido vazio.
    """
    return {
        "frame_index": frame.frame_index,
        "timestamp_s": frame.timestamp_s,
        "players": [asdict(jogador) for jogador in frame.players],
    }


__all__ = [
    "NAME_TO_ROLE",
    "Box",
    "FramePerception",
    "PlayerDetection",
    "Role",
    "UnknownClassNames",
    "as_role",
    "clip_box",
    "frame_to_dict",
    "ground_point",
    "player",
    "roles_from_names",
]
