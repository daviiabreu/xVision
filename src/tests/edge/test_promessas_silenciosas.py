"""Quatro promessas que, quebradas, mudam o numero sem mudar o relatorio.

Todas passaram por um teste de mutacao sem derrubar nada, e todas tem a mesma forma: o pipeline
continua reportando `ok`, o evento continua validando contra o schema, e o valor esta errado.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts import Clip
from edge.artifacts import read_manifest, sha256_of
from edge.frames.provider import MissingFrame, VideoSource
from edge.graph.runner import assemble
from edge.modules.base import ModuleOutput, PreprocessSpec


def _clipe() -> Clip:
    return Clip(clip_id="c", path="x.mp4", fps=30.0, frame_count=1, width=100, height=100)


def _saida(kind: str, registro: dict[str, object]) -> ModuleOutput:
    return ModuleOutput(kind=kind, frames=(0,), records=((registro,),))


def test_metres_never_survive_a_field_without_homography() -> None:
    """O runner declara isto como nao negociavel, e nenhum teste construia o caso.

    Um modulo de campo que produz registro sem homografia existe: e o quadro em que a deteccao de
    linhas nao fechou. Liberando `pitch_xy` nesse quadro, a feature mede em pixel achando que mede
    em metro, e o numero sai com a mesma cara de um medido em metro de verdade.
    """
    jogador: dict[str, object] = {
        "box": (0.0, 0.0, 10.0, 10.0),
        "confidence": 0.9,
        "pitch_xy": (12.0, 30.0),
    }

    com_campo_cego = assemble(
        _clipe(),
        {"players": _saida("players", jogador), "field": _saida("field", {"confidence": 0.5})},
    )

    assert com_campo_cego.frames[0].players[0].pitch_xy is None, (
        "coordenada em metros sobreviveu num quadro sem homografia"
    )


def test_a_transposed_preprocess_is_not_the_same_fingerprint() -> None:
    """A impressao digital entra na chave do cache, e 640x480 nao e 480x640.

    Iguais, uma spec transposta serve resultado computado noutra resolucao, e o unico sinal seria
    a deteccao sair deslocada.
    """
    deitado = PreprocessSpec(width=640, height=480, color="rgb", letterbox=False, normalize="unit")
    de_pe = PreprocessSpec(width=480, height=640, color="rgb", letterbox=False, normalize="unit")

    assert deitado.fingerprint != de_pe.fingerprint


def test_the_weight_digest_does_not_depend_on_the_order_of_the_files(tmp_path: Path) -> None:
    """O digest e o endereco do peso. Dependendo da ordem, reordenar blocos invalida todo o cache.

    Nenhum peso teria mudado, e a proxima execucao recomputaria os 633 clipes sem motivo.
    """
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"peso a")
    b.write_bytes(b"peso b")

    def manifesto(primeiro: Path, segundo: Path) -> str:
        return "\n".join(
            [
                'name = "m"',
                'version = "1"',
                'kind = "players"',
                'entrypoint = "a.bin"',
                "[[files]]",
                f'path = "{primeiro.name}"',
                f'sha256 = "{sha256_of(primeiro)}"',
                "[[files]]",
                f'path = "{segundo.name}"',
                f'sha256 = "{sha256_of(segundo)}"',
            ]
        )

    (tmp_path / "um.toml").write_text(manifesto(a, b), encoding="utf-8")
    (tmp_path / "dois.toml").write_text(manifesto(b, a), encoding="utf-8")

    assert (
        read_manifest(tmp_path / "um.toml").digest == read_manifest(tmp_path / "dois.toml").digest
    )


@pytest.mark.parametrize("indice", [-1, -10])
def test_a_negative_frame_index_is_refused_by_us_and_not_by_the_decoder(indice: int) -> None:
    """A guarda explicita existe para o comportamento nao depender do backend do OpenCV.

    Sem ela o indice negativo cai no ramo de busca e o erro que chega depende de qual codec esta
    instalado, o que e a definicao de comportamento nao especificado.
    """
    origem = VideoSource(Path("var/out/synthetic.mp4"))
    try:
        with pytest.raises(MissingFrame) as erro:
            origem.read((indice,))
    finally:
        origem.close()

    assert str(indice) in str(erro.value)
    assert "nao tem o quadro" in str(erro.value), (
        "a mensagem veio do decodificador, e nao da guarda: o comportamento passou a depender "
        "de qual codec esta instalado"
    )
