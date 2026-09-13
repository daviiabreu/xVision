"""O checkpoint que faz uma execucao longa poder ser retomada.

Os defeitos que estes testes replantam nao levantam excecao na hora: eles produzem um
`events.jsonl` com clipe repetido, ou um `run.json` cujo tempo por etapa cobre metade da execucao,
ou um JSON invalido no meio do arquivo que o avaliador so encontra horas depois. Todos passam pela
execucao com codigo de saida zero.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest

from mlops.checkpoint import (
    CHECKPOINT_FILE,
    Checkpoint,
    CheckpointError,
    read_checkpoint,
    truncate_to,
    write_checkpoint,
)


def _estado(**ajustes: object) -> Checkpoint:
    base = Checkpoint(config_sha256="abc123", population=10, started_at="2026-09-07T00:00:00Z")
    for nome, valor in ajustes.items():
        setattr(base, nome, valor)
    return base


class TestRoundTrip:
    def test_o_que_foi_escrito_volta_igual(self, tmp_path: Path) -> None:
        estado = _estado(
            selection_sha256="s" * 64,
            features_sha256="f" * 64,
            done=["a", "b", "c"],
            ok=2,
            failed=1,
            resumed=3,
            origens={"ball-break": 2},
            fontes={"a": "ball-break", "b": "ball-break"},
            tempos={"ball": [1.5, 2.5], "field": [1.0, 2.0]},
            identidade={"ball": ["yolo-ball", "0.1.0"], "field": ["yolo-field", "0.1.1"]},
            falhas={"field": [1, "insufficient-inliers"]},
        )

        write_checkpoint(tmp_path, estado)
        volta = read_checkpoint(tmp_path)

        assert volta == estado

    def test_a_escrita_passa_por_temporario_e_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem `replace`, ser interrompido gravando em cima do checkpoint bom deixaria a execucao
        sem nenhum checkpoint valido, que e pior do que nao ter: a pessoa acredita que tem.

        O teste observa o mecanismo, e nao so o resultado: `write_text` direto no destino final
        produziria exatamente o mesmo arquivo e passaria por uma assercao de existencia.
        """
        trocas: list[tuple[str, str]] = []
        original = Path.replace

        def espia(self: Path, alvo: Any) -> Path:
            trocas.append((self.name, Path(alvo).name))
            return original(self, alvo)

        monkeypatch.setattr(Path, "replace", espia)

        write_checkpoint(tmp_path, _estado(ok=1))

        assert trocas == [("progress.json.parcial", CHECKPOINT_FILE)]
        assert (tmp_path / CHECKPOINT_FILE).is_file()
        assert not (tmp_path / "progress.json.parcial").exists()

    def test_o_corte_do_parcial_tambem_passa_por_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Este arquivo vale mais do que o checkpoint: e o unico lugar onde os eventos estao."""
        parcial = tmp_path / "events.jsonl.parcial"
        parcial.write_text(
            "".join(json.dumps({"clip_id": str(i)}) + "\n" for i in range(4)), encoding="utf-8"
        )
        trocas: list[str] = []
        original = Path.replace

        def espia(self: Path, alvo: Any) -> Path:
            trocas.append(self.name)
            return original(self, alvo)

        monkeypatch.setattr(Path, "replace", espia)

        truncate_to(parcial, 2)

        assert trocas == ["events.jsonl.parcial.novo"]

    def test_eventos_escritos_conta_sucesso_e_nao_clipe_visto(self) -> None:
        """Clipe que falhou entra em `done`, para nao ser reprocessado, mas nao escreveu evento."""
        estado = _estado(done=["a", "b", "c"], ok=2, failed=1)

        assert estado.events_written == 2


class TestLeitura:
    @pytest.mark.parametrize("document", [None, [], 3, "checkpoint"])
    def test_non_object_json_is_a_checkpoint_error(self, tmp_path: Path, document: object) -> None:
        (tmp_path / CHECKPOINT_FILE).write_text(json.dumps(document))

        with pytest.raises(CheckpointError, match="progress.json"):
            read_checkpoint(tmp_path)

    def test_ausencia_de_checkpoint_e_erro_explicito(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="nao tem checkpoint para retomar"):
            read_checkpoint(tmp_path)

    def test_json_invalido_e_erro_explicito(self, tmp_path: Path) -> None:
        (tmp_path / CHECKPOINT_FILE).write_text("{nao e json", encoding="utf-8")

        with pytest.raises(CheckpointError, match="JSON invalido"):
            read_checkpoint(tmp_path)

    def test_schema_de_outra_versao_nao_e_lido_como_se_fosse_desta(self, tmp_path: Path) -> None:
        (tmp_path / CHECKPOINT_FILE).write_text(
            json.dumps({"schema_version": "0.9", "ok": 1}), encoding="utf-8"
        )

        with pytest.raises(CheckpointError, match="esperado '1.0'"):
            read_checkpoint(tmp_path)

    def test_campo_obrigatorio_faltando_nao_vira_zero_calado(self, tmp_path: Path) -> None:
        (tmp_path / CHECKPOINT_FILE).write_text(
            json.dumps({"schema_version": "1.0", "ok": 1}), encoding="utf-8"
        )

        with pytest.raises(CheckpointError, match="sem os campos"):
            read_checkpoint(tmp_path)

    def test_agregado_ausente_volta_vazio_e_nao_quebra(self, tmp_path: Path) -> None:
        (tmp_path / CHECKPOINT_FILE).write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "config_sha256": "a",
                    "population": 1,
                    "started_at": "z",
                    "done": [],
                    "ok": 0,
                }
            ),
            encoding="utf-8",
        )

        volta = read_checkpoint(tmp_path)

        assert volta.tempos == {} and volta.origens == {} and volta.resumed == 0


class TestTruncate:
    @pytest.mark.parametrize(
        "records",
        [
            [{"clip_id": "a"}, {"clip_id": "foreign"}],
            [{"clip_id": "a"}, {"clip_id": "a"}],
            [{"clip_id": "a"}, []],
            [{"clip_id": "a"}, {"clip_id": 2}],
        ],
    )
    def test_incompatible_prefix_preserves_the_partial(
        self, tmp_path: Path, records: list[object]
    ) -> None:
        partial = self._parcial(tmp_path, [json.dumps(record) + "\n" for record in records])
        before = partial.read_bytes()

        with pytest.raises(CheckpointError, match="prefixo"):
            truncate_to(partial, 2, expected_clips={"a", "b"})

        assert partial.read_bytes() == before

    def test_only_confirmed_prefix_must_match_checkpoint(self, tmp_path: Path) -> None:
        partial = self._parcial(
            tmp_path,
            [json.dumps({"clip_id": clip}) + "\n" for clip in ("a", "b", "unconfirmed")],
        )

        assert truncate_to(partial, 2, expected_clips={"a", "b"}) == 2
        assert [json.loads(line)["clip_id"] for line in partial.read_text().splitlines()] == [
            "a",
            "b",
        ]

    def _parcial(self, tmp_path: Path, linhas: list[str]) -> Path:
        caminho = tmp_path / "events.jsonl.parcial"
        caminho.write_text("".join(linhas), encoding="utf-8")
        return caminho

    def test_corta_o_evento_escrito_depois_do_checkpoint(self, tmp_path: Path) -> None:
        """O evento e valido, mas o checkpoint nao tem o tempo de etapa dele: manter faria o
        relatorio somar um clipe que ninguem consegue conferir."""
        parcial = self._parcial(
            tmp_path, [json.dumps({"clip_id": str(i)}) + "\n" for i in range(5)]
        )

        assert truncate_to(parcial, 3) == 3
        assert [json.loads(x)["clip_id"] for x in parcial.read_text().splitlines()] == [
            "0",
            "1",
            "2",
        ]

    def test_descarta_ultima_linha_truncada(self, tmp_path: Path) -> None:
        """Queda no meio da escrita. Anexar depois dela poria JSON invalido no meio do arquivo."""
        parcial = self._parcial(
            tmp_path,
            [json.dumps({"clip_id": "0"}) + "\n", json.dumps({"clip_id": "1"}) + "\n", '{"cli'],
        )

        assert truncate_to(parcial, 2) == 2
        assert parcial.read_text().count("\n") == 2
        for linha in parcial.read_text().splitlines():
            json.loads(linha)

    def test_recusa_parcial_com_menos_eventos_do_que_o_checkpoint_declara(
        self, tmp_path: Path
    ) -> None:
        parcial = self._parcial(tmp_path, [json.dumps({"clip_id": "0"}) + "\n"])

        with pytest.raises(CheckpointError, match="foi truncado ou substituido"):
            truncate_to(parcial, 4)

    def test_parcial_ausente_com_checkpoint_em_zero_e_valido(self, tmp_path: Path) -> None:
        assert truncate_to(tmp_path / "events.jsonl.parcial", 0) == 0

    def test_parcial_ausente_com_checkpoint_positivo_e_erro(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="nao existe, mas o checkpoint declara"):
            truncate_to(tmp_path / "events.jsonl.parcial", 3)

    def test_linha_vazia_no_meio_nao_conta_como_evento(self, tmp_path: Path) -> None:
        parcial = self._parcial(
            tmp_path,
            [json.dumps({"clip_id": "0"}) + "\n", "\n", json.dumps({"clip_id": "1"}) + "\n"],
        )

        assert truncate_to(parcial, 2) == 2


def test_todo_campo_do_checkpoint_sobrevive_ao_disco(tmp_path: Path) -> None:
    """Meta-teste: campo novo no dataclass que ninguem serializa volta com o padrao, em silencio.

    Foi o que aconteceu ao acrescentar `features_sha256`: o campo existia no dataclass, a guarda
    da retomada o comparava, e ele voltava vazio do disco porque `to_dict` e `read_checkpoint` nao
    o conheciam. Guarda que compara string vazia nao recusa nada, e a retomada voltava a aceitar
    tudo. O erro nao levanta e nao muda nenhuma contagem: ele so desliga uma verificacao.
    """
    campos = {f.name for f in fields(Checkpoint)}
    povoado = Checkpoint(
        config_sha256="c" * 64,
        population=7,
        started_at="2026-09-07T00:00:00.000Z",
        selection_sha256="s" * 64,
        features_sha256="f" * 64,
        done=["a", "b", "c"],
        ok=2,
        failed=1,
        resumed=3,
        origens={"ball-break": 2},
        fontes={"a": "ball-break", "b": "ball-break"},
        apoios={"a": "detected", "b": "detected"},
        tempos={"ball": [1.0, 2.0]},
        identidade={"ball": ["yolo-ball", "0.1.0"]},
        falhas={"ball": [1, "motivo"]},
    )
    assert campos == set(asdict(povoado)), "o dataclass ganhou campo que este teste nao povoa"

    write_checkpoint(tmp_path, povoado)
    de_volta = read_checkpoint(tmp_path)

    for campo in sorted(campos):
        esperado = getattr(povoado, campo)
        obtido = getattr(de_volta, campo)
        assert obtido == esperado, f"{campo} nao sobreviveu: {obtido!r} em vez de {esperado!r}"
