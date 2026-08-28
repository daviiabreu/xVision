"""Cache de percepcao em disco: a segunda execucao igual nao chama modulo nenhum."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from edge.cache.perception import PerceptionCache, perception_key, run_module, tuning_of
from edge.modules.base import FrameBatch, ModuleInputs, ModuleOutput, ModuleSpec, PreprocessSpec

SPEC = ModuleSpec(
    kind="players",
    name="espia",
    version="0.1.0",
    frames="coarse",
    preprocess=PreprocessSpec(width=8, height=8, color="rgb", letterbox=False, normalize="none"),
)
CHAVE_FIXADA = "b0b38c491e6a3a544a936584ce085f426c9eb57b6594b08c1881c32e73859e7c"
"""O valor mudou uma vez, em 26/08/2026, ao entrar o ajuste do modulo na chave.

O anterior era d667512351d29a1f. Mudar a chave invalida todo o cache em disco, e nesse caso isso e
o efeito desejado e nao dano colateral: as entradas antigas foram gravadas por um detector cujo
limiar de confianca nao aparecia em endereco nenhum, entao servir qualquer uma delas seria devolver
medida de um ajuste sob o nome de outro.

O teste continua existindo pelo motivo de sempre: mudanca acidental de formato de chave nao levanta
nada, so faz a proxima execucao recalcular tudo em silencio e pagar de novo o que ja estava pronto.
"""
REGISTRO = {"box": (0.0, 0.0, 1.0, 1.0), "confidence": 0.9}
REGISTRO_JSON = {"box": [0.0, 0.0, 1.0, 1.0], "confidence": 0.9}


class ModuloEspia:
    def __init__(self, spec: ModuleSpec = SPEC) -> None:
        self.spec = spec
        self.execucoes = 0

    def load(self) -> None: ...

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        self.execucoes += 1
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=tuple((dict(REGISTRO),) for _ in batch.indices),
        )

    def close(self) -> None: ...


def lote(indices: tuple[int, ...] = (0, 1)) -> FrameBatch:
    tensor = np.zeros((len(indices), 8, 8, 3), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=SPEC.preprocess)


def chave(
    spec: ModuleSpec = SPEC,
    indices: tuple[int, ...] = (0, 1),
    clip_id: str = "clipe-1",
    ctx: ModuleInputs | None = None,
    impl: str = "impls:Modulo",
) -> str:
    return perception_key(clip_id, spec, indices, ctx or {}, impl)


def _chave_em_subprocesso(hash_seed: str) -> str:
    source = (
        "from edge.cache.perception import perception_key;"
        "from edge.modules.base import ModuleSpec, PreprocessSpec;"
        "spec = ModuleSpec(kind='players', name='espia', version='0.1.0', frames='coarse',"
        " preprocess=PreprocessSpec(width=8, height=8, color='rgb', letterbox=False,"
        " normalize='none'));"
        "print(perception_key('clipe-1', spec, (0, 1), {}, 'impls:Modulo'))"
    )
    done = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        check=True,
        text=True,
        env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
    )
    return done.stdout.strip()


def test_perception_cache_key_survives_a_change_of_process() -> None:
    assert chave() == CHAVE_FIXADA
    assert {_chave_em_subprocesso(seed) for seed in ("0", "1", "random")} == {CHAVE_FIXADA}


def test_perception_cache_key_includes_the_module_name_and_version() -> None:
    assert chave() != chave(replace(SPEC, version="0.2.0"))
    assert chave() != chave(replace(SPEC, name="outro"))


def test_perception_cache_key_includes_the_implementation() -> None:
    assert chave() != chave(impl="impls:Outro")


def test_perception_cache_key_includes_preprocess_hash() -> None:
    outro = replace(SPEC, preprocess=replace(SPEC.preprocess, ops=("bilateral:5:50:50",)))

    assert chave() != chave(outro)


def test_perception_cache_key_includes_the_clip_and_the_indices_in_order() -> None:
    assert chave() != chave(clip_id="clipe-2")
    assert chave() != chave(indices=(0, 2))
    assert chave() != chave(indices=(1, 0))


def test_perception_cache_key_includes_the_upstream_outputs_it_declared() -> None:
    consumidor = replace(SPEC, kind="field", needs=("players",))
    antes = ModuleOutput(kind="players", frames=(0, 1), records=(({"box": [0, 0, 1, 1]},), ()))
    depois = ModuleOutput(kind="players", frames=(0, 1), records=(({"box": [0, 0, 2, 2]},), ()))

    assert chave(consumidor, ctx={"players": antes}) != chave(consumidor, ctx={"players": depois})
    assert chave(consumidor, ctx={"players": antes}) != chave(consumidor)


def test_perception_cache_key_ignores_what_the_module_did_not_declare() -> None:
    consumidor = replace(SPEC, kind="field", needs=("players",))
    jogadores = ModuleOutput(kind="players", frames=(0, 1), records=((), ()))
    bola = ModuleOutput(kind="ball", frames=(0, 1), records=((), ()))

    assert chave(consumidor, ctx={"players": jogadores}) == chave(
        consumidor, ctx={"players": jogadores, "ball": bola}
    )


def test_perception_cache_key_does_not_confuse_two_fields_that_carry_a_separator() -> None:
    com_barra = replace(SPEC, name="espia|0.1.0", version="")

    assert chave() != chave(com_barra)


def test_cache_hit_skips_module_execution_entirely(tmp_path: Path) -> None:
    cache = PerceptionCache(tmp_path)
    modulo = ModuloEspia()

    primeiro = run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")
    segundo = run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")

    assert modulo.execucoes == 1
    assert primeiro.frames == segundo.frames == (0, 1)
    assert segundo.records == ((REGISTRO_JSON,), (REGISTRO_JSON,))


def test_changing_module_version_invalidates_cache(tmp_path: Path) -> None:
    cache = PerceptionCache(tmp_path)
    modulo = ModuloEspia()

    run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")
    modulo.spec = replace(SPEC, version="0.2.0")
    run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")

    assert modulo.execucoes == 2


def test_corrupt_cache_entry_is_ignored_and_recomputed(tmp_path: Path) -> None:
    cache = PerceptionCache(tmp_path)
    modulo = ModuloEspia()
    run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")

    _reescreve(tmp_path, "{isto nao e json")
    recomputado = run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")

    assert modulo.execucoes == 2
    assert recomputado.frames == (0, 1)


def _sem_registro(quadros: int) -> list[list[dict[str, object]]]:
    return [[] for _ in range(quadros)]


ADULTERACOES: list[object] = [
    pytest.param({"kind": "players", "frames": [0, 1]}, id="sem_records"),
    pytest.param(
        {"kind": "players", "frames": [0, 1], "records": _sem_registro(1)},
        id="records_de_tamanho_diferente_de_frames",
    ),
    pytest.param(
        {"kind": "ball", "frames": [0, 1], "records": _sem_registro(2)},
        id="kind_de_outra_especie",
    ),
    pytest.param(
        {"kind": "players", "frames": [0, 9], "records": _sem_registro(2)},
        id="quadros_de_outro_lote",
    ),
    pytest.param([1, 2, 3], id="json_que_nem_e_objeto"),
]


@pytest.mark.parametrize("conteudo", ADULTERACOES)
def test_entry_that_json_accepts_but_the_contract_refuses_is_recomputed(
    tmp_path: Path, conteudo: object
) -> None:
    cache = PerceptionCache(tmp_path)
    modulo = ModuloEspia()
    run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")

    _reescreve(tmp_path, json.dumps(conteudo))
    recomputado = run_module(modulo, lote(), {}, cache=cache, clip_id="clipe-1")

    assert modulo.execucoes == 2
    assert recomputado.records == ((REGISTRO,), (REGISTRO,))


def test_record_that_json_cannot_serialize_is_refused_before_it_reaches_the_cache(
    tmp_path: Path,
) -> None:
    class ModuloComNumpy(ModuloEspia):
        def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
            self.execucoes += 1
            return ModuleOutput(
                kind=self.spec.kind,
                frames=batch.indices,
                records=tuple(({"score": np.float32(0.5)},) for _ in batch.indices),
            )

    with pytest.raises(TypeError, match="float32"):
        run_module(ModuloComNumpy(), lote(), {}, cache=PerceptionCache(tmp_path), clip_id="c")


def test_clear_removes_every_entry_and_reports_how_many(tmp_path: Path) -> None:
    cache = PerceptionCache(tmp_path)
    run_module(ModuloEspia(), lote((0,)), {}, cache=cache, clip_id="clipe-1")
    run_module(ModuloEspia(), lote((1,)), {}, cache=cache, clip_id="clipe-2")

    assert cache.clear() == 2
    assert cache.clear() == 0


def test_default_root_follows_the_write_root_and_never_the_partner_dump() -> None:
    """Comparar com literal quebraria para quem exporta XVISION_VAR_DIR, que e o caso de uso."""
    from edge.paths import cache_dir, data_root

    raiz = PerceptionCache().root

    assert raiz == cache_dir()
    assert data_root(required=False).resolve() not in raiz.resolve().parents


def _reescreve(raiz: Path, conteudo: str) -> None:
    entradas = sorted(raiz.rglob("*.json"))
    assert entradas, "o cache precisa ter gravado alguma coisa antes de ser adulterado"
    for entrada in entradas:
        entrada.write_text(conteudo, encoding="utf-8")


def test_swapping_the_weight_changes_the_key_even_with_the_same_manifest_path(
    tmp_path: Path,
) -> None:
    """Era o defeito declarado como parcial desde a primeira MR: a chave usava o caminho.

    Trocar o peso dentro de `models/<slug>.toml` sem trocar o nome do arquivo mantinha a chave e
    servia saida velha, calada.
    """
    from edge.artifacts import ArtifactFile, ArtifactManifest

    def manifesto(sha: str) -> ArtifactManifest:
        return ArtifactManifest(
            kind="players",
            name="demo",
            version="0.0.1",
            entrypoint="m.onnx",
            files=(ArtifactFile(path="m.onnx", sha256=sha, bytes=4),),
        )

    spec = replace(SPEC, artifact="models/players/demo-0.0.1.toml")
    antes = perception_key("c", spec, (0,), {}, "impl", artifact_digest=manifesto("a" * 64).digest)
    depois = perception_key("c", spec, (0,), {}, "impl", artifact_digest=manifesto("b" * 64).digest)

    assert antes != depois


def test_a_module_without_an_artifact_still_has_a_stable_key() -> None:
    """Stub e regra nao tem peso, e continuar funcionando sem digest e o caso mais comum hoje."""
    uma = perception_key("c", SPEC, (0,), {}, "impl")
    outra = perception_key("c", SPEC, (0,), {}, "impl")

    assert uma == outra


def test_run_module_serves_fresh_output_when_the_weight_behind_the_manifest_changes(
    tmp_path: Path,
) -> None:
    """A revisao reproduziu: com o parametro solto, `run_module` nunca o passava e o cache
    continuava servindo saida velha. O teste anterior media o parametro, e nao a ligacao."""
    chamadas = 0

    class Contado:
        spec = replace(SPEC, artifact=str(tmp_path / "m.toml"))

        def load(self) -> None: ...

        def close(self) -> None: ...

        def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
            nonlocal chamadas
            chamadas += 1
            return ModuleOutput(
                kind=self.spec.kind, frames=batch.indices, records=((),) * len(batch.indices)
            )

    def manifesto(peso: bytes) -> None:
        (tmp_path / "pesos").mkdir(exist_ok=True)
        (tmp_path / "pesos" / "m.onnx").write_bytes(peso)
        sha = hashlib.sha256(peso).hexdigest()
        (tmp_path / "m.toml").write_text(
            'schema_version = "1.0"\nkind = "players"\nname = "m"\nversion = "1"\n'
            f'entrypoint = "m.onnx"\n\n[[files]]\npath = "m.onnx"\nsha256 = "{sha}"\n'
            f"bytes = {len(peso)}\n",
            encoding="utf-8",
        )

    cache = PerceptionCache(tmp_path / "cache")
    entrada = lote((0,))
    modulo = Contado()

    manifesto(b"peso versao um")
    run_module(modulo, entrada, {}, cache=cache, clip_id="c")
    manifesto(b"peso versao dois")
    run_module(modulo, entrada, {}, cache=cache, clip_id="c")

    assert chamadas == 2, "o cache serviu saida do peso antigo"


def test_mudar_o_ajuste_do_modulo_muda_a_chave() -> None:
    """Sem isto, baixar conf de 0,25 para 0,10 servia a percepcao antiga e media o valor de antes.

    `implementation_id` hasheia o arquivo do modulo e o digest cobre o peso, e nenhum dos dois muda
    quando o ajuste vem do TOML. A execucao seguinte devolvia o resultado do limiar velho e o
    relatorio o publicava como se fosse o novo.
    """
    from edge.modules.ball import YoloBall

    frouxo, apertado = YoloBall(), YoloBall()
    frouxo.configure({"conf": 0.10})
    apertado.configure({"conf": 0.25})

    assert tuning_of(frouxo) != tuning_of(apertado)
    assert perception_key(
        "c1", frouxo.spec, (0, 1), {}, "impl", tuning=tuning_of(frouxo)
    ) != perception_key("c1", apertado.spec, (0, 1), {}, "impl", tuning=tuning_of(apertado))


def test_modulo_sem_ajuste_nao_muda_de_endereco() -> None:
    """Modulo que nao declara ajuste tem que continuar achando o que ja esta em disco."""
    from edge.modules.stubs import STUBS

    stub = next(m() for m in STUBS)

    assert tuning_of(stub) is None


def test_a_mesma_configuracao_da_a_mesma_chave() -> None:
    from edge.modules.ball import YoloBall

    um, outro = YoloBall(), YoloBall()
    um.configure({"conf": 0.10, "imgsz": 1600})
    outro.configure({"imgsz": 1600, "conf": 0.10})

    assert tuning_of(um) == tuning_of(outro)
