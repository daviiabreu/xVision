"""Demo descartavel: mostra a saida das duas features implementadas."""

from __future__ import annotations

import importlib.util
import pathlib

from contracts import Clip, Perception
from edge.graph.runner import assemble
from edge.modules.base import ModuleOutput
from edge.modules.identity import TeamIdentity
from edge.testing.pitch import clip_truth

# Carrega o fixture por caminho: `tests/edge/...` colidiria com o pacote real `edge` no sys.path.
_caminho = pathlib.Path(__file__).parent / "tests/edge/modules/identity/test_module.py"
_spec = importlib.util.spec_from_file_location("_fixture_identity", _caminho)
assert _spec is not None and _spec.loader is not None
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
CAIXAS, _batch, _inputs = _fixture.CAIXAS, _fixture._batch, _fixture._inputs


def clipe() -> Clip:
    g = clip_truth()
    return Clip(
        clip_id="c",
        path="x.mp4",
        fps=float(g.fps),
        frame_count=g.frame_count,
        width=g.width,
        height=g.height,
    )


print("=" * 68)
print("FEATURE 1: identity acumula cor por trajetoria e classifica equipes")
print("=" * 68)

modulo = TeamIdentity()
modulo.load()

for indice in range(3):
    saida = modulo.infer(_batch(indice), _inputs(indice))  # pyright: ignore[reportArgumentType]
    (registros,) = saida.records
    print(f"\nquadro {indice}  (amostras acumuladas por track ate aqui: {indice + 1})")
    for r in registros:
        team = r.get("team")
        waiting = "" if team else "   <- ainda sem amostras suficientes"
        print(f"   track_id={r['track_id']}  team={team!r}{waiting}")

print("\n>> Os dois torsos sao azul e vermelho; identity separa em team_a / team_b.")

print()
print("=" * 68)
print("FEATURE 2: o runner projeta o time de identity na percepcao")
print("=" * 68)


def percepcao_com(identity: dict[str, object]) -> Perception:
    return assemble(
        clipe(),
        {
            "players": ModuleOutput(
                kind="players",
                frames=(0,),
                records=(({"box": CAIXAS[0], "confidence": 0.9, "team": "detector"},),),
            ),
            "tracking": ModuleOutput(
                kind="tracking",
                frames=(0,),
                records=(({"detection_index": 0, "track_id": 7},),),
            ),
            "identity": ModuleOutput(kind="identity", frames=(0,), records=((identity,),)),
        },
    )


p1 = percepcao_com({"track_id": 7, "team": "team_b"})
print("\ncaso A - identity ja classificou:")
print('   detector diz "detector", identity diz "team_b"')
print(f"   -> percepcao: team={p1.frames[0].players[0].team!r}   (identity vence)")

p2 = percepcao_com({"track_id": 7})
print("\ncaso B - identity ainda sem amostras (sem a chave `team`):")
print('   detector diz "detector", identity nao opina')
team_fallback = p2.frames[0].players[0].team
print(f"   -> percepcao: team={team_fallback!r}   (fallback preservado, nao virou None)")

modulo.close()
print("\ncaso C - apos close(), a evidencia do clipe anterior e descartada:")
modulo.load()
saida = modulo.infer(_batch(0), _inputs(0))  # pyright: ignore[reportArgumentType]
(registros,) = saida.records
print(f"   primeiro quadro do novo clipe -> team={registros[0].get('team')!r} (recomeca do zero)")
