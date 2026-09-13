# TeamIdentity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classificar jogadores rastreados em `team_a` ou `team_b` durante o pipeline e preencher `PlayerDetection.team` com essa identidade temporal.

**Architecture:** Criar `TeamIdentity` como módulo de percepção com dependências em `players` e `tracking`. A instância acumula `TrackObservation` e cores LAB por `track_id` entre batches de um mesmo clipe; depois de cada batch usa `teams_from_color_samples` para produzir registros opcionais de `team`. O runner projeta esse dado nos jogadores montados, preferindo a identidade temporal ao campo `team` do detector.

**Tech Stack:** Python 3.12, NumPy, OpenCV, pytest, arquitetura `PerceptionModule` do xvision.

**Spec:** `docs/superpowers/specs/2026-09-08-team-identity-design.md`

## Global Constraints

- Implementar somente identificação de equipes; `jersey_number`, `marked`, `assist` e `occupied_area` não entram nesta entrega.
- Reusar `extract_shirt_color` e `teams_from_color_samples` de `edge.participants.identity`; não duplicar clustering de uniformes.
- Estado de cores e observações pertence somente a uma execução de clipe e é descartado em `close()`.
- `identity` continua com contrato mínimo de `track_id`; `team` é opcional.
- `config/pipeline.toml` continua com `StubIdentity`; somente `config/pipeline.models.toml` liga a implementação real.

---

## File Structure

- Create: `src/edge/modules/identity/module.py` — implementação `TeamIdentity`, conversão de saídas de `players`/`tracking`, armazenamento de amostras e emissão de registros `identity`.
- Create: `src/edge/modules/identity/__init__.py` — API pública do módulo com `TeamIdentity`.
- Create: `src/tests/edge/modules/identity/test_module.py` — testes de inferência temporal, registros incompletos e limpeza de estado.
- Modify: `src/edge/graph/runner.py` — coletar times de `identity` e usá-los ao construir `PlayerDetection`.
- Modify: `src/tests/edge/modules/test_records.py` — testes de `assemble()` para precedência e fallback de `team`.
- Modify: `src/config/pipeline.models.toml` — substituir `StubIdentity` por `TeamIdentity` no pipeline de modelos.

### Task 1: Criar o módulo de identidade temporal

**Files:**
- Create: `src/edge/modules/identity/module.py`
- Create: `src/edge/modules/identity/__init__.py`
- Test: `src/tests/edge/modules/identity/test_module.py`

**Interfaces:**
- Consumes: `FrameBatch`, `ModuleInputs`, `ModuleOutput`, `ModuleSpec`, `PreprocessSpec`, `check_preprocess`, `ModuleNotLoaded`; `TrackObservation`, `extract_shirt_color`, `teams_from_color_samples`; outputs `players` com `box`, `confidence`, `role` opcional e `tracking` com `detection_index`, `track_id`.
- Produces: `class TeamIdentity` com `spec.kind == "identity"`, `load()`, `infer(batch, ctx) -> ModuleOutput` e `close()`; cada registro é `{"track_id": int}` ou `{"track_id": int, "team": "team_a" | "team_b"}`.

- [ ] **Step 1: Write the failing tests for temporal classification and reset**

Create `src/tests/edge/modules/identity/test_module.py` with synthetic BGR batches in which each player torso has a solid color and tracking maps detection positions to tracks. Include this behavioral core:

```python
def test_identity_accumulates_color_samples_between_batches() -> None:
    module = TeamIdentity(minimum=2)
    module.load()
    first = module.infer(batch((0,), red_and_blue_frame()), inputs((0,), (1, 2)))
    second = module.infer(batch((1,), red_and_blue_frame()), inputs((1,), (1, 2)))

    assert all("team" not in item for item in first.records[0])
    teams = {item["track_id"]: item["team"] for item in second.records[0]}
    assert set(teams) == {1, 2}
    assert set(teams.values()) == {"team_a", "team_b"}


def test_identity_close_discards_evidence_from_previous_clip() -> None:
    module = TeamIdentity(minimum=2)
    module.load()
    module.infer(batch((0,), red_and_blue_frame()), inputs((0,), (1, 2)))
    module.infer(batch((1,), red_and_blue_frame()), inputs((1,), (1, 2)))
    module.close()
    module.load()

    output = module.infer(batch((0,), red_and_blue_frame()), inputs((0,), (1, 2)))

    assert all("team" not in item for item in output.records[0])
```

Also add tests that a missing matching tracking record and an unextractable shirt crop do not make `infer()` fail and still return valid per-frame `ModuleOutput`.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest src/tests/edge/modules/identity/test_module.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'edge.modules.identity'`.

- [ ] **Step 3: Implement the minimal TeamIdentity module**

Create `src/edge/modules/identity/module.py`. Define a preprocess spec with BGR pixels and a `ModuleSpec` whose dependencies are exactly `("players", "tracking")` and whose `kind` is `"identity"`. Implement the state lifecycle and inference shape:

```python
class TeamIdentity:
    spec = ModuleSpec(
        kind="identity",
        name="team-color",
        version="0.1.0",
        frames="window",
        preprocess=PREPROCESS,
        needs=("players", "tracking"),
    )

    def __init__(self, *, limit: int = DEFAULT_SAMPLE_LIMIT, minimum: int = 2) -> None:
        self._limit = limit
        self._minimum = minimum
        self._samples: dict[int, list[tuple[TrackObservation, LabColor]]] = {}
        self._pronto = False

    def load(self) -> None:
        self._samples = {}
        self._pronto = True

    def close(self) -> None:
        self._samples = {}
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        # Associar cada caixa ao track_id pelo detection_index, acumular cor válida e então
        # retornar registros de cada track visto no frame; anexar team somente se classificado.
```

Use o tensor BGR do batch para `extract_shirt_color`. Converter `box` para `Box`, `confidence` para `float` e preservar `role` como `str | None` ao montar `PlayerDetection` usado em `TrackObservation`. Usar `teams_from_color_samples(self._samples, limit=self._limit, minimum=self._minimum)` depois de acumular o batch. Não emitir registros para detecções sem `track_id`.

Create `src/edge/modules/identity/__init__.py`:

```python
"""A especie `identity`: classificacao temporal de equipes por cor de uniforme."""

from edge.modules.identity.module import TeamIdentity

__all__ = ["TeamIdentity"]
```

- [ ] **Step 4: Run the focused module tests**

Run: `pytest src/tests/edge/modules/identity/test_module.py -v`

Expected: PASS.

- [ ] **Step 5: Run module conformance tests**

Run: `pytest src/tests/edge/modules/test_inference.py src/tests/edge/modules/test_registry.py src/tests/edge/modules/test_records.py -v`

Expected: PASS; the new module honors the `PerceptionModule` protocol and `identity` records retain `track_id`.

- [ ] **Step 6: Commit the module implementation**

```bash
git add src/edge/modules/identity src/tests/edge/modules/identity
git commit -m "feat(identity): classifica equipes no pipeline"
```

### Task 2: Projetar a identidade de time na percepção montada

**Files:**
- Modify: `src/edge/graph/runner.py:342-380`
- Modify: `src/tests/edge/modules/test_records.py:141-163`

**Interfaces:**
- Consumes: registros `identity` de formato `{"track_id": int, "team": "team_a" | "team_b"}` opcional; `players` pode conter `team` opcional.
- Produces: `PlayerDetection.team` igual ao time de `identity` quando presente, senão igual ao valor de fallback de `players`.

- [ ] **Step 1: Write failing tests for identity-team precedence and detector fallback**

Extend `src/tests/edge/modules/test_records.py` with:

```python
def test_identity_team_overrides_team_emitted_by_player_detector() -> None:
    perception = assemble(
        clipe(),
        {
            "players": ModuleOutput(
                kind="players", frames=(0,),
                records=(({"box": (0.0, 0.0, 1.0, 1.0), "confidence": 0.9, "team": "detector"},),),
            ),
            "tracking": ModuleOutput(
                kind="tracking", frames=(0,), records=(({"detection_index": 0, "track_id": 7},),),
            ),
            "identity": ModuleOutput(
                kind="identity", frames=(0,), records=(({"track_id": 7, "team": "team_b"},),),
            ),
        },
    )

    assert perception.frames[0].players[0].team == "team_b"


def test_player_team_is_preserved_when_identity_has_no_team() -> None:
    # Mesmo setup, mas identity contém somente {"track_id": 7}.
    assert perception.frames[0].players[0].team == "detector"
```

- [ ] **Step 2: Run the two assembly tests to verify failure**

Run: `pytest src/tests/edge/modules/test_records.py -k "identity_team_overrides or player_team_is_preserved" -v`

Expected: the precedence test FAILS because `runner._player()` currently only reads `registro.get("team")`.

- [ ] **Step 3: Extend runner assembly without changing the public model**

In `src/edge/graph/runner.py`, replace the jersey-only identity lookup with separate mappings by `track_id`:

```python
camisas = {
    int(cast("Any", r["track_id"])): int(cast("Any", r["jersey_number"]))
    for r in por_quadro.get("identity", {}).get(indice, ())
    if "jersey_number" in r
}
times = {
    int(cast("Any", r["track_id"])): cast("Any", r["team"])
    for r in por_quadro.get("identity", {}).get(indice, ())
    if "team" in r
}
```

Pass `times` to `_player()`, add a `times: Mapping[int, object]` parameter, and set:

```python
team=times.get(track_id, registro.get("team")),
```

Do not change `PlayerDetection`, `REQUIRED["identity"]`, or the stub behavior.

- [ ] **Step 4: Run the focused assembly tests**

Run: `pytest src/tests/edge/modules/test_records.py -k "identity_team_overrides or player_team_is_preserved" -v`

Expected: PASS.

- [ ] **Step 5: Run graph and record regression tests**

Run: `pytest src/tests/edge/graph/test_runner.py src/tests/edge/graph/test_integration.py src/tests/edge/modules/test_records.py -v`

Expected: PASS.

- [ ] **Step 6: Commit runner projection changes**

```bash
git add src/edge/graph/runner.py src/tests/edge/modules/test_records.py
git commit -m "feat(identity): aplica time na percepcao"
```

### Task 3: Ligar o módulo real ao pipeline de modelos

**Files:**
- Modify: `src/config/pipeline.models.toml:61-63`
- Test: `src/tests/edge/modules/test_registry.py`

**Interfaces:**
- Consumes: `TeamIdentity` exportado em `edge.modules.identity`.
- Produces: o carregamento de `config/pipeline.models.toml` cria `TeamIdentity` para `modules.identity` sem alterar o pipeline stub.

- [ ] **Step 1: Write the failing configuration-resolution test**

In `src/tests/edge/modules/test_registry.py`, add a test that carrega `config/pipeline.models.toml` pelos utilitários de registry existentes e verifica:

```python
assert isinstance(modules["identity"], TeamIdentity)
assert modules["identity"].spec.needs == ("players", "tracking")
```

- [ ] **Step 2: Run the configuration-resolution test to verify failure**

Run: `pytest src/tests/edge/modules/test_registry.py -k "models and identity" -v`

Expected: FAIL because the TOML still resolve `edge.modules.stubs:StubIdentity`.

- [ ] **Step 3: Point the models configuration at TeamIdentity**

Change this TOML section only:

```toml
[modules.identity]
impl   = "edge.modules.identity:TeamIdentity"
frames = "dense"
```

Keep `config/pipeline.toml` unchanged and update the comment near the top of `pipeline.models.toml` to state that only `field` remains stub.

- [ ] **Step 4: Run registry and focused identity tests**

Run: `pytest src/tests/edge/modules/test_registry.py src/tests/edge/modules/identity/test_module.py -v`

Expected: PASS.

- [ ] **Step 5: Run the complete affected suite**

Run: `pytest src/tests/edge/modules src/tests/edge/graph src/tests/edge/participants/test_identity.py -v`

Expected: PASS.

- [ ] **Step 6: Commit configuration and tests**

```bash
git add src/config/pipeline.models.toml src/tests/edge/modules/test_registry.py
git commit -m "config: ativa identidade de equipes"
```

## Plan Self-Review

- **Spec coverage:** Task 1 implements the real stateful identity module, its use of existing shirt-color logic, tolerance to incomplete evidence, and reset between clips. Task 2 applies the optional identity result to `PlayerDetection.team` with fallback. Task 3 activates only the production-model configuration while preserving stubs in the base config.
- **Placeholder scan:** No TBD/TODO placeholders or undefined next steps remain.
- **Type consistency:** The module returns `ModuleOutput(kind="identity")`; all output records preserve mandatory `track_id`. The runner maps `track_id` to optional `team` and passes it to `_player` without changing public perception contracts.
