"""O video anotado: de onde vem o peso, quem pode baixar, e o que quebra se o script mudar.

`test_o_script_da_visao_superior_mantem_a_forma_que_o_render_espera` le a AST em vez de
importar, porque o CI nao instala torch nem ultralytics.
"""

from __future__ import annotations

import ast
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import PREFIXO, create_app, execution, render, store
from contracts import BallDetection, Box, FramePerception, Perception
from edge import video as video_renderer
from edge.testing.make_clip import make_synthetic_clip

from .apoio import cliente_autenticado

REPOSITORIO = Path(__file__).resolve().parents[3]
PROJETO = REPOSITORIO / "src"
PIPELINE = PROJETO / "config" / "pipeline.toml"
SCRIPT = REPOSITORIO / "src" / "models" / "top_view" / "create_top_view.py"
HOMOGRAFIA = REPOSITORIO / "src" / "models" / "top_view" / "homography_view.py"

ESPERADOS: tuple[str, ...] = ("IMGSZ", "CONFIDENCE", "PLAYER_DEVICE", "OUTPUT_DIR")
ASSINATURA: tuple[str, ...] = ("tracker", "video_path", "input_dir", "homography_model")


@pytest.fixture(autouse=True)
def isolamento(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("XVISION_VAR_DIR", str(tmp_path / "var"))
    yield


@pytest.fixture
def cliente() -> TestClient:
    return cliente_autenticado(create_app())


def _modulo(caminho: Path) -> ast.Module:
    return ast.parse(caminho.read_text(encoding="utf-8"))


def _funcao(arvore: ast.Module, nome: str) -> ast.FunctionDef | None:
    return next(
        (no for no in arvore.body if isinstance(no, ast.FunctionDef) and no.name == nome), None
    )


def _atribuidos(arvore: ast.Module) -> set[str]:
    return {
        alvo.id
        for no in arvore.body
        if isinstance(no, ast.Assign)
        for alvo in no.targets
        if isinstance(alvo, ast.Name)
    }


def test_o_script_da_visao_superior_mantem_a_forma_que_o_render_espera() -> None:
    assert SCRIPT.is_file(), f"{SCRIPT} saiu do lugar, e o render o procura ali"
    arvore = _modulo(SCRIPT)

    processa = _funcao(arvore, "process_video")
    assert processa is not None, "o script nao tem mais `process_video`, que e o que o render chama"
    assert tuple(argumento.arg for argumento in processa.args.args) == ASSINATURA, (
        "a assinatura de `process_video` mudou, e o render passa os quatro na ordem antiga"
    )

    faltando = set(ESPERADOS) - _atribuidos(arvore)
    assert not faltando, f"o script nao declara mais {', '.join(sorted(faltando))}"

    fonte = SCRIPT.read_text(encoding="utf-8")
    assert "PlayerTracker" in fonte, "o script nao usa mais `PlayerTracker`"
    assert render.SUFIXO in fonte, (
        f"o nome do arquivo de saida mudou: o render procura {render.SUFIXO}"
    )


def test_o_carregador_do_modelo_de_homografia_continua_onde_o_render_o_busca() -> None:
    carrega = _funcao(_modulo(HOMOGRAFIA), "load_model")

    assert carrega is not None, "`load_model` saiu de homography_view, e o render o chama"
    assert [argumento.arg for argumento in carrega.args.args] == ["path"]


@pytest.mark.parametrize("name", ["pipeline.toml", "pipeline.models.toml"])
def test_o_pipeline_do_repositorio_declara_os_dois_pesos_do_render(name: str) -> None:
    pipeline = PROJETO / "config" / name
    configuracao = render.read_config(pipeline)

    assert configuracao is not None, f"{pipeline} nao declara [{render.SECAO}]"
    for manifesto in (configuracao.players, configuracao.field):
        assert (PROJETO / manifesto).is_file(), f"{manifesto} nao existe"


def test_sem_a_secao_o_render_fica_desligado_em_vez_de_falhar(tmp_path: Path) -> None:
    arquivo = tmp_path / "pipeline.toml"
    arquivo.write_text('[modules.field]\nimpl = "edge.modules.stubs:StubField"\n', encoding="utf-8")

    assert render.read_config(arquivo) is None


@pytest.mark.parametrize(
    ("conteudo", "esperado"),
    [
        ('[render.top_view]\nplayers = "a.toml"\n', "field"),
        ('[render.top_view]\nfield = "a.toml"\n', "players"),
        ('[render.top_view]\nplayers = 42\nfield = "a.toml"\n', "espera texto"),
    ],
)
def test_secao_pela_metade_e_erro_e_nao_ausencia(
    tmp_path: Path, conteudo: str, esperado: str
) -> None:
    arquivo = tmp_path / "pipeline.toml"
    arquivo.write_text(conteudo, encoding="utf-8")

    with pytest.raises(render.RenderUnavailable, match=esperado):
        render.read_config(arquivo)


def test_peso_ausente_diz_o_caminho_em_vez_de_estourar_no_ultralytics(tmp_path: Path) -> None:
    configuracao = render.RenderConfig(players="nao/existe.toml", field="nao/existe.toml")

    with pytest.raises(render.RenderUnavailable, match="nao/existe.toml"):
        render.render(1, tmp_path / "clipe.mp4", configuracao)


def test_o_resultado_nao_promete_video_que_nao_existe(cliente: TestClient) -> None:
    video = store.criar("a.mp4", "video/mp4", principal_id=1)
    store.marcar(video.id, store.PROCESSANDO)
    store.marcar(video.id, store.FINALIZADO, resultados=[])

    corpo = cliente.get(f"{PREFIXO}/videos/{video.id}").json()

    assert corpo["render_url"] is None


def test_o_resultado_aponta_para_o_video_quando_o_render_saiu(cliente: TestClient) -> None:
    video = store.criar("a.mp4", "video/mp4", principal_id=1)
    store.marcar(video.id, store.PROCESSANDO)
    store.marcar(video.id, store.FINALIZADO, resultados=[])
    caminho = render.render_path(video.id)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(b"nao e um mp4 de verdade, e a rota nao precisa que seja")

    corpo = cliente.get(f"{PREFIXO}/videos/{video.id}").json()

    assert corpo["render_url"].endswith(f"{PREFIXO}/videos/{video.id}/render")
    resposta = cliente.get(corpo["render_url"])
    assert resposta.status_code == 200
    assert resposta.headers["content-type"] == "video/mp4"
    assert resposta.content == caminho.read_bytes()


def test_video_sem_render_responde_404_e_nao_arquivo_vazio(cliente: TestClient) -> None:
    video = store.criar("a.mp4", "video/mp4", principal_id=1)

    resposta = cliente.get(f"{PREFIXO}/videos/{video.id}/render")

    assert resposta.status_code == 404


def test_o_render_de_outro_dono_responde_404(cliente: TestClient) -> None:
    outro = store.criar_principal("servico", "dono-de-outro-video", token_hash="x" * 64)
    video = store.criar("a.mp4", "video/mp4", principal_id=outro.id)
    caminho = render.render_path(video.id)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(b"video de outra pessoa")

    resposta = cliente.get(f"{PREFIXO}/videos/{video.id}/render")

    assert resposta.status_code == 404


def test_o_render_sem_credencial_responde_401(cliente: TestClient) -> None:
    video = store.criar("a.mp4", "video/mp4", principal_id=1)
    anonimo = TestClient(cliente.app)  # pyright: ignore[reportArgumentType]

    resposta = anonimo.get(f"{PREFIXO}/videos/{video.id}/render")

    assert resposta.status_code == 401


class _ScriptFalso:
    IMGSZ = 960
    CONFIDENCE = 0.25
    PLAYER_DEVICE = "cpu"

    def __init__(self, saida: Path, *, produz: bool = True) -> None:
        self.OUTPUT_DIR = saida  # noqa: N803 - o nome e o do script
        self._produz = produz
        self.chamadas: list[tuple[Path, Path]] = []

    def PlayerTracker(self, weights: Path, **argumentos: object) -> object:  # noqa: N802
        return ("tracker", weights, argumentos)

    def process_video(
        self, tracker: object, video_path: Path, input_dir: Path, homography_model: object
    ) -> None:
        """Deixa o temporario que o script de verdade deixa: sem ele, o teste nao prova nada.

        `create_top_view.create_writer` grava `<stem>_temporario.mp4` antes de qualquer quadro, e
        `convert_to_h264` so o apaga depois de o ffmpeg terminar bem.
        """
        self.chamadas.append((video_path, input_dir))
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / f"{video_path.stem}_temporario.mp4").write_bytes(b"cru")
        if self._produz:
            (self.OUTPUT_DIR / f"{video_path.stem}{render.SUFIXO}").write_bytes(b"mp4")

    def load_model(self, path: Path) -> object:
        return ("modelo", path)


def _sem_pesos(monkeypatch: pytest.MonkeyPatch) -> None:
    def caminho(manifesto: str, especie: str) -> Path:
        return Path(manifesto)

    monkeypatch.setattr(video_renderer, "_weights", caminho)


@pytest.mark.parametrize("fps_factor", [1.0, 2.0])
def test_ball_is_added_to_the_original_top_view_without_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fps_factor: float
) -> None:
    pytest.importorskip("imageio_ffmpeg")
    _sem_pesos(monkeypatch)
    script = _ScriptFalso(tmp_path / "unused")

    def original_top_view(
        tracker: object, video_path: Path, input_dir: Path, homography_model: object
    ) -> None:
        capture = cv2.VideoCapture(str(video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(script.OUTPUT_DIR / f"{video_path.stem}{render.SUFIXO}"),
            cv2.VideoWriter.fourcc(*"mp4v"),
            capture.get(cv2.CAP_PROP_FPS) * fps_factor,
            (width + 160, height),
        )
        assert writer.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame[70:100, 30:80] = (210, 60, 40)
            side = np.full((height, 160, 3), (30, 90, 210), dtype=np.uint8)
            writer.write(np.concatenate((frame, side), axis=1))
        capture.release()
        writer.release()

    monkeypatch.setattr(script, "process_video", original_top_view)
    monkeypatch.setattr(video_renderer, "_script", lambda: (script, script))
    clip = make_synthetic_clip(tmp_path / "clip.mp4")
    evidence = Perception(
        "video-7",
        (
            FramePerception(
                0, 0, ball=BallDetection(Box(150, 150, 160, 160), 0.9, source="detected")
            ),
        ),
    )

    if fps_factor != 1:
        previous = render.render_path(7)
        previous.parent.mkdir(parents=True)
        previous.write_bytes(b"previous video")
        with pytest.raises(render.RenderUnavailable, match="FPS"):
            render.render(7, clip, render.RenderConfig(players="p", field="f"), perception=evidence)
        assert previous.read_bytes() == b"previous video"
        return

    target = render.render(
        7, clip, render.RenderConfig(players="p", field="f"), perception=evidence
    )

    source = cv2.VideoCapture(str(clip))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    source.release()
    capture = cv2.VideoCapture(str(target))
    for index in range(count):
        ok, frame = capture.read()
        assert ok
        assert frame.shape[1] == width + 160, "nao pode substituir o layout do top view original"
        assert np.abs(frame[110, width + 80].astype(int) - (30, 90, 210)).max() < 12
        assert np.abs(frame[85, 50].astype(int) - (210, 60, 40)).max() < 12
        if index == 0:
            region = frame[130:180, 130:200]
            assert ((region[:, :, 0] > 180) & (region[:, :, 2] > 180)).sum() > 50
    capture.release()


def test_sem_o_script_no_disco_o_render_diz_onde_ele_deveria_estar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sem_pesos(monkeypatch)
    monkeypatch.setattr(video_renderer, "MODELOS", tmp_path / "sem-modelos")

    with pytest.raises(render.RenderUnavailable, match="script da visao superior"):
        render.render(1, tmp_path / "c.mp4", render.RenderConfig(players="p", field="f"))


def test_o_video_termina_em_var_e_nao_sobra_nada_na_area_de_codigo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declarado = tmp_path / "video_tracking"
    _sem_pesos(monkeypatch)
    script = _ScriptFalso(declarado)
    monkeypatch.setattr(video_renderer, "_script", lambda: (script, script))
    clipe = tmp_path / "uploads" / "7.mp4"
    clipe.parent.mkdir(parents=True)
    clipe.write_bytes(b"clipe")

    destino = render.render(7, clipe, render.RenderConfig(players="p.toml", field="f.toml"))

    assert destino == render.render_path(7)
    assert destino.read_bytes() == b"mp4"
    assert declarado == script.OUTPUT_DIR, "o render nao devolveu o OUTPUT_DIR do script"
    assert not declarado.exists(), "o script nao pode virar deposito de video"
    assert list(destino.parent.iterdir()) == [destino], "a oficina ficou para tras em var/"
    assert script.chamadas == [(clipe, clipe.parent)]


def test_script_que_termina_sem_deixar_arquivo_e_erro_declarado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declarado = tmp_path / "video_tracking"
    _sem_pesos(monkeypatch)
    script = _ScriptFalso(declarado, produz=False)
    monkeypatch.setattr(video_renderer, "_script", lambda: (script, script))
    clipe = tmp_path / "8.mp4"
    clipe.write_bytes(b"clipe")

    with pytest.raises(render.RenderUnavailable, match="nao deixou"):
        render.render(8, clipe, render.RenderConfig(players="p.toml", field="f.toml"))

    assert declarado == script.OUTPUT_DIR
    assert not declarado.exists(), "o temporario do script sobreviveu a falha"
    assert not render.render_path(8).exists(), "falha nao pode deixar nome final no lugar"


def test_falha_no_meio_nao_deixa_video_pela_metade_com_o_nome_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`render_url` sai de `render_path(...).is_file()`, entao arquivo truncado vira 200."""
    _sem_pesos(monkeypatch)
    script = _ScriptFalso(tmp_path / "video_tracking")

    def explode(*_: object, **__: object) -> None:
        raise OSError("o disco encheu no meio")

    monkeypatch.setattr(video_renderer, "_script", lambda: (script, script))
    monkeypatch.setattr(video_renderer.os, "replace", explode)
    clipe = tmp_path / "9.mp4"
    clipe.write_bytes(b"clipe")

    with pytest.raises(OSError, match="disco encheu"):
        render.render(9, clipe, render.RenderConfig(players="p.toml", field="f.toml"))

    assert not render.render_path(9).exists()
    assert not (tmp_path / "video_tracking").exists()


def test_o_processamento_chama_a_visao_superior_com_o_clipe_que_chegou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ligacao entre `processar_video` e o render nao aparece em type checker nenhum.

    Sem este teste, apagar a chamada de `_renderizar` deixa a suite verde e a tela sem video.
    """
    chamadas: list[tuple[int, Path]] = []

    def espiao(
        video_id: int, clipe: Path, configuracao: render.RenderConfig, *, perception: Perception
    ) -> None:
        assert perception.clip_id == f"video-{video_id}" and perception.frames
        atual = store.ler(video_id)
        assert atual is not None and atual.etapa_atual == "render"
        chamadas.append((video_id, clipe))

    monkeypatch.setattr(render, "render", espiao)
    dono = store.criar_principal("servico", "dono-do-clipe", token_hash="d" * 64)
    video = store.criar("a.mp4", "video/mp4", principal_id=dono.id)
    clipe = make_synthetic_clip(tmp_path / "clipe.mp4")

    execution.processar_video(video.id, clipe, PIPELINE)

    guardado = store.ler(video.id)
    assert guardado is not None and guardado.estado == store.FINALIZADO
    assert chamadas == [(video.id, clipe)]


def test_visao_superior_que_falha_nao_derruba_a_analise_que_deu_certo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ilustracao ausente nao e falha de analise, e o resultado nao pode prometer video nenhum."""

    def recusa(*_: object, **__: object) -> None:
        raise render.RenderUnavailable("o peso de players nao esta pronto")

    monkeypatch.setattr(render, "render", recusa)
    dono = store.criar_principal("servico", "dono-do-clipe", token_hash="d" * 64)
    video = store.criar("a.mp4", "video/mp4", principal_id=dono.id)
    clipe = make_synthetic_clip(tmp_path / "clipe.mp4")

    execution.processar_video(video.id, clipe, PIPELINE)

    guardado = store.ler(video.id)
    assert guardado is not None and guardado.estado == store.FINALIZADO
    assert not render.render_path(video.id).exists()


def test_missing_shot_keeps_the_annotated_video_from_the_same_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("imageio_ffmpeg")
    import cv2

    def configured_render(
        clipe: Path, destino: Path, config: render.RenderConfig, *, perception: Perception
    ) -> Path:
        assert perception.frames and clipe.is_file()
        return make_synthetic_clip(destino)

    monkeypatch.setattr(video_renderer, "render_video", configured_render)
    config = tmp_path / "no-shot.toml"
    text = PIPELINE.read_text()
    start = text.index("[modules.shot]")
    end = text.index("[", start + 1)
    config.write_text(text[:start] + text[end:])
    owner = store.criar_principal("servico", "no-shot", token_hash="e" * 64)
    video = store.criar("clip.mp4", "video/mp4", principal_id=owner.id)
    clip = make_synthetic_clip(tmp_path / "clip.mp4")

    execution.processar_video(video.id, clip, config)

    saved = store.ler(video.id)
    assert saved is not None and saved.estado == store.FALHOU
    assert render.render_path(video.id).is_file(), "sem chute ainda existem deteccoes para mostrar"
    capture = cv2.VideoCapture(str(render.render_path(video.id)))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None


def test_dois_renders_ao_mesmo_tempo_nao_trocam_a_pasta_um_do_outro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OUTPUT_DIR` e atributo de modulo e `sys.stdout` e do processo: os dois sao globais.

    Com `XVISION_API_WORKERS` maior que um, dois desenhos simultaneos liam a oficina um do outro e
    o `finally` de quem saisse depois devolvia ao global uma pasta ja apagada, envenenando todo
    render seguinte. O cadeado serializa o desenho, que ja e a parte cara.
    """
    declarado = tmp_path / "video_tracking"
    _sem_pesos(monkeypatch)
    script = _ScriptFalso(declarado)
    vistos: list[Path] = []
    original = script.process_video

    def devagar(
        tracker: object, video_path: Path, input_dir: Path, homography_model: object
    ) -> None:
        vistos.append(script.OUTPUT_DIR)
        time.sleep(0.05)
        original(tracker, video_path, input_dir, homography_model)

    monkeypatch.setattr(script, "process_video", devagar)
    monkeypatch.setattr(video_renderer, "_script", lambda: (script, script))
    stdout = sys.stdout
    clipes: list[tuple[int, Path]] = []
    for numero in (7, 8):
        clipe = tmp_path / f"{numero}.mp4"
        clipe.write_bytes(b"clipe")
        clipes.append((numero, clipe))

    with ThreadPoolExecutor(max_workers=2) as pool:
        tarefas = [
            pool.submit(render.render, numero, clipe, render.RenderConfig(players="p", field="f"))
            for numero, clipe in clipes
        ]
        destinos = [tarefa.result() for tarefa in tarefas]

    assert sorted(destinos) == sorted(render.render_path(n) for n, _ in clipes)
    assert all(destino.read_bytes() == b"mp4" for destino in destinos)
    assert len(set(vistos)) == 2, "os dois desenhos usaram a mesma oficina"
    assert declarado == script.OUTPUT_DIR, "o global ficou apontando para uma oficina apagada"
    assert not declarado.exists(), "sobrou video na area de codigo"
    assert sys.stdout is stdout, "o redirect deixou o stdout do processo preso num buffer"


def test_renderer_refuses_evidence_from_another_video(tmp_path: Path) -> None:
    with pytest.raises(render.RenderUnavailable, match="outro video"):
        render.render(
            1, tmp_path / "clip.mp4", render.RenderConfig(), perception=Perception("video-2", ())
        )
