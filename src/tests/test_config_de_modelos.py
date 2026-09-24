"""A configuracao que roda os modelos de verdade precisa ser valida onde nao ha peso nenhum.

Ela e arquivo separado de `config/pipeline.toml` porque a suite inteira carregaria torch para
provar coisas que nao dependem de modelo, e porque um clone novo nao tem `var/models`. Isso cria
um risco proprio: arquivo que ninguem exercita apodrece calado, e o defeito so aparece quando
alguem tenta rodar os 633 clipes.

Estes testes fecham esse risco sem precisar de peso. Carregar um modulo instancia a classe e nao
chama `load`, entao nada toca disco: o que se verifica e que o `impl` importa, que a especie bate
com a secao, que o manifesto declarado existe, e que o grafo ordena.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge.artifacts import read_manifest
from edge.graph.framesets import read_framesets
from edge.graph.order import topological_order
from edge.graph.runner import edges
from edge.modules.ball.detector import DetectorConfig
from edge.modules.base import MODULE_KINDS, PerceptionModule
from edge.modules.registry import load_modules, read_config

MODELOS = Path("config/pipeline.models.toml")
PADRAO = Path("config/pipeline.toml")
PACOTE = Path("config/pipeline.players-pacote.toml")


@pytest.fixture(scope="module")
def modelos() -> dict[str, PerceptionModule]:
    return dict(load_modules(read_config(MODELOS)))


def test_a_configuracao_de_modelos_declara_as_sete_especies(
    modelos: dict[str, PerceptionModule],
) -> None:
    assert set(modelos) == MODULE_KINDS


def test_todo_impl_declarado_importa_e_bate_com_a_especie(
    modelos: dict[str, PerceptionModule],
) -> None:
    """`load_modules` recusa kind divergente: este teste garante que ele rodou sobre o arquivo."""
    for kind, modulo in modelos.items():
        assert modulo.spec.kind == kind


def test_o_grafo_da_configuracao_de_modelos_ordena_sem_ciclo(
    modelos: dict[str, PerceptionModule],
) -> None:
    """Sem conjunto ancorado nao ha barreira, e `players` pode rodar antes do `shot`.

    Era o contrario ate a entrega do instante: `players` vivia num conjunto ancorado no `shot`, o
    que impedia o `shot` de olhar para ele. Sem essa restricao o instante saia da maior quebra da
    trajetoria, mesmo quando a bola estava sozinha no meio do campo ou na arquibancada.
    """
    ordem = topological_order(edges(modelos, read_framesets(MODELOS)))

    assert set(ordem) == MODULE_KINDS
    assert ordem.index("ball") < ordem.index("shot")
    assert ordem.index("players") < ordem.index("shot")
    assert ordem.index("players") < ordem.index("tracking")
    assert ordem.index("tracking") < ordem.index("pose")
    assert ordem.index("shot") < ordem.index("pose")


def test_todo_manifesto_declarado_existe_e_e_legivel(modelos: dict[str, PerceptionModule]) -> None:
    """O peso nao vem no repositorio, mas o manifesto que o descreve vem e precisa ser valido."""
    declarados = [
        (kind, modulo.spec.artifact)
        for kind, modulo in modelos.items()
        if modulo.spec.artifact is not None
    ]

    assert declarados, "a configuracao de modelos nao declara peso nenhum"
    for kind, caminho in declarados:
        manifesto = read_manifest(Path(caminho))
        assert manifesto.kind == kind
        assert manifesto.files, f"{caminho} sem sha256 nao detecta troca de peso"


ORG_NO_HUB = "xvisionm11"
SEM_COPIA_NO_HUB = frozenset({"tdeed-soccernetball-test"})
"""Peso sem licenca declarada pelos autores: cada instalacao baixa da fonte original."""


def test_todo_peso_da_producao_diz_onde_esta_no_hub(modelos: dict[str, PerceptionModule]) -> None:
    """O manifesto e o unico lugar que diz de onde o peso vem.

    O repositorio privado tem o nome do manifesto e a versao como tag. Se um dos dois andar sem o
    outro, o `hf download` da pagina de operacao baixa outro arquivo, e o `artifact get` so recusa
    depois, quando o sha256 nao bate.
    """
    for modulo in modelos.values():
        if modulo.spec.artifact is None:
            continue
        manifesto = read_manifest(Path(modulo.spec.artifact))
        if manifesto.name in SEM_COPIA_NO_HUB:
            assert "hub" not in manifesto.source
            continue
        assert manifesto.source.get("hub") == f"{ORG_NO_HUB}/{manifesto.name}"
        assert manifesto.source.get("hub_revision") == f"v{manifesto.version}"


def test_as_duas_configuracoes_declaram_os_mesmos_conjuntos_de_quadros() -> None:
    """Conjunto divergente faria o mesmo modulo amostrar diferente em cada arquivo."""
    assert read_framesets(MODELOS) == read_framesets(PADRAO)


CONF_MEDIANA_DA_BOLA_CORRETA = 0.154
"""Confianca mediana da bola certa nos 38 quadros de contato anotados. p25 0,096, minimo 0,033."""


def test_o_limiar_de_confianca_da_bola_fica_abaixo_da_mediana_da_bola_correta() -> None:
    """Limiar acima da mediana zera o candidato certo antes de qualquer filtro nosso.

    Com o padrao 0,25 o detector descartava a bola correta em 21 a 23 dos 38 quadros de contato, e
    a dobra de ajuste ficava em 3/24 = 0,125, praticamente o acaso medido de 0,1307. Descer para
    0,10 leva a 9/24 = 0,375, com 6 clipes ganhos e nenhum perdido. O teste existe para a proxima
    pessoa que abrir o arquivo nao devolver o numero ao padrao sem medir de novo.
    """
    entrada = {c.kind: c for c in read_config(MODELOS)}["ball"]

    conf = entrada.options.get("conf")
    assert isinstance(conf, float)
    assert conf < CONF_MEDIANA_DA_BOLA_CORRETA


def test_o_ajuste_declarado_na_secao_da_bola_chega_ao_detector() -> None:
    """Ajuste que fica no TOML sem alcancar ninguem e pior que ajuste nenhum: quem le o arquivo
    acredita no valor e a execucao usa outro.

    O caminho e o de producao: `configure` separa o que e do modulo, e `load` entrega o resto ao
    `DetectorConfig`, que recusa nome que nao conhece. Validar so o detector deixava passar nome
    inventado, porque `configure` guarda o que nao reconhece em vez de reclamar na hora.
    """
    from edge.modules.ball import YoloBall

    entrada = {c.kind: c for c in read_config(MODELOS)}["ball"]

    modulo = YoloBall()
    modulo.configure(entrada.options)
    modulo.load()

    assert DetectorConfig(weights="x.pt").com_ajustes(
        {"conf": entrada.options["conf"]}
    ).conf == pytest.approx(entrada.options["conf"])


def test_a_configuracao_padrao_nao_exige_peso_de_disco() -> None:
    """Ela precisa rodar em clone novo: peso mora em `var/`, que e ignorado pelo git."""
    padrao = load_modules(read_config(PADRAO))

    assert all(modulo.spec.artifact is None for modulo in padrao.values())


def test_o_oraculo_so_difere_da_producao_no_modulo_do_instante() -> None:
    """As duas configuracoes tem que medir a mesma percepcao, senao a comparacao nao vale nada.

    O experimento que separa erro de instante de erro de percepcao troca `shot` pelo gabarito e
    deixa tudo igual. Elas divergiram uma vez e o defeito foi calado: `conf = 0.10` entrou so na de
    producao e o oraculo continuou rodando a bola a 0,25, entao a medida de `body_part` pelo caminho
    do oraculo comparava um detector com outro em vez de comparar dois instantes.
    """
    producao = {c.kind: c for c in read_config(Path("config/pipeline.models.toml"))}
    oraculo = {c.kind: c for c in read_config(Path("config/pipeline.oracle.toml"))}

    assert set(producao) == set(oraculo)
    assert oraculo["shot"].impl == "mlops.oracle:AnnotatedShot"
    assert producao["shot"].impl != oraculo["shot"].impl
    for kind in sorted(set(producao) - {"shot"}):
        assert producao[kind] == oraculo[kind], f"{kind} diverge entre producao e oraculo"


def test_as_duas_etapas_so_diferem_da_producao_no_instante() -> None:
    """A segunda etapa tem que medir a mesma percepcao da producao, com o instante vindo de fora.

    Se qualquer outra especie divergir, o numero medido na placa em duas etapas deixa de valer
    como numero da configuracao de producao, que e o que ele precisa representar.
    """
    producao = {c.kind: c for c in read_config(MODELOS)}
    etapas = {c.kind: c for c in read_config(Path("config/pipeline.models.duas-etapas.toml"))}

    assert set(producao) == set(etapas)
    assert etapas["shot"].impl == "mlops.spots:PrecomputedShot"
    assert etapas["shot"].options == {"spots": "../var/spots/tdeed.jsonl"}
    for kind in sorted(set(producao) - {"shot"}):
        assert producao[kind] == etapas[kind], f"{kind} diverge entre producao e duas etapas"
    assert read_framesets(MODELOS) == read_framesets(
        Path("config/pipeline.models.duas-etapas.toml")
    )


def test_a_configuracao_da_jetson_so_troca_bola_e_campo_por_plano() -> None:
    """A placa roda as duas etapas com plano TensorRT em `ball` e `field`, e nada mais muda.

    Plano sem o sha256 do ONNX de origem seria um binario opaco: nao daria para saber de que peso
    ele saiu. E `quantize` junto de `engine` e recusado pelo detector, porque a precisao ja foi
    fixada na construcao do plano.
    """
    etapas = {c.kind: c for c in read_config(Path("config/pipeline.models.duas-etapas.toml"))}
    jetson = {c.kind: c for c in read_config(Path("config/pipeline.models.jetson.toml"))}

    assert set(etapas) == set(jetson)
    for kind in ("ball", "field"):
        opcoes = dict(jetson[kind].options)
        assert str(opcoes.pop("engine")).startswith("../var/engines/"), kind
        assert len(str(opcoes.pop("engine_onnx_sha256"))) == 64, kind
        assert "quantize" not in opcoes, kind
        esperado = {k: v for k, v in etapas[kind].options.items() if k != "quantize"}
        assert opcoes == esperado, kind
    for kind in sorted(set(etapas) - {"ball", "field"}):
        assert etapas[kind] == jetson[kind], f"{kind} diverge entre duas etapas e jetson"


def test_a_janela_so_difere_do_oraculo_no_frameset_da_pose() -> None:
    """Taxa nativa ancorada no chute e a unica variavel dessa configuracao."""
    oraculo = {c.kind: c for c in read_config(Path("config/pipeline.oracle.toml"))}
    janela = {c.kind: c for c in read_config(Path("config/pipeline.oracle-janela.toml"))}

    assert janela["pose"].frames == "window"
    assert oraculo["pose"].frames == "dense"
    for kind in sorted(set(oraculo) - {"pose"}):
        assert oraculo[kind] == janela[kind], f"{kind} diverge entre oraculo e janela"


def test_o_pacote_de_jogadores_so_difere_da_producao_no_impl_das_duas_especies() -> None:
    """As duas implementacoes de `players` e `tracking` precisam ser comparaveis, e nao parecidas.

    Depois do merge com o `develop` convivem `edge.modules.players:YoloPlayers` e o
    `PlayersModule` do pacote da MR !31, e o mesmo par em `tracking`. Escolher entre elas e medir,
    e `xvision eval-ball` mede: a metrica da bola le a distancia ate a caixa de jogador mais
    proxima, entao detector que perde gente move o numero.

    O que este teste protege e a validade da comparacao. Se o peso, o frameset ou o ajuste da bola
    divergirem junto com o `impl`, o numero passa a medir duas mudancas somadas e nao diz de qual
    delas veio a diferenca. E o mesmo modo de falha que fez o oraculo rodar a bola a 0,25 enquanto
    a producao rodava a 0,10, calado.
    """
    producao = {c.kind: c for c in read_config(MODELOS)}
    pacote = {c.kind: c for c in read_config(PACOTE)}
    trocadas = {"players", "tracking"}

    assert set(producao) == set(pacote)
    assert pacote["players"].impl == "edge.participants.module:PlayersModule"
    assert pacote["tracking"].impl == "edge.participants.module:TrackingModule"
    for kind in sorted(trocadas):
        assert producao[kind].impl != pacote[kind].impl, f"{kind} nao trocou de implementacao"
        assert producao[kind].artifact == pacote[kind].artifact, f"{kind}: peso diferente"
        assert producao[kind].frames == pacote[kind].frames, f"{kind}: frameset diferente"
    for kind in sorted(set(producao) - trocadas):
        assert producao[kind] == pacote[kind], f"{kind} diverge entre producao e pacote"


def test_o_impl_do_pacote_de_jogadores_importa_e_bate_com_a_especie() -> None:
    """Mesma garantia que `modelos` tem: o arquivo nao pode apodrecer calado por ninguem rodar.

    `load_modules` instancia sem chamar `load`, entao nada toca disco. O que se prova aqui e que os
    dois `impl` do pacote resolvem, que o registro consegue construi-los sem argumento e que a
    especie de cada um bate com a secao onde esta declarado.
    """
    pacote = dict(load_modules(read_config(PACOTE)))

    assert set(pacote) == MODULE_KINDS
    for kind, modulo in pacote.items():
        assert modulo.spec.kind == kind
    assert set(topological_order(edges(pacote, read_framesets(PACOTE)))) == MODULE_KINDS


def test_nenhuma_especie_volta_para_o_peso_generico_do_coco() -> None:
    """O detector generico era a causa raiz, e a regressao para ele nao levantaria nada.

    O `yolov8n.pt` do COCO ve bola pela classe 32, `sports ball`, que reune bola de basquete, de
    tenis e de futebol vistas de perto, enquanto o objeto do problema tem 13 px a cinquenta metros.
    Medido no quadro do contato dos 24 clipes da dobra de ajuste, trocar so o peso da bola pelo
    dedicado leva a taxa de 3 para 11 em 24, ganhando 8 clipes e perdendo nenhum.

    `players` continua com peso de futebol e nao com o COCO por outro motivo: `person` nao distingue
    goleiro, arbitro e gandula, e a metrica da bola mede distancia ate a caixa de gente mais
    proxima, entao arbitro contado como jogador aproxima o contato de quem nao chutou.
    """
    for arquivo in (
        "pipeline.models.toml",
        "pipeline.oracle.toml",
        "pipeline.oracle-janela.toml",
        "pipeline.players-pacote.toml",
    ):
        for entrada in read_config(Path("config") / arquivo):
            if entrada.kind not in {"ball", "players"}:
                continue
            assert entrada.artifact is not None
            assert "yolov8n-8.4.0" not in entrada.artifact, f"{arquivo}: {entrada.kind} no COCO"


def test_a_bola_tem_peso_proprio_e_nao_o_de_jogadores() -> None:
    """A Roboflow separa os dois modelos, e a medida aqui diz o mesmo.

    O peso de quatro classes tem a bola dominada por jogador em 372 imagens de treino, e mediu 7 em
    24 na dobra de ajuste. O dedicado, de classe unica, mediu 11 em 24 no mesmo conjunto.
    """
    entradas = {c.kind: c for c in read_config(Path("config/pipeline.models.toml"))}

    assert entradas["ball"].artifact != entradas["players"].artifact


def test_o_campo_de_producao_usa_o_modulo_real_e_o_manifesto_versionado() -> None:
    entrada = {c.kind: c for c in read_config(MODELOS)}["field"]

    assert entrada.impl == "edge.modules.field:YoloField"
    assert entrada.artifact == ("models/manifests/field/pitch-keypoints-yolo11n-pose-0.1.0.toml")
    assert entrada.options["method"] == "ransac"
    assert entrada.options["min_inliers"] == 6


def test_o_onnx_do_campo_registra_o_pt_pai_e_os_shapes_fixos() -> None:
    manifesto = read_manifest(
        Path("models/manifests/field/pitch-keypoints-yolo11n-pose-onnx-0.1.0.toml")
    )

    assert manifesto.entrypoint == "best.onnx"
    assert manifesto.source["parent_pt_sha256"] == (
        "650a84407ae9833106d87b39a2063d2ab8b92e8ac12c2d4f36ffd09f0de37674"
    )
    assert manifesto.source["input"] == "images:float32:1x3x544x960:NCHW"
    assert manifesto.source["output"] == "output0:float32:1x92x10710"


def test_o_onnx_da_bola_registra_o_pt_pai_e_os_shapes_fixos() -> None:
    manifesto = read_manifest(
        Path("models/manifests/ball/yolo-football-ball-detection-onnx-0.1.0.toml")
    )

    assert manifesto.entrypoint == "yolo-football-ball-detection.onnx"
    assert manifesto.source["parent_pt_sha256"] == (
        "fb37942448e7de08745e8aab148d0794f680a738ddd55e5f17abe9ab2d6313fb"
    )
    assert manifesto.source["input"] == "images:float32:1x3x992x1600:NCHW"
    assert manifesto.source["output"] == "output0:float32:1x5x32550"


@pytest.mark.parametrize("config_name", ["models", "oracle", "oracle-janela", "players-pacote"])
def test_rejected_ball_ranker_is_disabled_in_model_configs(config_name: str) -> None:
    ball = next(
        c for c in read_config(Path(f"config/pipeline.{config_name}.toml")) if c.kind == "ball"
    )

    assert ball.options["ranking"] is False
