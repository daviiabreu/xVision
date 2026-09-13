"""O instante da finalizacao a partir da trajetoria da bola, no lugar do `StubShot`.

Este modulo e a ancora do pipeline inteiro: o frameset `window` abre em volta do quadro que ele
apontar, e e nessa janela que `pose`, `tracking` e `identity` medem. Instante errado move a janela
inteira, entao a tolerancia sai da configuracao e nao de gosto. Com `before_s = 1.5` e
`after_s = 0.5` a 59,94 fps, a janela e 90 quadros antes e 30 depois: erro de ate 30 quadros ainda
preserva dois tercos do contexto anterior, e erro acima de 120 tira o contato de dentro da janela.

Ele nao olha um pixel. Toda a decisao sai de `edge.geometry.contact` sobre as caixas que a especie
`ball` produziu, e por isso a `PreprocessSpec` declarada e a menor que o contrato aceita: o lote
chega e nao e lido. Declarar 1920x1080 aqui custaria memoria de quadro cru sem comprar nada.

Tres decisoes que valem explicacao.

A primeira e `needs = ("ball",)` e nao `("ball", "players")`. A proximidade de um jogador filtraria
quebra causada por quique no gramado, e `edge.geometry.contact` aceita esse filtro pelo parametro
`candidates`. O custo e que o runner pula a etapa inteira quando um modulo de `needs` falha, entao
exigir `players` dobraria a superficie de falha de um clipe para refinar um caso. O filtro entra
quando `players` for modulo de verdade com confiabilidade medida, e nao antes.

A segunda e que `shooter_track_id` sai sempre ausente, e isso e estrutural e nao esquecimento. O
identificador de trajetoria vem da especie `tracking`, que roda no frameset `window`, que e
ancorado no `shot`. Um `shot` que dependesse de `tracking` fecharia ciclo no grafo e o
`topological_order` recusaria a execucao antes de comecar. Quem precisa saber de quem e o chute
identifica o jogador mais proximo da bola no quadro do contato, sobre a `Perception` ja montada.

A terceira e o ponto medio como degradacao declarada. Sem quebra utilizavel o modulo poderia nao
emitir nada, e ai `process_clip` levanta `NoShotDetected` e o clipe inteiro sai sem evento, sem
nenhuma feature medida. Emitir o meio do trecho amostrado com `source` dizendo que foi chute e com
confianca baixa mantem o resto do pipeline rodando e deixa o numero rastreavel: quem le a execucao
consegue separar clipe medido de clipe estimado, que e exatamente o que um zero calado impediria.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from edge.geometry.contact import Contact, Observation, find_contact
from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)

BALL = "ball"
PLAYERS = "players"
MEDIDO = "ball-break"
ESTIMADO = "midpoint"
DETECTADA = "detected"
INTERPOLADA = "interpolated"
DESCONHECIDA = "unknown"
AUSENTE = "absent"
AJUSTES: Mapping[str, str] = {
    "estimar_quando_nao_medir": "_estimar",
    "exigir_jogador_perto": "_exigir_jogador",
    "exigir_bola_medida": "_exigir_medida",
}
"""O vocabulario que `configure` aceita, mapeado no atributo interno.

Nome errado levanta em vez de ser ignorado, pelo mesmo motivo do resto do projeto: quem escreveu
`exigir_bola_detectada` no TOML leria o arquivo, veria a execucao terminar sem reclamar e concluiria
que a politica mudou.
"""
"""Vocabulario de `ball_support`, o mesmo que o registro da bola usa para `source`.

`absent` e o quadro escolhido sem posicao nenhuma, que acontece no caminho `midpoint`: o ponto medio
do trecho nao precisa ter bola. `unknown` e posicao sem procedencia declarada.
"""
FALLBACK_CONFIDENCE = 0.05

MARGEM_LARGURA = 0.6
MARGEM_ALTURA = 0.35
"""Quanto a caixa do jogador cresce para dizer que a bola esta com ele.

Sao fracoes do proprio lado, e nao pixels, porque a caixa de um jogador varia de umas 30 px perto
do fundo do campo a mais de 200 px em close. Mais largura que altura porque a bola sai de lado no
chute e fica na altura do pe, e nao acima da cabeca.
"""

PREPROCESS = PreprocessSpec(width=32, height=32, color="gray", letterbox=False, normalize="none")
"""O menor lote que o contrato aceita, porque o modulo decide pela `ball` e nao pelo quadro."""


class BallBreakShot:
    spec = ModuleSpec(
        kind="shot",
        name="ball-break",
        version="0.1.0",
        frames="coarse",
        preprocess=PREPROCESS,
        needs=(BALL,),
        optional_needs=(PLAYERS,),
    )

    def __init__(
        self,
        *,
        estimar_quando_nao_medir: bool = True,
        exigir_jogador_perto: bool = True,
        exigir_bola_medida: bool = False,
    ) -> None:
        self._pronto = False
        self._estimar = estimar_quando_nao_medir
        self._exigir_jogador = exigir_jogador_perto
        self._exigir_medida = exigir_bola_medida

    def configure(self, options: Mapping[str, object]) -> None:
        """Os tres ajustes deste modulo por configuracao, para poderem ser medidos sem
        editar `src`.

        Sem isto qualquer comparacao entre politicas exigia trocar o codigo, e um numero medido com
        codigo diferente do que roda nao vale como comparacao.
        """
        for nome in options:
            if nome not in AJUSTES:
                raise ValueError(
                    f"shot.{nome} nao existe: os ajustes sao {', '.join(sorted(AJUSTES))}"
                )
        for nome, valor in options.items():
            if not isinstance(valor, bool):
                raise ValueError(f"shot.{nome} espera booleano, recebeu {valor!r}")
            setattr(self, AJUSTES[nome], valor)

    def load(self) -> None:
        self._pronto = True

    def close(self) -> None:
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        if not batch.indices:
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())

        registro = self._registro(batch, ctx)
        return _saida(self.spec.kind, batch.indices, registro)

    def _registro(self, batch: FrameBatch, ctx: ModuleInputs) -> Mapping[str, object] | None:
        """`ctx.get` e nao `ctx[...]`: a suite de conformidade chama `infer` com o contexto vazio.

        Um modulo que indexasse direto reprovaria os sete testes de uma vez, e o contexto vazio nao
        e caso de teste artificial: e o que acontece quando a especie de que ele depende falhou.
        """
        limite = max(batch.indices) + 1
        saida_da_bola = ctx.get(BALL)
        bola = observacoes(saida_da_bola)
        procedencia = procedencias(saida_da_bola)
        if self._exigir_medida:
            bola = tuple(o for o in bola if procedencia.get(o.frame_index) == DETECTADA)
        candidatos = _com_jogador_perto(bola, ctx.get(PLAYERS)) if self._exigir_jogador else None
        contato = find_contact(bola, frame_count=limite, candidates=candidatos)
        if contato is not None:
            return _medido(contato, procedencia)
        if not self._estimar:
            return None
        return _estimado(batch.indices, procedencia)


def _com_jogador_perto(
    bola: Sequence[Observation], players: ModuleOutput | None
) -> frozenset[int] | None:
    """Os quadros em que a bola esta com algum jogador, que sao os unicos onde ha contato.

    Sem esta restricao o instante sai da maior quebra da trajetoria, e a trajetoria de uma bola
    mal detectada e ruido: medido em 40 clipes do parceiro, o envelope dela tem 1386 px de
    diagonal na mediana, num quadro de 1920. O limite de salto do rastreador nao segura isso
    porque ele e em pixel por segundo e a amostragem e de 5 fps, o que autoriza 900 px entre duas
    amostras.

    Contato e a bola encostando em alguem. Um quadro em que ela esta sozinha no meio do campo, ou
    na arquibancada, nao pode ser o instante, por mais que a trajetoria quebre ali.

    `None` quando a especie `players` nao veio: sem jogador nao ha restricao a aplicar, e inventar
    uma faria o modulo se calar em todo clipe.
    """
    if players is None:
        return None
    caixas = {
        indice: tuple(
            (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            for r in registros
            if (c := cast("Any", r.get("box"))) is not None
        )
        for indice, registros in zip(players.frames, players.records, strict=True)
    }
    perto: set[int] = set()
    for o in bola:
        for x1, y1, x2, y2 in caixas.get(o.frame_index, ()):
            largura, altura = x2 - x1, y2 - y1
            if (
                x1 - largura * MARGEM_LARGURA <= o.x <= x2 + largura * MARGEM_LARGURA
                and y1 - altura * MARGEM_ALTURA <= o.y <= y2 + altura * MARGEM_ALTURA
            ):
                perto.add(o.frame_index)
                break
    return frozenset(perto)


def procedencias(saida: ModuleOutput | None) -> dict[int, str]:
    """Procedencia da posicao da bola por quadro, na mesma leitura que `observacoes` faz.

    Le `registros[0]` como `observacoes`, para os dois lados enxergarem a mesma bola. Registro sem
    `source` entra como `desconhecida` em vez de `detectada`: um detector que nao declara nao pode
    ser lido como se tivesse declarado medida.
    """
    if saida is None:
        return {}
    fontes: dict[int, str] = {}
    for indice, registros in zip(saida.frames, saida.records, strict=True):
        if not registros or registros[0].get("box") is None:
            continue
        fonte = registros[0].get("source")
        fontes[indice] = DESCONHECIDA if fonte is None else str(fonte)
    return fontes


def observacoes(saida: ModuleOutput | None) -> tuple[Observation, ...]:
    """O centro de cada caixa de bola, um por quadro.

    `assemble` ja descarta tudo menos `registros[0]` de cada quadro, entao usar o primeiro aqui
    mantem os dois lados enxergando a mesma bola. Registro sem `box` e ignorado em vez de estourar,
    porque `check_records` roda depois do `infer` e um detector meio quebrado nao pode derrubar a
    etapa antes de a validacao ter chance de dizer o que faltou.
    """
    if saida is None:
        return ()
    observacoes: list[Observation] = []
    for indice, registros in zip(saida.frames, saida.records, strict=True):
        if not registros:
            continue
        caixa = registros[0].get("box")
        if caixa is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in cast("Any", caixa))
        observacoes.append(Observation(indice, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return tuple(observacoes)


def _medido(contato: Contact, procedencia: Mapping[int, str]) -> dict[str, object]:
    return {
        "frame_index": contato.frame_index,
        "confidence": round(contato.confidence, 6),
        "source": MEDIDO,
        "turn_deg": round(contato.turn_deg, 3),
        "speed_ratio": None
        if contato.speed_ratio == float("inf")
        else round(contato.speed_ratio, 3),
        **_suporte(contato.frame_index, procedencia),
    }


def _estimado(indices: Sequence[int], procedencia: Mapping[int, str]) -> dict[str, object]:
    escolhido = indices[len(indices) // 2]
    return {
        "frame_index": escolhido,
        "confidence": FALLBACK_CONFIDENCE,
        "source": ESTIMADO,
        "reason": "a trajetoria da bola nao sustentou uma quebra, e o instante e o meio do trecho",
        **_suporte(escolhido, procedencia),
    }


def _suporte(quadro: int, procedencia: Mapping[int, str]) -> dict[str, object]:
    """Se a posicao da bola no quadro escolhido foi medida ou inventada, e quantas de cada
    no trecho.

    O rastreador da bola preenche lacuna curta por interpolacao linear, e o instante do chute e
    exatamente onde a trajetoria vira: uma reta tracada por cima da virada passa longe do pe. Ate
    aqui o modulo tratava as duas iguais, porque `observacoes` le so a caixa e joga fora o campo
    `source` que o registro da bola carrega.

    Isso deixou de ser detalhe quando o ranking de candidatos entrou: medido no piloto anotado, as
    posicoes interpoladas foram de 15 para 88 em 328 quadros, e tres dos 24 quadros de contato
    passaram a nao ter deteccao real. Sem `ball_support`, um instante decidido sobre posicao
    inventada sai com a mesma cara de um decidido sobre medida, e nenhuma metrica agregada separa
    os dois.

    Nao muda a previsao. A spec manda primeiro preservar a procedencia e reproduzir as contagens,
    e so depois testar uma regra que use isso para decidir.
    """
    contagem = Counter(procedencia.values())
    return {
        "ball_support": procedencia.get(quadro, AUSENTE),
        "ball_detected_frames": contagem.get(DETECTADA, 0),
        "ball_interpolated_frames": contagem.get(INTERPOLADA, 0),
    }


def _saida(
    kind: str, indices: Sequence[int], registro: Mapping[str, object] | None
) -> ModuleOutput:
    """Um registro so, no quadro amostrado mais proximo do instante, e vazio nos outros.

    O quadro que carrega o registro nao precisa ser o instante: `_anchor_frame` e `_frame_index`
    leem `frame_index` de dentro do registro e ignoram a posicao. E o instante refinado costuma cair
    entre duas amostras, entao amarra-lo a uma delas desfaria o refino.
    """
    if registro is None:
        return ModuleOutput(kind=kind, frames=tuple(indices), records=((),) * len(indices))
    alvo = cast("int", registro["frame_index"])
    portador = min(range(len(indices)), key=lambda i: abs(indices[i] - alvo))
    registros = tuple((registro,) if posicao == portador else () for posicao in range(len(indices)))
    return ModuleOutput(kind=kind, frames=tuple(indices), records=registros)
