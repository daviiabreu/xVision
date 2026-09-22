"""O instante da finalizacao a partir da trajetoria da bola, no lugar do `StubShot`.

Este modulo e a ancora declarada do pipeline: o frameset `window` abre em volta do quadro que ele
apontar. Hoje nenhum modulo do config de producao roda nessa janela, porque as sete especies foram
uniformizadas no `dense`, e o `window` sobrevive como janela declarada que a config
`oracle-janela` usa para `pose`. Ela continua sendo a medida do contexto que o instante precisa
acertar: com `before_s = 1.5` e `after_s = 0.5` a 59,94 fps sao 90 quadros antes e 30 depois, entao
erro de ate 30 quadros ainda preserva dois tercos do contexto anterior e erro acima de 120 tira o
contato de dentro dela. A tolerancia sai dessa conta e nao de gosto.

Ele nao olha um pixel. Toda a decisao sai de `edge.geometry.contact` sobre as caixas que a especie
`ball` produziu, e por isso a `PreprocessSpec` declarada e a menor que o contrato aceita: o lote
chega e nao e lido. Declarar 1920x1080 aqui custaria memoria de quadro cru sem comprar nada.

Tres decisoes que valem explicacao.

A primeira e `needs = ("ball",)` e nao `("ball", "players")`. A proximidade de um jogador filtraria
quebra causada por quique no gramado, e `edge.geometry.contact` aceita esse filtro pelo parametro
`candidates`. O custo e que o runner pula a etapa inteira quando um modulo de `needs` falha, entao
exigir `players` dobraria a superficie de falha de um clipe para refinar um caso. O filtro entra
quando `players` for modulo de verdade com confiabilidade medida, e nao antes.

A segunda e que `shooter_track_id` sai sempre ausente, e o motivo nao e mais o ciclo no grafo.
Quando `tracking` rodava na janela ancorada no `shot`, a dependencia fechava ciclo de verdade; a
uniformizacao dos conjuntos removeu isso, e medido no config de producao `shot` dependendo de
`tracking` ordena limpo. O motivo que fica e o simples: este modulo decide sobre a trajetoria da
bola, nao sobre caixas de jogador, e amarra-lo a `tracking` faria a validade do grafo depender de
qual frameset o TOML escolhe. Quem precisa saber de quem e o chute usa `edge/features/touch.py`,
sobre a `Perception` ja montada.

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

from edge.geometry.contact import (
    Candidate,
    Contact,
    Observation,
    best_contact,
    rank_contacts,
)
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
MISTA = "mixed"
"""Vocabulario de `ball_support`, o mesmo que o registro da bola usa para `source`, mais tres.

`mixed` existe porque o instante nao cai numa amostra. `find_contact` refina a quebra pela
intersecao das duas retas ajustadas e devolve um quadro entre amostras: no conjunto `dense`, de
passo quatro, tres de cada quatro valores refinados nao coincidem com amostra. Sem `mixed`, a
resolucao pela amostra mais proxima teria de escolher um dos dois lados e esconder o outro.

`absent` e o quadro escolhido sem posicao alguma por perto, que acontece no caminho `midpoint`: o
ponto medio do trecho nao precisa ter bola. `unknown` e posicao sem procedencia declarada.
"""
AJUSTES = frozenset({"estimar_quando_nao_medir", "exigir_jogador_perto", "exigir_bola_medida"})
"""O vocabulario que `configure` aceita.

Nome errado levanta em vez de ser ignorado, pelo mesmo motivo do resto do projeto: quem escreveu
`exigir_bola_detectada` no TOML leria o arquivo, veria a execucao terminar sem reclamar e concluiria
que a politica mudou.

A atribuicao e por ramo explicito e nao por `setattr`: com `setattr`, renomear um atributo interno
criaria um atributo novo com o nome antigo, `infer` leria o renomeado, e a opcao do TOML viraria
exatamente o silencio que este conjunto existe para impedir. O pyright nao ve atraves de `setattr`.
"""
FALLBACK_CONFIDENCE = 0.05

CANDIDATOS_REGISTRADOS = 5
"""Quantas quebras o registro guarda com score, contando a escolhida.

Cinco responde a pergunta que o erro medido deixa em aberto, se o contato certo estava entre as
quebras descartadas, sem guardar score de coisa nenhuma: toda amostra interior da trajetoria e uma
quebra de algum tamanho, e a mediana por clipe e 54.

Nada no pipeline le este campo. Ele existe para o diagnostico de `mlops.instant` poder separar
erro de escolha, em que a quebra certa estava na lista e perdeu no score, de erro de trajetoria,
em que ela nunca chegou a existir. As duas causas pedem trabalho diferente e ate aqui saiam com a
mesma cara na mediana.
"""

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
        for nome, valor in options.items():
            if nome not in AJUSTES:
                raise ValueError(
                    f"shot.{nome} nao existe: os ajustes sao {', '.join(sorted(AJUSTES))}"
                )
            if not isinstance(valor, bool):
                raise ValueError(f"shot.{nome} espera booleano, recebeu {valor!r}")
        if "estimar_quando_nao_medir" in options:
            self._estimar = bool(options["estimar_quando_nao_medir"])
        if "exigir_jogador_perto" in options:
            self._exigir_jogador = bool(options["exigir_jogador_perto"])
        if "exigir_bola_medida" in options:
            self._exigir_medida = bool(options["exigir_bola_medida"])

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
        return saida_unica(self.spec.kind, batch.indices, registro)

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
        ordenados = rank_contacts(bola, frame_count=limite, candidates=candidatos)
        contato = best_contact(ordenados)
        if contato is not None:
            return _medido(contato, procedencia, ordenados)
        if not self._estimar:
            return None
        return _estimado(batch.indices, procedencia, ordenados)


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


def _medido(
    contato: Contact, procedencia: Mapping[int, str], ordenados: Sequence[Candidate]
) -> dict[str, object]:
    return {
        "frame_index": contato.frame_index,
        "confidence": round(contato.confidence, 6),
        "source": MEDIDO,
        "turn_deg": round(contato.turn_deg, 3),
        "speed_ratio": None
        if contato.speed_ratio == float("inf")
        else round(contato.speed_ratio, 3),
        "candidates": _candidatos(ordenados),
        "candidate_pool": _pool(ordenados),
        **suporte(contato.frame_index, procedencia),
    }


def _estimado(
    indices: Sequence[int], procedencia: Mapping[int, str], ordenados: Sequence[Candidate]
) -> dict[str, object]:
    """O ponto medio tambem registra o que foi considerado, e por isso a lista pode vir cheia.

    Cair no ponto medio nao quer dizer que nao havia quebra nenhuma: quer dizer que nenhuma passou
    de `MIN_SCORE`. A diferenca importa no diagnostico, porque clipe sem candidato e problema de
    deteccao da bola e clipe com quatro candidatos fracos e problema de limiar.
    """
    escolhido = indices[len(indices) // 2]
    return {
        "frame_index": escolhido,
        "confidence": FALLBACK_CONFIDENCE,
        "source": ESTIMADO,
        "reason": "a trajetoria da bola nao sustentou uma quebra, e o instante e o meio do trecho",
        "candidates": _candidatos(ordenados),
        "candidate_pool": _pool(ordenados),
        **suporte(escolhido, procedencia),
    }


def _pool(ordenados: Sequence[Candidate]) -> list[int]:
    """O quadro de TODA quebra pontuada, sem score, ordenado como a escolha as viu.

    Serve a uma pergunta que a lista de cinco nao responde: quanto do acerto e criterio e quanto e
    a largura da tolerancia. Com 54 quebras por clipe na mediana e uma tolerancia de 30 quadros,
    sortear quatro delas ao acaso ja cai dentro da tolerancia na maioria dos clipes, e um teto
    apresentado sem essa conta parece maior do que e.

    So o quadro, e nao o registro inteiro: o score das descartadas nao entra em nenhuma conta, e
    guardar os tres campos de 133 quebras seria guardar a trajetoria de novo com outro nome.
    """
    return [candidato.frame_index for candidato in ordenados]


def _candidatos(ordenados: Sequence[Candidate]) -> list[dict[str, object]]:
    """As `CANDIDATOS_REGISTRADOS` maiores quebras, na ordem em que a escolha as viu."""
    return [
        {
            "frame_index": candidato.frame_index,
            "sample_index": candidato.sample_index,
            "score": round(candidato.score, 6),
        }
        for candidato in ordenados[:CANDIDATOS_REGISTRADOS]
    ]


def suporte(quadro: int, procedencia: Mapping[int, str]) -> dict[str, object]:
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
        "ball_support": _apoio(quadro, procedencia),
        "ball_detected_frames": contagem.get(DETECTADA, 0),
        "ball_interpolated_frames": contagem.get(INTERPOLADA, 0),
    }


def _apoio(quadro: int, procedencia: Mapping[int, str]) -> str:
    """A procedencia no quadro escolhido, resolvida pelas amostras que o cercam quando ele cai fora.

    Ler `procedencia[quadro]` direto era um defeito: `find_contact` refina a quebra pela intersecao
    das retas e devolve quadro entre amostras, entao o `get` errava o dicionario no caso normal e
    o registro saia `ball-break` com apoio `absent`. Medido com passo quatro, isso acontecia em
    tres de cada quatro contatos, e os testes nao pegavam porque usavam passo doze com o vertice em
    cima da amostra.
    """
    if quadro in procedencia:
        return procedencia[quadro]
    anteriores = [indice for indice in procedencia if indice <= quadro]
    posteriores = [indice for indice in procedencia if indice >= quadro]
    if not anteriores or not posteriores:
        return AUSENTE
    fontes = {procedencia[max(anteriores)], procedencia[min(posteriores)]}
    return fontes.pop() if len(fontes) == 1 else MISTA


def saida_unica(
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
