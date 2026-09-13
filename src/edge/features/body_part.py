"""Com que perna o finalizador tocou a bola, decidido por quem balancou e nao por quem estava perto.

E a metrica com mais folga do conjunto. A maioria vale 0,519 sobre os 604 registros, contra 0,894
de `looked_before` e 0,876 de `goalkeaper_position`, entao e onde um modelo tem mais espaco para
provar valor. E o gabarito e o mais confiavel que existe ali: `cortes_metadados.foot` reproduz
`match_shots.body_part` em 633 de 633.

Esse mesmo numero e a razao de a barra nao ser a maioria. O metadado do corte ja resolve o rotulo
sem tocar em video, entao `config/labels.toml` declara `baseline_metadata = 1.0` para ela, e o
relatorio imprime os dois lados. Uma feature de video aqui nao acrescenta informacao ao parceiro:
ela prova que a cadeia de percepcao mede o que diz medir, sobre o rotulo em que isso e verificavel.

A regra por distancia da bola ao tornozelo nao cabe nesta escala, e a medida e direta: sobre os 38
clipes com o instante anotado a mao, o finalizador tem 93,4 px de altura na imagem (mediana; 23,5
minimo, 131,1 maximo) e os dois tornozelos ficam a 10,7 px um do outro (mediana; 2,0 minimo). A
bola tem cerca de 24 px de diametro, entao ela encosta nos dois ao mesmo tempo. Sobre a dobra de
ajuste, a regra por distancia acertava 6 dos 19 clipes que respondia, abaixo da maioria, que vale
8 de 19, e a perna certa saia em 4 de 15.

A regra desta versao le a assinatura fisica do chute: a perna que finaliza balanca e a de apoio
sustenta o peso. Ganha o tornozelo com maior velocidade media numa janela em torno do contato.
Sobre a mesma dobra de ajuste isso acerta 9 de 19 contra 6 de 19 da regra por distancia, com a
mesma cobertura de 19 de 24, e a maioria vale 8 de 19. Onde a regra se aplica de verdade, os
clipes de perna, sao 9 de 13 contra 4 de 15.

Tres clipes de diferenca nao sustentam manchete, e o relatorio precisa dizer isso: nos 14 clipes
que as duas regras respondem, a nova acerta sozinha em 7 e a antiga em 4, o que da p = 0,55 no
teste de McNemar. O que esta medido e que a velocidade nao e pior que a distancia, e que ela e a
unica das duas com justificativa fisica na escala em que o clipe chega.

**Ela empata com a constante, e nao ganha dela.** Sobre os 38 clipes anotados: 14 de 32 = 0,438,
IC 95% de Wilson [0,282; 0,607], contra 10 de 31 = 0,323 da regra por distancia e exatamente 14
de 32 da maioria. Na dobra reservada, que so foi lida depois de os dois limiares estarem
congelados, sao 5 de 13 = 0,385 contra 6 de 13 da maioria, ou seja abaixo dela por um clipe. A
leitura honesta e que a mudanca troca uma regra que perdia da constante por uma que a alcanca, e
que 38 clipes nao separam 0,438 de 0,462. Passar da constante depende do que ainda nao foi feito:
a cabecada, que hoje entra inteira na conta de erro, e a pose no conjunto `window`, que hoje nao
existe porque `players` so roda no `dense` e a velocidade fica amostrada a 15 fps.

Quem e o finalizador continua conferido aqui, e nao herdado da especie `pose`. Quando o evento
declara `shooter_track_id`, a feature procura esse corpo; sem ele, com um corpo so a bola nao e
consultada e, com mais de um, a proximidade da bola separa finalizador de marcador.

Nao mede angulo, nao decide se o chute foi bom e nao inventa quando a pose nao sustenta: confianca
de keypoint abaixo do piso sai `unavailable` com motivo, porque tornozelo ocluso e ausencia e nao
chute do outro pe.

**Entrega parcial declarada**: a classe `CABECA` nao e produzida. O separador disponivel era a
distancia da bola a cabeca contra a distancia aos tornozelos, e sobre a dobra de ajuste nenhum
limiar dela paga o que custa. Varrida a razao de 0,3 a 1,5, a taxa total cai de 9 para 8, 7, 5, 5,
5 e 6 em 19: no ponto em que a porta comeca a acertar cabecada ela ja disse `CABECA` 6 vezes para
1 acerto. O motivo tem numero: cabeca e tornozelo ficam a 67,5 px um do outro (mediana sobre os
38 clipes) e a bola detectada hoje cai a 55,0 px do tornozelo mais proximo no quadro do contato
(mediana sobre a dobra de ajuste), ou seja o erro da bola e da ordem da distancia que ele
precisaria resolver. Enquanto isso, toda cabecada coberta sai contada como erro, e sao 6 dos 19
clipes que a feature responde na dobra de ajuste.

A hipotese a testar na proxima sprint e direta: com um detector de bola cujo erro mediano no
contato caia abaixo de 20 px, refazer esta mesma varredura de limiar e verificar se a porta da
cabecada passa a somar em vez de subtrair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contracts import FeatureValue, FramePerception, Perception, PlayerDetection
from contracts.perception import Keypoint
from edge.features.base import FeatureSpec, ShotOutput, unavailable

PERNA_ESQUERDA = "PERNA_ESQUERDA"
PERNA_DIREITA = "PERNA_DIREITA"
CABECA_NAO_E_SEPARAVEL = """A classe CABECA nao e produzida, e a medida que fecha isso e do teto.

A razao entre a distancia da bola ate a cabeca e ate o tornozelo mais proximo, medida no quadro do
contato anotado com o detector de bola ja corrigido, tem mediana 9,32 nas cabecadas e 4,51 nos
chutes de perna. Ela esta invertida em relacao a fisica: nas cabecadas a bola fica MAIS LONGE da
cabeca. Varrendo todos os cortes possiveis e escolhendo o melhor OLHANDO A RESPOSTA, o acerto e
24 de 32, exatamente igual a responder sempre PERNA. Ou seja o teto do metodo empata com a
constante, e qualquer corte escolhido honestamente fica abaixo.

Medido tambem no pipeline: a porta com corte em 1,0 disparou em 9 clipes, todos de perna, e em
nenhuma das 8 cabecadas verdadeiras. Sobre a dobra de ajuste, so velocidade da 9 de 20 e velocidade
com porta da 4 de 20.

O que falta para a cabecada voltar a ser possivel nao e limiar. E a bola no quadro certo do lance:
a inversao sugere que em cabecada, que quase sempre vem de cruzamento e area cheia, o corpo
escolhido como finalizador nao e o que cabeceou. Testar isso pede escolher o finalizador por outra
coisa que nao proximidade da bola, e essa e a hipotese da proxima sprint."""

TORNOZELO_ESQUERDO = "ankle_left"
TORNOZELO_DIREITO = "ankle_right"
LADOS = ((PERNA_ESQUERDA, TORNOZELO_ESQUERDO), (PERNA_DIREITA, TORNOZELO_DIREITO))

MIN_CONFIANCA = 0.3
"""Piso da confianca do keypoint. Abaixo dele o estimador esta chutando a posicao do ponto.

O piso nao e limiar varrido: a confianca do tornozelo e bimodal e ele cai no vao vazio entre os
dois modos. Medido sobre as 206 amostras de tornozelo da janela na dobra de ajuste, p10 = 0,016,
p25 = 0,597 e a mediana e 0,932, ou seja o estimador ou enxerga o pe ou declara que nao enxerga.

Ele custa cobertura, e o numero esta medido: em 5 dos 24 clipes de ajuste os dois tornozelos ficam
abaixo do piso na janela inteira, e ali a feature sai `unavailable`. Desligar o piso levaria a
cobertura a 24 de 24 e a taxa a 12 de 24, contra 9 de 19 com ele. Nao foi adotado: cinco clipes
nao bastam para passar por cima do proprio estimador dizendo que nao viu o pe, e a alternativa e
publicar decisao tomada sobre ponto que ninguem mediu.
"""

JANELA_S = 0.25
"""Meia-largura, em segundos, da janela em que a velocidade do tornozelo e medida.

Ha patamar largo, varrido na dobra de ajuste sobre os clipes de perna: 0,750 com 0,10 s, 0,833 com
0,15 s e 0,20 s, 0,692 com 0,25 s, 0,769 com 0,30 s, 0,692 com 0,35 s e 0,40 s, e 0,538 com 0,50 s.
Tudo entre 0,10 s e 0,40 s cabe em dois clipes de diferenca, e so depois de meio segundo a medida
desaba. E o que a fisica preve: o balanco do chute dura cerca de um quarto de segundo, e passando
disso entra a corrida, em que os dois tornozelos alternam e a media empata.

0,25 s e o meio desse patamar e a menor janela que ainda cobre 19 dos 24 clipes. 0,30 s daria um
acerto a mais, e escolher por um clipe seria calibrar no ruido.
"""

MAX_DISTANCIA_QUADROS = 12
"""Quantos quadros a busca pela bola pode se afastar do contato, para separar dois corpos.

A especie `ball` roda no conjunto grosso, a 5 fps, entao o quadro do contato quase nunca tem
amostra de bola. Doze quadros e o passo do grosso a 59,94 fps: alem disso a bola ja saiu do pe e a
comparacao passaria a medir para onde ela foi, e nao de onde ela saiu.
"""


@dataclass(frozen=True)
class _Amostra:
    """Um quadro em que os dois tornozelos do finalizador se sustentam, com o instante em segundos.

    O tempo vem de `FramePerception.timestamp_s` e nao do indice do quadro porque a janela e uma
    duracao fisica, o balanco do chute, e o conjunto de quadros que a alimenta muda de taxa entre
    `dense` e `window` sem que a duracao mude.
    """

    timestamp_s: float
    tornozelos: Mapping[str, tuple[float, float]]


class BodyPart:
    spec = FeatureSpec(
        name="body_part",
        version="0.2.0",
        requires=("pose",),
        unit="classe",
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        com_pose = _quadros_com_pose(perception)
        if not com_pose:
            return unavailable(self.spec, "nenhum jogador tem keypoint na janela do contato")
        referencia = min(com_pose, key=lambda quadro: abs(quadro.frame_index - shot.frame_index))

        finalizador = _finalizador(referencia, perception, shot)
        if finalizador is None:
            return unavailable(
                self.spec,
                f"{len([j for j in referencia.players if j.keypoints])} corpos com keypoint no "
                "quadro do contato e a especie ball sem amostra para separar o finalizador",
            )

        amostras = _amostras(_trilha(com_pose, finalizador))
        janela = _na_janela(amostras, referencia.timestamp_s)
        velocidades = _velocidades(amostras, janela)
        if velocidades is None:
            return unavailable(
                self.spec,
                f"o finalizador tem {len(janela)} quadros com os dois tornozelos acima de "
                f"{MIN_CONFIANCA} de confianca na janela de {JANELA_S} s em torno do contato, "
                "e a comparacao precisa de dois",
            )

        escolhida = max(velocidades, key=lambda parte: velocidades[parte])
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=escolhida,
            confidence=_confianca(velocidades),
            reason=None,
            evidence={
                "frame_do_corpo": referencia.frame_index,
                "quadros_na_janela": len(janela),
                "velocidade_px_s": {parte: round(v, 2) for parte, v in velocidades.items()},
            },
        )


def _quadros_com_pose(perception: Perception) -> tuple[FramePerception, ...]:
    return tuple(q for q in perception.frames if any(j.keypoints for j in q.players))


def _finalizador(
    referencia: FramePerception, perception: Perception, shot: ShotOutput
) -> PlayerDetection | None:
    """Entre os corpos com keypoint do quadro de referencia, o do finalizador.

    Quando o evento declara `shooter_track_id`, ele ganha porque ja resolveu a associacao. Com um
    corpo so a resposta e ele, e exigir bola ali custava cobertura sem comprar nada: 7 dos 38
    clipes anotados nao tem amostra de bola a menos de 12 quadros do contato, e todos os 7 tem
    exatamente um corpo com keypoint. Com mais de um corpo a bola volta a ser obrigatoria.

    E por isso que `requires` declara so `pose`. Declarar `ball` tambem barraria a feature inteira
    em `_run_one` antes de `extract` rodar, e o codigo aqui nunca teria chance de aplicar a regra do
    corpo unico: os mesmos 7 clipes voltariam a sair `unavailable`, agora por uma porta que a
    propria funcao sabe abrir.
    """
    corpos = [jogador for jogador in referencia.players if jogador.keypoints]
    if shot.shooter_track_id is not None:
        declarado = next(
            (jogador for jogador in corpos if jogador.track_id == shot.shooter_track_id), None
        )
        if declarado is not None:
            return declarado
    if len(corpos) == 1:
        return corpos[0]
    bola = _bola_perto_de(perception, shot.frame_index)
    if bola is None:
        return None
    return min(corpos, key=lambda jogador: _distancia(_centro_da_caixa(jogador), bola))


def _trilha(
    com_pose: Sequence[FramePerception], finalizador: PlayerDetection
) -> tuple[FramePerception, ...]:
    """Os quadros em que o corpo do finalizador aparece, ele proprio embutido em `players[0]`.

    O casamento e por `track_id` quando o rastreador deu um, e por proximidade da caixa quando
    nao deu. Sem casar corpo com corpo, a velocidade sairia da diferenca entre o tornozelo de um
    jogador num quadro e o de outro no seguinte, que e ruido com cara de balanco.
    """
    quadros: list[FramePerception] = []
    for quadro in com_pose:
        corpo = _mesmo_corpo(quadro, finalizador)
        if corpo is not None:
            quadros.append(_so_com(quadro, corpo))
    return tuple(quadros)


def _mesmo_corpo(quadro: FramePerception, finalizador: PlayerDetection) -> PlayerDetection | None:
    corpos = [jogador for jogador in quadro.players if jogador.keypoints]
    if not corpos:
        return None
    if finalizador.track_id is not None:
        return next((j for j in corpos if j.track_id == finalizador.track_id), None)
    if len(corpos) == 1:
        return corpos[0]
    alvo = _centro_da_caixa(finalizador)
    return min(corpos, key=lambda jogador: _distancia(_centro_da_caixa(jogador), alvo))


def _so_com(quadro: FramePerception, corpo: PlayerDetection) -> FramePerception:
    return FramePerception(
        frame_index=quadro.frame_index,
        timestamp_s=quadro.timestamp_s,
        players=(corpo,),
        ball=quadro.ball,
        field=quadro.field,
    )


def _amostras(trilha: Sequence[FramePerception]) -> tuple[_Amostra, ...]:
    """So os quadros em que os DOIS tornozelos se sustentam, porque a medida e uma comparacao.

    Aceitar um quadro em que so um lado passa o piso faria as duas medias sairem de conjuntos de
    quadros diferentes, e o que a comparacao mediria em parte seria a diferenca entre os instantes,
    nao entre as pernas. Na dobra de ajuste isso nao move o numero, 9 de 19 pareado e despareado, e
    o pareamento fica porque a alternativa e um resultado que depende de qual quadro cada lado
    pegou.
    """
    saida: list[_Amostra] = []
    for quadro in trilha:
        keypoints = quadro.players[0].keypoints
        esquerdo = _keypoint(keypoints, TORNOZELO_ESQUERDO)
        direito = _keypoint(keypoints, TORNOZELO_DIREITO)
        if esquerdo is not None and direito is not None:
            saida.append(
                _Amostra(
                    quadro.timestamp_s,
                    {
                        PERNA_ESQUERDA: (esquerdo.x, esquerdo.y),
                        PERNA_DIREITA: (direito.x, direito.y),
                    },
                )
            )
    return tuple(saida)


def _na_janela(amostras: Sequence[_Amostra], centro_s: float) -> tuple[_Amostra, ...]:
    return tuple(a for a in amostras if abs(a.timestamp_s - centro_s) <= JANELA_S)


def _velocidades(
    amostras: Sequence[_Amostra], janela: Sequence[_Amostra]
) -> dict[str, float] | None:
    """Media do modulo da velocidade de cada tornozelo na janela, em px/s.

    A media ganha do maximo por margem medida na dobra de ajuste com o piso de confianca desligado,
    13 de 24 contra 11 de 24: a 15 fps o pico do balanco cai entre duas amostras, e o maximo passa
    a medir qual dos dois lados teve o erro de estimativa mais alto naquele quadro.

    A diferenca e centrada, e as vizinhas saem da trilha inteira e nao da janela: usar so o que
    esta dentro dela deixaria as amostras da borda com diferenca de um lado so, que e outra
    grandeza.
    """
    if len(janela) < 2:
        return None
    dentro = {a.timestamp_s for a in janela}
    medias: dict[str, float] = {}
    for parte, _ in LADOS:
        valores = [
            v
            for i, a in enumerate(amostras)
            if a.timestamp_s in dentro and (v := _velocidade_em(amostras, i, parte)) is not None
        ]
        if not valores:
            return None
        medias[parte] = sum(valores) / len(valores)
    return medias


def _velocidade_em(amostras: Sequence[_Amostra], i: int, parte: str) -> float | None:
    antes = amostras[max(i - 1, 0)]
    depois = amostras[min(i + 1, len(amostras) - 1)]
    intervalo = depois.timestamp_s - antes.timestamp_s
    if intervalo <= 0:
        return None
    return _distancia(antes.tornozelos[parte], depois.tornozelos[parte]) / intervalo


def _keypoint(keypoints: Sequence[Keypoint], nome: str) -> Keypoint | None:
    """O ponto pedido, e so quando ele se sustenta: abaixo do piso o estimador esta chutando."""
    return next(
        (
            k
            for k in keypoints
            if k.name == nome and (k.confidence is None or k.confidence >= MIN_CONFIANCA)
        ),
        None,
    )


def _centro_da_caixa(jogador: PlayerDetection) -> tuple[float, float]:
    caixa = jogador.box
    return ((caixa.x1 + caixa.x2) / 2.0, (caixa.y1 + caixa.y2) / 2.0)


def _bola_perto_de(perception: Perception, contato: int) -> tuple[float, float] | None:
    candidatos = [
        (abs(quadro.frame_index - contato), quadro)
        for quadro in perception.frames
        if quadro.ball is not None
    ]
    if not candidatos:
        return None
    distancia, quadro = min(candidatos, key=lambda item: item[0])
    if distancia > MAX_DISTANCIA_QUADROS or quadro.ball is None:
        return None
    caixa = quadro.ball.box
    return ((caixa.x1 + caixa.x2) / 2.0, (caixa.y1 + caixa.y2) / 2.0)


def _confianca(velocidades: Mapping[str, float]) -> float:
    """O quanto o tornozelo escolhido se destaca do outro, e nao o quanto ele e rapido.

    Dois tornozelos a 90 e a 92 px/s sao um empate tecnico, e reportar confianca alta ali seria
    afirmar uma separacao que a medida nao tem. Nesta feature o numero tem uso pratico imediato:
    ele e o unico jeito de o consumidor separar os clipes em que a regra decidiu dos clipes em que
    ela desempatou ruido, e a acuracia de 0,438 sobre 32 clipes nao distingue os dois sozinha.
    """
    melhor, segunda = max(velocidades.values()), min(velocidades.values())
    if melhor + segunda <= 0:
        return 0.0
    return round((melhor - segunda) / (melhor + segunda), 4)


def _distancia(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
