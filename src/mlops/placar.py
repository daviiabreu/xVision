"""O placar de uma configuracao: percepcao da bola, instante e features, na mesma tabela.

Existe porque medir uma correcao pede tres comandos e a comparacao entre eles some no terminal. Sem
juntar, uma mudanca que sobe a taxa da bola e derruba `body_part` passa por vitoria, e foi
exatamente esse o risco medido: nos clipes com bola boa a acuracia de tres classes cai (3/11 contra
7/20) enquanto a de duas sobe (0,833 contra 0,480), entao qual numero se olha decide a conclusao.

`body_part` sai nas duas formas de proposito. Tres classes e o rotulo do parceiro e o que a entrega
promete; duas classes separa o que a resolucao permite do que ela nao permite, porque a cabeca fica
a cerca de 80 px dos tornozelos enquanto os dois tornozelos ficam a 12,8 px um do outro. Publicar so
a de tres esconde que metade do problema esta resolvida; publicar so a de duas esconde que a outra
metade nao esta.

Nao roda nada: recebe os caminhos do que ja foi medido e monta a tabela. Assim o placar entra em
teste sem GPU e sem clipe.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mlops.instant import wilson

CABECA = "CABECA"
PE = "PE"


def colapsa(rotulo: str) -> str:
    """Junta as duas pernas numa classe so, que e a distincao que a resolucao sustenta."""
    return CABECA if rotulo == CABECA else PE


@dataclass(frozen=True)
class Linha:
    nome: str
    acertos: int
    n: int
    referencia: float
    referencia_nome: str

    @property
    def taxa(self) -> float:
        return self.acertos / self.n if self.n else 0.0

    @property
    def intervalo(self) -> tuple[float, float]:
        return wilson(self.acertos, self.n)

    @property
    def ganha(self) -> bool:
        """Ganhar e o limite inferior do intervalo passar da referencia, e nao a media passar.

        Com 14 a 38 clipes a media sozinha sobe e desce por um acerto. Exigir o limite inferior
        evita declarar vitoria que o proximo clipe anotado desfaz.
        """
        return self.intervalo[0] > self.referencia


def previsoes(eventos: Path, feature: str) -> dict[str, str]:
    """As previsoes de uma feature numa execucao, ignorando clipe em que ela saiu indisponivel."""
    saida: dict[str, str] = {}
    if not eventos.is_file():
        return saida
    for linha in eventos.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        evento = json.loads(linha)
        for f in evento.get("features", ()):
            if f.get("name") == feature and f.get("value"):
                saida[str(evento["clip_id"])] = str(f["value"])
    return saida


def maioria(verdades: Sequence[str]) -> float:
    if not verdades:
        return 0.0
    contagem: dict[str, int] = {}
    for v in verdades:
        contagem[v] = contagem.get(v, 0) + 1
    return max(contagem.values()) / len(verdades)


def linha_de_feature(
    nome: str, previsto: Mapping[str, str], verdade: Mapping[str, str]
) -> Linha | None:
    comuns = sorted(set(previsto) & set(verdade))
    if not comuns:
        return None
    return Linha(
        nome=nome,
        acertos=sum(1 for c in comuns if previsto[c] == verdade[c]),
        n=len(comuns),
        referencia=maioria([verdade[c] for c in comuns]),
        referencia_nome="maioria",
    )


def linha_de_bola(relatorio: Mapping[str, object], acaso: float) -> Linha:
    return Linha(
        nome="bola no contato",
        acertos=int(str(relatorio["encostada"])),
        n=int(str(relatorio["n"])),
        referencia=acaso,
        referencia_nome="acaso",
    )


def tabela(linhas: Sequence[Linha]) -> str:
    largura = max((len(linha.nome) for linha in linhas), default=0)
    saida = [f"  {'medida':<{largura}}  {'acerto':>10}  {'IC95':>16}  referencia"]
    for linha in linhas:
        lo, hi = linha.intervalo
        marca = "PASSA" if linha.ganha else "  -  "
        saida.append(
            f"  {linha.nome:<{largura}}  {linha.acertos:>4}/{linha.n:<5} "
            f"[{lo:.3f}, {hi:.3f}]  {linha.referencia_nome} {linha.referencia:.3f}  {marca}"
        )
    return "\n".join(saida)
