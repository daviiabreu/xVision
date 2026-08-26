# Dados locais

Esta pasta e uma das tres raizes que `edge/paths.py` define, e a unica das tres em que **nada que
o sistema produz e escrito**. Aqui entra so o que o parceiro mandou.

- `data/footstats/` — o dump da Footstats, material restrito de 3,5 GB. E o valor padrao de
  `data_root()`, e quem guarda o dump em outro lugar aponta `XVISION_DATA_DIR` para la em vez de
  copiar. Os clipes de finalizacao ficam sob essa raiz.
- `var/` — tudo que o sistema escreve: `var/cache/perception`, `var/out`, `var/runs` e
  `var/models`. Apagar `var/` inteiro deixa o sistema funcionando, so mais lento.

Nenhuma das duas e versionada, e a separacao nao e organizacao: misturar o que escrevemos com o
dump faz o contexto de um `docker build` passar a incluir video de partida.

Os pesos de modelo tambem nao vem no Git. O manifesto em `models/<especie>/` descreve cada um, e
`uv run xvision artifact get <manifesto>` diz o diretorio exato sob `var/models/` onde por o
arquivo e confere o sha256 depois que ele estiver la.

Nao inclua neste arquivo nomes de partidas, identificadores, estatisticas ou qualquer conteudo
fornecido pela Footstats.
