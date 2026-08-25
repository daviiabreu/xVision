# Backend

O caminho do clipe de vídeo ao evento de finalização, e a plataforma sobre a qual os modelos
entram. Esta página é do backend; o README da raiz é do projeto inteiro e a documentação
avaliada é publicada pelo site em `docs/`.

As áreas, e o que cada uma faz:

| Área | Responsabilidade | O que ela não faz |
| --- | --- | --- |
| `contracts/` | o acordo entre a borda e quem consome o evento | não importa ninguém |
| `edge/` | decode, módulos de percepção, features e o pipeline que os liga | não lê CSV nem serve HTTP |
| `mlops/` | conjunto do parceiro, rótulos, partição e avaliação | não roda pipeline |
| `api/` | o serviço HTTP que o frontend consome, com [documentação própria](api/docs/README.md) | não lê CSV |
| `cli/` | ponto de entrada de linha de comando | ninguém o importa |
| `interface/` | o frontend, quando a branch dele for mesclada | é TypeScript, não entra nos portões de Python |

`cli` e `api` são pontos de entrada e ninguém os importa: a fronteira entre as áreas é verificada
por teste, não confiada à disciplina.

`interface/` ainda não existe aqui, e o lugar já está reservado: a branch `feat/frontend` põe o
app em `src/interface/`. Os testes que varrem `src/` definem área como pasta **com Python
dentro**, então o dia em que ele entrar nenhum portão falha em cima de quem fizer o merge.

## Como rodar

Requisito único: [uv](https://docs.astral.sh/uv/). Ele instala o próprio Python, então não é
preciso ter a versão certa na máquina.

```bash
git clone <URL do repositório>
cd g02
uv sync --frozen --all-extras
```

Para processar um clipe, começando pelo de exemplo, que é sintético e vem do próprio código:

```bash
uv run xvision clip
uv run xvision run --clip var/out/synthetic.mp4
```

A saída é uma linha por etapa, com o tempo de cada uma, e o caminho do evento no fim:

```text
  ball       ok            2.7 ms
  field      ok            0.1 ms
  players    ok            0.1 ms
  shot       ok            0.1 ms
  identity   ok            4.1 ms
  pose       ok            0.3 ms
  tracking   ok            0.5 ms
var/runs/synthetic/events.jsonl
```

Tudo que o sistema produz vai para `var/`, que não é versionado: `var/runs/` guarda as execuções,
`var/cache/` o cache de percepção e `var/out/` os clipes gerados. Apagar `var/` inteiro não
quebra nada, só deixa a próxima execução mais lenta.

Se preferir contêiner, sem instalar nada além do Docker:

```bash
docker compose run --rm edge run --clip /data/<clipe>   # a borda, por linha de comando
docker compose up api                                   # o serviço HTTP, na porta 8080
```

### Com o conjunto do parceiro

O material da Footstats é restrito e não vem no repositório. Com ele em `data/footstats/`:

```bash
uv run xvision data check
uv run xvision run --all --limit 20
uv run xvision eval var/runs/<execução>
```

O `data check` conta o que existe e como está particionado. O `eval` casa os eventos com o
gabarito do parceiro e imprime uma linha por métrica, com a acurácia ao lado da acurácia de um
chute constante, que é o piso contra o qual qualquer modelo precisa ser comparado.

## Por onde começa quem vai implementar um módulo de percepção

Os sete módulos são stubs: respondem do gabarito da fixture sintética, não de vídeo. Trocar um
stub por um modelo de verdade não exige entender o pipeline inteiro, e três provas ao vivo
mostram cada peça isolada, rodando:

```bash
uv run python -m contracts.demo      # monta um evento e valida contra o schema
uv run python -m edge.frames.demo --clip var/out/synthetic.mp4   # decode único e cache
uv run python -m edge.graph.demo     # monta o grafo da configuração, roda e mostra as etapas
```

O molde de um módulo está em `src/edge/testing/example_module.py`, o de uma feature em
`src/edge/testing/example_feature.py`, e o contrato que o módulo cumpre

está em `src/edge/modules/base.py`. Um módulo novo precisa de três coisas: declarar a `ModuleSpec`,
implementar `infer` devolvendo as chaves que `src/edge/modules/records.py` exige para a espécie, e
entrar em `config/pipeline.toml`. A suíte de conformidade (`--extra conformance`) roda contra ele
sem que você escreva teste nenhum.

## O que foi entregue, o que está parcial e o que ficou para depois

| Item | Situação | Evidência |
| --- | --- | --- |
| Contrato do evento de finalização | entregue | `src/contracts/`, JSON Schema v1 como fonte única, validação nos dois lados |
| Interface que todo modelo cumpre | entregue | `src/edge/modules/base.py` e a suíte de conformidade em `conformance.py` |
| Entrega de quadro com decode único | entregue | `src/edge/frames/provider.py` |
| Runner do grafo, com falha e skip declarados | entregue | `src/edge/graph/runner.py` |
| Plataforma de features com degradação | entregue | `src/edge/features/base.py` |
| Do clipe ao evento por linha de comando | entregue | `uv run xvision run --clip` |
| Serviço HTTP que serve o frontend | entregue | `src/api/`, contrato em `src/api/docs/openapi.json` |
| Leitura do conjunto do parceiro | entregue | `src/mlops/catalog.py`, `uv run xvision data check` |
| Rótulos e partição por partida | entregue | `config/labels.toml`, `config/split.toml` |
| Avaliação contra o gabarito | entregue | `src/mlops/evaluate.py`, `uv run xvision eval` |
| Peso de modelo endereçado por conteúdo | entregue | `src/edge/artifacts.py`, `uv run xvision artifact` |
| Ambiente reproduzível em contêiner | entregue | `Dockerfile`, `compose.yaml` |
| Infraestrutura de produção declarada | entregue, não implantada | `infra/`, com `terraform validate` e `plan` no CI |
| Os sete módulos de percepção | **parcial** | são stubs: respondem do gabarito da fixture, não de vídeo |
| Features de nível 1, 2 e 3 | **parcial** | a plataforma existe e nenhuma feature está declarada em `config/pipeline.toml`; a cobertura sai 0% no `eval` |
| PostgreSQL em produção | adiado | a topologia declara o armazenamento sem amarrar provedor, e escolher um é decisão que o time ainda não tomou |
| Transporte da borda para a nuvem | adiado | o contrato está pronto; o transporte depende da frente de serviços |
| Imagem para a Jetson | adiado | ninguém tem a placa nesta sprint; a armadilha da OpenCV duplicada está no `Dockerfile` |
| Compartilhar decode entre processos | adiado | a borda roda um fluxo por vez, então não há caso de uso |

Duas medidas que limitam o que dá para prometer, e que valem para quem for definir critério de
aceite. A distância de referência do parceiro tem grão de 5 metros, com dez valores distintos
em 2585 finalizações, então erro médio abaixo de uns 2,5 m mede o rótulo e não o modelo. E a
marcação de tempo é em segundos inteiros, o que a 59,94 fps são cerca de 60 quadros.
