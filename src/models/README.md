# Modelos

Uma pasta por modelo, com o código que o treina e o que o avalia. O manifesto do peso **não** mora
aqui: ele fica em `models/` na raiz, indexado por espécie, porque a maioria dos pesos que o projeto
usa é de terceiros e não tem código nosso que os produza.

| pasta | o que é |
| --- | --- |
| `ball/` | fine tuning do detector da bola sobre o SoccerNet, mais os scripts de avaliação |
| `top_view/` | desenha a visão superior a partir dos outros modelos; é consumidor, não modelo |
| `track_players/` | treino e análise do detector de jogadores |

**Nada aqui roda no runtime da borda.** É por isso que `src/models` está fora da lista de pacotes do
wheel em `pyproject.toml`, fora de `pyright.include` e fora de `coverage.run.source`: são scripts que
exigem GPU e dataset local, e medir cobertura de linha neles produz um número que não diz nada.

Quem consome os pesos em produção é `src/edge`, pelo manifesto, com o sha256 conferido.
