# Modelos

Uma pasta por modelo, com o código que o treina e o que o avalia. Os manifestos versionados ficam
em `manifests/`, indexados por espécie. Eles descrevem tanto pesos produzidos pelo time quanto
pesos de terceiros, sem guardar o binário no Git.

| pasta | o que é |
| --- | --- |
| `ball/` | fine tuning do detector da bola sobre o SoccerNet, mais os scripts de avaliação |
| `manifests/` | identidade, versão e digest dos pesos consumidos em execução |
| `top_view/` | desenha a visão superior a partir dos outros modelos; é consumidor, não modelo |
| `track_players/` | treino e análise do detector de jogadores |

**O extra `training` declara o que estes scripts importam**, com uma excecao deliberada: o
`roboflow`, que so o `track_players/train_player_detector.py` usa. Medido, acrescenta-lo ao lock
arrasta `typer` de 0.27.1 para 0.25.1 e quebra o `pyright` com 33 erros; prendendo o `typer`, ele
puxa `httpx2` de 2.12 para 2.3 junto com `idna` e `opencv-python-headless`. Um script de operador
que roda uma vez nao paga esse preco para o projeto inteiro, entao ele entra por cima, com
`uv run --with roboflow`, e o proprio script diz isso na docstring.

**Nada aqui roda no runtime da borda.** É por isso que `src/models` está fora da lista de pacotes do
wheel em `pyproject.toml`, fora de `pyright.include` e fora de `coverage.run.source`: são scripts que
exigem GPU e dataset local, e medir cobertura de linha neles produz um número que não diz nada.

Quem consome os pesos em produção é `edge`, pelo manifesto, com o sha256 conferido.
