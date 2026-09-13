# Modelos

Uma pasta por modelo, com o código que o treina e o que o avalia. Os manifestos versionados ficam
em `manifests/`, indexados por espécie. Eles descrevem tanto pesos produzidos pelo time quanto
pesos de terceiros, sem guardar o binário no Git.

| pasta | o que é |
| --- | --- |
| `ball/` | fine tuning do detector da bola sobre o SoccerNet, mais os scripts de avaliação |
| `field_keypoints/` | detector de pontos do campo e seu `best.pt` |
| `homography/` | cálculo da homografia, projeção e desenho da geometria do campo |
| `manifests/` | identidade, versão e digest dos pesos consumidos em execução |
| `top_view/` | executa a pipeline e renderiza a visão superior |
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

## Pipeline da visão superior

A implementação está separada por responsabilidade:

- `field_keypoints/model.py`: carrega o modelo e detecta pontos do campo em pixels.
- `homography/geometry.py`: calcula a homografia e projeta as posições.
- `top_view/pipeline.py`: coordena as etapas.
- `top_view/create_top_view.py`: renderiza o resultado.

Um decoder entrega cada quadro a dois workers: jogadores/tracking e detecção de pontos do campo.
A bola está fora desta pipeline por enquanto.
As inferências do mesmo quadro rodam em paralelo; o próximo quadro só avança quando elas
terminam. Isso limita a memória das imagens e preserva o estado temporal dos rastreadores.

Depois, a identificação existente resolve os times sobre o clipe completo. Os resultados
são unidos por `frame_index`, as posições são projetadas e a marcação é calculada em metros.
Sem homografia válida, não há marcação em metros naquele quadro.
`create_top_view.py` desenha os resultados prontos, incluindo os pares de marcação.

Execute a partir de `src/`, com o ambiente do projeto e os pesos disponíveis:

```bash
python models/top_view/create_top_view.py /caminho/video.mp4 \
  --output-dir /caminho/resultados
```

Os pesos são encontrados automaticamente em `models/track_players/best.pt` e
`models/field_keypoints/best.pt`. Treino e preparação de datasets não fazem parte dessa pipeline.
A API e o runtime em `edge` não são modificados;
a chamada existente com quatro argumentos continua funcionando.

A identificação reutilizada ainda relê o vídeo e gera seu vídeo intermediário, removido
pela pipeline após essa etapa. O render final também relê o vídeo. O decode compartilhado
é o das inferências, sem guardar os frames de todo o clipe em memória. O paralelismo não
garante aceleração na mesma GPU: o ganho e o consumo de VRAM precisam ser medidos no hardware.

Para rodar a partir da raiz do repositório:

```bash
cd src
python models/top_view/create_top_view.py ../video.mp4 --output-dir ../resultados
```

A saída será `resultados/video_tracking_visao_superior.mp4`. Os logs mostram o carregamento,
o tempo acumulado de cada modelo, o tempo real da inferência paralela, a identificação, a
homografia, a projeção, o render, a conversão H.264 e o tempo total. Como os modelos rodam em
paralelo, a soma dos tempos acumulados pode ser maior que o tempo real de parede.
