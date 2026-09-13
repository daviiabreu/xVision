# TeamIdentity Design

## Goal

Integrar a classificação temporal de equipes ao pipeline de percepção para preencher `PlayerDetection.team` com `"team_a"`, `"team_b"` ou `None`, sem implementar reconhecimento de número de camisa.

## Scope

Esta mudança cobre somente a frente de identificação de times atribuída a Roberto. As features `marked`, `assist` e `occupied_area` não fazem parte desta entrega.

## Architecture

Um novo módulo de percepção, `TeamIdentity`, será responsável por transformar detecções de jogadores e seus `track_id`s em registros de identidade por frame. Ele depende de `players` e `tracking`, usa os frames BGR originais do `FrameBatch` para extrair a cor do tronco e delega a agrupação para `edge.participants.identity.teams_from_color_samples`.

A instância do módulo manterá observações e cores por `track_id` enquanto processa os batches de um único clipe. O estado será inicializado em `load()` e descartado em `close()`, impedindo que evidências de um clipe contaminem o próximo. Em um batch sem amostras suficientes para separar dois grupos, a saída conterá somente `track_id`; quando a separação se tornar possível, emitirá também `team` para as trajetórias classificadas.

O `runner` continuará como orquestrador: ao montar `PlayerDetection`, ele lerá `team` dos registros de `identity` e o usará antes do valor eventualmente emitido pelo detector de jogadores. Registros sem `team` preservam o valor de `players` como fallback.

## Data Flow

1. `players` emite caixas e confianças por frame.
2. `tracking` associa cada `detection_index` a um `track_id`.
3. `TeamIdentity.infer()` combina os dois outputs, cria `TrackObservation` e extrai cores das caixas nos frames correspondentes.
4. O módulo acumula as amostras por trajetória, executa `teams_from_color_samples`, e emite registros `{"track_id": int}` ou `{"track_id": int, "team": "team_a" | "team_b"}`.
5. `runner.assemble()` incorpora o time de `identity` no `PlayerDetection.team`.

## Error Handling

- Jogador sem `track_id`, caixa inválida ou recorte de camisa sem cor utilizável não interrompe a inferência; apenas não adiciona evidência de cor.
- Menos de duas trajetórias com amostras válidas resulta em nenhum `team` emitido.
- O contrato mínimo de `identity` continua exigindo apenas `track_id`; `team` é opcional.
- O módulo não reusa estado após `close()` e só processa chamadas quando carregado, como os módulos existentes.

## Configuration

`config/pipeline.models.toml` passa a apontar `[modules.identity]` para `edge.modules.identity:TeamIdentity`, no mesmo frameset temporal usado por `tracking`. `config/pipeline.toml` mantém `StubIdentity`, preservando testes rápidos e execuções que não usam os modelos reais.

## Testing

Cobrir o módulo em isolamento com batches sintéticos e outputs `players`/`tracking`:

- registros incluem `track_id` mesmo quando a classificação ainda não é possível;
- cores de duas trajetórias acumuladas em batches separados classificam ambas de forma estável;
- `close()` limpa as amostras e uma execução seguinte não recebe classificação por evidência anterior.

Cobrir o `runner.assemble()` verificando que `identity.team` vence o `team` opcional de `players`, enquanto a ausência de `identity.team` mantém o fallback de `players`.
