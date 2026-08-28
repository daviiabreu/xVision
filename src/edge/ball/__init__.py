"""Analise da bola na borda: rastreamento agora e velocidade mais adiante.

- `tracking`: modelo e rastreador da bola. Nao depende de outras frentes, por isso comeca primeiro.
- `velocity`: calculo da velocidade da bola. Depende da homografia (px -> metros do gramado) e so
  sera iniciado depois que essa frente ficar pronta.
"""

__all__: list[str] = []
