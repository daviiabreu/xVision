"""Pipeline de pre-processamento propria da bola.

Esqueleto inicial. A bola ocupa poucos pixels em plano aberto, entao esta frente tende a exigir
passos proprios (recorte de ROI em maior resolucao e realce seletivo de objeto pequeno) alem dos
passos comuns de decode e conversao de cor.

A pipeline e desenhada para, no futuro, ser concatenada com as outras numa unica passada:
os passos comuns rodam uma vez por quadro e sao reusados, e cada frente puxa apenas os passos que
declara. Aqui ficam somente os passos especificos da bola.
"""

from __future__ import annotations

__all__: list[str] = []
