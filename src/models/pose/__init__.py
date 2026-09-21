"""Treino do estimador de pose que a especie `pose` da borda carrega.

`dataset.py` monta os recortes, `train.py` ajusta o peso e `validate.py` mede o que sai. As
constantes que ligam o treino a producao vem de `edge.modules.pose` e de `mlops.pose_quality` em
vez de serem redigitadas aqui: dataset recortado com margem diferente da borda treina o modelo para
um enquadramento que ele nunca vai ver, e a divergencia sai como acuracia baixa sem causa visivel.
"""
