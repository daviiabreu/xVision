"""O dataset de recortes precisa descrever o que a borda ve, e o espelhamento nao pode trocar lado.

Dois defeitos moram aqui e nenhum levanta excecao: margem divergente da producao treina para um
enquadramento que o modelo nunca vai ver, e `flip_idx` errado ensina a trocar tornozelo esquerdo por
direito, que e o erro que inverte a resposta de `body_part`.

O modulo e carregado pelo caminho do arquivo porque `models/` fica fora dos pacotes que o
`pyproject.toml` empacota, como em `tests/test_coverage_scope.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from edge.modules.pose import COCO_KEYPOINTS, MARGEM

SRC = Path(__file__).resolve().parents[2]


def _carrega(nome: str) -> ModuleType:
    arquivo = SRC / "models" / "pose" / f"{nome}.py"
    chave = f"models_pose_{nome}"
    spec = importlib.util.spec_from_file_location(chave, arquivo)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[chave] = modulo
    spec.loader.exec_module(modulo)
    return modulo


dataset = _carrega("dataset")


def _keypoints(**posicoes: tuple[float, float, int]) -> tuple[tuple[float, float, int], ...]:
    """Um corpo completo dentro da caixa de 40 por 130 px, com os pontos nomeados trocados."""
    return tuple(posicoes.get(nome, (120.0, 300.0, 2)) for nome in COCO_KEYPOINTS)


def _anotacao(**mudancas: object) -> dict[str, object]:
    crus: list[float] = []
    for x, y, v in _keypoints():
        crus += [x, y, float(v)]
    base: dict[str, object] = {
        "id": 1,
        "image_id": 7,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": [100.0, 200.0, 40.0, 130.0],
        "keypoints": crus,
    }
    base.update(mudancas)
    return base


def _documento(*anotacoes: dict[str, object]) -> dict[str, object]:
    return {
        "images": [{"id": 7, "file_name": "a.jpg", "width": 1920, "height": 1080}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": list(anotacoes),
    }


def _escreve(tmp_path: Path, documento: dict[str, object]) -> Path:
    import json

    caminho = tmp_path / "keypoints.json"
    caminho.write_text(json.dumps(documento), encoding="utf-8")
    return caminho


def test_o_flip_idx_declarado_e_o_que_os_nomes_derivam() -> None:
    """Se este teste quebrar, NAO conserte a constante ate entender qual dos dois lados mudou:
    ou os nomes do contrato mudaram, ou a constante foi editada."""
    assert dataset._flip_idx_dos_nomes(COCO_KEYPOINTS) == dataset.FLIP_IDX


def test_o_espelhamento_troca_exatamente_os_pares_laterais() -> None:
    flip_idx = cast(tuple[int, ...], dataset.FLIP_IDX)
    for indice, nome in enumerate(COCO_KEYPOINTS):
        destino = COCO_KEYPOINTS[flip_idx[indice]]
        if nome.endswith("_left"):
            assert destino == nome.removesuffix("_left") + "_right"
        elif nome.endswith("_right"):
            assert destino == nome.removesuffix("_right") + "_left"
        else:
            assert destino == nome, f"{nome} nao tem lado e nao pode se mover no espelhamento"


def test_o_espelhamento_e_a_sua_propria_inversa() -> None:
    for indice in range(len(COCO_KEYPOINTS)):
        assert dataset.FLIP_IDX[dataset.FLIP_IDX[indice]] == indice


def test_o_recorte_usa_a_margem_da_producao() -> None:
    """Margem diferente da borda treina o modelo para um enquadramento que ele nunca vai ver."""
    instancia = dataset.Instancia(
        image_id=7,
        arquivo="a.jpg",
        largura=1920,
        altura=1080,
        box=(100.0, 200.0, 140.0, 330.0),
        keypoints=_keypoints(),
    )

    recorte = dataset.recortar(instancia)

    assert recorte is not None
    assert recorte.origem == (int(100 - 40 * MARGEM), int(200 - 130 * MARGEM))
    assert recorte.tamanho == (
        int(140 + 40 * MARGEM) - int(100 - 40 * MARGEM),
        int(330 + 130 * MARGEM) - int(200 - 130 * MARGEM),
    )


def test_o_recorte_e_cortado_pela_borda_da_imagem() -> None:
    dentro = tuple((20.0, 100.0, 2) for _ in COCO_KEYPOINTS)
    instancia = dataset.Instancia(
        image_id=7,
        arquivo="a.jpg",
        largura=160,
        altura=340,
        box=(0.0, 0.0, 40.0, 130.0),
        keypoints=dentro,
    )

    recorte = dataset.recortar(instancia)

    assert recorte is not None
    assert recorte.origem == (0, 0)


def test_ponto_fora_do_recorte_vira_ausente_e_nao_coordenada_na_moldura() -> None:
    """Grudar o ponto na borda ensinaria que todo tornozelo cortado mora na moldura."""
    longe = _keypoints(nose=(5.0, 5.0, 2))
    instancia = dataset.Instancia(
        image_id=7,
        arquivo="a.jpg",
        largura=1920,
        altura=1080,
        box=(100.0, 200.0, 140.0, 330.0),
        keypoints=longe,
    )

    recorte = dataset.recortar(instancia)

    assert recorte is not None
    nariz = recorte.keypoints[COCO_KEYPOINTS.index("nose")]
    assert nariz == (0.0, 0.0, 0), "o ponto fora do recorte tem de sair do rotulo"


def test_recorte_que_perde_um_tornozelo_e_descartado() -> None:
    """Sem os dois pes a instancia nao ensina o que separa perna esquerda de direita."""
    instancia = dataset.Instancia(
        image_id=7,
        arquivo="a.jpg",
        largura=1920,
        altura=1080,
        box=(100.0, 200.0, 140.0, 330.0),
        keypoints=_keypoints(ankle_left=(5.0, 5.0, 2)),
    )

    assert dataset.recortar(instancia) is None


def test_a_linha_yolo_tem_a_caixa_e_os_dezessete_pontos_normalizados() -> None:
    recorte = dataset.Recorte(
        origem=(0, 0),
        tamanho=(100, 200),
        box=(20.0, 40.0, 80.0, 160.0),
        keypoints=tuple((50.0, 100.0, 2) for _ in COCO_KEYPOINTS),
    )

    campos = dataset.linha_yolo(recorte).split()

    assert len(campos) == 5 + 3 * len(COCO_KEYPOINTS)
    assert campos[0] == "0"
    assert float(campos[1]) == pytest.approx(0.5)
    assert float(campos[2]) == pytest.approx(0.5)
    assert float(campos[3]) == pytest.approx(0.6)
    assert float(campos[4]) == pytest.approx(0.6)
    coordenadas = [float(v) for v in campos[1:5]] + [
        float(campos[5 + i * 3 + eixo]) for i in range(len(COCO_KEYPOINTS)) for eixo in (0, 1)
    ]
    assert all(0.0 <= valor <= 1.0 for valor in coordenadas)
    assert campos[7] == "2", "a visibilidade e inteira e nao entra na normalizacao"


def test_a_instancia_pequena_demais_nao_entra(tmp_path: Path) -> None:
    caminho = _escreve(
        tmp_path, _documento(_anotacao(bbox=[100.0, 200.0, 10.0, dataset.MIN_ALTURA_PX - 1.0]))
    )

    assert dataset.ler_instancias(caminho) == []


def test_a_instancia_grande_demais_nao_entra(tmp_path: Path) -> None:
    caminho = _escreve(
        tmp_path, _documento(_anotacao(bbox=[0.0, 0.0, 300.0, dataset.MAX_ALTURA_PX + 1.0]))
    )

    assert dataset.ler_instancias(caminho) == []


def test_a_caixa_quase_quadrada_nao_entra(tmp_path: Path) -> None:
    caminho = _escreve(tmp_path, _documento(_anotacao(bbox=[100.0, 200.0, 120.0, 130.0])))

    assert dataset.ler_instancias(caminho) == []


def test_a_instancia_sem_tornozelo_nao_entra(tmp_path: Path) -> None:
    crus: list[float] = []
    for nome in COCO_KEYPOINTS:
        crus += [10.0, 10.0, 0.0 if nome == "ankle_right" else 2.0]
    caminho = _escreve(tmp_path, _documento(_anotacao(keypoints=crus)))

    assert dataset.ler_instancias(caminho) == []


def test_a_multidao_nao_entra(tmp_path: Path) -> None:
    caminho = _escreve(tmp_path, _documento(_anotacao(iscrowd=1)))

    assert dataset.ler_instancias(caminho) == []


def test_a_instancia_de_transmissao_entra(tmp_path: Path) -> None:
    caminho = _escreve(tmp_path, _documento(_anotacao()))

    instancias = dataset.ler_instancias(caminho)

    assert len(instancias) == 1
    assert instancias[0].altura_da_caixa == pytest.approx(130.0)


def test_a_fonte_sem_a_categoria_person_levanta(tmp_path: Path) -> None:
    documento = _documento(_anotacao())
    documento["categories"] = [{"id": 1, "name": "cat"}]

    with pytest.raises(dataset.DatasetError, match="person"):
        dataset.ler_instancias(_escreve(tmp_path, documento))


def test_o_split_nao_se_move_entre_execucoes() -> None:
    """Se movesse, o numero reservado passaria a ser medido sobre algo que o modelo ja viu."""
    primeiro = [dataset.split_de(i, seed=1) for i in range(200)]
    segundo = [dataset.split_de(i, seed=1) for i in range(200)]

    assert primeiro == segundo
    assert set(primeiro) == set(dataset.SPLITS), "os tres conjuntos precisam receber imagem"


def test_a_semente_muda_a_particao() -> None:
    a = [dataset.split_de(i, seed=1) for i in range(200)]
    b = [dataset.split_de(i, seed=2) for i in range(200)]

    assert a != b


def test_o_dataset_yaml_declara_o_flip_idx_e_os_seventeen_pontos(tmp_path: Path) -> None:
    texto = dataset.dataset_yaml(tmp_path)

    assert f"kpt_shape: [{len(COCO_KEYPOINTS)}, 3]" in texto
    assert f"flip_idx: {list(dataset.FLIP_IDX)}" in texto
    for nome in COCO_KEYPOINTS:
        assert nome in texto
