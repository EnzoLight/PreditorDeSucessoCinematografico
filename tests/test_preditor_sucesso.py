import unittest
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from unittest.mock import patch

from preditor_sucesso import (
    Preparador,
    RegressaoLogistica,
    analisar_mensagem_com_llm,
    dividir_por_tempo,
    metricas,
    responder_natural_com_llm,
    validar_campos,
)


class TestPreditor(unittest.TestCase):
    def test_regressao_logistica_aprende_separacao(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        modelo = RegressaoLogistica(epocas=1800).fit(X, y)
        probs = modelo.predict_proba(np.array([[-1.5], [1.5]]))
        self.assertLess(probs[0], 0.5)
        self.assertGreater(probs[1], 0.5)

    def test_divisao_temporal_nao_cruza_anos(self):
        df = pd.DataFrame({"release_year": np.repeat(np.arange(2000, 2020), 3)})
        treino, validacao, teste = dividir_por_tempo(df)
        self.assertLess(treino.release_year.max(), validacao.release_year.min())
        self.assertLess(validacao.release_year.max(), teste.release_year.min())

    def test_metricas_confusao(self):
        result = metricas(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.4, 0.9]))
        self.assertEqual(result["matriz_confusao"], [[1, 1], [1, 1]])
        self.assertEqual(result["acuracia"], 0.5)

    def test_validacao_rejeita_orcamento_invalido_e_normaliza(self):
        campos = {"ano": "2027", "duracao": "110", "orcamento_usd": 50000000,
                  "franquia": "não", "diretor": " A ", "roteirista": "B",
                  "generos": "drama, comedy"}
        validos, erros = validar_campos(campos)
        self.assertEqual(erros, [])
        self.assertFalse(validos["franquia"])
        self.assertEqual(validos["generos"], ["drama", "comedy"])
        campos["orcamento_usd"] = 100
        _, erros = validar_campos(campos)
        self.assertIn("orcamento_usd", erros)

    def test_preparador_frequencias_sao_do_treino(self):
        train = pd.DataFrame({
            "release_year": [2000, 2001], "runtime": [90, 100], "budget": [1e6, 2e6],
            "is_franchise": [0, 1], "director": ["A", "A"], "writer": ["W", "W"],
            "genres_list": [["drama"], ["comedy"]],
        })
        prep = Preparador.fit(train)
        film = {"ano": 2027, "duracao": 100, "orcamento_usd": 1e6, "franquia": False,
                "diretor": "Novo", "roteirista": "Novo", "generos": ["drama"]}
        self.assertEqual(prep.freq_diretor.get(film["diretor"], 0), 0)
        self.assertEqual(prep.filme(film).shape[1], 6 + len(prep.generos))

    def test_intencao_geral_nao_vira_pergunta_de_campos(self):
        resposta = {"intent": "general", "fields": {}}
        with patch("preditor_sucesso._consultar_llm_json", return_value=resposta):
            analise = analisar_mensagem_com_llm("Como tu sabe?", {}, list(),
                                               {"probabilidade": 0.525}, [])
        self.assertEqual(analise["intent"], "general")

    def test_extrai_campos_e_mantem_intencao_de_filme(self):
        resposta = {
            "intent": "film",
            "fields": {"ano": 2027, "duracao": 123, "orcamento_usd": 40000,
                       "franquia": False, "diretor": "Calvin Henrris",
                       "roteirista": "Brubeu Nadaver", "generos": ["ação"]},
        }
        with patch("preditor_sucesso._consultar_llm_json", return_value=resposta):
            analise = analisar_mensagem_com_llm("dados do filme", {}, list(), None, [])
        self.assertEqual(analise["intent"], "film")
        self.assertEqual(analise["fields"]["generos"], ["ação"])

    def test_resposta_conversacional_e_texto_livre(self):
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="É uma estimativa, não uma certeza."))]
        )
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://example.test/v1", "LLM_MODEL": "teste"}), \
             patch("preditor_sucesso._llm_client", return_value=fake_client):
            resposta = responder_natural_com_llm("Como tu sabe?", "followup", {}, [])
        self.assertEqual(resposta, "É uma estimativa, não uma certeza.")


if __name__ == "__main__":
    unittest.main()
