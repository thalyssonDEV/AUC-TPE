### Relatório

**Qual modelo apresentou melhor equilíbrio e estabilidade?**

O modelo de **Regressão Logística** foi, sem dúvida, o que apresentou o **melhor equilíbrio**. Suas métricas de Treino (`AUC 0.9086`) e Validação (`AUC 0.9086`) são praticamente idênticas, indicando que ele não decorou os dados.

Em termos de **estabilidade**, embora todos os modelos tenham tido um desvio padrão (CV Std) baixo, a Regressão Logística foi a mais estável no teste de generalização (com ruído), sofrendo a menor queda de performance.

---

**Algum apresentou overfitting ou underfitting?**

* **Overfitting:** Sim. O **Random Forest** (`AUC Treino 1.0000`) e o **XGBoost** (`AUC Treino 0.9981`) apresentaram **claro overfitting**. Eles alcançaram a perfeição nos dados de treino, mas não conseguiram manter esse nível nos dados de validação e teste.
* **Underfitting:** Não. Nenhum modelo apresentou underfitting, já que todos tiveram um desempenho excelente (`AUC de teste > 0.91`), muito acima de um resultado aleatório.

---

**Qual generalizou melhor?**

Depende do critério:

1.  **Melhor Performance:** O **XGBoost** foi o que melhor generalizou para os dados de teste "limpos", alcançando o maior `AUC (0.9674)` e a maior `Acurácia (91%)`.
2.  **Melhor Robustez:** O **Regressão Logística** foi o que melhor generalizou para um cenário com *dados ruidosos ou inesperados*. Ele foi o modelo mais robusto, pois sua performance caiu muito menos (queda de ~3.5% no AUC) em comparação com o Random Forest (~11.6%) e o XGBoost (~13.7%).
