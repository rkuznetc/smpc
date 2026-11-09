# Ноутбуки команды

Исходные исследовательские ноутбуки (Google Colab), сохранены без изменений. Код из них перенесён в пакет `src/smpc`.

| Ноутбук | Что внутри | Где теперь в пакете |
|---|---|---|
| `01_psi_exploration.ipynb` | Знакомство с данными, проверка OpenMined PSI на 100 записях и на полном объёме, PSI с обменом ролями, PSI-CA | `protocols/openmined.py`, `data.py` |
| `02_psi_psica_full.ipynb` | PSI и PSI-CA на полном объёме с проверкой по эталону | `protocols/openmined.py`, `reference.py` |
| `03_pis_groups_prototype.ipynb` | PIS по бинам признака: HMAC-псевдонимы, агрегаты, Paillier, WoE и IV | `protocols/pis_prototype.py`, `metrics.py`, `data.assign_bins` |
| `04_psi_ckks_l1_100_features.ipynb` | ECDH-PSI, препроцессинг всех признаков, L1-отбор до 100, CKKS-градиент приближённой логистической регрессии, сравнение с открытым эталоном | `ml/features.py`, `ml/approx.py`, `he/ckks.py`, `ml/checkpoint.py`; описание в `docs/ckks_experiment.md` |
| `05_psi_ckks_compare_bank_telecom.ipynb` | Финальное сравнение «только банк» против «банк + весь телеком» открыто и по CKKS, с контрольными точками на Google Drive | `experiments/bank_vs_joint.py`; описание в `docs/ckks_experiment.md` |

Ожидаемые файлы данных в ноутбуках: `/content/bank_psi.parquet`, `/content/telecom_train.parquet` (в `01` также `bank_selected.parquet`).

Ноутбуки `04` и `05` работают на всех столбцах таблиц. В ноутбуке `05` ячейки 9.1–9.3 (итоговые метрики) сохранены без выводов: результаты ещё не получены. Запуск того же эксперимента из пакета: `smpc-compare` (см. README в корне).
