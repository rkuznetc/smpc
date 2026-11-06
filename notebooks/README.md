# Ноутбуки команды

Исходные исследовательские ноутбуки (Google Colab), сохранены без изменений. Код из них перенесён в пакет `src/smpc`.

| Ноутбук | Что внутри | Где теперь в пакете |
|---|---|---|
| `01_psi_exploration.ipynb` | Знакомство с данными, проверка OpenMined PSI на 100 записях и на полном объёме, PSI с обменом ролями, PSI-CA | `protocols/openmined.py`, `data.py` |
| `02_psi_psica_full.ipynb` | PSI и PSI-CA на полном объёме с проверкой по эталону | `protocols/openmined.py`, `reference.py` |
| `03_pis_groups_prototype.ipynb` | PIS по бинам признака: HMAC-псевдонимы, агрегаты, Paillier, WoE и IV | `protocols/pis_prototype.py`, `metrics.py`, `data.assign_bins` |

Ожидаемые файлы данных в ноутбуках: `/content/bank_psi.parquet`, `/content/telecom_train.parquet` (в `01` также `bank_selected.parquet`).
