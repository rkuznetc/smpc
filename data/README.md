# Данные

Файлы данных в репозиторий не добавляются (размер и правила соревнования Kaggle). Их нужно положить в эту папку вручную.

Источник: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data), таблица заявок, разделённая командой на две стороны.

| Файл | Сторона | Обязательные колонки |
|---|---|---|
| `bank_psi.parquet` | банк | `SK_ID_CURR`, `TARGET` (остальные признаки банка не читаются протоколами) |
| `telecom_train.parquet` | телеком | `SK_ID_CURR`, `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`, `OBS_60_CNT_SOCIAL_CIRCLE`, `DEF_60_CNT_SOCIAL_CIRCLE`, `DAYS_LAST_PHONE_CHANGE` |

Имена файлов можно поменять параметрами демо `--bank-file` и `--telecom-file`.
