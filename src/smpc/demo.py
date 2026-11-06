"""Демо MVP: данные -> PSI -> PSI-CA -> PIS (прототип) -> сводка.

Запуск:
    python -m smpc --data-dir data
    python -m smpc --data-dir data --sample 50000 --feature DEF_30_CNT_SOCIAL_CIRCLE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smpc import reference
from smpc.channel import Channel
from smpc.data import BANK_FILE, TELECOM_FEATURES, TELECOM_FILE, load_bank, load_telecom, subsample
from smpc.metrics import woe_iv
from smpc.parties import Bank, Telecom
from smpc.protocols import REGISTRY, PISPrototype, ProtocolResult


def build_protocol(key: str, feature: str, key_bits: int):
    if key not in REGISTRY:
        raise SystemExit(f"Неизвестный протокол {key!r}; доступны: {', '.join(REGISTRY)}")
    if key == "pis":
        return PISPrototype(feature=feature, key_bits=key_bits)
    return REGISTRY[key]()


def check_against_reference(key: str, result: ProtocolResult, bank: Bank, telecom: Telecom, feature: str) -> bool:
    """Сверка выхода банка с открытым JOIN (только для лаборатории)."""
    if key == "psi":
        return set(result.bank_output) == reference.intersection(bank, telecom)
    if key == "psi_ca":
        return result.bank_output == len(reference.intersection(bank, telecom))
    if key == "pis":
        expected = reference.bin_stats(bank, telecom, feature)
        got = result.bank_output[["bin", "N", "defaults"]]
        return got.reset_index(drop=True).equals(expected.reset_index(drop=True))
    return False


def describe_output(key: str, result: ProtocolResult) -> str:
    if key == "psi":
        return f"{len(result.bank_output)} общих ID"
    if key == "psi_ca":
        return f"число общих = {result.bank_output}"
    if key == "pis":
        return f"агрегаты по {len(result.bank_output)} бинам"
    return str(result.bank_output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MVP: PSI, PSI-CA и прототип PIS для банка и телекома")
    parser.add_argument("--data-dir", default="data", help="папка с parquet-файлами")
    parser.add_argument("--bank-file", default=BANK_FILE)
    parser.add_argument("--telecom-file", default=TELECOM_FILE)
    parser.add_argument("--feature", default="DAYS_LAST_PHONE_CHANGE", choices=TELECOM_FEATURES,
                        help="признак телекома для PIS")
    parser.add_argument("--protocols", default="psi,psi_ca,pis", help="через запятую: " + ",".join(REGISTRY))
    parser.add_argument("--sample", type=int, default=None,
                        help="случайная подвыборка N строк у каждой стороны (по умолчанию — полный объём)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--key-bits", type=int, default=2048, help="длина ключа Paillier в PIS")
    parser.add_argument("--verbose", action="store_true", help="печатать журнал сообщений канала")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    bank = Bank(subsample(load_bank(data_dir / args.bank_file), args.sample, args.seed))
    telecom = Telecom(subsample(load_telecom(data_dir / args.telecom_file, [args.feature]), args.sample, args.seed + 1))

    common = reference.intersection(bank, telecom)
    print(f"Банк: {len(bank):,} | Телеком: {len(telecom):,} | Общих (эталон): {len(common):,} | "
          f"Только банк: {len(bank) - len(common):,} | Только телеком: {len(telecom) - len(common):,}")
    print()

    rows, leakage_lines, all_ok, pis_result = [], [], True, None
    for key in [k.strip() for k in args.protocols.split(",") if k.strip()]:
        protocol = build_protocol(key, args.feature, args.key_bits)
        channel = Channel()
        print(f"Запуск {protocol.name}...", flush=True)
        result = protocol.run(bank, telecom, channel)
        ok = check_against_reference(key, result, bank, telecom, args.feature)
        all_ok &= ok
        rows.append({
            "Протокол": result.name,
            "Результат банка": describe_output(key, result),
            "Совпало с эталоном": "да" if ok else "НЕТ",
            "Время, с": round(result.seconds, 2),
            "Трафик, КБ": round(result.bytes_sent / 1024, 1),
        })
        leakage_lines.append(f"{result.name}:")
        leakage_lines += [f"  {who}: {what}" for who, what in result.leakage.items()]
        if args.verbose:
            print(channel.summary().to_string(index=False))
        if key == "pis":
            pis_result = result

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("Кто что узнаёт:")
    print("\n".join(leakage_lines))

    if pis_result is not None:
        table, iv = woe_iv(pis_result.bank_output)
        print()
        print(f"PIS по признаку {args.feature} (проверка Paillier: {pis_result.details['homomorphic_check']}):")
        print(table[["bin", "N", "defaults", "WoE", "IV_contrib"]].round(4).to_string(index=False))
        print(f"IV = {iv:.5f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
