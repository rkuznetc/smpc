"""Сравнение «банк» против «банк + телеком» (логика ноутбука 05).

Порядок работы (как в ноутбуке):
  1. ECDH-PSI по ID (готовый PSIProtocol), строки обеих сторон выравниваются по общим ID.
  2. Препроцессинг признаков каждой стороны; fit только на train.
  3. L1-логистическая регрессия на СОВМЕСТНОЙ матрице (банк + весь телеком)
     отбирает до max_features признаков. Из них берутся банковские признаки.
  4. Модели:
       «Только банк»          — банковские признаки, отобранные на шаге 3;
       «Банк + весь телеком»  — те же банковские признаки + все признаки телекома.
  5. Два варианта обучения (backend):
       open — открытый приближённый градиент (контроль, без шифрования);
       ckks — тот же градиент над шифротекстами CKKS (демо-расшифрование агрегатов).
  6. Порог F1 подбирается на validation; метрики считаются на test.

Ограничения: открытое сопоставление после PSI, обе стороны в одном процессе,
открытый L1-отбор, открытое обновление весов, демо-расшифрование агрегатов.
Это исследовательский прогон, а не защищённая двухсторонняя система.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from smpc.channel import Channel
from smpc.data import BANK_FILE, TELECOM_FILE, align_to_ids, load_bank_full, load_telecom_full
from smpc.he.ckks import CKKSGradient, CKKSParams, DemoAggregateDecryptor, make_context_pair
from smpc.metrics import best_f1_threshold, classification_row
from smpc.ml.approx import ApproxState, plain_gradient, run_epochs
from smpc.ml.base import ApproxLogisticModel
from smpc.ml.checkpoint import fingerprint, load_state, save_state
from smpc.ml.features import FeatureBlock, SplitIndices, build_block, l1_select, make_splits
from smpc.parties import Bank, Telecom
from smpc.protocols import PSIProtocol

BACKENDS = ("open", "ckks")
BANK_ONLY = "Только банк"
BANK_PLUS_TELECOM = "Банк + весь телеком"


@dataclass
class ExperimentConfig:
    data_dir: Path = Path("data")
    bank_file: str = BANK_FILE
    telecom_file: str = TELECOM_FILE
    max_rows: int | None = None  # случайная стратифицированная подвыборка после PSI
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    max_features: int | None = 100
    l1_c: float = 0.1
    epochs: int = 3
    batch_size: int = 1024
    learning_rate: float = 0.15
    reg_l2: float = 1e-3
    backends: tuple[str, ...] = ("open",)
    checkpoint_dir: Path | None = None
    check_rows: int = 128
    ckks: CKKSParams = field(default_factory=CKKSParams)

    def __post_init__(self) -> None:
        unknown = [b for b in self.backends if b not in BACKENDS]
        if unknown:
            raise ValueError(f"Неизвестные backend: {unknown}; доступны: {BACKENDS}")


@dataclass
class ModelSpec:
    key: str
    label: str
    train: sp.csr_matrix
    val: sp.csr_matrix
    test: sp.csr_matrix
    names: list[str]


@dataclass
class PreparedData:
    y: np.ndarray
    splits: SplitIndices
    models: list[ModelSpec]
    selected_features: list[str]
    n_common: int
    psi_bytes: int
    psi_seconds: float


@dataclass
class ExperimentResult:
    table: pd.DataFrame
    deltas: dict[str, dict[str, float]]
    selected_features: list[str]
    n_common: int
    details: dict[str, Any]


def prepare(config: ExperimentConfig) -> PreparedData:
    """Загрузка, PSI, признаки и L1-отбор. Всё, что нужно для обучения."""
    data_dir = Path(config.data_dir)
    bank = load_bank_full(data_dir / config.bank_file)
    telecom = load_telecom_full(data_dir / config.telecom_file)

    channel = Channel()
    start = time.perf_counter()
    outcome = PSIProtocol().run(Bank(bank), Telecom(telecom), channel)
    psi_seconds = time.perf_counter() - start
    common = list(outcome.bank_output)
    if not common:
        raise ValueError("Пересечение пустое: нечего обучать")

    bank = align_to_ids(bank, common)
    telecom = align_to_ids(telecom, common)
    if config.max_rows is not None and len(bank) > config.max_rows:
        _, chosen = train_test_split(
            np.arange(len(bank)), test_size=config.max_rows, random_state=config.seed,
            stratify=bank["TARGET"].to_numpy(),
        )
        bank = bank.iloc[chosen].reset_index(drop=True)
        telecom = telecom.iloc[chosen].reset_index(drop=True)
    assert bank["ID"].equals(telecom["ID"]), "ID сторон должны совпадать после выравнивания"

    y = bank["TARGET"].to_numpy(dtype=np.int64)
    splits = make_splits(y, config.seed, config.test_size, config.val_size)
    bank_block = build_block(bank, "bank__", splits)
    telecom_block = build_block(telecom, "telecom__", splits)

    joint_train = sp.hstack([bank_block.train, telecom_block.train], format="csr", dtype=np.float32)
    joint_names = bank_block.names + telecom_block.names
    selected, _ = l1_select(
        joint_train, y[splits.train], joint_names, config.max_features, config.l1_c, config.seed
    )
    n_bank = len(bank_block.names)
    bank_idx = np.array([j for j in selected if j < n_bank], dtype=int)
    if bank_idx.size == 0:
        raise ValueError("L1 не выбрал ни одного банковского признака: увеличьте l1_c")

    models = _model_specs(bank_block, telecom_block, bank_idx)
    return PreparedData(
        y=y,
        splits=splits,
        models=models,
        selected_features=[joint_names[j] for j in selected],
        n_common=len(common),
        psi_bytes=outcome.bytes_sent,
        psi_seconds=psi_seconds,
    )


def _model_specs(bank: FeatureBlock, telecom: FeatureBlock, bank_idx: np.ndarray) -> list[ModelSpec]:
    bank_names = [bank.names[j] for j in bank_idx]
    only = ModelSpec(
        key="bank_only",
        label=BANK_ONLY,
        train=bank.train[:, bank_idx].tocsr(),
        val=bank.val[:, bank_idx].tocsr(),
        test=bank.test[:, bank_idx].tocsr(),
        names=bank_names,
    )
    joint = ModelSpec(
        key="bank_plus_telecom",
        label=BANK_PLUS_TELECOM,
        train=sp.hstack([only.train, telecom.train], format="csr"),
        val=sp.hstack([only.val, telecom.val], format="csr"),
        test=sp.hstack([only.test, telecom.test], format="csr"),
        names=bank_names + telecom.names,
    )
    return [only, joint]


def verify_gradient(gradient, X, y: np.ndarray, state: ApproxState, rows: int, atol: float = 3e-3) -> float:
    """Сверка CKKS-градиента с открытым на небольшом пакете. Возвращает max-ошибку."""
    m = min(rows, X.shape[0])
    g_he, b_he = gradient(state.weights, state.bias, X[:m], y[:m])
    g_plain, b_plain = plain_gradient(state.weights, state.bias, X[:m], y[:m])
    error = float(max(np.max(np.abs(g_he - g_plain)), abs(b_he - b_plain)))
    if error > atol:
        raise RuntimeError(f"CKKS-градиент отличается от открытого на {error:.2e} (допуск {atol:.0e})")
    return error


def _evaluate(model: ApproxLogisticModel, spec: ModelSpec, y_val: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    threshold = best_f1_threshold(y_val, model.predict_proba(spec.val))
    test_proba = model.predict_proba(spec.test)
    return {
        **classification_row(y_test, test_proba, threshold),
        "Признаков": spec.train.shape[1],
        "Порог": threshold,
        "Эпох": model.state.epoch,
    }


def _train_open(config: ExperimentConfig, spec: ModelSpec, y_train: np.ndarray) -> tuple[ApproxLogisticModel, float]:
    model = ApproxLogisticModel(
        epochs=config.epochs, learning_rate=config.learning_rate, reg_l2=config.reg_l2
    )
    tic = time.perf_counter()
    model.fit(spec.train, y_train)
    return model, time.perf_counter() - tic


def _train_ckks(
    config: ExperimentConfig,
    spec: ModelSpec,
    y_train: np.ndarray,
    gradient: CKKSGradient,
) -> tuple[ApproxLogisticModel, float, float, dict[str, Any]]:
    fp = fingerprint(
        spec.train, y_train, spec.names,
        {"batch_size": config.batch_size, "learning_rate": config.learning_rate, "reg_l2": config.reg_l2},
    )
    path = None if config.checkpoint_dir is None else Path(config.checkpoint_dir) / f"ckks_{spec.key}.npz"
    state = load_state(path, fp) if path is not None else None
    if state is None:
        state = ApproxState.start(y_train, spec.train.shape[1])
    check_error = 0.0
    if state.epoch == 0:
        check_error = verify_gradient(gradient, spec.train, y_train, state, config.check_rows)

    model = ApproxLogisticModel(
        gradient=gradient,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        reg_l2=config.reg_l2,
        on_epoch=None if path is None else (lambda s: save_state(path, s, fp, spec.names)),
    )
    tic = time.perf_counter()
    model.fit(spec.train, y_train, state=state)
    seconds = time.perf_counter() - tic

    # Ошибка весов относительно открытого эталона с тем же числом эпох.
    reference = ApproxState.start(y_train, spec.train.shape[1])
    run_epochs(reference, spec.train, y_train, plain_gradient, model.state.epoch, config.learning_rate, config.reg_l2)
    weight_error = float(max(
        np.max(np.abs(model.weights - reference.weights)), abs(model.bias - reference.bias)
    ))
    return model, seconds, weight_error, {"gradient_check": check_error, "checkpoint": str(path) if path else None}


def run(config: ExperimentConfig) -> ExperimentResult:
    prepared = prepare(config)
    y_train = prepared.y[prepared.splits.train].astype(float)
    y_val = prepared.y[prepared.splits.val].astype(float)
    y_test = prepared.y[prepared.splits.test].astype(float)

    ckks_gradient = None
    if "ckks" in config.backends:
        private, public = make_context_pair(config.ckks)
        ckks_gradient = CKKSGradient(public, DemoAggregateDecryptor(private), config.batch_size)

    rows, details = [], {"psi_seconds": prepared.psi_seconds, "psi_bytes": prepared.psi_bytes, "backends": {}}
    for backend in config.backends:
        for spec in prepared.models:
            if backend == "open":
                model, seconds = _train_open(config, spec, y_train)
                extra = {}
            else:
                model, seconds, weight_error, extra = _train_ckks(config, spec, y_train, ckks_gradient)
                extra["weight_error"] = weight_error
            row = {"Backend": backend, "Модель": spec.label, **_evaluate(model, spec, y_val, y_test)}
            row["Время, с"] = seconds
            rows.append(row)
            details["backends"].setdefault(backend, {})[spec.key] = extra

    table = pd.DataFrame(rows)
    deltas = {}
    for backend in config.backends:
        part = table[table["Backend"] == backend].set_index("Модель")
        if BANK_ONLY in part.index and BANK_PLUS_TELECOM in part.index:
            deltas[backend] = {
                metric: float(part.loc[BANK_PLUS_TELECOM, metric] - part.loc[BANK_ONLY, metric])
                for metric in ("ROC-AUC", "PR-AUC")
            }
    return ExperimentResult(
        table=table,
        deltas=deltas,
        selected_features=prepared.selected_features,
        n_common=prepared.n_common,
        details=details,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сравнение «банк» против «банк + телеком» (ноутбук 05)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--bank-file", default=BANK_FILE)
    parser.add_argument("--telecom-file", default=TELECOM_FILE)
    parser.add_argument("--max-rows", type=int, default=None, help="подвыборка после PSI (по умолчанию — все)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=100, help="0 — без ограничения")
    parser.add_argument("--l1-c", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--reg-l2", type=float, default=1e-3)
    parser.add_argument("--backends", default="open", help="через запятую: open,ckks")
    parser.add_argument("--checkpoint-dir", default=None, help="папка для контрольных точек CKKS")
    parser.add_argument("--check-rows", type=int, default=128)
    parser.add_argument("--out", default=None, help="сохранить таблицу метрик в CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ExperimentConfig(
        data_dir=Path(args.data_dir),
        bank_file=args.bank_file,
        telecom_file=args.telecom_file,
        max_rows=args.max_rows,
        seed=args.seed,
        max_features=args.max_features or None,
        l1_c=args.l1_c,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reg_l2=args.reg_l2,
        backends=tuple(b.strip() for b in args.backends.split(",") if b.strip()),
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        check_rows=args.check_rows,
    )
    result = run(config)

    telecom_selected = [f for f in result.selected_features if f.startswith("telecom__")]
    print(f"Общих клиентов после PSI: {result.n_common:,}; признаков после L1: {len(result.selected_features)}")
    print(f"Признаки телекома после L1 ({len(telecom_selected)}): {', '.join(telecom_selected) or '—'}")
    print()
    print(result.table.round(4).to_string(index=False))
    print()
    for backend, delta in result.deltas.items():
        print(f"[{backend}] Прирост от телекома: ROC-AUC {delta['ROC-AUC']:+.4f}, PR-AUC {delta['PR-AUC']:+.4f}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.table.to_csv(out, index=False)
        print(f"Таблица сохранена: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
