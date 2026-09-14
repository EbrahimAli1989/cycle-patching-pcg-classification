"""Evaluation metrics for record-level and cycle-level predictions.

`MetricsCalculator.from_confusion_matrix` computes the exact same formulas as
the original `Utils_fns.multiclass_metrics_special` (extracted verbatim in
the first, flat-script version of this repository's `metrics.py`), now
wrapped to return a `ClassificationReport` dataclass instead of a
tuple-of-dicts, for a friendlier public API. `as_row()` reproduces the
`[Recall, Specificity, Precision, F1_score, Accuracy,
MatthewsCorrelationCoefficient, Kappa]` ordering used for the results CSV
written by the original `train.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassificationReport:
    """Class-averaged classification metrics for one confusion matrix."""

    recall: float
    specificity: float
    precision: float
    f1_score: float
    accuracy: float
    matthews_corrcoef: float
    kappa: float
    per_class: dict = field(default_factory=dict, repr=False)

    def as_row(self) -> list:
        """`[Recall, Specificity, Precision, F1_score, Accuracy,
        MatthewsCorrelationCoefficient, Kappa]`, matching the column order
        of the results CSV written by the original scripts.
        """
        return [
            self.recall, self.specificity, self.precision, self.f1_score,
            self.accuracy, self.matthews_corrcoef, self.kappa,
        ]

    @staticmethod
    def columns() -> list:
        return [
            'Recall', 'Specificity', 'Precision', 'F1_score', 'Accuracy',
            'MatthewsCorrelationCoefficient', 'Kappa',
        ]


class MetricsCalculator:
    """Computes Recall, Specificity, Precision, F1, Accuracy, Matthews
    Correlation Coefficient and Kappa from a confusion matrix."""

    @staticmethod
    def from_confusion_matrix(confusion_matrix: np.ndarray, delta: float = 1e-6) -> ClassificationReport:
        result, reference_result, row = MetricsCalculator._compute(confusion_matrix, delta)
        return ClassificationReport(
            recall=row[0], specificity=row[1], precision=row[2], f1_score=row[3],
            accuracy=row[4], matthews_corrcoef=row[5], kappa=row[6],
            per_class=reference_result,
        )

    @staticmethod
    def _compute(confMatrix: np.ndarray, delta: float):
        row, col = confMatrix.shape
        if row != col:
            raise ValueError('Confusion matrix dimension is wrong')

        n_class = row

        if n_class == 2:
            TP = confMatrix[0, 0]
            FN = confMatrix[0, 1]
            FP = confMatrix[1, 0]
            TN = confMatrix[1, 1]
        else:
            TP = np.zeros(n_class)
            FN = np.zeros(n_class)
            FP = np.zeros(n_class)
            TN = np.zeros(n_class)
            for i in range(n_class):
                TP[i] = confMatrix[i, i]
                FN[i] = np.sum(confMatrix[i, :]) - confMatrix[i, i]
                FP[i] = np.sum(confMatrix[:, i]) - confMatrix[i, i]
                TN[i] = np.sum(confMatrix) - TP[i] - FP[i] - FN[i]

        P = TP + FN
        N = FP + TN
        if n_class == 2:
            accuracy = (TP + TN) / (P + N + delta)
            Error = 1 - accuracy
            Result = {'Accuracy': accuracy, 'Error': Error}
        else:
            accuracy = TP / (P + N + delta)
            Error = FP / (P + N + delta)
            Result = {'Accuracy': np.sum(accuracy), 'Error': np.sum(Error)}

        ReferenceResult = {
            'AccuracyOfSingle': (TP / P).tolist(),
            'ErrorOfSingle': (1 - (TP / P)).tolist(),
        }

        Recall = TP / (P + delta)
        Specificity = TN / (N + delta)
        Precision = TP / (TP + FP + delta)
        FPR = 1 - Specificity
        beta = 1
        F1_score = ((1 + beta ** 2) * (Recall * Precision)) / (beta ** 2 * (Precision + Recall + delta))

        temp = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        MCC = [(TP * TN - FP * FN) / np.sqrt(temp + delta), (FP * FN - TP * TN) / np.sqrt(temp + delta)]
        MCC = np.max(MCC)

        # Kappa calculation by 2x2 matrix shape
        pox = np.sum(accuracy)
        Px = np.sum(P)
        TPx = np.sum(TP)
        FPx = np.sum(FP)
        TNx = np.sum(TN)
        FNx = np.sum(FN)
        Nx = np.sum(N)
        pex = (Px * (TPx + FPx) + Nx * (FNx + TNx)) / (TPx + TNx + FPx + FNx + delta) ** 2
        kappa_overall = [(pox - pex) / (1 - pex), (pex - pox) / (1 - pox)]
        kappa_overall = np.max(kappa_overall)

        # Kappa calculation by n_class x n_class matrix shape
        po = accuracy
        pe = (P * (TP + FP) + N * (FN + TN)) / (TP + TN + FP + FN + delta) ** 2
        kappa = [(po - pe) / (1 - pe), (pe - po) / (1 - po)]
        kappa = np.max(kappa)

        ReferenceResult['AccuracyInTotal'] = accuracy.tolist()
        ReferenceResult['ErrorInTotal'] = Error.tolist()
        ReferenceResult['Recall'] = Recall.tolist()
        ReferenceResult['Specificity'] = Specificity.tolist()
        ReferenceResult['Precision'] = Precision.tolist()
        ReferenceResult['FalsePositiveRate'] = FPR.tolist()
        ReferenceResult['F1_score'] = F1_score.tolist()
        ReferenceResult['MatthewsCorrelationCoefficient'] = MCC.tolist()
        ReferenceResult['Kappa'] = kappa.tolist()
        ReferenceResult['TruePositive'] = TP.tolist()
        ReferenceResult['FalsePositive'] = FP.tolist()
        ReferenceResult['FalseNegative'] = FN.tolist()
        ReferenceResult['TrueNegative'] = TN.tolist()

        Result['Recall'] = np.mean(Recall)
        Result['Specificity'] = np.mean(Specificity)
        Result['Precision'] = np.mean(Precision)
        Result['FalsePositiveRate'] = np.mean(FPR)
        Result['F1_score'] = np.mean(F1_score)
        Result['MatthewsCorrelationCoefficient'] = np.mean(MCC)
        Result['Kappa'] = kappa_overall
        R = [
            Result['Recall'], Result['Specificity'], Result['Precision'], Result['F1_score'],
            np.sum(ReferenceResult['AccuracyInTotal']), Result['MatthewsCorrelationCoefficient'], Result['Kappa'],
        ]
        return Result, ReferenceResult, R
