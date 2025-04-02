from typing import List


def confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str]) -> dict:
    """
    Builds a confusion matrix.

    Returns:
    dict: Confusion matrix with true labels as keys and dictionaries of predicted labels as values.
    """
    matrix = {cls: {pred_cls: 0 for pred_cls in classes} for cls in classes}
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    return matrix


def calculate_precision(
    y_true: List[str],
    y_pred: List[str],
    confusion_matrix: dict,
    classes: List[str],
    average: str = "binary",
) -> float:
    """
    Calculate and return the precision score.

    Parameters:
    average (str): Type of averaging performed on the data.
                    'binary' for binary classification,
                    'macro' for unweighted mean per label,
                    'weighted' for weighted mean per label.

    Returns:
    float: Precision score.
    """
    if average not in ["binary", "macro", "weighted"]:
        raise ValueError("average must be 'binary', 'macro', or 'weighted'.")

    if average == "binary":
        if len(classes) != 2:
            raise ValueError(
                "binary precision is only applicable for binary classification. please provide a list of two classes."
            )

        positive_class = classes[1]
        TP = confusion_matrix[positive_class][positive_class]
        FP = sum(confusion_matrix[cls][positive_class] for cls in classes) - TP
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        return precision

    precisions = []
    for cls in classes:
        TP = confusion_matrix[cls][cls]
        FP = sum(confusion_matrix[other_cls][cls] for other_cls in classes) - TP
        precision_cls = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        precisions.append(precision_cls)

    if average == "macro":
        return sum(precisions) / len(classes)
    elif average == "weighted":
        weights = [y_true.count(cls) for cls in classes]
        total = sum(weights)
        weighted_precision = (
            sum(p * w for p, w in zip(precisions, weights)) / total
            if total != 0
            else 0.0
        )
        return weighted_precision


def calculate_recall(
    y_true: List[str],
    y_pred: List[str],
    confusion_matrix: dict,
    classes: List[str],
    average: str = "binary",
) -> float:
    """
    Calculate and return the recall score.

    Parameters:
    average (str): Type of averaging performed on the data.
                    'binary' for binary classification,
                    'macro' for unweighted mean per label,
                    'weighted' for weighted mean per label.

    Returns:
    float: Recall score.
    """
    if average not in ["binary", "macro", "weighted"]:
        raise ValueError("average must be 'binary', 'macro', or 'weighted'.")

    if average == "binary":
        if len(classes) != 2:
            raise ValueError(
                "please provide at least 1 positive label and 1 negative label."
            )
        positive_class = classes[1]
        TP = confusion_matrix[positive_class][positive_class]
        FN = sum(confusion_matrix[positive_class][cls] for cls in classes) - TP
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        return recall

    recalls = []
    for cls in classes:
        TP = confusion_matrix[cls][cls]
        FN = sum(confusion_matrix[cls][other_cls] for other_cls in classes) - TP
        recall_cls = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        recalls.append(recall_cls)

    if average == "macro":
        return sum(recalls) / len(classes)
    elif average == "weighted":
        weights = [y_true.count(cls) for cls in classes]
        total = sum(weights)
        weighted_recall = (
            sum(r * w for r, w in zip(recalls, weights)) / total if total != 0 else 0.0
        )
        return weighted_recall


def calculate_f1(
    y_true: List[str],
    y_pred: List[str],
    confusion_matrix: dict,
    classes: List[str],
    average: str = "binary",
) -> float:
    """
    Calculate and return the F1 score.

    Parameters:
    average (str): Type of averaging performed on the data.
                    'binary' for binary classification,
                    'macro' for unweighted mean per label,
                    'weighted' for weighted mean per label.

    Returns:
    float: F1 score.
    """
    if average not in ["binary", "macro", "weighted"]:
        raise ValueError("average must be 'binary', 'macro', or 'weighted'.")

    if average == "binary":
        precision = calculate_precision(
            average="binary",
            y_true=y_true,
            y_pred=y_pred,
            confusion_matrix=confusion_matrix,
            classes=classes,
        )
        recall = calculate_recall(
            average="binary",
            y_true=y_true,
            y_pred=y_pred,
            confusion_matrix=confusion_matrix,
            classes=classes,
        )
        if (precision + recall) == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    precisions = []
    recalls = []
    for cls in classes:
        TP = confusion_matrix[cls][cls]
        FP = sum(confusion_matrix[other_cls][cls] for other_cls in classes) - TP
        FN = sum(confusion_matrix[cls][other_cls] for other_cls in classes) - TP

        precision_cls = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall_cls = TP / (TP + FN) if (TP + FN) != 0 else 0.0

        precisions.append(precision_cls)
        recalls.append(recall_cls)

    f1_scores = []
    for prec, rec in zip(precisions, recalls):
        if (prec + rec) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * prec * rec / (prec + rec))

    if average == "macro":
        return sum(f1_scores) / len(classes)
    elif average == "weighted":
        weights = [y_true.count(cls) for cls in classes]
        total = sum(weights)
        weighted_f1 = (
            sum(f1 * w for f1, w in zip(f1_scores, weights)) / total
            if total != 0
            else 0.0
        )
        return weighted_f1


def calculate_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate and return the accuracy score.

    Returns:
    float: Accuracy score.
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    accuracy = correct / total if total != 0 else 0.0
    return accuracy


def calculate_kappa(y_true: List[str], y_pred: List[str], classes: List[str]):
    """
    Calculate and return Cohen's Kappa score.

    Returns:
    float: Cohen's Kappa score.
    """
    total = len(y_true)
    if total == 0:
        return 0.0

    # Observed agreement
    observed_agreement = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    p_o = observed_agreement / total

    # Expected agreement
    true_counts = {cls: y_true.count(cls) for cls in classes}
    pred_counts = {cls: y_pred.count(cls) for cls in classes}
    p_e = sum(
        (true_counts.get(cls, 0) * pred_counts.get(cls, 0)) / (total**2)
        for cls in classes
    )

    if (1 - p_e) == 0:
        return 0.0  # Avoid division by zero

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def print_metrics(average: str = "binary"):
    """
    Calculate and print all the metrics.

    Parameters:
    average (str): Type of averaging performed on the data.
                    Applicable for Precision, Recall, and F1 Score.
    """
    precision = calculate_precision(average=average)
    recall = calculate_recall(average=average)
    f1 = calculate_f1(average=average)
    accuracy = calculate_accuracy()
    kappa = calculate_kappa()

    print(f"Precision ({average}): {precision:.2f}")
    print(f"Recall ({average}): {recall:.2f}")
    print(f"F1 Score ({average}): {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Cohen's Kappa: {kappa:.2f}")
