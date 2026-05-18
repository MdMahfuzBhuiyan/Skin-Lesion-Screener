"""ISIC 2019 multi-class → binary label mapping."""

CLASS_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
BINARY_LABELS = ("benign", "malignant")


def row_to_binary_label(row, malignant: list[str], benign: list[str]) -> str | None:
    """Return 'benign' or 'malignant' from a one-hot ground-truth row."""
    active = [c for c in CLASS_COLUMNS if int(row.get(c, 0)) == 1]
    if len(active) != 1:
        return None
    label = active[0]
    if label in malignant:
        return "malignant"
    if label in benign:
        return "benign"
    return None
