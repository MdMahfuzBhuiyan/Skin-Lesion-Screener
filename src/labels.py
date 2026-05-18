"""Multi-class → binary label mapping (DermaMNIST / MedMNIST)."""

BINARY_LABELS = ("benign", "malignant")

# DermaMNIST class names (MedMNIST v3)
DERMAMNIST_CLASS_NAMES = [
    "akiec",   # 0 — actinic keratoses / intraepithelial carcinoma
    "bcc",     # 1 — basal cell carcinoma
    "bkl",     # 2 — benign keratosis
    "df",      # 3 — dermatofibroma
    "mel",     # 4 — melanoma
    "nv",      # 5 — melanocytic nevi
    "vasc",    # 6 — vascular lesions
]


def class_id_to_binary(
    class_id: int,
    malignant_ids: list[int],
    benign_ids: list[int],
) -> str | None:
    if class_id in malignant_ids:
        return "malignant"
    if class_id in benign_ids:
        return "benign"
    return None
