"""Decoupled post-hoc AUC decomposition toolkit."""

from .cluster_students import (
    assign_clusters,
    cluster_students_with_irt,
    compute_person_fit,
    fit_irt,
)
from .plot_auc_matrix import (
    compute_auc_matrix_by_cluster,
    plot_auc_matrix,
)

__all__ = [
    "assign_clusters",
    "cluster_students_with_irt",
    "compute_person_fit",
    "fit_irt",
    "compute_auc_matrix_by_cluster",
    "plot_auc_matrix",
]
