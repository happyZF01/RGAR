try:
    from ._C import tree_search as tree_search
except Exception as exc:  # pragma: no cover
    raise ImportError("tree_vq_ext extension is not built. Run setup.py build_ext --inplace") from exc
