"""
Repository Discovery and File Scanner (Task 1).
Safely traverses a repository path, respects exclusion patterns, prevents symlink escapes,
and collects candidate source files for language parsing.
"""

import os
import fnmatch
from pathlib import Path
from typing import List, Dict, Set, Optional, Any


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".vscode",
    ".idea",
    "node_modules",
    "build",
    "dist",
    ".logfire",
}

DEFAULT_EXCLUDED_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.egg-info*",
    "*.egg",
    ".DS_Store",
    "Thumbs.db",
    "*.lock",
}

SUPPORTED_EXTENSIONS = {
    ".py",
}


def load_gitignore_patterns(repo_root: Path) -> List[str]:
    """Loads patterns from .gitignore in the repository root if it exists."""
    gitignore_file = repo_root / ".gitignore"
    patterns = []
    if gitignore_file.is_file():
        try:
            with open(gitignore_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Remove trailing comments
                        pattern = line.split()[0]
                        patterns.append(pattern)
        except Exception:
            pass
    return patterns


def is_path_excluded(
    rel_path: str,
    name: str,
    is_dir: bool,
    custom_exclusions: Optional[Set[str]] = None,
    gitignore_patterns: Optional[List[str]] = None,
) -> bool:
    """Checks whether a given path or filename matches any exclusion rules."""
    excluded_dirs = DEFAULT_EXCLUDED_DIRS | (custom_exclusions or set())
    if is_dir and name in excluded_dirs:
        return True

    # Check against default patterns
    for pat in DEFAULT_EXCLUDED_PATTERNS:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_path, pat):
            return True

    # Check gitignore patterns
    if gitignore_patterns:
        norm_rel = rel_path.replace("\\", "/")
        for pat in gitignore_patterns:
            clean_pat = pat.rstrip("/")
            if fnmatch.fnmatch(norm_rel, clean_pat) or fnmatch.fnmatch(name, clean_pat):
                return True
            if clean_pat in norm_rel.split("/"):
                return True

    return False


class DiscoveredFile:
    def __init__(self, rel_path: str, abs_path: str, extension: str, size_bytes: int):
        self.rel_path = rel_path
        self.abs_path = abs_path
        self.extension = extension
        self.size_bytes = size_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "abs_path": self.abs_path,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
        }


def scan_repository(
    repo_path: str,
    custom_exclusions: Optional[Set[str]] = None,
    supported_extensions: Optional[Set[str]] = None,
) -> List[DiscoveredFile]:
    """
    Safely scans a repository directory for supported source files.
    Validates path, checks symlink boundaries, and respects exclusion rules.
    """
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    exts = supported_extensions or SUPPORTED_EXTENSIONS
    gitignore_pats = load_gitignore_patterns(root)
    discovered: List[DiscoveredFile] = []

    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=False):
        current_dir = Path(dirpath)

        # Symlink boundary check: do not follow symlinks outside root
        try:
            current_dir.resolve().relative_to(root)
        except ValueError:
            # Traversal escaped repository root via symlink
            dirnames.clear()
            continue

        # Prune excluded directories in-place so os.walk does not descend into them
        dirnames[:] = [
            d for d in dirnames
            if not is_path_excluded(
                rel_path=str((current_dir / d).relative_to(root)),
                name=d,
                is_dir=True,
                custom_exclusions=custom_exclusions,
                gitignore_patterns=gitignore_pats,
            )
        ]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in exts:
                continue

            file_abs = current_dir / fname
            # Prevent following symlinked files that point outside repository
            try:
                if file_abs.is_symlink():
                    file_abs.resolve().relative_to(root)
            except (ValueError, RuntimeError):
                continue

            rel_path = str(file_abs.relative_to(root)).replace("\\", "/")

            if is_path_excluded(
                rel_path=rel_path,
                name=fname,
                is_dir=False,
                custom_exclusions=custom_exclusions,
                gitignore_patterns=gitignore_pats,
            ):
                continue

            try:
                size_bytes = file_abs.stat().st_size
                discovered.append(
                    DiscoveredFile(
                        rel_path=rel_path,
                        abs_path=str(file_abs),
                        extension=ext,
                        size_bytes=size_bytes,
                    )
                )
            except (OSError, PermissionError):
                # Unreadable file - safely skip without aborting scan
                continue

    # Return sorted for deterministic ordering
    discovered.sort(key=lambda x: x.rel_path)
    return discovered
