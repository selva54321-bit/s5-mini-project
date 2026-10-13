"""
Git Repository Source Module.
Provides utilities to clone remote Git repositories to a temporary local directory
so the existing code ingestion pipeline can scan them.

Supports HTTPS and SSH URLs for GitHub, GitLab, Bitbucket, and any standard Git remote.
"""

import os
import re
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("graph.git_source")


class GitSourceError(Exception):
    """Raised when a git clone or checkout operation fails."""
    pass


def _validate_git_url(url: str) -> bool:
    """
    Validates that the URL looks like a git remote.
    Accepts HTTPS, SSH (git@host:org/repo), and git:// protocols.
    """
    url = url.strip()
    # SSH style: git@github.com:org/repo.git
    if re.match(r"^git@[\w.\-]+:[\w.\-/]+$", url):
        return True
    # HTTPS / git:// style
    parsed = urlparse(url)
    if parsed.scheme in ("https", "http", "git", "ssh"):
        return bool(parsed.netloc and parsed.path)
    return False


def _extract_repo_name(url: str) -> str:
    """Extracts a clean repository name from a git URL."""
    url = url.strip().rstrip("/")
    # Remove .git suffix
    if url.endswith(".git"):
        url = url[:-4]
    # Get last path component
    if ":" in url and "@" in url:
        # SSH format: git@github.com:org/repo
        name = url.split(":")[-1].split("/")[-1]
    else:
        name = url.split("/")[-1]
    # Sanitize
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", name) or "unknown_repo"


def _extract_org(url: str) -> str:
    """Extracts the organization/owner from a git URL."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    if ":" in url and "@" in url:
        # SSH: git@github.com:org/repo
        path_part = url.split(":")[-1]
        parts = path_part.split("/")
        return parts[-2] if len(parts) >= 2 else "unknown"
    else:
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        return parts[-2] if len(parts) >= 2 else "unknown"


def _extract_provider(url: str) -> str:
    """Extracts the provider (github, gitlab, bitbucket, etc.) from a git URL."""
    url = url.strip()
    if ":" in url and "@" in url:
        host = url.split("@")[1].split(":")[0]
    else:
        parsed = urlparse(url)
        host = parsed.netloc or ""

    host_lower = host.lower()
    if "github" in host_lower:
        return "github"
    elif "gitlab" in host_lower:
        return "gitlab"
    elif "bitbucket" in host_lower:
        return "bitbucket"
    elif "azure" in host_lower:
        return "azure"
    return "git"


def clone_repository(
    git_url: str,
    branch: Optional[str] = None,
    depth: Optional[int] = 1,
    target_dir: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """
    Clones a remote git repository to a local directory.

    Args:
        git_url: HTTPS or SSH URL of the repository.
        branch: Optional branch/tag to checkout. Defaults to the remote HEAD.
        depth: Shallow clone depth. Set to None for full history. Defaults to 1 (shallow).
        target_dir: Optional specific directory to clone into. If None, uses a temp directory.

    Returns:
        Tuple of (local_path, repo_name, org, provider)

    Raises:
        GitSourceError: If the URL is invalid or clone fails.
    """
    git_url = git_url.strip()

    if not _validate_git_url(git_url):
        raise GitSourceError(f"Invalid git URL: {git_url}")

    # Check git is available
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True, check=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise GitSourceError("git is not installed or not available on PATH")

    repo_name = _extract_repo_name(git_url)
    org = _extract_org(git_url)
    provider = _extract_provider(git_url)

    # Determine target directory
    if target_dir:
        clone_path = os.path.abspath(target_dir)
    else:
        # Create a temp directory under the project's workspace
        base_tmp = os.path.join(tempfile.gettempdir(), "graph_rag_repos")
        os.makedirs(base_tmp, exist_ok=True)
        clone_path = os.path.join(base_tmp, f"{org}_{repo_name}")

    # If directory already exists with content, remove it for a fresh clone
    if os.path.isdir(clone_path) and os.listdir(clone_path):
        logger.info(f"Removing existing directory for fresh clone: {clone_path}")
        shutil.rmtree(clone_path)

    # Build the clone command
    cmd = ["git", "clone"]
    if depth is not None:
        cmd.extend(["--depth", str(depth)])
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend(["--single-branch", git_url, clone_path])

    logger.info(f"Cloning repository: {git_url} -> {clone_path}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large repos
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise GitSourceError(f"git clone failed (exit {result.returncode}): {stderr}")
    except subprocess.TimeoutExpired:
        raise GitSourceError(f"git clone timed out after 300 seconds for: {git_url}")

    if not os.path.isdir(clone_path):
        raise GitSourceError(f"Clone directory not created: {clone_path}")

    logger.info(f"Successfully cloned {git_url} to {clone_path}")
    return clone_path, repo_name, org, provider


def is_git_url(source: str) -> bool:
    """
    Determines whether the source string is a git URL (as opposed to a local path).
    """
    source = source.strip()
    if source.startswith(("https://", "http://", "git://", "ssh://", "git@")):
        return True
    return False


def cleanup_cloned_repo(local_path: str) -> None:
    """Removes a previously cloned repository directory."""
    if os.path.isdir(local_path):
        shutil.rmtree(local_path, ignore_errors=True)
        logger.info(f"Cleaned up cloned repo at: {local_path}")
