"""Artifact IO errors."""


class ArtifactError(Exception):
    """Base error for artifact path / IO failures."""


class MissingConfigError(ArtifactError, FileNotFoundError):
    """Run config file is missing."""


class CorruptArtifactError(ArtifactError):
    """Config / result / index is not a mapping or cannot be parsed."""
