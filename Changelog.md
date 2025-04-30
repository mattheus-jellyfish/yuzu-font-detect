# Changelog

All notable changes to the YuzuMarker.FontDetection project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Changelog.md file to track changes
- Added pyproject.toml for modern Python dependency management
- Added virtual environment (.venv) setup for isolated development
- Added L4 GPU-specific installation option for CUDA 12.1 dependencies
- Added README_UPGRADE.md with uv sync commands for dependency management

### Changed
- Updated dependency specifications to be more flexible
- Updated PyTorch configuration for broader compatibility
- Updated Lightning (formerly PyTorch Lightning)
- Updated Gradio to latest version
- Updated Pillow to latest stable version 
- Updated all dependencies to versions compatible with modern environments
- Simplified dependency management by removing uv-specific configuration

### Fixed
- Resolved deprecated API calls in PyTorch and Lightning libraries
- Fixed compatibility issues with newer CUDA versions
- Fixed virtual environment setup script for broader compatibility 