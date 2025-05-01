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
- Added English language font support with dedicated corpus generator
- Added English fonts specification to configs/font.yml with proper format
- Generated sample font images for English fonts in the dataset
- Added persistent workers support to FontDataModule for better performance
- Added optimizations specifically for NVIDIA L4 GPU on GCP Colab Enterprise
- Added mixed precision training for faster training on L4 GPU
- Added robust error handling for torch.compile to prevent training failures

### Changed
- Updated dependency specifications to be more flexible
- Fixed torch installation requirements
- Modified layout.py to work without libraqm dependency
- Used uv to install and manage dependencies
- Reduced text_size_min from 15 to 9 to support more fonts
- Improved path normalization in font loading to prevent path format issues
- Enhanced the EnglishCorpusGenerator to better handle fonts with limited character support
- Optimized DataLoader worker count based on GPU environment
- Improved torch.compile usage with better backend selection for L4 GPU
- Removed unnecessary multiprocessing.freeze_support() on Linux environments

### Fixed
- Fixed issues with language and direction parameters in the font rendering functions
- Fixed font skipping issues during dataset generation
- Fixed UnqualifiedFontException by improving character validation and fallback mechanisms
- Addressed path format inconsistencies that caused font loading problems
- Fixed multiprocessing issues on macOS by adding proper guards to train.py
- Added worker limits for macOS to prevent crashes during data loading
- Fixed handling of None stroke_color values by using text_color as fallback
- Fixed compatibility issues with training on different GPU architectures

## [0.1.0] - 2023-06-18
### Added
- Initial release of the YuzuMarker.FontDetection module 