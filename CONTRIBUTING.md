# Contributing to Fracttalix

Thank you for your interest in contributing to Fracttalix.

## How to contribute

### Reporting bugs

Open an issue at https://github.com/thomasbrennan/Fracttalix/issues with:

- A clear description of the problem
- Steps to reproduce
- Python version and OS
- Whether NumPy is installed (and version if so)

### Suggesting features

Open an issue with the `enhancement` label. Describe the use case and how the feature fits within the existing three-channel architecture.

### Submitting changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run the test suite: `python fracttalix_sentinel_v1200.py --test`
5. Ensure all 75 tests pass
6. Submit a pull request with a clear description of the change

### Code style

- The detector is a single-file module by design. Do not split it into multiple files.
- All pipeline steps inherit from `DetectorStep` and use the `@register_step` decorator.
- New steps must be additive — do not modify existing step behavior.
- Include tests for new functionality. Follow the existing `T01`–`T75` naming convention.
- Type annotations on all public API methods.

### Architecture constraints

- **Zero required dependencies.** The core must run on the Python 3.8+ standard library alone. Optional dependencies (NumPy, Matplotlib, etc.) must degrade gracefully.
- **Backward compatibility.** All v7.x kwargs and v8.0 pipeline behavior must be preserved.
- **Three-channel model.** Extensions must fit within the structural/rhythmic/temporal channel architecture.

### Running tests

```bash
# Run full test suite (pure Python)
python fracttalix_sentinel_v1200.py --test

# Run benchmark suite
python fracttalix_sentinel_v1200.py --benchmark
```

### Getting help

Open an issue or start a discussion at https://github.com/thomasbrennan/Fracttalix/issues.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
