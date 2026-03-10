# Contributing to Fracttalix Sentinel

Thank you for your interest in contributing to Fracttalix Sentinel.

## Reporting bugs

Open a GitHub issue with:

- Python version (`python3 --version`)
- Sentinel version (`python3 fracttalix_sentinel_v1200.py --version`)
- Minimal reproducing code or data
- Expected vs. actual behavior

## Suggesting enhancements

Open a GitHub issue describing the use case and proposed behavior. For changes to the detection pipeline or theoretical model, reference the relevant FRM paper or axiom.

## Development setup

```bash
git clone https://github.com/thomasbrennan/Fracttalix.git
cd Fracttalix
pip install -e ".[full]"

# Run the test suite
python3 fracttalix_sentinel_v1200.py --test
```

No required dependencies beyond Python 3.8+. The optional `[full]` extras (numpy, matplotlib, numba, tqdm) enable FFT-based features and visualization.

## Pull request process

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused — one logical change per PR.
3. Run the test suite and confirm all tests pass before submitting.
4. Write a clear PR description explaining *what* changed and *why*.
5. New detection steps or alert types should include corresponding tests.

## Code style

- Follow existing patterns in `fracttalix_sentinel_v1200.py`.
- New pipeline steps subclass `DetectorStep` and implement `execute()`.
- Configuration additions go into `SentinelConfig` as frozen dataclass fields with sensible defaults.
- All new features should be additive — do not remove or alter existing step behavior.

## Theoretical contributions

Fracttalix Sentinel implements the Fractal Rhythm Model (FRM). Contributions that extend the theoretical framework should reference the relevant FRM axioms and papers. The AI layer system (`ai-layers/`) documents falsifiable claims for verification.

## License

By contributing, you agree that your contributions will be released under the CC0 1.0 Universal public domain dedication, consistent with the project license.
