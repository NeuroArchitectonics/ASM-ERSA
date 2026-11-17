# Contributing to ASM-ERSA

Thank you for your interest in contributing to ASM-ERSA! This document provides guidelines for contributing.

## License Agreement

By contributing to ASM-ERSA, you agree that your contributions are submitted under the Contributor License Agreement (CLA) framework: you grant the project owner the right to distribute your contributions under the [ASM-ERSA 自定义社区许可协议](LICENSE.txt) and any subsequent version. Note that this project is **not** distributed under an OSI-approved open-source license.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, PyTorch version, etc.)

### Suggesting Enhancements

1. Create an issue describing your enhancement
2. Explain the motivation and expected benefits
3. Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/[your-username]/asm_ersa.git
cd asm_ersa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions/classes
- Keep functions focused and modular

## Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=asm_ersa
```

## Documentation

- Update docstrings for any changed code
- Update README.md if needed
- Add examples for new features

## Questions?

Feel free to open an issue or contact us at 1393475440@qq.com.

Thank you for contributing! 🎉
