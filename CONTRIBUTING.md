# Contributing to Trading Strategy Framework

We welcome contributions to the Trading Strategy Framework! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow these guidelines to help maintain a positive and inclusive community.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/trading-strategy.git
   cd trading-strategy
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Process

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards
3. Write or update tests for your changes
4. Run the test suite:
   ```bash
   pytest
   ```

4. Run code quality checks:
   ```bash
   black src tests
   flake8 src tests
   mypy src
   ```

5. Commit your changes:
   ```bash
   git commit -m "feat: add new momentum indicator"
   ```

### Commit Message Convention

We follow the conventional commits specification:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test-related changes
- `chore:` Maintenance tasks

### Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Create a pull request with:
   - Clear title and description
   - Reference to related issues
   - Screenshots or examples if applicable

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints where possible

### Documentation

- All public functions and classes must have docstrings
- Follow Google-style docstrings
- Include examples in docstrings when helpful

### Testing

- Write unit tests for all new functionality
- Maintain minimum 80% code coverage
- Use pytest fixtures for common test data
- Mark tests appropriately (`@pytest.mark.unit`, `@pytest.mark.integration`)

## Project Structure

```
trading-strategy/
├── src/                    # Source code
│   ├── strategies/         # Trading strategies
│   ├── backtesting/        # Backtesting engine
│   ├── risk/              # Risk management
│   ├── data/              # Data handling
│   ├── portfolio/         # Portfolio optimization
│   └── analytics/         # Performance analytics
├── tests/                 # Test suite
├── research/              # Jupyter notebooks
├── config/                # Configuration files
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

## Types of Contributions

### New Trading Strategies

When adding new strategies:
- Inherit from `BaseStrategy`
- Implement required abstract methods
- Include comprehensive tests
- Add documentation and examples
- Consider risk management implications

### Performance Improvements

- Profile your changes
- Include benchmarks
- Document performance gains
- Ensure backward compatibility

### Bug Fixes

- Include regression tests
- Reference the issue being fixed
- Verify the fix doesn't break existing functionality

### Documentation

- Keep documentation up to date with code changes
- Include practical examples
- Use clear, concise language
- Consider adding diagrams for complex concepts

## Review Process

All submissions require review. We use GitHub pull requests for this purpose. Reviewers will check for:

- Code quality and style compliance
- Test coverage and quality
- Documentation completeness
- Performance implications
- Security considerations

## Performance Guidelines

- Vectorize operations using NumPy/Pandas when possible
- Avoid unnecessary loops in hot paths
- Profile performance-critical code
- Consider memory usage in large-scale operations
- Use appropriate data structures

## Security Guidelines

- Never commit API keys or sensitive information
- Use environment variables for configuration secrets
- Validate all external inputs
- Follow secure coding practices
- Report security vulnerabilities privately

## Financial Data Guidelines

- Respect data provider terms of service
- Handle missing data gracefully
- Implement proper error handling for market data
- Consider timezone handling for global markets
- Validate data integrity

## Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Join our community Slack (if applicable)
- Attend virtual meetups and discussions

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation
- Annual contributor recognition

Thank you for contributing to the Trading Strategy Framework!