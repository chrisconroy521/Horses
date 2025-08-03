# Contributing to Ragozin Sheets Parser

Thank you for your interest in contributing to the Ragozin Sheets Parser! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- OpenAI API key (for testing)

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ragozin-sheets-parser.git
   cd ragozin-sheets-parser
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   python setup_env.py
   ```

## 🛠️ Development Workflow

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Testing
- Write tests for new features
- Ensure all tests pass before submitting
- Test with different Python versions (3.8+)

### Git Workflow
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Commit with descriptive messages: `git commit -m "Add feature: description"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Create a Pull Request

## 📁 Project Structure

```
├── gpt_parser_alternative.py    # Main GPT parser
├── streamlit_app.py            # Streamlit frontend
├── api.py                      # FastAPI backend
├── setup_env.py                # Environment setup script
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .github/workflows/         # CI/CD workflows
└── README.md                  # Project documentation
```

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
- Sample PDF file (if applicable)

## 💡 Feature Requests

For feature requests, please:
- Describe the feature clearly
- Explain the use case
- Provide examples if possible
- Consider implementation complexity

## 🔧 Development Guidelines

### Adding New Features
1. Create a feature branch
2. Implement the feature
3. Add tests
4. Update documentation
5. Test thoroughly
6. Submit PR

### Code Review Process
- All PRs require review
- Address review comments promptly
- Ensure CI checks pass
- Update documentation as needed

## 📝 Documentation

- Keep README.md updated
- Add docstrings to new functions
- Update API documentation if needed
- Include examples for new features

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest flake8

# Run linting
flake8 .

# Run tests (when available)
pytest
```

### Manual Testing
```bash
# Start the application
python start.py

# Test with sample PDF
# Upload cd062525.pdf through the web interface
```

## 🔐 Security

- Never commit API keys or sensitive data
- Use environment variables for configuration
- Validate all user inputs
- Follow security best practices

## 📞 Support

If you need help:
- Check existing issues
- Search documentation
- Create a new issue with details
- Join discussions in issues/PRs

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to the Ragozin Sheets Parser! 🐎 