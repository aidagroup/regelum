
# Documentation Guide

This repository uses MkDocs with Material theme for documentation. The setup includes special handling for Jupyter notebooks, LaTeX math rendering, and custom styling.

## Structure

```
docs/
├── src/                   # Source markdown and notebook files
│   ├── _internal/         # Internal assets
│   │   └── javascripts/   # Custom JavaScript (e.g., MathJax config)
│   ├── stylesheets/       # Custom CSS
│   └── learn/             # Tutorial notebooks
├── overrides/             # Theme customization for rendering
└── README.md              # This file
```

## Features

### 1. Jupyter Notebooks
- Notebooks in `src/` are automatically rendered as documentation pages
- Requirements cells handling:
  - First two cells must have the `requirements` tag
  - These cells will be hidden in the rendered documentation
  - First cell typically contains LaTeX preamble
  - Second cell contains package installation commands
- Interactive options:
  - Each notebook page includes a Google Colab badge automatically
  - Users can download notebooks to run locally

### 2. Math Rendering
- Uses MathJax for LaTeX math rendering
- Supports both inline (`$...$`) and display math (`$$...$$`)
- Comprehensive LaTeX preamble with predefined commands:
  - Automatically injected via `overrides/main.html` template
  - Includes extensive mathematical notation (vectors, sets, operators)
  - Common abbreviations and control theory symbols
  - Consistent styling across all documentation pages
- Custom MathJax configuration in `_internal/javascripts/mathjax.js` (do not edit)

## Configuration

Key configuration files:

1. [`mkdocs.yaml`](../mkdocs.yaml): Main configuration file
   - Site structure and navigation
   - Theme settings
   - Extensions and plugins
   - Custom assets

2. [`src/stylesheets/jupyter-code-wrapper.css`](./src/stylesheets/jupyter-code-wrapper.css): Jupyter notebook styling
   - Hides input/output prompts
   - Custom code block styling

3. [`overrides/main.html`](./overrides/main.html): Custom theme template
   - LaTeX preamble injection
   - Interactive notebook badges for opening in Google Colab. The badges are automatically added to each notebook ipynb.

## Adding Content

### Markdown Files
1. Add `.md` files to `src/`
2. Update `nav` section in `mkdocs.yaml`

### Jupyter Notebooks
1. Create notebook in `src/`
2. Add required tags to first two cells:
   ```python
   # Cell 1: LaTeX preamble (with "requirements" tag)
   $$
   % LaTeX definitions
   $$

   # Cell 2: Package installation (with "requirements" tag)
   !pip install required-packages
   ```
3. Add to navigation in `mkdocs.yaml`

## Building Documentation

```bash
pip install -e ".[dev]"
mkdocs serve --port 8000 # Serve documentation locally on port 8000
```

## Plugins

- `literate-nav`: Auto-generated navigation
- `autorefs`: Automatic reference linking
- `macros`: Template variables and macros
- `section-index`: Section index pages
- `mkdocs-jupyter`: Jupyter notebook integration

## Notes

- Ensure all notebooks are properly tagged to hide requirement cells
- Keep LaTeX preamble consistent across notebooks
- Check Colab compatibility for interactive notebooks
