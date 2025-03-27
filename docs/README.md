# Regelum Documentation

This directory contains the documentation for the Regelum framework.

## Building the Documentation

### Using Docker

The easiest way to build and serve the documentation is using Docker and the provided `docker-compose.yml` file.

#### Serving Documentation Locally (Development)

To serve the documentation locally with hot-reloading:

```bash
cd docs
docker compose up mkdocs
```

This will start a development server at [http://localhost:8000](http://localhost:8000).

#### Building Documentation

To build the documentation as static files:

```bash
cd docs
docker compose up mkdocs-build
```

The built documentation will be available in the `site` directory.

### Using pip/uv

If you prefer not to use Docker, you can build the documentation using pip or uv directly:

#### Setup (First Time)

```bash
# Using pip
pip install mkdocs mkdocs-material mkdocs-jupyter mdx-include pymdown-extensions
pip install mkdocs-literate-nav mkdocs-section-index mkdocs-autorefs mkdocs-macros-plugin
pip install mkdocs-gen-files mkdocstrings mkdocstrings-python

# Or using uv
uv pip install mkdocs mkdocs-material mkdocs-jupyter mdx-include pymdown-extensions
uv pip install mkdocs-literate-nav mkdocs-section-index mkdocs-autorefs mkdocs-macros-plugin
uv pip install mkdocs-gen-files mkdocstrings mkdocstrings-python
```

#### Serving Documentation Locally

```bash
cd /path/to/regelum
mkdocs serve
```

#### Building Documentation

```bash
cd /path/to/regelum
mkdocs build
```

## Documentation Structure

- `src/learn/`: Contains the tutorials and guides
  - `pendulum_pd_tutorial.md`: Getting started tutorial with PD controller
  - `basic_concepts.md`: Introduction to basic concepts
  - `advanced_graph.md`: Deep dive into the Graph system
  - `advanced_node.md`: Deep dive into the Node system
  - `energy_based_control.md`: Tutorial on energy-based control
  - `how_it_works.md`: Overview of how Regelum works
  
- `mkdocs.yml`: MkDocs configuration file
- `docker-compose.yml`: Docker configuration for building docs

## MkDocs Plugins

The documentation uses several MkDocs plugins:

- `literate-nav`: Auto-generated navigation from directory structure
- `section-index`: Support for section index pages
- `autorefs`: Automatic reference linking within the documentation
- `macros`: Support for template variables and macros
- `mkdocs-jupyter`: Integration for Jupyter notebooks
- `mdx-include`: Include markdown files within other markdown files
- `pymdown-extensions`: Various Markdown extensions for enhanced formatting
- `gen-files`: Dynamically generate documentation files
- `mkdocstrings`: API documentation generation from docstrings
- `mkdocstrings-python`: Python handler for mkdocstrings
