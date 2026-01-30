# Website Source

This directory contains the Quarto source files for the project website.

## Structure

- `index.qmd` - Home page (Milestone 4)
- `eda.qmd` - Exploratory Data Analysis page (Milestone 1)
- `nlp.qmd` - Natural Language Processing page (Milestone 2)
- `ml.qmd` - Machine Learning page (Milestone 3)
- `conclusion.qmd` - Conclusion and recommendations (Milestone 4)
- `_quarto.yml` - Quarto configuration
- `styles.css` - Custom CSS styling

## Building the Website

### Prerequisites

Install Quarto: https://quarto.org/docs/get-started/

### Render the Website

```bash
# From this directory
quarto render

# Or from project root
cd website-source
quarto render
```

The rendered website will be output to the `../docs/` directory (configured for GitHub Pages).

### Preview While Developing

```bash
quarto preview
```

## Workflow

1. **Milestone 1 (Week 3)**: Edit `eda.qmd` with EDA findings
2. **Milestone 2 (Week 5)**: Edit `nlp.qmd` with NLP analysis
3. **Milestone 3 (Week 6)**: Edit `ml.qmd` with ML results
4. **Milestone 4 (Week 8)**: Complete `index.qmd` and `conclusion.qmd`

## Resources

- [Quarto Documentation](https://quarto.org/)
- [Quarto Gallery](https://quarto.org/docs/gallery/)
