# Website Requirements

## Overview

Your final deliverable includes a self-contained website that presents your analysis, findings, and visualizations. The website serves as the primary means of communicating your work and should be professional, well-organized, and easy to navigate.

## Required Structure

The website must have the following structure in the `website/` directory:

1. **Landing and project description page** - `website/index.html`
   - You populate this during the **final submission (Milestone 4)**
   - Should include: project overview, team members, high-level problem statement, dataset description

2. **Exploratory Data Analysis page** - `website/eda.html`
   - You populate this during **Milestone 1**
   - Should include: statistical analysis, temporal patterns, visualizations, key findings from EDA questions

3. **NLP page** - `website/nlp.html`
   - You populate this during **Milestone 2**
   - Should include: sentiment analysis results, topic modeling, text mining insights, NLP visualizations

4. **ML page** - `website/ml.html`
   - You populate this during **Milestone 3**
   - Should include: model descriptions, evaluation metrics, feature importance, prediction results

5. **Conclusion page** - `website/conclusion.html`
   - You populate this during **final submission (Milestone 4)**
   - Should include: summary of findings, business recommendations, answers to all 10 questions, future work

## Technology Choices

**You have complete freedom in how you build your website.** Choose any technology stack you're comfortable with:

### Option 1: HTML/CSS/JavaScript (Manual)
- Write HTML pages directly
- Use CSS frameworks like Bootstrap, Tailwind, or custom styles
- Add interactivity with JavaScript
- Use visualization libraries like D3.js, Chart.js, Plotly.js

### Option 2: Quarto (Recommended for simplicity)
- **Most straightforward approach** for data science projects
- Generate plots and tables in your Python/PySpark code
- Use Quarto to render content combining markdown, code, and visualizations
- Automatically creates professional-looking HTML pages

**Quarto Resources:**
- [Quarto Documentation](https://quarto.org/)
- [Quarto Gallery Examples](https://quarto.org/docs/gallery/)
- [Quarto for Python](https://quarto.org/docs/computations/python.html)

**Using the Provided Quarto Template:**

This repository includes a complete Quarto website template in the `website-source/` directory with all required pages pre-configured:

- `index.qmd` - Landing page
- `eda.qmd` - EDA analysis
- `nlp.qmd` - NLP analysis
- `ml.qmd` - ML models
- `conclusion.qmd` - Final synthesis

**Step 1: Install Quarto**

Download and install Quarto from: https://quarto.org/docs/get-started/

Or on Linux:
```bash
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.553/quarto-1.4.553-linux-amd64.deb
sudo dpkg -i quarto-1.4.553-linux-amd64.deb
```

**Step 2: Generate Your Plots and Data**

In your analysis code, save visualizations and results:
```python
import matplotlib.pyplot as plt

# Generate visualization
plt.figure(figsize=(10, 6))
# ... your plot code ...
plt.savefig('data/plots/eda_temporal_patterns.png', dpi=300, bbox_inches='tight')

# Save results
results_df.to_csv('data/csv/eda_statistics.csv', index=False)
```

**Step 3: Edit the .qmd Files**

Edit the placeholder files in `website-source/` with your actual content:

```markdown
## Temporal Patterns

Our analysis reveals distinct posting patterns across time periods.

![Posting patterns over time](../data/plots/eda_temporal_patterns.png)

## Key Statistics

```{python}
#| echo: false
import pandas as pd
df = pd.read_csv('../data/csv/eda_statistics.csv')
print(df.head(10).to_markdown(index=False))
```
```

**Step 4: Build the Website**

```bash
# Navigate to website source
cd website-source

# Render the entire website
quarto render

# Or preview with auto-reload during development
quarto preview
```

The website will be generated in the `docs/` directory (configured for GitHub Pages).

**Step 5: Verify Output**

Check that `docs/index.html` and other pages were created:
```bash
ls -la docs/
```

You can open `docs/index.html` in a browser to view your site locally.

**Step 6: Enable GitHub Pages (Optional)**

To publish your website on GitHub Pages:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select **Deploy from a branch**
4. Select branch: **main** and folder: **/docs**
5. Click **Save**

Your website will be published at: `https://your-username.github.io/your-repo-name/`

### Option 3: React/Vue/Other Frameworks
- Use modern JavaScript frameworks
- Create interactive dashboards
- Integrate with D3.js or other visualization libraries
- Host as static site

### Option 4: Jupyter Book or Similar
- Convert Jupyter notebooks to HTML
- Use tools like Jupyter Book, Sphinx, or MkDocs

## Best Practices

1. **Navigation**: Include clear navigation between all pages
2. **Responsive design**: Ensure website works on different screen sizes
3. **Visualizations**: Use high-quality, clearly labeled plots
4. **Writing**: Provide clear explanations and insights, not just plots
5. **Code**: Do NOT include raw code in the website (unless showing example snippets)
6. **Performance**: Optimize images and assets for fast loading
7. **Consistency**: Maintain consistent styling across all pages

## Incremental Development

Build your website incrementally with each milestone:
- **Milestone 1**: Create `eda.html` with EDA findings
- **Milestone 2**: Add `nlp.html` with NLP results
- **Milestone 3**: Add `ml.html` with ML models and results
- **Milestone 4**: Complete with `index.html` and `conclusion.html`, polish all pages

This incremental approach helps you:
- Get feedback early and often
- Avoid last-minute rush
- Build a coherent narrative throughout the project

## Serving Your Website (Optional)

You do **not** need to host your website online. The graders will view it locally from your repository. However, if you wish to host it:
- GitHub Pages (free, easy)
- Netlify (free tier available)
- Vercel (free tier available)
- Any static site hosting service

## Questions?

If you're unsure about website requirements, ask on the course discussion board or during office hours.
