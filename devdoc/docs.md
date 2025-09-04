# Documentation Guide

This document provides a detailed guide on how to configure and update the documentation for the `unfolding` project. The documentation is built using [Hugo](https://gohugo.io/) and automatically deployed to GitHub Pages via a GitHub Actions workflow.

## Configuration

### Hugo Setup

The static documentation site is generated using Hugo. Here's a breakdown of the configuration:

- **Main Configuration:** The primary Hugo configuration is located in `docs/hugo.toml`. It contains settings like the `baseURL`, `title`, and the theme.
- **Content:** All documentation source files are Markdown files (`.md`) located in the `docs/content/` directory. The structure of this directory dictates the structure of the final documentation site.
- **Theme:** The site uses the `relearn` theme, which is included as a git submodule in `docs/themes/relearn`. This submodule ensures that the theme is version-controlled and can be easily updated.

To build the site locally, you can run the following commands:

```bash
cd docs
hugo
```

The generated static site will be placed in the `docs/public/` directory.

### GitHub Workflow

The automatic deployment process is handled by a GitHub Actions workflow defined in `.github/workflows/gh-pages.yml`.

- **Trigger:** The workflow is automatically triggered on any push to the `main` branch.
- **Process:** The workflow consists of two main jobs:
    1.  **`build`:** This job checks out the repository, including the theme submodule. It then sets up the Hugo environment and runs the `hugo` command to build the static site. The resulting `docs/public` directory is then uploaded as a build artifact.
    2.  **`deploy`:** This job runs after the `build` job has successfully completed. It downloads the build artifact and uses the official `actions/deploy-pages@v4` action to deploy the site to GitHub Pages.
- **Permissions:** The workflow is configured with specific permissions to allow it to read the repository content and write to the GitHub Pages deployment.

## How to Update the Documentation

Here is a step-by-step tutorial for updating the documentation.

### 1. Prerequisites

- **Git:** You must have Git installed to clone the repository and manage changes.
- **Hugo (for local preview):** To preview your changes locally before pushing them, you need to install Hugo. You can find detailed installation instructions on the [Hugo website](https://gohugo.io/getting-started/installing/).
- **Repository Setup:** After cloning the repository, make sure to initialize and update the git submodules to fetch the theme:

  ```bash
  git submodule update --init --recursive
  ```

### 2. Edit Content

- The documentation pages are Markdown files located in the `docs/content/` directory.
- To **edit an existing page**, simply open the corresponding `.md` file in a text editor and make your changes.
- To **add a new page**, create a new `.md` file within the `docs/content/` directory. You can organize pages in subdirectories. It's often easiest to copy the front matter (the section at the top between `---` markers) from an existing page and modify it.

### 3. Preview Changes Locally (Recommended)

- To ensure your changes are rendered correctly, it's highly recommended to preview them locally.
- Open a terminal and navigate to the `docs` directory:

  ```bash
  cd docs
  ```

- Start the Hugo development server:

  ```bash
  hugo server
  ```

- Open your web browser and navigate to `http://localhost:1313/`. You will see a live preview of the documentation site. The server will automatically rebuild and reload the page whenever you save a file.

### 4. Commit and Push Changes

- Once you are satisfied with your changes, you need to commit them to the repository.
- Stage your changes:

  ```bash
  git add .
  ```

- Commit the changes with a clear and descriptive message:

  ```bash
  git commit -m "docs: Describe the new feature X"
  ```

- Push your commit to the `main` branch on GitHub:

  ```bash
  git push origin main
  ```

### 5. Automatic Deployment

- Pushing your changes to the `main` branch will automatically trigger the GitHub Actions workflow.
- You can monitor the progress of the workflow in the "Actions" tab of the project's GitHub repository.
- Once the workflow completes successfully (usually within a few minutes), your changes will be live on the GitHub Pages site.
