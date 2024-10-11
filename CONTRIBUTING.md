# Contribution Guidelines
Welcome to **OntoAligner**!

We appreciate your interest in contributing to this project. Whether you're a developer, researcher, or enthusiast, your contributions are invaluable. Before you start contributing, please take your time to review the following guidelines.

## 1. How to Contribute?

### 1.1. Reporting Bugs

If you encounter a bug or unexpected behavior, please help us by reporting it. Use the [GitHub Issue Tracker](https://github.com/HamedBabaei/OntoAligner/issues) to create a detailed bug report. Include the following information:
- A descriptive title
- Clear instructions on how to reproduce the bug
- A screenshot of the bug or unexpected behavior
- The expected behavior
- The actual behavior
- Your operating system and Python version

### 1.2. Adding a New Ontology Matching System
If you have developed a new ontology matching system and want to add it to the library, we would love to hear from you. Please open an issue and provide a clear description, suggestions on how to integrate it, your implementation details, and any related publications if available.

### 1.3. Improving Documentation
If you have suggestions to make our documentation clearer or more professional, or if you think additional sections are needed, feel free to let us know. Alternatively, you can make the changes directly and submit a pull request for us to review.

### 1.4. Commit Guidelines

#### 1.4.1. Functional Best Practices

- Use **one commit per logical change** for efficient review.
- Make smaller code changes to facilitate quicker reviews and easier troubleshooting.
- Avoid mixing whitespace changes with functional code changes, and keep unrelated functional changes in separate commits to speed up the review process.

#### 1.4.2. Stylistic Best Practices

- Use the imperative mood in the commit subject (e.g., "Add preprocessing step" instead of "Adding preprocessing step").
- Keep the subject line concise, preferably under 60 characters.
- Focus on **what** the change does in the commit message, and avoid explaining **how** (which belongs in the code or documentation).
- If a commit references an issue or pull request, format it as follows: "Add LLaMA-3 [#42]".
- Optionally, prepend your commit message with one of the following emoji codes for clarity:

| Code           | Emoji | Use for                        |
|----------------|-------|--------------------------------|
| `:fire:`       | 🔥    | Remove code or files           |
| `:bug:`        | 🐛    | Fix a bug or issue             |
| `:sparkles:`   | ✨    | Add feature or improvements    |
| `:memo:`       | 📝    | Add or update documentation    |
| `:tada:`       | 🎉    | Start a project                |
| `:recycle:`    | ♻️    | Refactor code                  |
| `:pencil2:`    | ✏️    | Minor changes   or improvement |
| `:bookmark:`   | 🔖    | Version release                |
| `:adhesive_bandage:` | 🩹 | Non-critical fix               |
| `:test_tube:`  | 🧪    | Test-related changes           |
| `:boom:`       | 💥    | Introduce breaking changes     |

## 2. How to Submit a Pull Request (PR)

To contribute changes to the library, please follow these steps:

1. Fork the `OntoAligner` repository.
2. Create a new branch for your changes.
3. Implement your changes.
4. Update the documentation to reflect your changes.
5. Ensure your code adheres to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
6. Format the code using `ruff` (`ruff check --fix .`).
7. Write tests to validate the functionality if necessary.
8. If your changes involve code, run the tests and ensure they all pass.
9. Open a pull request with your changes to the `dev` branch.
10. Be responsive to feedback during the review process.

## 3. License

By contributing to OntoAligner, you agree that your contributions will be licensed under the [MIT License](https://github.com/HamedBabaei/OntoAligner/blob/main/LICENSE).

We look forward to your contributions!
