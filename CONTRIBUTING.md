# Contribution Guidelines
Welcome to **OntoAligner**!

We appreciate your interest in contributing to this project. Whether you're a developer, researcher, or enthusiast, your contributions are invaluable. Before you start contributing, please take your time to review the following guidelines.

## 1. How to Contribute?


| Contribution      | How?                                                                                                                                                                                                                                                                                                                                          |
|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Reporting Bugs | If you encounter a bug or unexpected behavior, please report it via [GitHub Issue Tracker](https://github.com/HamedBabaei/sciknoworg/issues) with details and  screenshots of the bug report. Also mention what was your expected behavior and also your operating system and python version that you are using.                              |
| 2. Adding a New Ontology Matching System | If you have developed a new ontology matching system and want to add it to the library, we would love to hear from you. Please open an [issue](https://github.com/HamedBabaei/sciknoworg/issues) and provide a clear description, suggestions on how to integrate it, your implementation details, and any related publications if available. |
| 3. Improving Documentation| If you have suggestions to make our documentation clearer or more professional, or if you think additional sections are needed, feel free to let us know. Alternatively, you can make the changes directly and submit a pull request for us to review.                                                                                        |


### 2. Commit Guidelines

For efficient code review, *`make one commit per logical change`* and keep changes small to speed up the process and simplify troubleshooting. Avoid mixing whitespace changes with functional updates, and separate unrelated functional changes into distinct commits. Use the imperative mood in commit messages (e.g., "Add preprocessing step") and keep the subject concise, ideally under 60 characters. Focus on what the change does, not how it’s done, and format references to issues or PRs like "Add LLaMA-3.2 [#2]". Optionally, use emoji codes for clarity.


**This is an Optional** but feel free to use the following emoji codes in your message.

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

## 3. How to Submit a Pull Request (PR)

To contribute changes to the library, please follow these steps:

1. Fork the `OntoAligner` repository.
2. Clone your forked repository.
```bash
git clone git@github.com:USERNAME/OntoAligner.git
cd OntoAligner
```
3. Create a virtual environment with python=3.10, activate it, install the required dependencies and install the pre-commit configuration:
```bash
conda create -n my_env python=3.9
conda activate my_env
pip install -r requirements.txt
pre-commit install
```
4. Create a branch and commit your changes:
```bash
git switch -c <name-your-branch>
# do your changes
git add .
git commit -m "your commit msg"
git push
```
5. Update the documentation to reflect your changes.
6. Ensure your code adheres to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
7. Format the code using `ruff check --fix .`.
8. Open a pull request with your changes to the `dev` branch.
9. Be responsive to feedback during the review process.



We look forward to your contributions!
