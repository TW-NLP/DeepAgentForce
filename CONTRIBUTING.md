# Contributing

Thanks for your interest in **DeepAgentForce**! Contributions of all kinds are welcome — bug reports, documentation improvements, new skills/tools/MCP integrations, bug fixes, and new features.

## 🐛 Reporting Issues

Before opening an issue, please search the [existing issues](https://github.com/TW-NLP/DeepAgentForce/issues) to avoid duplicates. When reporting, please include where applicable:

- A clear description of the problem and the expected behavior;
- Steps to reproduce (endpoint or skill/tool involved, request payload);
- Your environment (OS, Python version, deployment mode — Docker or local);
- Full error logs or example input/output.

## 🔀 Submitting a Pull Request

1. Fork the repo and create a feature branch off `master`: `git checkout -b feat/your-feature`.
2. Make your changes, keeping the code style consistent with the surrounding code.
3. Run the test suites under `scripts/` and make sure they pass before committing.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (e.g. `feat: ...`, `fix: ...`, `docs: ...`).
5. Push the branch and open a Pull Request describing the motivation, approach, and testing. If it relates to an issue, reference it with `Closes #<number>`.

## 📐 Code & Docs Conventions

- Python code follows PEP 8; match the naming and comment style of the file you're editing.
- When changing configuration or API behavior, update both [README.md](README.md) and [README_CN.md](README_CN.md) accordingly.
- When adding a new skill, tool, or MCP integration, keep the relevant tables in the README in sync.

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [Apache-2.0 License](LICENSE).
