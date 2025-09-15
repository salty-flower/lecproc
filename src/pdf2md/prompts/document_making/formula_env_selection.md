# Typst Environment Selection

When encountering mathematical content, your task is to choose the correct enclosing syntax (`$...$`, `$$...$$`, or ` ```typst `) based on the following rules.

## Rule 1: Inline Math (`$...$`)
Use a single dollar sign for formulas or symbols that are part of a sentence and flow with the text.
- **Example:** The equation is `$f(x) = x^2$`, which is a simple parabola.

## Rule 2: Block-Level Math (`$$...$$`)
Use double dollar signs for equations that are displayed on their own line. This is the most common choice for important formulas.
- **Use for:**
  - Complex expressions involving `sum`, `integral`, `mat`.
  - Multi-line equations (use `\ ` for a rendered newline).
  - Any equation that was originally displayed or centered in the source document.
- **⚠️ CRITICAL SOURCE CODE CONSTRAINT:** The entire `$$...$$` block MUST be written on a single line in the Markdown source. DO NOT use the `\n` character within it.

## Rule 3: Typst Code Block (` ```typst `)
This is for special cases and should be used sparingly.
- **Use for:**
  - Typst code that uses commands like `#let`, `#show`, or `#set`. These are invalid inside `$$...$$`.
