# Mathematical Formulas in **Typst**

Write all mathematical formulas and symbols using **Typst**.
You are PROHIBITED to write LaTeX for formulas.
Detailed instructions attached below.
PAY FULL ATTENTION TO THE DETAILS. DO NOT HOLD BACK. GIVE IT YOU ALL.

## General Differences from LaTeX

### Symbols

- No backslashes. Symbols, functions are to be directly called by their names. Example: $lim_(x->0) sin(x)/x, quad lim_(x->0) (x+1)/x$
- Intuitive ASCII composition for symbols will work. Example: $<=>$ for "IFF sign", $<=$ for less than or equal to sign, $>= $ for greater than or equal to, $!= $ for not equal to
- Common decorations: $hat(p) approx p$, $overline(X)$, $Pr("Some event" bar "Another event to condition on")$

### Styling

- Use double-quotes for text within math mode. Example: $X_"Whatever text you would like to give"$
- More styles example: $bold(P) = mono("Some other text")$, $cal(C)="I am a calligraphic C"$

### Grouping

- Consecutive letters and numbers are Typst language variables: compiler treats consecutive letters (eg. `abc`, `B2`) as known function/symbol.
  - To mean literally consecutive letters, we must explicitly write text using quotes (`"Some English text!"`)
 (use this when it's natural language; font will be same as main content, i.e. outside of math environment)
  - or with spacing (`a b c`, `B 2`)
 (use this when you concatenate some math symbols; font is same as math sans, i.e. exactly like you expect `abc` in LaTeX).
  - This applies heavily to subscripts and superscripts.

#### Subscripts and Superscripts

- Explicitly insert space for better semantics, especially between superscripts/subscripts and normal tokens: ${f_1^2 (x), f_2^2 (x), f_3^2 (x), ..., f_(n+1)^2 (x)}$
- Group with parentheses, not curly braces; to render a pair of visible parenthesis, write one more layer of parentheses.
  - Example: $bold(P)^((n))$ gives a bold P superscripted by "(n)" - a visible pair of parentheses enclosing the letter "n".

### Lining

Use one backslash followed by space, i.e. "\ " (backslash followed by a space) for a new line inside a formula.
No need to put "\n".
To align, use "&" exactly as you use it in LaTeX.

### MISC

- Cases: within a case, contents automatically get concatenated; use comma to separate different cases, not semicolon.
  eg. $xi_i = cases(+1", with" p=1/2, -1", with" p=1/2)$

## Final Remarks on Output

In this document, when backtick is used to enclose something,
it means the content inside is something valid within math environment, but does not form a complete formula example.

In your output, do not use backtick (`) before or after dollar signs, and in general, not inside math environment.
