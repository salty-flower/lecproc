# Mathematical Formulas in **Typst**

Write all mathematical formulas and symbols using **Typst**.
You are PROHIBITTED to write LaTeX for formulas.
Detailed instructions attached below.
PAY FULL ATTENTION TO THE DETAILS. DO NOT HOLD BACK. GIVE IT YOU ALL.

## General Differences from LaTeX

- No backslashes. Symbols, functions are to be directly called by their names. Example: $lim_(x->0) sin(x)/x$
- Parentheses, not curly braces. Example: $bold(P)^((n))$
- Intuitive ASCII composition for symbols will work. Example: $<=>$ for "IFF sign", $<=$ for less than or equal to sign, $>= $ for greater than or equal to, $!= $ for not equal to
- Use double-quotes for text within math mode. Example: $X_"Whatever text you would like to give"$
- Consecutive letters are Typst language variables: compiler treats consecutive letters (`abc`) as known function/symbol. To mean literally consecutive letters, we must explicitly write text using quotes (`"Some English text!"`) (use this when it's natural language; font will be same as main content, i.e. outside of math environment) or with spacing (`a b c`) (use this when you concatenate some math symbols; font is same as math sans, i.e. exactly like you expect `abc` in LaTeX). This applies heavily to subscripts and superscripts.
