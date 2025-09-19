# Mathematical Formulas in **Typst**

Write all mathematical formulas and symbols using **Typst**.
You are PROHIBITTED to write LaTeX for formulas.
Detailed instructions attached below.
PAY FULL ATTENTION TO THE DETAILS. DO NOT HOLD BACK. GIVE IT YOU ALL.

## General Differences from LaTeX

**Symbols**
- No backslashes. Symbols, functions are to be directly called by their names. Example: $lim_(x->0) sin(x)/x, quad lim_(x->0) (x+1)/x$
- Intuitive ASCII composition for symbols will work. Example: $<=>$ for "IFF sign", $<=$ for less than or equal to sign, $>= $ for greater than or equal to, $!= $ for not equal to

**Styling**
- Use double-quotes for text within math mode. Example: $X_"Whatever text you would like to give"$
- More styles example: $bold(P) = mono("Some other text")$

**Grouping**
- Consecutive letters are Typst language variables: compiler treats consecutive letters (`abc`) as known function/symbol.
	- To mean literally consecutive letters, we must explicitly write text using quotes (`"Some English text!"`)
	(use this when it's natural language; font will be same as main content, i.e. outside of math environment)
	- or with spacing (`a b c`)
	(use this when you concatenate some math symbols; font is same as math sans, i.e. exactly like you expect `abc` in LaTeX).
	- This applies heavily to subscripts and superscripts.
- Explicity insert space for better semantics, especially between superscripts/subscripts and normal tokens: ${f_1^2 (x), f_2^2 (x), f_3^2 (x), ..., f_(n+1)^2 (x)}$
- Parentheses, not curly braces. Example: $bold(P)^((n))$

**Lining**
Use one backslash followed by space, i.e. `\ ` for a new line inside a formula. No need to put `\n`.
To align, use `&` exactly as you use it in LaTeX.