from typing import Literal, TypedDict

from pydantic import BaseModel


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class FileInner(TypedDict):
    file_data: str


class FilePart(TypedDict):
    type: Literal["file"]
    file: FileInner


ContentPart = TextPart | FilePart


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: list[ContentPart]


class UserMessage(TypedDict):
    role: Literal["user"]
    content: list[ContentPart]


class BookWideContext(BaseModel):
    title: str | None = None
    author: str | None = None
    description: str | None = None
    language: str | None = None
    publisher: str | None = None
    year: int | None = None

    table_of_contents: str | list[str] | None = None

    heading_preference: str | None = None
    notation_preference: str | None = None
    figure_preference: str | None = None


def compose_pdf_user_messages(
    pdf_file_name: str, base64_pdf: str, general_context: str | None = None
) -> list[SystemMessage | UserMessage]:
    draft: list[SystemMessage | UserMessage | None] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
                    You are an expert editor specializing in mathematical writing.
                    Your task is to convert PDF files to clean, readable, Obsidian-flavored Markdown.
                    Preserve headings, lists, tables and links.

                    ## TikZ for Diagrams

                    Try writing diagrams with TikZ.

                    ### Available Packages

                    amsmath. amstext. amsfonts. amssymb. array.
                    tikz-cd. circuitikz. chemfig. pgfplots. tikz-3dplot.

                    ### Complete Example
                    ```tikz
                    \\usepackage{amsmath}
                    \\begin{document}
                    \\begin{tikzpicture}

                    % Draw states in triangular arrangement
                    \\node[draw, circle, minimum size=1.5cm] (rainy) at (0,3) {R};
                    \\node[draw, circle, minimum size=1.5cm] (sunny) at (-2,0) {S};
                    \\node[draw, circle, minimum size=1.5cm] (cloudy) at (2,0) {C};

                    % Self-loops
                    \\draw[->, thick] (rainy) to [out=120, in=60, looseness=8] node[above] {0.2} (rainy);
                    \\draw[->, thick] (sunny) to [out=200, in=240, looseness=8] node[left] {0.6} (sunny);
                    \\draw[->, thick] (cloudy) to [out=340, in=20, looseness=8] node[right] {0.2} (cloudy);

                    % Transitions from rainy
                    \\draw[->, thick] (rainy) to [bend right=20] node[left] {0.6} (sunny);
                    \\draw[->, thick] (rainy) to [bend left=20] node[right] {0.2} (cloudy);

                    % Transitions from sunny
                    \\draw[->, thick] (sunny) to [bend right=20] node[left] {0.1} (rainy);
                    \\draw[->, thick] (sunny) to [bend right=20] node[below] {0.3} (cloudy);

                    % Transitions from cloudy
                    \\draw[->, thick] (cloudy) to [bend left=20] node[right] {0.7} (rainy);
                    \\draw[->, thick] (cloudy) to [bend right=30] node[below] {0.1} (sunny);

                    \\end{tikzpicture}
                    \\end{document}
                    ```

                    ## Descriptive Text for Pictures

                    If it's a picture, write a descriptive text right at where the picture is.

                    ## LaTeX formatting
                    Always write formulas and symbols in LaTeX format.

                    ### Vectors and matrices
                    For vectors and matrices, write them in Boldface: `\\bm{the symbol}`.
                    When it's Latin and upright preferred, use `\\mathbf{the content}`.

                    ### Border matrices
                    $\\bordermatrix$ isn't supported in MathJax. To bypass this, use $\\mybordermatrix{<whatever content in the main matrix>}{<the leftmost column, serving as column headers>}{<the topmost row, serving as row headers>}$
                    Example:
                        $$
                        \\mathbf{P} =
                        \\mybordermatrix{
                        P_0 & P_1 & P_2 & \\cdots & P_m \\\\
                        P_m & P_0 & P_1 & \\cdots & P_{m-1} \\\\
                        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
                        P_1 & P_2 & P_3 & \\cdots & P_0
                        }{0 \\\\ 1 \\\\ \\vdots \\\\ m}{0&1&2&&\\ldots&&m}
                        $$

                    Other matrices, for example, `pmatrix`, are still supported, so you don't need `mybordermatrix` for vectors.

                    Do not introduce any new information or opinions.
                    """,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                    Here are some general context and my preferences:
                    {general_context}
                    """,
                },
            ],
        }
        if general_context
        else None,
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Please process the PDF file named {pdf_file_name}:",
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:application/pdf;base64,{base64_pdf}",
                    },
                },
            ],
        },
    ]

    return [f for f in draft if f is not None]
