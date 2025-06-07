#################################################################################################
# Copyright (c) 2022-2025 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

from .._environment import DISABLE_TQDM


def opt_progress_bar(fn, total):
    if DISABLE_TQDM:
        return fn
    try:
        import tqdm  # type: ignore[import-untyped]

        return tqdm.tqdm(fn, total=total)

    except ImportError:
        return fn


try:
    from rich.console import Console
    from rich.table import Table

    def _print_rich_table(title, headers, values):
        table = Table(show_header=bool(title), header_style="bold", title=title)
        for h in headers:
            table.add_column(h, justify="center")
        for r in values:
            table.add_row(*r)
        return table

    def print_table(title, headers, values, has_footer=False):
        console = Console()
        console.print(_print_rich_table(title, headers, values))

except ImportError:
    # Minimal table print

    import sys

    def bold(text):
        if sys.stdout.isatty():  # Only add ANSI codes if output is a terminal
            return f"\033[1m{text}\033[0m"
        return text

    def _print_ascii_table(header, rows, title=None, footer=None):
        # Combine header, rows, footer (if any) to calculate col widths
        all_rows = [header] + rows + ([footer] if footer else [])
        columns = list(zip(*all_rows))
        col_widths = [max(len(str(cell)) for cell in col) for col in columns]

        def format_row(row, bold_row=False):
            formatted = " | ".join(
                str(cell).ljust(width) for cell, width in zip(row, col_widths)
            )
            formatted = f"| {formatted} |"
            return bold(formatted) if bold_row else formatted

        def separator(char="-", junction="+"):
            return (
                junction + junction.join(char * (w + 2) for w in col_widths) + junction
            )

        # Title
        if title:
            title_lines = title.split("\n")

            total_width = sum(col_widths) + 3 * len(col_widths) + 1
            print("|" + "=" * (total_width - 2) + "|")

            for t in title_lines:
                print("|" + str(t).center(total_width - 2) + "|")

            print("|" + "=" * (total_width - 2) + "|")

        # Top border
        print(separator("="))

        # Header
        print(format_row(header, bold_row=True))
        print(separator("="))

        # Rows
        for row in rows:
            print(format_row(row))

        # Footer (if any)
        if footer:
            print(separator("="))
            print(format_row(footer, bold_row=True))

        # Bottom border
        print(separator("="))

        print()

    def print_table(title, headers, values, has_footer=False):
        if has_footer:
            rows = values[:-1]
            footer = values[-1]
        else:
            rows = values
            footer = None

        _print_ascii_table(header=headers, rows=rows, footer=footer, title=title)
