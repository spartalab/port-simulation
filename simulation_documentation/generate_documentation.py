import re
import sys
import subprocess
import os
import webbrowser
from pathlib import Path

try:
    import markdown
except ImportError:
    print("The 'markdown' package is required. Install it via 'pip install markdown'.", file=sys.stderr)
    sys.exit(1)

# List of module directories
MODULES = [
    "simulation_classes",
    "simulation_handler",
    "simulation_analysis",
    "main",
]

def get_venv_python() -> str:
    """Return the path to the current virtualenv's python executable."""
    # If we're already inside a venv, sys.executable is ideal.
    # Otherwise, fall back to .venv/<Scripts|bin>/python
    exe = Path(sys.executable)
    if "Scripts" in exe.parts or "bin" in exe.parts:
        return str(exe)
    venv_dir = Path(__file__).parent / ".venv"
    if os.name == "nt":
        return str(venv_dir / "Scripts" / "python.exe")
    else:
        return str(venv_dir / "bin" / "python")

def generate_docs(output_dir: str = "./simulation_documentation"):
    """Run pdoc to generate documentation for all modules and packages."""
    python_exe = get_venv_python()

    # build the command
    cmd = [
        python_exe, "-m", "pdoc",
        *MODULES,
        "--docformat", "google",
        "--no-include-undocumented",
        "-o", output_dir
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent)
    

    print(f"Running: {cmd!r} with PYTHONPATH={env['PYTHONPATH']}")
    subprocess.run(cmd, check=True, env=env)
    print(f"Documentation generated into {output_dir}")

def inject_readme(html_path: str = "./simulation_documentation/index.html", readme_path: str = "README.md"):
    """Inject the README.md contents into the generated pdoc index.html."""
    html_path = Path(html_path)
    readme_path = Path(readme_path)

    html = html_path.read_text(encoding="utf-8")
    md_text = readme_path.read_text(encoding="utf-8")
    md_html = markdown.markdown(md_text, extensions=["fenced_code", "tables", "toc"])

    # adjust image paths if needed
    md_html = md_html.replace(
        'src="simulation_documentation/home_logo.png"',
        'src="home_logo.png"'
    )

    insert_html = f"""
<section class="module-info">
{md_html}
</section>
"""

    pattern = re.compile(r'(<main class="pdoc">)(.*?)(</main>)', re.DOTALL)
    if not pattern.search(html):
        print("Failed to locate the <main class=\"pdoc\"> element.", file=sys.stderr)
        sys.exit(1)

    new_html, count = pattern.subn(lambda m: f"{m.group(1)}{insert_html}{m.group(3)}", html)
    html_path.write_text(new_html, encoding="utf-8")
    print(f"Successfully injected README.md into '{html_path}' ({count} replacements performed)")

def open_index(html_path: str = "./simulation_documentation/index.html"):
    abs_path = Path(html_path).resolve()
    url = f"file://{abs_path}"
    print(f"Opening {url} in the default web browser...")
    webbrowser.open(url)

if __name__ == "__main__":
    print("Generating documentation using pdoc…")
    generate_docs()

    print("Injecting README.md into the documentation…")
    html_file = sys.argv[1] if len(sys.argv) > 1 else "./simulation_documentation/index.html"
    readme_file = sys.argv[2] if len(sys.argv) > 2 else "README.md"
    inject_readme(html_file, readme_file)

    print("Opening the documentation index in the web browser…")
    open_index(html_file)
