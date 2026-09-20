"""Story-012 TEST-001: docs images exist; every content page has frontmatter;
optional hugo build check when hugo is available."""
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT = os.path.join(ROOT, "docs", "content")
STATIC_IMG = os.path.join(ROOT, "docs", "static", "images")


def test_all_figure_references_resolve():
    img_re = re.compile(r"src=\"(/images/[^\"$]+)\"")
    refs = []
    for dirpath, _, files in os.walk(CONTENT):
        for fn in files:
            if fn.endswith(".md"):
                text = open(os.path.join(dirpath, fn)).read()
                refs.extend(img_re.findall(text))
    assert refs, "no figure references found in docs content"
    for ref in refs:
        path = os.path.join(ROOT, "docs", "static", ref.lstrip("/"))
        assert os.path.exists(path), f"missing docs figure: {ref}"


def test_content_pages_have_frontmatter():
    for fn in os.listdir(CONTENT):
        if fn.endswith(".md"):
            first = open(os.path.join(CONTENT, fn)).readline().strip()
            assert first == "---", f"{fn} missing Hugo frontmatter"


def test_hugo_build():
    if shutil.which("hugo") is None:
        import pytest

        pytest.skip("hugo not installed")
    out_dir = os.path.join(ROOT, "docs", "_check_docs_build")
    r = subprocess.run(
        ["hugo", "--source", "docs", "--destination", out_dir],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.exists(os.path.join(out_dir, "index.html"))
    shutil.rmtree(out_dir)
