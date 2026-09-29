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


def test_example_pages_embed_config_and_guide_link():
    """Story-041 FR-001..003: every example page embeds its bundle
    configuration in a fenced ```toml block and links the shared
    [method, parameters, and k-path] guide."""
    examples = os.path.join(CONTENT, "examples")
    pages = sorted(
        fn for fn in os.listdir(examples)
        if fn.endswith(".md") and fn != "_index.md"
    )
    assert len(pages) == 16, pages
    for fn in pages:
        text = open(os.path.join(examples, fn)).read()
        assert "```toml" in text, f"{fn}: no ```toml configuration block"
        assert "/guide/method-parameters-kpaths/" in text, (
            f"{fn}: missing method-parameters-kpaths guide link"
        )


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
    # Figure shortcodes must resolve against the baseURL subpath: a raw
    # src="/images/..." in the built HTML 404s on the Pages deployment.
    examples = os.path.join(out_dir, "examples", "index.html")
    html = open(examples, encoding="utf-8").read()
    srcs = re.findall(r'<img src="([^"]+images/[^"]+)"', html)
    assert srcs, "no figure images rendered on the examples page"
    for src in srcs:
        assert not src.startswith("/images/"), f"unsubpathed figure: {src}"
        built = os.path.join(out_dir, src.split("/unfolding/", 1)[-1])
        assert os.path.exists(built), f"missing built figure: {src}"
    shutil.rmtree(out_dir)
