# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import logging
import os
import re
import sys
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
print(os.path.abspath(REPO_ROOT))
sys.path.append(os.path.abspath(REPO_ROOT))

# -- Project information -----------------------------------------------------

project = 'FastVideo'
copyright = f'{datetime.datetime.now().year}, FastVideo Team'
author = 'the FastVideo Team'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "autodoc2",
    "myst_parser",
    "sphinxarg.ext",
    "sphinx_design",
    "sphinx_togglebutton",
]
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]
autodoc2_packages = [
    {
        "path": "../../fastvideo",
        "exclude_dirs": ["__pycache__", "third_party"],
    },
]
autodoc2_output_dir = "api"
autodoc2_render_plugin = "myst"
autodoc2_hidden_objects = ["dunder", "private", "inherited"]
autodoc2_docstring_parser_regexes = [
    (".*", "docs.source.autodoc2_docstring_parser"),
]
autodoc2_sort_names = True
autodoc2_index_template = None
autodoc2_skip_module_regexes = [
    "fastvideo.dataset",
    "fastvideo.distill",
    "fastvideo.data_preprocess",
    "fastvideo.models",
    "fastvideo.sample",
    "fastvideo.utils",
    "fastvideo.distill_adv",
    "fastvideo.train",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = ["**/*.template.md", "**/*.inc.md"]

# Exclude the prompt "$" when copying code
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = project
html_theme = 'sphinx_book_theme'
html_logo = '../../assets/logos/icon_simple.svg'
html_theme_options = {
    'path_to_docs': 'docs/source',
    'repository_url': 'https://github.com/hao-ai-lab/FastVideo/',
    'use_repository_button': True,
    'use_edit_page_button': True,
    # Prevents the full API being added to the left sidebar of every page.
    # Reduces build time by 2.5x and reduces build size from ~225MB to ~95MB.
    'collapse_navbar': True,
    # Makes API visible in the right sidebar on API reference pages.
    'show_toc_level': 3,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_js_files = ["custom.js"]
html_css_files = ["custom.css"]

myst_url_schemes = {
    'http': None,
    'https': None,
    'mailto': None,
    'ftp': None,
    "gh-issue": {
        "url":
        "https://github.com/hao-ai-lab/FastVideo/issues/{{path}}#{{fragment}}",
        "title": "Issue #{{path}}",
        "classes": ["github"],
    },
    "gh-pr": {
        "url":
        "https://github.com/hao-ai-lab/FastVideo/pull/{{path}}#{{fragment}}",
        "title": "Pull Request #{{path}}",
        "classes": ["github"],
    },
    "gh-dir": {
        "url": "https://github.com/hao-ai-lab/FastVideo/tree/main/{{path}}",
        "title": "{{path}}",
        "classes": ["github"],
    },
    "gh-file": {
        "url": "https://github.com/hao-ai-lab/FastVideo/blob/main/{{path}}",
        "title": "{{path}}",
        "classes": ["github"],
    },
}

# see https://docs.readthedocs.io/en/stable/reference/environment-variables.html # noqa
READTHEDOCS_VERSION_TYPE = os.environ.get('READTHEDOCS_VERSION_TYPE')
if READTHEDOCS_VERSION_TYPE == "tag":
    # remove the warning banner if the version is a tagged release
    header_file = os.path.join(os.path.dirname(__file__),
                               "_templates/sections/header.html")
    # The file might be removed already if the build is triggered multiple times
    # (readthedocs build both HTML and PDF versions separately)
    if os.path.exists(header_file):
        os.remove(header_file)


# Generate additional rst documentation here.
def setup(app):
    from docs.source.generate_examples import generate_examples
    generate_examples()


_cached_base: str = ""
_cached_branch: str = ""


def get_repo_base_and_branch(pr_number: str) -> tuple[str | None, str | None]:
    global _cached_base, _cached_branch
    if _cached_base and _cached_branch:
        return _cached_base, _cached_branch

    url = f"https://api.github.com/repos/hao-ai-lab/FastVideo/pulls/{pr_number}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        _cached_base = data['head']['repo']['full_name']
        _cached_branch = data['head']['ref']
        return _cached_base, _cached_branch
    else:
        logger.error("Failed to fetch PR details: %s", response)
        return None, None


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None

    # Get path from module name
    file = Path(f"{info['module'].replace('.', '/')}.py")
    path = REPO_ROOT / file
    if not path.exists():
        path = REPO_ROOT / file.with_suffix("") / "__init__.py"
    if not path.exists():
        return None

    # Get the line number of the object
    with open(path) as f:
        lines = f.readlines()
    name = info['fullname'].split(".")[-1]
    pattern = fr"^( {{4}})*((def|class) )?{name}\b.*"
    for lineno, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        if re.match(pattern, line):
            break

    # If the line number is not found, return None
    if lineno == len(lines):
        return None

    # If the line number is found, create the URL
    filename = path.relative_to(REPO_ROOT)
    if "checkouts" in path.parts:
        # a PR build on readthedocs
        pr_number = REPO_ROOT.name
        base, branch = get_repo_base_and_branch(pr_number)
        if base and branch:
            return f"https://github.com/{base}/blob/{branch}/{filename}#L{lineno}"
    # Otherwise, link to the source file on the main branch
    return f"https://github.com/hao-ai-lab/FastVideo/blob/main/{filename}#L{lineno}"


# Mock out external dependencies here, otherwise the autodoc pages may be blank.
autodoc_mock_imports = [
    "blake3",
    "compressed_tensors",
    "cpuinfo",
    "cv2",
    "torch",
    "huggingface_hub",
    "torchvision",
    "transformers",
    "psutil",
    "prometheus_client",
    "sentencepiece",
    "vllm._C",
    "PIL",
    "numpy",
    'triton',
    "tqdm",
    "tensorizer",
    "pynvml",
    "outlines",
    "xgrammar",
    "librosa",
    "soundfile",
    "gguf",
    "lark",
    "decord",
]

for mock_target in autodoc_mock_imports:
    if mock_target in sys.modules:
        logger.info(
            "Potentially problematic mock target (%s) found; "
            "autodoc_mock_imports cannot mock modules that have already "
            "been loaded into sys.modules when the sphinx build starts.",
            mock_target)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "typing_extensions":
    ("https://typing-extensions.readthedocs.io/en/latest", None),
    "aiohttp": ("https://docs.aiohttp.org/en/stable", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "psutil": ("https://psutil.readthedocs.io/en/stable", None),
}

navigation_with_keys = False
