# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/_autosummary
	find $(SOURCEDIR) -name "*.pyc" -delete
	find $(SOURCEDIR) -name "__pycache__" -type d -exec rm -rf {} +

# Quick development build (without warnings as errors)
dev:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -E

# Production build with all checks
prod: clean
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) -E

# Live reload development server
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --open-browser --reload-delay 1

# Generate API documentation
apidoc:
	sphinx-apidoc -o $(SOURCEDIR) ../src/innovate --force --module-first

# Check external links
linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

# Build PDF documentation
pdf:
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex"
	cd $(BUILDDIR)/latex && make all-pdf

# Build EPUB documentation  
epub:
	$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub"

# Coverage check
coverage:
	$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage"
	@echo
	@echo "Testing of coverage in the sources finished, look at the " \
		"results in $(BUILDDIR)/coverage/python.txt."

# Doctest - test code examples in documentation
doctest:
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest"

# Build all formats
all: clean html pdf epub

# Install documentation dependencies
install-deps:
	pip install sphinx sphinx-rtd-theme myst-parser nbsphinx
	pip install sphinx-autobuild sphinx-copybutton

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)