# Minimal makefile for Sphinx documentation
# Ensure indentation is done via tabs: check with "cat -e -t -v Makefile"
# where tabs are "^I" and line endings are "$"
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ChaosMagPy
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/classes/
	rm -rf $(SOURCEDIR)/functions/
	rm -rf $(SOURCEDIR)/gallery/

html-noplot:
	@$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

.PHONY: help Makefile clean html-noplot

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
