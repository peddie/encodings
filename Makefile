all: encodings.html encodings-text.html

COMMAND=pandoc --self-contained -V slidy-url=slidy -t slidy

%.html : %.md
	$(COMMAND) -o $*.html $*.md
