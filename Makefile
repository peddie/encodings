all: encodings.html encodings-text.html

COMMAND=pandoc --self-contained -V slidy-url=slidy -t slidy+simple_tables+table_captions

%.html : %.md
	$(COMMAND) -o $*.html $*.md
