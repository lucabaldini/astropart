DOC := astropart


all:
	pdflatex $(DOC)
	#makeindex $(DOC)
	bibtex $(DOC)
	pdflatex $(DOC)
	pdflatex $(DOC)

clean:
	rm -f *.{aux,bbl,blg,log,out,mw,toc,~}

cleanall: clean
	rm -f $(DOC).pdf
