# Main Makefile for the project
# all: Compile the LaTex document
# clean: Remove all generated files and directories
# cleanall: Remove all generated files, directories, and the final PDF

all:
		cd notes; make
		#cd docs; make

clean:
		rm -rf .pytest_cache astropartnotes/__pycache__ tests/__pycache__
		cd notes; make clean
		#cd docs; make clean

cleanall:
		rm -rf .pytest_cache astropartnotes/__pycache__ tests/__pycache__
		cd notes; make cleanall
		#cd docs; make cleanall
