clean:
		rm -rf .pytest_cache astropartnotes/__pycache__ tests/__pycache__
		cd latex; make clean
		cd docs; make clean
