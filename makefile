doctests:
	pytest --doctest-modules agsutil/ -W ignore

doctestsaccept:
	pytest --doctest-modules agsutil/ -W ignore --accept

mkdocs_serve:
	cp README.md docs/index.md & mkdocs serve --livereload