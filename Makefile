test: python3 -m unittest discover tests

test2: coverage run --source=<package> setup.py test

autre_test: python -m unittest -v -b

upload: python3 setup.py sdist bdist_wheel
			twine upload --repository-url https://test.pypi.org/legacy/ dist/*
