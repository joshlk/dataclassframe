
# Build and push to PyPi
Requires: `pip install twine`
Don't forget to increment version number

Bump version (major, minor or patch):

```shell script
bump2version micro
```

Commit to master and get build-id from Azure Pipeline URL:
https://dev.azure.com/josh0282/k-means-constrained/_build?definitionId=1

Download distributions (artifacts)

```shell script
make download-dists ID=$BUILD_ID
```

Upload to test PyPi

```shell script
make check-dist
make test-pypi
```

Activate virtual env (might need to `make venv-create`)

```shell script
source k-means-env/bin/activate
```

Test install (in virtual env):

```shell script
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple k-means-constrained
```

Then push to real PyPI:

```shell script
make pypi-upload
```
