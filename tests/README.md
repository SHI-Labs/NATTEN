# Unit tests

Simply run:
```
make test
```
or:
```
python -m unittest discover -v -s ./tests
```

Presently each CPP module is tested through `gradcheck`.
If CUDA is available, CPU and CUDA modules are tested against each other (`allclose` test).
