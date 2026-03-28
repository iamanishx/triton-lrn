mental model:

- big tensor problem
- spliting it into tiles/chunks
- the grid says how many Triton program instances to launch
- each program instance handles one tile/chunk

```
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
```
computes how many Triton program instances to launch.

- `n_elements` = total vector length
- `meta["BLOCK_SIZE"]` = how many elements one program instance handles
- `triton.cdiv(a, b)` = ceil division, meaning `ceil(a / b)`

So:

- if `n_elements = 1024`
- and `BLOCK_SIZE = 1024`

then:

- `triton.cdiv(1024, 1024) = 1`
- grid becomes `(1,)