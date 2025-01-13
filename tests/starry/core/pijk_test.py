from jaxoplanet.starry.core.polynomials import Pijk


def test_n2ijk():
    assert Pijk.n2ijk(5) == (1, 0, 1)
    assert Pijk.n2ijk(8) == (0, 2, 0)


def test_additon():
    a = Pijk({(0, 0, 0): 1, (1, 0, 0): 2})
    b = Pijk({(0, 0, 0): 3, (1, 1, 0): 4})
    c = a + b
    assert c.data == {(0, 0, 0): 4, (1, 0, 0): 2, (1, 1, 0): 4}
