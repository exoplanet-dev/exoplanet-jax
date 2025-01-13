# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.10", "3.11", "3.12"]


@nox.session(python=ALL_PYTHON_VS)
def test(session):
    session.install(".[test,test-math]")
    session.run("pytest", "-n", "auto", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def test_x64(session):
    session.install(".[test,test-math]")
    env = {"JAX_ENABLE_X64": "1"}
    session.run("pytest", "-n", "auto", *session.posargs, env=env)


@nox.session(python=["3.10"])
def comparison(session):
    session.install(".[test,comparison]", "numpy<1.22")
    session.run("python", "-c", "import starry")
    session.run("python", "-c", "import theano")
    if session.posargs:
        args = session.posargs
    else:
        args = ("tests/starry",)
    session.run(
        "pytest",
        "-n",
        "auto",
        *args,
        env={"JAX_ENABLE_X64": "True"},
    )


@nox.session
def docs(session):
    session.install(".[docs]")
    with session.chdir("docs"):
        session.run(
            "python",
            "-m",
            "sphinx",
            "-T",
            "-E",
            "-W",
            "--keep-going",
            "-b",
            "dirhtml",
            "-d",
            "_build/doctrees",
            "-D",
            "language=en",
            ".",
            "_build/dirhtml",
        )
