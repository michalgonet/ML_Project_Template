from setuptools import find_packages, setup

E_DOT = '-e .'


def get_requirements(file_path: str) -> list[str]:
    """
    this function will return the list of requirements
    """
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [r.replace("\n", "") for r in requirements]
        if E_DOT in requirements:
            requirements.remove(E_DOT)

    return requirements


setup(
    name='AnyML',
    version='0.0.1',
    author='Michal Gonet',
    author_email='michal.gonet.mail@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
