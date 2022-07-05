from distutils.core import setup

setup(
    name='robustness',
    version='0.1',
    description='Robustness check tool for semantic segmentation',
    long_description=open('README.md').read(),
    url='',
    author="Andreas Bär and Edgard Moreira Minete",
    author_email="andreas.baer@tu-bs.de",
    licence='MIT',
    packages=['robustness/attacks', 'robustness/corruptions', 'robustness/helper'],
    package_data={
            "robustness/corruptions/frost": [
                "frost1.png", "frost2.png", "frost3.png",
                "frost4.jpg", "frost5.jpg", "frost6.jpg"
            ],
        },
)
