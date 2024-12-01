document.addEventListener('DOMContentLoaded', function () {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach((item, index) => {
        if (index < 4) {
            item.style.display = 'none';
        }
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const githubLink = document.querySelector('a[href="https://github.com/sciknoworg/OntoAligner"]');
    if (githubLink) {
        githubLink.innerHTML = '<i class="fab fa-github"></i> Github';
    }

    const pypiLink = document.querySelector('a[href="https://pypi.org/project/OntoAligner/"]');
    if (pypiLink) {
        pypiLink.innerHTML = '<i class="fab fa-python"></i> Pypi';
    }
});
