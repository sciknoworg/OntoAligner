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
        // githubLink.innerHTML = '<i class="fab fa-github"></i> Github';
        githubLink.innerHTML = '<img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"  style="width: 25px; height: 25px; vertical-align: middle; margin-right: -1px; margin-bottom: 5px;"/> Github';

    }

    const pypiLink = document.querySelector('a[href="https://pypi.org/project/OntoAligner/"]');
    if (pypiLink) {
        // pypiLink.innerHTML = '<i class="fab fa-python"></i> Pypi';
        pypiLink.innerHTML = '<img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png" style="width: 18px; height: 18px; vertical-align: middle; margin-right: 3px;"/> Pypi';
    }
});
