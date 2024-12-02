// Rolagem suave para links de âncoras
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Menu Hamburguer
const menuButton = document.getElementById('menuButton');
const menuItems = document.getElementById('menuItems');

// Controle de exibição do menu
let isMenuOpen = false;

menuButton.addEventListener('click', () => {
    isMenuOpen = !isMenuOpen;
    toggleMenu();
});

// Fechar menu ao clicar fora dele ou ao rolar a página
document.addEventListener('click', (e) => {
    if (!menuButton.contains(e.target) && !menuItems.contains(e.target)) {
        isMenuOpen = false;
        toggleMenu();
    }
});

window.addEventListener('scroll', () => {
    isMenuOpen = false;
    toggleMenu();
});

// Função para exibir/ocultar menu
function toggleMenu() {
    if (isMenuOpen) {
        menuItems.classList.add('show');
    } else {
        menuItems.classList.remove('show');
    }
}

// Exibir tópicos ao passar o mouse (sem travar o menu)
menuItems.addEventListener('mouseover', () => {
    if (!isMenuOpen) {
        menuItems.classList.add('show');
    }
});

menuItems.addEventListener('mouseout', () => {
    if (!isMenuOpen) {
        menuItems.classList.remove('show');
    }
});
