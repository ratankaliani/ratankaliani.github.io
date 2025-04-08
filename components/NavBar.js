class NavBar extends HTMLElement {
    connectedCallback() {
        this.innerHTML = `
        <div class="header-nav">
          <a href="index.html">Home</a>
          <a href="blog.html">Blog</a>
          <a href="https://curius.app/ratan-kaliani">Curiosities</a>
        </div>
      `;
    }
}

customElements.define('nav-bar', NavBar);