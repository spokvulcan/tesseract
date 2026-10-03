// Runs in Safari, on the page being shared: hands "Read in Tesseract" the
// page's HTML, so its readable text is found without the app ever fetching
// the page. It reads a copy, so the page you see doesn't change. Sections a
// page collapses but lets you find in it (hidden="until-found", as on
// Wikipedia's mobile site) are part of the text.
var ExtensionPreprocessingJS = {
    run: function (parameters) {
        var copy = document.documentElement.cloneNode(true);
        copy.querySelectorAll("[hidden='until-found']").forEach(function (element) {
            element.removeAttribute("hidden");
        });
        // Scripts and styles aren't text; a page's JSON-LD metadata (its
        // headline) stays for the title.
        copy.querySelectorAll("script:not([type='application/ld+json']), style, noscript, template")
            .forEach(function (element) {
                element.remove();
            });
        parameters.completionFunction({
            title: document.title,
            url: document.URL,
            html: copy.outerHTML
        });
    },
    finalize: function () {}
};
