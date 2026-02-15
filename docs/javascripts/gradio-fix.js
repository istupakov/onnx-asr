document$.subscribe(function () {
  const hasGradio = document.querySelector("gradio-app");
  const hasReloaded = sessionStorage.getItem("gradio-manual-reload");

  if (hasGradio) {
    if (!hasReloaded) {
      hasGradio.remove();
      sessionStorage.setItem("gradio-manual-reload", "true");
      console.log("Gradio page detected: performing one-time hard reload.");
      window.location.reload();
    }
  } else {
    sessionStorage.removeItem("gradio-manual-reload");
  }
});
