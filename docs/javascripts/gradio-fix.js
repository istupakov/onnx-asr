let isInitialPage = true;

document$.subscribe(function () {
  if (isInitialPage) {
    isInitialPage = false;
    return;
  }

  if (document.querySelector("gradio-app")) {
    window.location.reload();
  }
});
